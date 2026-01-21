# ============================================================================
# PPO学習ロジック検証ガイド
# ============================================================================
# 
# 問題: PPO（K10, K20モデル）が傷病度に関わらず直近隊と同じ配車をしている
# 目標: 原因を特定し、軽症系でカバレッジを考慮した配車を実現する
#
# ============================================================================

## 1. 検証すべき項目

### 1.1 ハイブリッドモードの動作確認

```python
# ems_environment.py の該当箇所を確認
if self.hybrid_mode:
    severity = current_incident.get('severity', '')
    
    if severity in self.severe_conditions:
        # 重症系：直近隊運用（これはOK）
        action = get_closest_action()
        reward = 0.0  # 学習対象外
    else:
        # 軽症系：PPOで学習（ここが問題）
        action = ppo_action
        reward = calculate_reward()

# 確認ポイント:
# - self.hybrid_mode が True になっているか
# - self.severe_conditions に正しい値が設定されているか
# - 軽症系で ppo_action が使われているか
```

### 1.2 PPOの行動選択の確認

```python
# PPOの select_action メソッドで以下をログ出力
def select_action(self, state, action_mask, deterministic=True):
    with torch.no_grad():
        logits = self.actor(state)
        
        # ★確認ポイント1: logitsの分布
        print(f"Action logits: min={logits.min():.3f}, max={logits.max():.3f}")
        print(f"Action logits std: {logits.std():.3f}")
        
        # マスク適用
        masked_logits = logits.clone()
        masked_logits[~action_mask] = -1e9
        
        # 確率分布を取得
        probs = F.softmax(masked_logits, dim=-1)
        
        # ★確認ポイント2: 確率分布の偏り
        top5_probs, top5_indices = torch.topk(probs, 5)
        print(f"Top 5 actions: {top5_indices.tolist()}")
        print(f"Top 5 probs: {top5_probs.tolist()}")
        
        # ★確認ポイント3: 直近隊との比較
        closest_action = self._get_closest_action(...)
        print(f"Closest action: {closest_action}")
        print(f"Selected action: {action}")
        print(f"Match: {action == closest_action}")
        
        return action, log_prob, value
```

### 1.3 報酬計算の確認

```python
# 報酬計算で以下をログ出力
def _calculate_reward(self, dispatch_result):
    severity = dispatch_result['severity']
    response_time = dispatch_result['response_time_minutes']
    
    # カバレッジ計算
    coverage_before = self.current_coverage
    coverage_after = self._predict_coverage_after_dispatch(...)
    coverage_loss = coverage_before - coverage_after
    
    # ★確認ポイント: 各報酬成分の値
    print(f"Severity: {severity}")
    print(f"Response time: {response_time:.2f} min")
    print(f"Coverage loss: {coverage_loss:.4f}")
    
    # 報酬計算
    time_reward = -response_time * self.time_weight
    coverage_reward = -coverage_loss * self.coverage_weight
    total_reward = time_reward + coverage_reward
    
    print(f"Time reward: {time_reward:.3f}")
    print(f"Coverage reward: {coverage_reward:.3f}")
    print(f"Total reward: {total_reward:.3f}")
    
    return total_reward
```

## 2. 確認手順

### Step 1: 基本動作の確認

```bash
# 1エピソードだけ実行してログを確認
python train_ppo.py --config config_verification_exploration.yaml --n_episodes 1 --debug
```

確認項目:
- [ ] ハイブリッドモードが有効か
- [ ] 重症系で直近隊が選択されているか
- [ ] 軽症系でPPOのactionが使われているか

### Step 2: PPO行動分布の確認

```bash
# PPOの行動分布をログ出力
python train_ppo.py --config config_verification_exploration.yaml --log_action_distribution
```

確認項目:
- [ ] PPOの出力確率が一様でないか（学習していれば偏りがあるはず）
- [ ] 直近隊以外のactionにも確率が割り当てられているか
- [ ] entropy係数を上げた場合に分布が広がるか

### Step 3: 報酬計算の確認

```bash
# 報酬の各成分をログ出力
python train_ppo.py --config config_verification_exploration.yaml --log_reward_components
```

確認項目:
- [ ] カバレッジ報酬が計算されているか
- [ ] カバレッジ報酬の値が応答時間報酬と比較して十分大きいか
- [ ] 直近隊以外を選んだ場合にカバレッジ報酬が高くなるか

### Step 4: 学習進行の確認

```bash
# 100エピソードごとに直近隊との一致率を確認
python train_ppo.py --config config_verification_exploration.yaml --log_closest_agreement
```

確認項目:
- [ ] 学習が進むにつれて直近隊との一致率が変化するか
- [ ] entropy係数が高い場合に一致率が下がるか
- [ ] カバレッジ報酬を強化した場合に一致率が下がるか

## 3. 予想される問題と解決策

### 問題1: カバレッジ報酬が計算されていない

症状:
- coverage_reward が常に 0
- または coverage_loss が常に 0

解決策:
```python
# カバレッジ計算が実装されているか確認
def _calculate_coverage_loss(self, selected_ambulance, available_ambulances, ...):
    # この関数が呼ばれているか確認
    # 戻り値が 0 でないか確認
```

### 問題2: 応答時間報酬が支配的

症状:
- time_reward >> coverage_reward
- 直近隊が常に最高報酬

解決策:
- coverage_weight を 10倍以上に増加
- time_weight を 1/10 以下に削減
- 軽症系で応答時間が13分以内ならペナルティをほぼ0に

### 問題3: 探索が行われていない

症状:
- action分布がほぼ決定的（1つのactionに99%以上）
- entropy が 0 に近い

解決策:
- entropy_coef を 0.05 以上に増加
- 学習初期に確率的選択を強制
- temperature を導入して探索を促進

### 問題4: 学習が収束していない

症状:
- 報酬が変動し続ける
- actor_loss が減少しない

解決策:
- 学習率を下げる
- batch_size を増やす
- 報酬のスケールを調整

## 4. 期待される正常動作

### 学習が正常な場合:

```
Episode 0-100:
  - 直近隊との一致率: 80-90%
  - Action entropy: 高い
  - 探索的な行動選択

Episode 100-500:
  - 直近隊との一致率: 60-70%
  - カバレッジ報酬が上昇
  - 一部の状況で直近隊以外を選択

Episode 500-1000:
  - 直近隊との一致率: 40-60%
  - カバレッジ維持が改善
  - 軽症系で戦略的な配車

Episode 1000+:
  - 安定した配車パターン
  - 重症RTは傷病度考慮と同等
  - 全体RTは改善
```

### 学習が異常な場合（現状）:

```
全エピソード:
  - 直近隊との一致率: 95%以上
  - Action entropy: 低い
  - 常に同じ行動を選択
  → PPOを使う意味がない
```

## 5. 実装修正が必要な場合

上記の検証で問題が見つかった場合、以下のファイルを修正:

1. `ems_environment.py`
   - 報酬計算のロジック
   - カバレッジ計算の実装

2. `ppo_agent.py`
   - 行動選択のロジック
   - 探索の制御

3. `state_encoder.py`
   - カバレッジ情報の状態への埋め込み

4. `config.yaml`
   - 報酬の重み設定
   - 探索パラメータ
