# dispatch_strategies.py 修正パッチ

## 修正箇所1: 状態次元のハードコード問題（最重要）

### 場所: 1116-1120行目

```python
# ===== 修正前 =====
if self.compact_mode:
    print(f"  コンパクトモード有効: Top-{self.top_k}選択")
    self.action_dim = self.top_k
    # state_dim = severity_features(2) + (top_k × features_per_ambulance(3)) + global_features(5)
    self.state_dim = 2 + (self.top_k * 3) + 5  # ← 古い37次元の計算式

# ===== 修正後 =====
if self.compact_mode:
    print(f"  コンパクトモード有効: Top-{self.top_k}選択")
    self.action_dim = self.top_k
    # 状態次元はCompactStateEncoderから取得（46次元）
    # 候補隊(top_k×4) + グローバル(5) + 傷病度(1) = 46
    self.state_dim = None  # 後でStateEncoderから取得
```

### 理由
- 学習時: 46次元（候補隊40 + グローバル5 + 傷病度1）
- テスト時: 37次元（古い計算式）
- この不一致がRuntimeErrorの原因


## 修正箇所2: ID対応表のパス問題

### 場所: 30行目と1216行目

```python
# ===== 修正前（30行目）=====
fix_dir = PROJECT_ROOT / ".05_Ambulance_RL_fix_from_v11_3"

# ===== 修正後 =====
# fix_dirを使用しない、またはPROJECT_ROOTを使用
fix_dir = PROJECT_ROOT
```

```python
# ===== 修正前（1216行目）=====
mapping_file = fix_dir / "id_mapping_proposal.json"

# ===== 修正後（2つの選択肢）=====
# 選択肢A: PROJECT_ROOTを使用
mapping_file = PROJECT_ROOT / "id_mapping_proposal.json"

# 選択肢B: コンパクトモードではID対応表を使用しない
# （コンパクトモードではTop-K選択後にインデックス0-9を使うため、
#   192台全体へのマッピングは不要）
```

### 重要な注意
コンパクトモード（46次元設計）では、ID対応表は**本質的に不要**です：
- Top-K救急車を移動時間順にソート
- PPOは0-9のインデックスで選択
- 選択後、Top-Kリストから直接救急車を取得

したがって、ID対応表のエラーは無視しても動作します（ログに警告が出るだけ）。


## 修正箇所3: _load_id_mapping メソッドの改善（オプション）

### 場所: 1214-1251行目

```python
# ===== 修正後 =====
def _load_id_mapping(self):
    """Phase 1で生成されたID対応表を読み込む"""
    # コンパクトモードではID対応表は不要
    # （Top-K選択でインデックス0-9を使用するため）
    if self.compact_mode:
        print("  コンパクトモード: ID対応表は不要（Top-Kインデックス使用）")
        self.id_mapping_loaded = False
        return
    
    # 従来モード用のID対応表読み込み
    mapping_file = PROJECT_ROOT / "id_mapping_proposal.json"
    
    if not mapping_file.exists():
        print("  ⚠️ 警告: id_mapping_proposal.json が見つかりません")
        print("  フォールバックモードで動作します")
        self.id_mapping_loaded = False
        return
    
    try:
        with open(mapping_file, 'r', encoding='utf-8') as f:
            mapping_data = json.load(f)
        
        self.validation_id_to_action = mapping_data.get('string_to_int', {})
        int_to_string = mapping_data.get('int_to_string', {})
        self.action_to_validation_id = {int(k): v for k, v in int_to_string.items()}
        self.id_mapping_loaded = True
        
        print(f"  ✓ ID対応表読み込み完了: {len(self.validation_id_to_action)}件")
        
    except Exception as e:
        print(f"  ⚠️ ID対応表の読み込みエラー: {e}")
        print("  フォールバックモードで動作します")
        self.id_mapping_loaded = False
```


## 完全な修正手順

### 手順1: 1119-1120行目を修正

Cursorで`dispatch_strategies.py`を開き、1119-1120行目を以下に変更：

```python
            # state_dim = severity_features(2) + (top_k × features_per_ambulance(3)) + global_features(5)
            self.state_dim = 2 + (self.top_k * 3) + 5
```
↓
```python
            # 状態次元はCompactStateEncoderから取得（46次元）
            self.state_dim = None  # 後でStateEncoderから取得
```

### 手順2: 30行目のfix_dirを修正（オプション）

```python
fix_dir = PROJECT_ROOT / ".05_Ambulance_RL_fix_from_v11_3"
```
↓
```python
fix_dir = PROJECT_ROOT  # 古いパスを使用しない
```

### 手順3: テスト実行

```bash
python baseline_comparison.py
```

期待される出力：
```
CompactStateEncoder初期化(46次元版):
  状態次元: 46
CompactStateEncoderを使用 (Top-10)
状態次元: 46  ← 46になるはず
PPOエージェント初期化
  状態次元: 46
  行動次元: 10
```
