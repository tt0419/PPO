# コンパクト状態空間 実装仕様書

**目的**: 状態空間を999次元から37次元に削減し、行動空間を192次元から10次元に削減する  
**対象者**: Cursor経由でClaude Opus 4.5に実装を依頼するための仕様書

---

## 変更対象ファイル一覧

| ファイル | 変更種別 | 変更規模 |
|----------|----------|----------|
| `state_encoder.py` | **全面改修** | 大 |
| `ems_environment.py` | **部分修正** | 中 |
| `config.yaml` | **設定追加** | 小 |
| `ppo_agent.py` | 変更不要 | - |
| `reward_designer.py` | 変更不要 | - |
| `trainer.py` | 変更不要 | - |
| `train_ppo.py` | 変更不要 | - |

---

## 1. state_encoder.py の変更

### 1.1 変更概要

現在のStateEncoderクラスを維持しつつ、新しい`CompactStateEncoder`クラスを追加する。
configの設定に応じて、どちらのエンコーダーを使用するか切り替える。

### 1.2 新しいクラス: CompactStateEncoder

```python
class CompactStateEncoder:
    """
    コンパクトな状態エンコーダー
    999次元 → 37次元に削減
    
    状態ベクトル構造（37次元）:
    [0-1]   傷病度（2次元）: is_severe, is_mild
    [2-31]  Top-10救急車（30次元）: 各3特徴量 × 10台
    [32-36] グローバル統計（5次元）
    """
```

### 1.3 実装すべきメソッド

#### `__init__(self, config, top_k=10, travel_time_matrix=None, grid_mapping=None)`

**引数**:
- `config`: 設定辞書
- `top_k`: 考慮する上位救急車数（デフォルト10）
- `travel_time_matrix`: 移動時間行列（numpy配列）
- `grid_mapping`: H3インデックス→行列インデックスの辞書

**初期化すべき属性**:
```python
self.top_k = top_k
self.features_per_ambulance = 3  # 移動時間, カバレッジ損失, 署距離
self.severity_features = 2       # is_severe, is_mild
self.global_features = 5         # 利用可能数, カバレッジ率, 時刻, 6分以内数, 平均移動時間
self.state_dim = self.severity_features + (self.top_k * self.features_per_ambulance) + self.global_features
# state_dim = 2 + 30 + 5 = 37
```

#### `encode_state(self, state_dict, grid_mapping=None) -> np.ndarray`

**引数**:
- `state_dict`: 環境の状態辞書（以下のキーを含む）
  - `ambulances`: 救急車状態の辞書 `{amb_id: {'current_h3': str, 'status': str, 'station_h3': str, ...}}`
  - `pending_call`: 現在の事案 `{'h3_index': str, 'severity': str, ...}` または None
  - `time_of_day`: 時刻（0-24の浮動小数点）

**戻り値**: 37次元のnumpy配列（float32）

**処理フロー**:
```python
features = np.zeros(self.state_dim, dtype=np.float32)

# 1. 傷病度をエンコード（2次元）
severity = state_dict.get('pending_call', {}).get('severity', '')
features[0] = 1.0 if severity in ['重症', '重篤', '死亡'] else 0.0  # is_severe
features[1] = 1.0 if severity in ['軽症', '中等症'] else 0.0        # is_mild

# 2. Top-K救急車の情報を取得してエンコード（30次元）
top_k_ambulances = self._get_top_k_ambulances(state_dict['ambulances'], state_dict.get('pending_call'))
for i, amb_info in enumerate(top_k_ambulances):
    base_idx = 2 + i * 3
    features[base_idx + 0] = amb_info['travel_time_normalized']    # 移動時間（0-1）
    features[base_idx + 1] = amb_info['coverage_loss']             # カバレッジ損失（0-1）
    features[base_idx + 2] = amb_info['station_distance_normalized'] # 署距離（0-1）

# 3. グローバル統計をエンコード（5次元）
global_idx = 2 + self.top_k * 3  # = 32
features[global_idx + 0] = available_count / 192.0          # 利用可能救急車数
features[global_idx + 1] = coverage_rate                     # カバレッジ率
features[global_idx + 2] = time_of_day / 24.0               # 時刻
features[global_idx + 3] = within_6min_count / self.top_k   # 6分以内到達可能数
features[global_idx + 4] = avg_travel_time / 30.0           # 平均移動時間（分）

return features
```

#### `_get_top_k_ambulances(self, ambulances, incident) -> List[Dict]`

**処理**:
1. 利用可能な救急車（status == 'available'）のみを抽出
2. 各救急車について、事案への移動時間を計算
3. 移動時間順にソート
4. 上位K台を返す
5. K台に満たない場合はダミーデータで埋める

**戻り値の構造**:
```python
[
    {
        'amb_id': int,                      # 救急車ID
        'travel_time': float,               # 移動時間（秒）
        'travel_time_normalized': float,    # 正規化移動時間（0-1、30分を1.0とする）
        'coverage_loss': float,             # カバレッジ損失（0-1）
        'station_distance': float,          # 署からの距離（km）
        'station_distance_normalized': float # 正規化署距離（0-1、10kmを1.0とする）
    },
    ...
]
```

#### `_calculate_coverage_loss(self, amb_id, ambulances, incident_h3) -> float`

**処理**:
1. 指定された救急車を除いた残りの救急車リストを作成
2. 残りの救急車でカバーできるグリッド数を計算
3. カバレッジ損失 = 1 - (残りカバレッジ / 現在カバレッジ)

**注意**: 計算コストが高い場合は簡易版を実装
```python
# 簡易版: 近隣の利用可能救急車数に基づく
nearby_count = sum(1 for a in ambulances.values() 
                   if a['status'] == 'available' 
                   and self._is_nearby(a, ambulances[amb_id]))
return 1.0 / (nearby_count + 1)  # 近隣が多いほど損失小
```

#### `get_top_k_ambulance_ids(self, ambulances, incident) -> List[int]`

**目的**: Top-K救急車のIDリストを返す（ems_environment.pyで使用）

**戻り値**: `[amb_id_1, amb_id_2, ..., amb_id_k]`（移動時間順）

### 1.4 プロパティ

```python
@property
def state_dim(self) -> int:
    """状態ベクトルの次元数を返す"""
    return self.severity_features + (self.top_k * self.features_per_ambulance) + self.global_features
```

### 1.5 既存のStateEncoderとの共存

ファイルの構造:
```python
# state_encoder.py

class StateEncoder:
    """既存のエンコーダー（999次元）- 変更なし"""
    ...

class CompactStateEncoder:
    """新しいコンパクトエンコーダー（37次元）"""
    ...

def create_state_encoder(config, **kwargs):
    """設定に応じてエンコーダーを作成するファクトリ関数"""
    encoding_mode = config.get('state_encoding', {}).get('mode', 'full')
    
    if encoding_mode == 'compact':
        top_k = config.get('state_encoding', {}).get('top_k', 10)
        return CompactStateEncoder(config, top_k=top_k, **kwargs)
    else:
        return StateEncoder(config, **kwargs)
```

---

## 2. ems_environment.py の変更

### 2.1 変更箇所一覧

| 箇所 | 変更内容 |
|------|----------|
| `__init__` | action_dimとstate_encoderの初期化変更 |
| 新規属性 | `current_top_k_ids`の追加 |
| `_get_observation` | Top-K IDの更新処理追加 |
| `step` | actionをTop-Kインデックスとして解釈 |
| `get_action_mask` | Top-K用のマスク生成 |
| `get_optimal_action` | Top-Kモード対応 |

### 2.2 __init__ の変更

**変更前（L176-199）**:
```python
# 状態・行動空間の次元
self.action_dim = len(self.ambulance_data)  # 実際の救急車数

from .state_encoder import StateEncoder
self.state_encoder = StateEncoder(
    config=self.config,
    max_ambulances=self.action_dim,
    travel_time_matrix=response_matrix,
    grid_mapping=self.grid_mapping
)

self.state_dim = self.state_encoder.state_dim
```

**変更後**:
```python
# コンパクトモードの設定を読み込み
state_encoding_config = self.config.get('state_encoding', {})
self.compact_mode = state_encoding_config.get('mode', 'full') == 'compact'
self.top_k = state_encoding_config.get('top_k', 10)

if self.compact_mode:
    # コンパクトモード: action_dim = top_k
    self.action_dim = self.top_k
    
    from .state_encoder import CompactStateEncoder
    self.state_encoder = CompactStateEncoder(
        config=self.config,
        top_k=self.top_k,
        travel_time_matrix=response_matrix,
        grid_mapping=self.grid_mapping
    )
    
    # Top-K救急車のIDを保持するリスト
    self.current_top_k_ids = []
    
    print(f"コンパクトモード有効: Top-{self.top_k}")
else:
    # 従来モード: action_dim = 全救急車数
    self.action_dim = len(self.ambulance_data)
    
    from .state_encoder import StateEncoder
    self.state_encoder = StateEncoder(
        config=self.config,
        max_ambulances=self.action_dim,
        travel_time_matrix=response_matrix,
        grid_mapping=self.grid_mapping
    )
    
    self.current_top_k_ids = None  # 従来モードでは使用しない

self.state_dim = self.state_encoder.state_dim
```

### 2.3 _get_observation の変更

**現在の実装を探して、以下の処理を追加**:

```python
def _get_observation(self) -> np.ndarray:
    """状態を取得"""
    state_dict = self._build_state_dict()
    
    # ★★★ コンパクトモード: Top-K IDを更新 ★★★
    if self.compact_mode:
        self.current_top_k_ids = self.state_encoder.get_top_k_ambulance_ids(
            state_dict['ambulances'],
            state_dict.get('pending_call')
        )
    
    return self.state_encoder.encode_state(state_dict)
```

### 2.4 step の変更

**変更箇所**: actionを実際の救急車IDに変換する処理を追加

```python
def step(self, action: int) -> StepResult:
    """行動を実行"""
    
    # ★★★ コンパクトモード: actionをTop-K内のインデックスとして解釈 ★★★
    if self.compact_mode:
        if self.current_top_k_ids and action < len(self.current_top_k_ids):
            actual_ambulance_id = self.current_top_k_ids[action]
        else:
            # フォールバック: Top-1を選択
            actual_ambulance_id = self.current_top_k_ids[0] if self.current_top_k_ids else 0
            print(f"警告: action={action} がTop-K範囲外。Top-1を選択。")
    else:
        # 従来モード: actionがそのまま救急車ID
        actual_ambulance_id = action
    
    # 以降、actual_ambulance_id を使用して配車処理
    # （既存のactionをactual_ambulance_idに置き換え）
    ...
```

**注意**: 既存のstep()メソッド内で`action`を使っている箇所を`actual_ambulance_id`に置き換える必要がある。特に以下の箇所:
- `self._dispatch_ambulance(action, ...)` → `self._dispatch_ambulance(actual_ambulance_id, ...)`
- ログ出力での行動番号表示

### 2.5 get_action_mask の変更

**変更前（L2151-2222）**: 192次元のマスク

**変更後**:
```python
def get_action_mask(self) -> np.ndarray:
    """利用可能な行動のマスクを取得"""
    
    if self.compact_mode:
        # ★★★ コンパクトモード: Top-K用のマスク ★★★
        mask = np.ones(self.action_dim, dtype=bool)  # action_dim = top_k
        
        # Top-Kに満たない場合は残りを無効化
        if self.current_top_k_ids:
            valid_count = len(self.current_top_k_ids)
            if valid_count < self.action_dim:
                mask[valid_count:] = False
        else:
            # Top-Kが未設定の場合は全て有効（初期状態）
            pass
        
        return mask
    
    else:
        # ★★★ 従来モード: 既存のロジック ★★★
        mask = np.zeros(self.action_dim, dtype=bool)
        
        for amb_id, amb_state in self.ambulance_states.items():
            if amb_id < self.action_dim and amb_state['status'] == 'available':
                mask[amb_id] = True
        
        # 以下、既存のcoverage_awareモードのフィルタリング処理...
        ...
        
        return mask
```

### 2.6 get_optimal_action の変更

**目的**: 教師あり学習で使用する最適行動を返す

**変更後**:
```python
def get_optimal_action(self) -> Optional[int]:
    """最適な行動を返す（直近隊 = 最短移動時間）"""
    
    if self.compact_mode:
        # ★★★ コンパクトモード: 常にaction=0（Top-1）が最適 ★★★
        return 0
    
    else:
        # ★★★ 従来モード: 既存のロジック ★★★
        # ... 既存の実装 ...
```

### 2.7 新規メソッド: get_actual_ambulance_id（オプション）

**目的**: デバッグやログ出力用に、actionから実際の救急車IDを取得

```python
def get_actual_ambulance_id(self, action: int) -> int:
    """actionから実際の救急車IDを取得"""
    if self.compact_mode:
        if self.current_top_k_ids and action < len(self.current_top_k_ids):
            return self.current_top_k_ids[action]
        return self.current_top_k_ids[0] if self.current_top_k_ids else 0
    else:
        return action
```

---

## 3. config.yaml の変更

### 3.1 追加する設定項目

```yaml
# 状態空間エンコーディング設定
state_encoding:
  # モード: 'full'（従来の999次元）または 'compact'（37次元）
  mode: 'compact'
  
  # コンパクトモードの設定
  top_k: 10  # 考慮する上位救急車数
  
  # 各救急車の特徴量
  ambulance_features:
    - travel_time       # 移動時間（必須）
    - coverage_loss     # カバレッジ損失（必須）
    - station_distance  # 署からの距離（必須）
  
  # カバレッジ損失の計算設定
  coverage_loss:
    method: 'simple'  # 'simple' または 'full'
    # simple: 近隣救急車数に基づく簡易計算
    # full: 完全なカバレッジ計算（計算コスト高）
  
  # 正規化設定
  normalization:
    max_travel_time_minutes: 30   # これを1.0とする
    max_station_distance_km: 10   # これを1.0とする
```

### 3.2 既存設定との互換性

`mode: 'full'` を設定すれば従来の999次元モードで動作する。
設定がない場合のデフォルトは `'full'` とし、後方互換性を維持する。

---

## 4. 変更不要なファイルの確認

### 4.1 ppo_agent.py

**理由**: state_dimとaction_dimは環境から渡される引数として受け取るため、変更不要。

```python
# train_ppo.py での初期化（変更なし）
agent = PPOAgent(
    state_dim=env.state_dim,    # 環境から取得（37または999）
    action_dim=env.action_dim,  # 環境から取得（10または192）
    config=config['ppo']
)
```

### 4.2 reward_designer.py

**理由**: 報酬計算は応答時間と傷病度に基づいており、状態空間・行動空間の変更とは独立している。

### 4.3 trainer.py

**理由**: 学習ループは環境とエージェントのインターフェースを使用しており、内部の次元数には依存しない。

### 4.4 train_ppo.py

**理由**: 環境とエージェントの初期化時にstate_dim/action_dimを取得するため、変更不要。

---

## 5. 実装の優先順位

### Phase 1: 最小限の実装（必須）

1. **state_encoder.py**: `CompactStateEncoder`クラスの実装
   - `__init__`
   - `encode_state`
   - `_get_top_k_ambulances`
   - `get_top_k_ambulance_ids`
   - `state_dim`プロパティ

2. **ems_environment.py**: 基本的な変更
   - `__init__`のaction_dim/state_encoder初期化
   - `_get_observation`のTop-K ID更新
   - `step`のaction→actual_ambulance_id変換
   - `get_action_mask`のコンパクトモード対応
   - `get_optimal_action`のコンパクトモード対応

3. **config.yaml**: 設定追加
   - `state_encoding`セクション

### Phase 2: 追加機能（推奨）

1. `_calculate_coverage_loss`の本格実装
2. デバッグ・ログ出力の強化
3. `create_state_encoder`ファクトリ関数

### Phase 3: 検証（必須）

1. 単体テスト
2. 学習実行テスト
3. 従来モードとの比較

---

## 6. テスト計画

### 6.1 単体テスト

```python
# test_compact_state_encoder.py

def test_state_dim():
    """状態次元が37であることを確認"""
    encoder = CompactStateEncoder(config, top_k=10)
    assert encoder.state_dim == 37

def test_encode_state_shape():
    """エンコード結果の形状を確認"""
    state = encoder.encode_state(state_dict)
    assert state.shape == (37,)
    assert state.dtype == np.float32

def test_top_k_ambulances():
    """Top-K救急車が正しく取得できることを確認"""
    top_k = encoder._get_top_k_ambulances(ambulances, incident)
    assert len(top_k) == 10
    # 移動時間順にソートされていることを確認
    for i in range(len(top_k) - 1):
        assert top_k[i]['travel_time'] <= top_k[i+1]['travel_time']

def test_top_k_ids():
    """Top-K IDリストの取得を確認"""
    ids = encoder.get_top_k_ambulance_ids(ambulances, incident)
    assert len(ids) == 10
    assert all(isinstance(id, int) for id in ids)
```

### 6.2 統合テスト

```python
# test_compact_mode_integration.py

def test_environment_initialization():
    """環境が正しく初期化されることを確認"""
    config['state_encoding'] = {'mode': 'compact', 'top_k': 10}
    env = EMSEnvironment(config_path)
    assert env.state_dim == 37
    assert env.action_dim == 10
    assert env.compact_mode == True

def test_step_with_compact_mode():
    """step()がコンパクトモードで正しく動作することを確認"""
    env.reset()
    action = 0  # Top-1を選択
    result = env.step(action)
    assert result.observation.shape == (37,)

def test_action_mask_compact():
    """行動マスクがTop-K用に生成されることを確認"""
    mask = env.get_action_mask()
    assert mask.shape == (10,)
```

---

## 7. 期待される結果

### 7.1 次元数の比較

| 項目 | 変更前 | 変更後 |
|------|--------|--------|
| 状態次元 | 999 | 37 |
| 行動次元 | 192 | 10 |
| 削減率 | - | 96% / 95% |

### 7.2 学習効率

- 8000エピソードで十分な学習が期待できる
- 学習曲線が早期に収束する見込み

### 7.3 性能

- 重症系: action=0（Top-1）を学習 → 直近隊と同等
- 軽症系: action>0も選択 → カバレッジ考慮を学習する可能性

---

## 8. 注意事項

### 8.1 互換性

- `mode: 'full'`設定で従来の999次元モードが動作することを確認
- 既存の学習済みモデルは使用不可（state_dim/action_dimが異なるため）

### 8.2 デバッグ

- Top-K IDリストが正しく更新されているか確認
- actionからactual_ambulance_idへの変換が正しいか確認
- 報酬計算が正しく行われているか確認

### 8.3 ログ出力

コンパクトモードでは以下の情報をログ出力することを推奨:
- Top-K救急車のID一覧
- 選択されたaction（Top-K内インデックス）
- 実際に配車された救急車ID

---

*作成日: 2024年12月*
*目的: Cursor/Claude Opus 4.5への実装依頼用仕様書*
