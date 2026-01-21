# ems_environment.py 変更ガイド

## 概要

コンパクトモード（37次元状態、10次元行動）をサポートするための変更。
既存の機能を維持しつつ、設定で切り替え可能にする。

---

## 変更箇所一覧

| 行番号（目安） | メソッド/箇所 | 変更内容 |
|---------------|---------------|----------|
| L176-199 | `__init__` | action_dim, state_encoderの初期化変更 |
| 新規 | 属性追加 | `compact_mode`, `top_k`, `current_top_k_ids` |
| - | `_get_observation` | Top-K ID更新処理の追加 |
| L850-956 | `step` | actionをTop-Kインデックスとして解釈 |
| L2151-2222 | `get_action_mask` | コンパクトモード対応 |
| L958-998 | `get_optimal_action` | コンパクトモード対応 |

---

## 1. __init__ の変更

### 変更箇所: L176-199付近

**変更前**:
```python
# 状態・行動空間の次元
self.action_dim = len(self.ambulance_data)  # 実際の救急車数

# ★★★【修正提案】★★★
# StateEncoderの初期化をここで行い、インスタンスをクラス変数として保持する
response_matrix = self.travel_time_matrices.get('response', None)
if response_matrix is None:
    print("警告: responseフェーズの移動時間行列が見つかりません。")

# StateEncoderを初期化して、self.state_encoderとして保持
from .state_encoder import StateEncoder
self.state_encoder = StateEncoder(
    config=self.config,
    max_ambulances=self.action_dim,
    travel_time_matrix=response_matrix,
    grid_mapping=self.grid_mapping
)

# StateEncoderインスタンスから状態次元を取得する
self.state_dim = self.state_encoder.state_dim
```

**変更後**:
```python
# ========== コンパクトモードの設定 ==========
state_encoding_config = self.config.get('state_encoding', {})
self.compact_mode = state_encoding_config.get('mode', 'full') == 'compact'
self.top_k = state_encoding_config.get('top_k', 10)

# 移動時間行列の取得
response_matrix = self.travel_time_matrices.get('response', None)
if response_matrix is None:
    print("警告: responseフェーズの移動時間行列が見つかりません。")

# ========== 状態・行動空間の次元設定 ==========
if self.compact_mode:
    # コンパクトモード: action_dim = top_k, state_dim = 37
    self.action_dim = self.top_k
    
    from .state_encoder import CompactStateEncoder
    self.state_encoder = CompactStateEncoder(
        config=self.config,
        top_k=self.top_k,
        travel_time_matrix=response_matrix,
        grid_mapping=self.grid_mapping
    )
    
    # Top-K救急車のIDを保持するリスト（step()で使用）
    self.current_top_k_ids = []
    
    print(f"★ コンパクトモード有効: Top-{self.top_k}")
    print(f"  状態次元: {self.state_encoder.state_dim}")
    print(f"  行動次元: {self.action_dim}")
else:
    # 従来モード: action_dim = 全救急車数, state_dim = 999
    self.action_dim = len(self.ambulance_data)
    
    from .state_encoder import StateEncoder
    self.state_encoder = StateEncoder(
        config=self.config,
        max_ambulances=self.action_dim,
        travel_time_matrix=response_matrix,
        grid_mapping=self.grid_mapping
    )
    
    # 従来モードではTop-K IDは使用しない
    self.current_top_k_ids = None

self.state_dim = self.state_encoder.state_dim

print(f"状態空間次元: {self.state_dim}")
print(f"行動空間次元: {self.action_dim}")
```

---

## 2. _get_observation の変更

### 現在のメソッドを探す

`_get_observation` または `get_observation` メソッドを探し、以下の処理を追加。

**追加する処理**:
```python
def _get_observation(self) -> np.ndarray:
    """状態を取得"""
    state_dict = self._build_state_dict()
    
    # ========== コンパクトモード: Top-K IDを更新 ==========
    if self.compact_mode:
        self.current_top_k_ids = self.state_encoder.get_top_k_ambulance_ids(
            state_dict['ambulances'],
            state_dict.get('pending_call')
        )
    
    return self.state_encoder.encode_state(state_dict)
```

**注意**: `_build_state_dict()` が存在しない場合は、状態辞書を構築している箇所を探して同様の処理を追加。

---

## 3. step の変更

### 変更箇所: L850-956付近（stepメソッド内）

stepメソッドの**最初の部分**に以下の処理を追加:

```python
def step(self, action: int) -> StepResult:
    """行動を実行"""
    
    # ========== コンパクトモード: actionをTop-K内インデックスとして解釈 ==========
    if self.compact_mode:
        if self.current_top_k_ids and action < len(self.current_top_k_ids):
            actual_ambulance_id = self.current_top_k_ids[action]
        else:
            # フォールバック: Top-1を選択（エラー時）
            if self.current_top_k_ids:
                actual_ambulance_id = self.current_top_k_ids[0]
            else:
                actual_ambulance_id = 0
            print(f"警告: action={action}がTop-K範囲外。actual_ambulance_id={actual_ambulance_id}を使用")
    else:
        # 従来モード: actionがそのまま救急車ID
        actual_ambulance_id = action
    
    # 以降、action の代わりに actual_ambulance_id を使用
    # （既存のコードで action を使っている箇所を actual_ambulance_id に置き換え）
```

### 置き換えが必要な箇所

stepメソッド内で `action` を救急車IDとして使用している箇所を特定し、`actual_ambulance_id` に置き換える。

主な箇所:
1. `self._dispatch_ambulance(action, ...)` → `self._dispatch_ambulance(actual_ambulance_id, ...)`
2. `self.ambulance_states[action]` → `self.ambulance_states[actual_ambulance_id]`
3. ログ出力で救急車IDを表示している箇所

**例**:
```python
# 変更前
if action in self.ambulance_states:
    amb_state = self.ambulance_states[action]
    ...

# 変更後
if actual_ambulance_id in self.ambulance_states:
    amb_state = self.ambulance_states[actual_ambulance_id]
    ...
```

---

## 4. get_action_mask の変更

### 変更箇所: L2151-2222付近

**変更後**:
```python
def get_action_mask(self) -> np.ndarray:
    """利用可能な行動のマスクを取得"""
    
    # ========== コンパクトモード ==========
    if self.compact_mode:
        # Top-K用のマスク（基本的に全てTrue）
        mask = np.ones(self.action_dim, dtype=bool)
        
        # Top-Kに満たない場合は残りを無効化
        if self.current_top_k_ids:
            valid_count = len(self.current_top_k_ids)
            if valid_count < self.action_dim:
                mask[valid_count:] = False
        
        return mask
    
    # ========== 従来モード（既存のロジック）==========
    mask = np.zeros(self.action_dim, dtype=bool)
    
    # 基本マスク：利用可能な救急車
    for amb_id, amb_state in self.ambulance_states.items():
        if amb_id < self.action_dim and amb_state['status'] == 'available':
            mask[amb_id] = True
    
    # coverage_awareモードでアクションマスクが有効な場合、追加フィルタリング
    # （既存のロジックをそのまま維持）
    if (self.reward_designer.mode == 'coverage_aware' and 
        self.pending_call is not None):
        
        action_mask_config = self.reward_designer.config.get('reward', {}).get('core', {}).get('action_mask', {})
        if action_mask_config.get('enabled', False):
            # ... 既存のフィルタリング処理 ...
            pass
    
    return mask
```

---

## 5. get_optimal_action の変更

### 変更箇所: L958-998付近

**変更後**:
```python
def get_optimal_action(self) -> Optional[int]:
    """
    現在の事案に対して最適な救急車を選択（最近接）
    
    Returns:
        最適な救急車のID（コンパクトモードではTop-K内インデックス）、または None
    """
    if self.pending_call is None:
        return None
    
    # ========== コンパクトモード ==========
    if self.compact_mode:
        # Top-1（最短移動時間）が最適
        # action=0 が常に最短移動時間の救急車
        return 0
    
    # ========== 従来モード（既存のロジック）==========
    best_action = None
    min_travel_time = float('inf')
    
    # 全ての救急車をチェック
    for amb_id, amb_state in self.ambulance_states.items():
        # 利用可能な救急車のみ対象
        if amb_state['status'] != 'available':
            continue
        
        try:
            # 現在位置から事案発生地点への移動時間を計算
            travel_time = self._calculate_travel_time(
                amb_state['current_h3'],
                self.pending_call['h3_index']
            )
            
            # より近い救急車を発見
            if travel_time < min_travel_time:
                min_travel_time = travel_time
                best_action = amb_id
                
        except Exception as e:
            continue
    
    return best_action
```

---

## 6. 新規メソッド（オプション）

### get_actual_ambulance_id

デバッグやログ出力用に、actionから実際の救急車IDを取得するヘルパーメソッド:

```python
def get_actual_ambulance_id(self, action: int) -> int:
    """
    actionから実際の救急車IDを取得
    
    Args:
        action: 行動番号（コンパクトモードではTop-K内インデックス）
    
    Returns:
        実際の救急車ID
    """
    if self.compact_mode:
        if self.current_top_k_ids and action < len(self.current_top_k_ids):
            return self.current_top_k_ids[action]
        return self.current_top_k_ids[0] if self.current_top_k_ids else 0
    else:
        return action
```

---

## 7. インポートの追加

ファイル先頭のインポート部分に追加（必要に応じて）:

```python
# 既存のインポートに追加
from .state_encoder import StateEncoder, CompactStateEncoder
```

または、使用箇所で動的にインポートする方式でも可:
```python
if self.compact_mode:
    from .state_encoder import CompactStateEncoder
    ...
else:
    from .state_encoder import StateEncoder
    ...
```

---

## 8. _build_state_dict の確認

`_get_observation` で使用する `_build_state_dict()` メソッドが存在するか確認。
存在しない場合は、状態辞書を構築している箇所を特定し、以下の形式でデータを準備:

```python
state_dict = {
    'ambulances': self.ambulance_states,  # {amb_id: {'current_h3': str, 'status': str, ...}}
    'pending_call': self.pending_call,     # {'h3_index': str, 'severity': str, ...} or None
    'time_of_day': self._get_current_time_of_day(),  # 0-24の浮動小数点
    'episode_step': self.episode_step
}
```

---

## 変更のチェックリスト

- [ ] `__init__`: compact_mode, top_k, current_top_k_ids の属性追加
- [ ] `__init__`: CompactStateEncoder の初期化
- [ ] `__init__`: action_dim を top_k に設定（コンパクトモード時）
- [ ] `_get_observation`: current_top_k_ids の更新処理追加
- [ ] `step`: action → actual_ambulance_id の変換処理追加
- [ ] `step`: 既存の action 使用箇所を actual_ambulance_id に置換
- [ ] `get_action_mask`: コンパクトモード用のマスク生成
- [ ] `get_optimal_action`: コンパクトモードで action=0 を返す
- [ ] インポート文の追加

---

## テスト確認項目

1. **初期化テスト**
   - `config['state_encoding']['mode'] = 'compact'` で `env.action_dim == 10`
   - `config['state_encoding']['mode'] = 'full'` で `env.action_dim == 192`

2. **ステップ実行テスト**
   - `action=0` で Top-1 救急車が配車される
   - `action=5` で Top-6 救急車が配車される

3. **行動マスクテスト**
   - コンパクトモードで `mask.shape == (10,)`
   - 従来モードで `mask.shape == (192,)`

4. **最適行動テスト**
   - コンパクトモードで `get_optimal_action() == 0`
