"""
data_cache.py
CSVデータのキャッシュと効率的な読み込みを管理するクラス
"""

import pandas as pd
import os
from typing import Optional, Dict, List
import logging
from pathlib import Path

class EmergencyDataCache:
    """救急事案データのキャッシュクラス"""
    
    def __init__(self):
        self._cached_data: Optional[pd.DataFrame] = None
        self._cache_file_path: Optional[str] = None
        self._cache_file_mtime: Optional[float] = None
        self.logger = logging.getLogger(__name__)
        
        # パスの優先順位リスト
        self.possible_paths = [
            "C:/Users/tetsu/OneDrive - Yokohama City University/30_データカタログ/tfd_data/hanso_special_wards.csv",
            "C:/Users/hp/OneDrive - Yokohama City University/30_データカタログ/tfd_data/hanso_special_wards.csv"
        ]
    
    def _find_data_file(self) -> Optional[str]:
        """データファイルのパスを探す"""
        for path in self.possible_paths:
            if os.path.exists(path):
                return path
        return None
    
    def _should_reload_cache(self, file_path: str) -> bool:
        """キャッシュの再読み込みが必要かチェック"""
        if self._cached_data is None:
            return True
        
        if self._cache_file_path != file_path:
            return True
        
        # ファイルの更新時刻をチェック
        current_mtime = os.path.getmtime(file_path)
        if self._cache_file_mtime != current_mtime:
            return True
        
        return False
    
    def load_data(self, force_reload: bool = False) -> pd.DataFrame:
        """
        救急事案データを読み込み（キャッシュ機能付き）
        
        Args:
            force_reload: 強制的に再読み込みするかどうか
            
        Returns:
            読み込まれたDataFrame
        """
        # ファイルパスを探す
        file_path = self._find_data_file()
        if file_path is None:
            raise FileNotFoundError("救急事案データファイルが見つかりません")
        
        # キャッシュの確認
        if not force_reload and not self._should_reload_cache(file_path):
            self.logger.info("キャッシュからデータを取得")
            return self._cached_data  # copy()を削除してメモリ使用量を削減
        
        # データの読み込み
        self.logger.info(f"CSVファイルを読み込み中: {os.path.basename(file_path)}")
        start_time = pd.Timestamp.now()
        
        try:
            # CSVファイルの読み込み
            df = pd.read_csv(file_path, encoding='utf-8')
            self.logger.info(f"読み込み完了: {len(df):,}行")
            
            # 日付変換（一度だけ実行）
            self.logger.info("日付データを変換中...")
            df['出場年月日時分'] = pd.to_datetime(df['出場年月日時分'], errors='coerce')
            
            # NaTを除外
            before_dropna = len(df)
            df = df.dropna(subset=['出場年月日時分'])
            if before_dropna > len(df):
                self.logger.info(f"無効な日付データを除外: {before_dropna - len(df)}件")
            
            # 「その他」の事案を除外
            before_filter = len(df)
            df = df[df['収容所見程度'] != 'その他'].copy()
            if before_filter > len(df):
                self.logger.info(f"「その他」の事案を除外: {before_filter - len(df)}件")
            
            # 座標の有効性確認
            before_coord = len(df)
            df = df.dropna(subset=['Y_CODE', 'X_CODE'])
            if before_coord > len(df):
                self.logger.info(f"無効な座標データを除外: {before_coord - len(df)}件")
            
            # データ型の最適化（メモリ使用量削減）
            print("データ型を最適化中...")
            if 'Y_CODE' in df.columns:
                df['Y_CODE'] = pd.to_numeric(df['Y_CODE'], errors='coerce').astype('float32')
            if 'X_CODE' in df.columns:
                df['X_CODE'] = pd.to_numeric(df['X_CODE'], errors='coerce').astype('float32')
            
            # 文字列カラムのカテゴリ化
            if '収容所見程度' in df.columns:
                df['収容所見程度'] = df['収容所見程度'].astype('category')
            if '救急事案番号キー' in df.columns:
                df['救急事案番号キー'] = df['救急事案番号キー'].astype('string')
            
            # メモリ使用量を表示
            memory_usage_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
            print(f"メモリ使用量最適化完了 (約{memory_usage_mb:.1f}MB)")
            
            # キャッシュに保存
            self._cached_data = df
            self._cache_file_path = file_path
            self._cache_file_mtime = os.path.getmtime(file_path)
            
            end_time = pd.Timestamp.now()
            self.logger.info(f"データ処理完了: {len(df):,}件 (処理時間: {(end_time - start_time).total_seconds():.2f}秒)")
            
            return df  # copy()を削除してメモリ使用量を削減
            
        except Exception as e:
            self.logger.error(f"データ読み込みエラー: {e}")
            raise
    
    def get_period_data(self, start_date: str, end_date: str, area_filter: Optional[List[str]] = None) -> pd.DataFrame:
        """
        指定期間のデータを取得
        
        Args:
            start_date: 開始日（YYYYMMDD形式の文字列）
            end_date: 終了日（YYYYMMDD形式の文字列）
            area_filter: 出場先区市のフィルタリングリスト（例: ["目黒区", "渋谷区", "世田谷区"]）
            
        Returns:
            指定期間のDataFrame
        """
        # キャッシュからデータを取得
        df = self.load_data()
        
        # エリアフィルタリング（指定方面用）
        if area_filter is not None:
            if '出場先区市' in df.columns:
                before_area_filter = len(df)
                df = df[df['出場先区市'].isin(area_filter)]
                self.logger.info(f"エリアフィルタ適用: {before_area_filter}件 → {len(df)}件 (対象: {', '.join(area_filter)})")
            else:
                self.logger.warning("'出場先区市'カラムが見つかりません。エリアフィルタをスキップします。")
        
        # 日付文字列を変換
        start_str = str(start_date)
        end_str = str(end_date)
        
        start_date_str = f"{start_str[:4]}-{start_str[4:6]}-{start_str[6:8]}"
        end_date_str = f"{end_str[:4]}-{end_str[4:6]}-{end_str[6:8]}"
        
        start_datetime = pd.to_datetime(start_date_str)
        end_datetime = pd.to_datetime(end_date_str) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        
        # 期間でフィルタリング（メモリ効率版 - 段階的条件適用）
        # 大きなデータセットでメモリ断片化を避けるため、条件を段階的に適用
        datetime_col = df['出場年月日時分']
        
        # 1段階目: 開始日時以降のデータを抽出
        start_mask = datetime_col >= start_datetime
        temp_df = df[start_mask]
        
        # 2段階目: 終了日時以前のデータを抽出
        end_mask = temp_df['出場年月日時分'] <= end_datetime
        filtered_df = temp_df[end_mask]
        
        area_info = f" (エリア限定: {', '.join(area_filter)})" if area_filter else ""
        self.logger.info(f"期間 {start_date_str} ～ {end_date_str}{area_info}: {len(filtered_df)}件")
        
        return filtered_df
    
    def get_datetime_range_data(self, start_datetime: pd.Timestamp, end_datetime: pd.Timestamp) -> pd.DataFrame:
        """
        指定の日時範囲のデータを取得
        
        Args:
            start_datetime: 開始日時
            end_datetime: 終了日時
            
        Returns:
            指定範囲のDataFrame
        """
        # キャッシュからデータを取得
        df = self.load_data()
        
        # 日時範囲でフィルタリング（メモリ効率版 - 段階的条件適用）
        # 大きなデータセットでメモリ断片化を避けるため、条件を段階的に適用
        datetime_col = df['出場年月日時分']
        
        # 1段階目: 開始日時以降のデータを抽出
        start_mask = datetime_col >= start_datetime
        temp_df = df[start_mask]
        
        # 2段階目: 終了日時以前のデータを抽出
        end_mask = temp_df['出場年月日時分'] < end_datetime
        filtered_df = temp_df[end_mask]
        
        self.logger.info(f"日時範囲 {start_datetime} ～ {end_datetime}: {len(filtered_df)}件")
        
        return filtered_df
    
    def get_cache_info(self) -> Dict:
        """キャッシュの情報を取得"""
        if self._cached_data is None:
            return {"cached": False}
        
        return {
            "cached": True,
            "file_path": self._cache_file_path,
            "file_mtime": self._cache_file_mtime,
            "total_records": len(self._cached_data),
            "date_range": {
                "start": self._cached_data['出場年月日時分'].min(),
                "end": self._cached_data['出場年月日時分'].max()
            },
            "severity_distribution": self._cached_data['収容所見程度'].value_counts().to_dict()
        }
    
    def clear_cache(self):
        """キャッシュをクリア"""
        self._cached_data = None
        self._cache_file_path = None
        self._cache_file_mtime = None
        self.logger.info("データキャッシュをクリアしました")


# グローバルキャッシュインスタンス
_global_cache = EmergencyDataCache()

def get_emergency_data_cache() -> EmergencyDataCache:
    """グローバルキャッシュインスタンスを取得"""
    return _global_cache

def load_emergency_data(force_reload: bool = False) -> pd.DataFrame:
    """救急事案データを読み込み（グローバルキャッシュ使用）"""
    return _global_cache.load_data(force_reload)

def get_period_emergency_data(start_date: str, end_date: str, area_filter: Optional[List[str]] = None) -> pd.DataFrame:
    """指定期間の救急事案データを取得（グローバルキャッシュ使用）"""
    return _global_cache.get_period_data(start_date, end_date, area_filter)

def get_datetime_range_emergency_data(start_datetime: pd.Timestamp, end_datetime: pd.Timestamp) -> pd.DataFrame:
    """指定日時範囲の救急事案データを取得（グローバルキャッシュ使用）"""
    return _global_cache.get_datetime_range_data(start_datetime, end_datetime)


if __name__ == "__main__":
    # テスト用コード
    import logging
    logging.basicConfig(level=logging.INFO)
    
    cache = EmergencyDataCache()
    
    # 初回読み込み
    print("=== 初回読み込み ===")
    data1 = cache.load_data()
    print(f"データ件数: {len(data1)}")
    
    # キャッシュからの読み込み
    print("\n=== キャッシュからの読み込み ===")
    data2 = cache.load_data()
    print(f"データ件数: {len(data2)}")
    
    # キャッシュ情報表示
    print("\n=== キャッシュ情報 ===")
    info = cache.get_cache_info()
    for key, value in info.items():
        print(f"{key}: {value}")
    
    # 期間データの取得テスト
    print("\n=== 期間データ取得テスト ===")
    period_data = cache.get_period_data("20230401", "20230401")
    print(f"2023年4月1日のデータ: {len(period_data)}件")