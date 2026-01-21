import pandas as pd
import h3
import json
from pathlib import Path

def generate_192_ambulance_mapping():
    """192台（デイタイム除外）のID対応表を生成"""
    print("=" * 80)
    print("192台用ID対応表生成")
    print("=" * 80)
    
    # ValidationSimulatorと同じロジックでデータ読み込み
    ambulance_data = pd.read_csv("data/tokyo/import/amb_place_master.csv", encoding='utf-8')
    ambulance_data = ambulance_data[ambulance_data['special_flag'] == 1]
    
    print(f"\n元データ: {len(ambulance_data)}台")
    
    # 「救急隊なし」を除外
    if 'team_name' in ambulance_data.columns:
        ambulance_data = ambulance_data[ambulance_data['team_name'] != '救急隊なし'].copy()
        print(f"「救急隊なし」除外後: {len(ambulance_data)}台")
    
    # デイタイム救急を除外
    if 'team_name' in ambulance_data.columns:
        ambulance_data = ambulance_data[~ambulance_data['team_name'].astype(str).str.contains("デイタイム救急")].copy()
        print(f"「デイタイム救急」除外後: {len(ambulance_data)}台")
    
    # ValidationSimulator形式のID生成
    validation_ids = []
    
    for index, row in ambulance_data.iterrows():
        h3_index = h3.latlng_to_cell(row['latitude'], row['longitude'], 9)
        
        num_ambulances = 0
        if 'amb' in row and pd.notna(row['amb']):
            try:
                amb_value = int(float(str(row['amb'])))
                if amb_value > 0:
                    num_ambulances = 1
            except ValueError:
                continue
        
        if num_ambulances <= 0:
            continue
        
        team_name = row.get('team_name', f"Station_{h3_index}")
        if not team_name:
            team_name = f"Station_{h3_index}"
        
        for i in range(num_ambulances):
            amb_id = f"{team_name}_{i}"
            validation_ids.append(amb_id)
    
    print(f"\nValidationSimulator ID数: {len(validation_ids)}台")
    
    # EMSEnvironment形式のID（0から始まる整数）
    ems_ids = list(range(len(validation_ids)))
    
    print(f"EMSEnvironment ID数: {len(ems_ids)}台")
    
    # 対応表を作成
    string_to_int = {}
    int_to_string = {}
    
    for ems_id, val_id in zip(ems_ids, validation_ids):
        string_to_int[val_id] = ems_id
        int_to_string[str(ems_id)] = val_id
    
    print(f"\n対応表作成: {len(string_to_int)}件のマッピング")
    
    # サンプル表示
    print(f"\nサンプルマッピング（最初の5件）:")
    for val_id, ems_id in list(string_to_int.items())[:5]:
        print(f"  '{val_id}' → {ems_id}")
    
    # 最後の5件も表示（デイタイムが除外されているか確認）
    print(f"\nサンプルマッピング（最後の5件）:")
    for val_id, ems_id in list(string_to_int.items())[-5:]:
        print(f"  '{val_id}' → {ems_id}")
    
    # デイタイムが含まれていないか確認
    daytime_check = [vid for vid in validation_ids if 'デイタイム' in vid]
    if daytime_check:
        print(f"\n⚠️ 警告: デイタイム救急が{len(daytime_check)}件含まれています！")
        for dt in daytime_check[:5]:
            print(f"  - {dt}")
    else:
        print(f"\n✅ デイタイム救急は含まれていません")
    
    # JSON保存
    mapping_output = {
        'string_to_int': string_to_int,
        'int_to_string': int_to_string,
        'mapping_count': len(string_to_int),
        'validation_ids_count': len(validation_ids),
        'ems_ids_count': len(ems_ids)
    }
    
    output_path = Path("id_mapping_proposal.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(mapping_output, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ ID対応表を保存: {output_path}")
    print(f"   台数: {len(string_to_int)}台")
    print(f"   デイタイム: 除外済み")
    
    return mapping_output

if __name__ == "__main__":
    result = generate_192_ambulance_mapping()
    
    print("\n" + "=" * 80)
    print("完了")
    print("=" * 80)
    print("\n次のステップ:")
    print("  1. baseline_comparison.py を実行")
    print("  2. 'マッピング失敗' 警告が消えることを確認")
    print("=" * 80)