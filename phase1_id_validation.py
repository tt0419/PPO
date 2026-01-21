import pandas as pd
import json
import h3
from pathlib import Path
from collections import defaultdict
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def analyze_validation_simulator_ids():
    """ValidationSimulatorが生成するIDを分析"""
    print("=" * 80)
    print("Phase 1-A: ValidationSimulator ID構造分析")
    print("=" * 80)
    
    # ValidationSimulatorと同じロジックでIDを生成してみる
    ambulance_data = pd.read_csv("data/tokyo/import/amb_place_master.csv", encoding='utf-8')
    ambulance_data = ambulance_data[ambulance_data['special_flag'] == 1]
    
    # 「救急隊なし」を除外
    if 'team_name' in ambulance_data.columns:
        ambulance_data = ambulance_data[ambulance_data['team_name'] != '救急隊なし'].copy()
    
    print(f"\n総救急署数: {len(ambulance_data)}")
    
    # ValidationSimulatorと同じロジックでID生成をシミュレート
    simulated_ids = []
    id_to_info = {}
    
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
            simulated_ids.append(amb_id)
            id_to_info[amb_id] = {
                'team_name': team_name,
                'h3_index': h3_index,
                'index_within_team': i,
                'section': row.get('section', 1)
            }
    
    print(f"生成される救急車ID総数: {len(simulated_ids)}")
    
    # ID末尾の数字を抽出（PPOStrategyが行う変換）
    idx_extraction = {}
    idx_conflicts = defaultdict(list)
    
    for amb_id in simulated_ids:
        if '_' in amb_id:
            idx = int(amb_id.split('_')[-1])
            idx_extraction[amb_id] = idx
            idx_conflicts[idx].append(amb_id)
    
    print(f"\nID末尾抽出結果:")
    print(f"  ユニークなインデックス数: {len(idx_conflicts)}")
    print(f"  実際の救急車数: {len(simulated_ids)}")
    
    # 重複の深刻度チェック
    conflicts = {idx: ids for idx, ids in idx_conflicts.items() if len(ids) > 1}
    
    if conflicts:
        print(f"\n⚠️  重複検出: {len(conflicts)}個のインデックスで重複")
        print(f"  最大重複数: {max(len(ids) for ids in conflicts.values())}台")
        
        print("\n  重複例（最初の5件）:")
        for i, (idx, ids) in enumerate(list(conflicts.items())[:5]):
            print(f"    インデックス {idx}: {len(ids)}台")
            for amb_id in ids[:3]:  # 最大3台まで表示
                print(f"      - {amb_id}")
    else:
        print("\n✓ 重複なし: 全てのIDがユニーク")
    
    # 結果をJSONで保存
    output = {
        'total_ambulances': len(simulated_ids),
        'unique_indices': len(idx_conflicts),
        'id_list': simulated_ids,
        'id_to_info': id_to_info,
        'idx_extraction': idx_extraction,
        'conflicts': {k: v for k, v in conflicts.items()}
    }
    
    output_path = Path("validation_ids_analysis.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n結果を保存: {output_path}")
    
    return output

def analyze_ems_environment_ids():
    """EMSEnvironmentが使用するIDを分析"""
    print("\n" + "=" * 80)
    print("Phase 1-B: EMSEnvironment ID構造分析")
    print("=" * 80)
    
    try:
        from reinforcement_learning.environment.ems_environment import EMSEnvironment
        
        # 環境初期化
        config_path = "reinforcement_learning/experiments/config_hybrid_mode.yaml"
        
        # 複数の可能性のあるパスを試す
        possible_paths = [
            config_path,
            f"reinforcement_learning/{config_path}",
            "config.yaml",
            "reinforcement_learning/config.yaml"
        ]
        
        config_path = None
        for path in possible_paths:
            if Path(path).exists():
                config_path = path
                break
        
        if config_path is None:
            print("⚠️  設定ファイルが見つかりません。EMSEnvironmentの分析をスキップします。")
            return None
        
        print(f"設定ファイル: {config_path}")
        env = EMSEnvironment(config_path, mode='train')
        
        # EMSEnvironmentではreset()を呼ぶまでambulance_statesが初期化されない
        print("環境をリセットして救急車状態を初期化中...")
        env.reset()
        
        print(f"\n総救急車数: {len(env.ambulance_states)}")
        print(f"行動空間次元: {env.action_dim}")
        print(f"状態空間次元: {env.state_dim}")
        
        # IDの構造を確認
        print("\n救急車ID構造:")
        sample_ids = list(env.ambulance_states.keys())[:10]
        print(f"  サンプルID（最初の10台）: {sample_ids}")
        print(f"  IDの型: {type(sample_ids[0])}")
        print(f"  IDの範囲: {min(env.ambulance_states.keys())} ～ {max(env.ambulance_states.keys())}")
        
        # ambulance_dataの内容を確認
        print("\n元データ（ambulance_data）:")
        if hasattr(env, 'ambulance_data'):
            print(f"  行数: {len(env.ambulance_data)}")
            if 'team_name' in env.ambulance_data.columns:
                print(f"  team_nameサンプル:")
                for name in env.ambulance_data['team_name'].head(5):
                    print(f"    - {name}")
        
        # 結果を保存（ambulance_statesの値は複雑なのでシリアライズ可能な形式に変換）
        ambulance_states_sample = {}
        for k, v in list(env.ambulance_states.items())[:5]:
            ambulance_states_sample[int(k)] = {
                'id': v.get('id'),
                'name': v.get('name'),
                'status': v.get('status'),
                'station_h3': v.get('station_h3'),
                'current_h3': v.get('current_h3')
            }
        
        output = {
            'total_ambulances': len(env.ambulance_states),
            'action_dim': env.action_dim,
            'state_dim': env.state_dim,
            'id_list': [int(k) for k in env.ambulance_states.keys()],
            'id_type': str(type(sample_ids[0])),
            'ambulance_states_sample': ambulance_states_sample
        }
        
        output_path = Path("ems_environment_ids_analysis.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"\n結果を保存: {output_path}")
        
        return output
        
    except Exception as e:
        print(f"\n❌ EMSEnvironment分析エラー: {e}")
        import traceback
        traceback.print_exc()
        return None


def compare_id_structures(validation_output, ems_output):
    """両者のID構造を比較"""
    print("\n" + "=" * 80)
    print("Phase 1-C: ID構造比較")
    print("=" * 80)
    
    if validation_output is None or ems_output is None:
        print("⚠️  比較に必要なデータが不足しています")
        return
    
    val_count = validation_output['total_ambulances']
    ems_count = ems_output['total_ambulances']
    
    print(f"\n救急車数比較:")
    print(f"  ValidationSimulator: {val_count}台")
    print(f"  EMSEnvironment: {ems_count}台")
    
    if val_count == ems_count:
        print("  ✓ 台数一致")
    else:
        print(f"  ⚠️  台数不一致（差分: {abs(val_count - ems_count)}台）")
    
    print(f"\nID形式比較:")
    print(f"  ValidationSimulator: 文字列型（例: '{validation_output['id_list'][0]}'）")
    print(f"  EMSEnvironment: {ems_output['id_type']}型（例: {ems_output['id_list'][0]}）")
    
    # 重複問題の深刻度
    if validation_output.get('conflicts'):
        total_conflicts = len(validation_output['conflicts'])
        max_conflict = max(len(ids) for ids in validation_output['conflicts'].values())
        
        print(f"\n⚠️  重大な問題:")
        print(f"  - ValidationSimulatorの{val_count}台の救急車が")
        print(f"  - PPOStrategyでは{validation_output['unique_indices']}台に圧縮される")
        print(f"  - 最大{max_conflict}台が同じインデックスに重複")
        print(f"  - これにより、PPOエージェントは正しい状態を認識できません")
    
    # 比較結果を保存
    comparison = {
        'validation_count': val_count,
        'ems_count': ems_count,
        'count_match': val_count == ems_count,
        'validation_unique_indices': validation_output.get('unique_indices', 0),
        'conflict_severity': {
            'has_conflicts': bool(validation_output.get('conflicts')),
            'num_conflicts': len(validation_output.get('conflicts', {})),
            'max_conflict_size': max(
                (len(ids) for ids in validation_output.get('conflicts', {}).values()),
                default=0
            )
        }
    }
    
    output_path = Path("id_comparison_result.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
    
    print(f"\n比較結果を保存: {output_path}")
    
    return comparison


def generate_id_mapping_proposal(validation_output, ems_output):
    """ValidationSimulatorとEMSEnvironment間のID対応表を提案"""
    print("\n" + "=" * 80)
    print("Phase 1-D: ID対応表の提案")
    print("=" * 80)
    
    if validation_output is None or ems_output is None:
        print("⚠️  対応表作成に必要なデータが不足しています")
        return None
    
    validation_ids = validation_output['id_list']
    ems_ids = ems_output['id_list']
    
    print(f"\nValidationSimulator ID数: {len(validation_ids)}")
    print(f"EMSEnvironment ID数: {len(ems_ids)}")
    
    # 文字列ID → 整数IDへの対応表を作成
    id_mapping = {}
    reverse_mapping = {}
    
    for i, val_id in enumerate(validation_ids):
        if i < len(ems_ids):
            ems_id = ems_ids[i]
            id_mapping[val_id] = ems_id
            reverse_mapping[ems_id] = val_id
    
    print(f"\n対応表作成: {len(id_mapping)}個のマッピング")
    print("\nサンプルマッピング（最初の5件）:")
    for i, (val_id, ems_id) in enumerate(list(id_mapping.items())[:5]):
        print(f"  {val_id} → {ems_id}")
    
    # 対応表を保存
    output = {
        'string_to_int': id_mapping,
        'int_to_string': reverse_mapping,
        'mapping_count': len(id_mapping),
        'validation_ids_count': len(validation_ids),
        'ems_ids_count': len(ems_ids)
    }
    
    output_path = Path("id_mapping_proposal.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n対応表を保存: {output_path}")
    
    return output


def main():
    """Phase 1のメイン実行"""
    print("\n" + "=" * 80)
    print("Phase 1: 検証と準備（コード変更なし）")
    print("=" * 80)
    print("\nこのスクリプトは既存コードを一切変更しません")
    print("現状のID構造を分析し、問題の深刻度を把握します")
    print("=" * 80)
    
    # Phase 1-A: ValidationSimulator ID分析
    validation_output = analyze_validation_simulator_ids()
    
    # Phase 1-B: EMSEnvironment ID分析
    ems_output = analyze_ems_environment_ids()
    
    # Phase 1-C: 比較分析
    comparison = compare_id_structures(validation_output, ems_output)
    
    # Phase 1-D: 対応表の提案
    if validation_output and ems_output:
        mapping = generate_id_mapping_proposal(validation_output, ems_output)
    
    # 最終サマリー
    print("\n" + "=" * 80)
    print("Phase 1 完了")
    print("=" * 80)
    print("\n生成されたファイル:")
    print("  1. validation_ids_analysis.json - ValidationSimulatorのID構造")
    print("  2. ems_environment_ids_analysis.json - EMSEnvironmentのID構造")
    print("  3. id_comparison_result.json - 比較結果")
    print("  4. id_mapping_proposal.json - ID対応表の提案")
    
    print("\n次のステップ:")
    print("  これらのファイルを確認し、問題の深刻度を把握してください")
    print("  問題が確認できたら、Phase 2の修正に進みます")
    print("=" * 80)


if __name__ == "__main__":
    main()
