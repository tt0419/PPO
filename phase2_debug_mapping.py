import json
from pathlib import Path

def analyze_mapping_mismatch():
    """ID対応表と実際の救急車の不一致を分析"""
    print("=" * 80)
    print("Phase 2デバッグ: マッピング問題の検証")
    print("=" * 80)

    # Phase 1で生成されたファイルを読み込み
    mapping_file = Path("id_mapping_proposal.json")
    validation_file = Path("validation_ids_analysis.json")
    ems_file = Path("ems_environment_ids_analysis.json")

    if not all([mapping_file.exists(), validation_file.exists(), ems_file.exists()]):
        print("❌ Phase 1の出力ファイルが見つかりません")
        print("先に phase1_id_validation.py を実行してください")
        return

    with open(mapping_file, 'r', encoding='utf-8') as f:
        mapping_data = json.load(f)

    with open(validation_file, 'r', encoding='utf-8') as f:
        validation_data = json.load(f)

    with open(ems_file, 'r', encoding='utf-8') as f:
        ems_data = json.load(f)

    # 対応表の内容
    string_to_int = mapping_data['string_to_int']
    int_to_string = mapping_data['int_to_string']

    print(f"\n1. 台数の確認:")
    print(f"   ValidationSimulator: {validation_data['total_ambulances']}台")
    print(f"   EMSEnvironment（学習時）: {ems_data['total_ambulances']}台")
    print(f"   ID対応表のマッピング数: {len(string_to_int)}件")

    # 問題点の特定
    print(f"\n2. 問題の特定:")

    # ValidationSimulatorの実際のID
    validation_ids = set(validation_data['id_list'])

    # ID対応表に含まれるValidationSimulatorのID
    mapped_validation_ids = set(string_to_int.keys())

    # EMSEnvironmentのアクション番号
    ems_actions = set(ems_data['id_list'])

    # ID対応表に含まれるアクション番号
    mapped_actions = set(int(k) for k in int_to_string.keys())

    print(f"\n   ValidationSimulator側:")
    print(f"     実際のID数: {len(validation_ids)}")
    print(f"     対応表のID数: {len(mapped_validation_ids)}")

    missing_in_mapping = validation_ids - mapped_validation_ids
    if missing_in_mapping:
        print(f"     ⚠️  対応表に含まれないID: {len(missing_in_mapping)}個")
        print(f"     例: {list(missing_in_mapping)[:5]}")
    else:
        print(f"     ✓ 全てのIDが対応表に含まれる")

    print(f"\n   EMSEnvironment側:")
    print(f"     学習時のアクション数: {len(ems_actions)}")
    print(f"     対応表のアクション数: {len(mapped_actions)}")

    missing_actions = ems_actions - mapped_actions
    if missing_actions:
        print(f"     ⚠️  対応表に含まれないアクション: {len(missing_actions)}個")
        print(f"     例: {sorted(list(missing_actions))[:10]}")
    else:
        print(f"     ✓ 全てのアクションが対応表に含まれる")

    # 台数不一致の影響
    print(f"\n3. 台数不一致の影響:")
    diff = ems_data['total_ambulances'] - validation_data['total_ambulances']
    print(f"   学習時は{ems_data['total_ambulances']}台で学習")
    print(f"   実行時は{validation_data['total_ambulances']}台しかない")
    print(f"   差分: {diff}台")

    if diff > 0:
        print(f"\n   ⚠️  問題:")
        print(f"   - PPOエージェントは0-{ems_data['total_ambulances']-1}のアクションを出力")
        print(f"   - しかし、{validation_data['total_ambulances']}-{ems_data['total_ambulances']-1}に対応する救急車が存在しない")
        print(f"   - これらのアクションが選択されると「マッピング失敗」になる")
        
        # 実際に存在しないアクション番号
        missing_action_range = list(range(validation_data['total_ambulances'], ems_data['total_ambulances']))
        print(f"\n   存在しないアクション番号: {missing_action_range[:10]}... (最初の10個)")

    # EMSEnvironmentの「救急隊なし」の影響
    print(f"\n4. 「救急隊なし」の影響:")

    ems_sample = ems_data.get('ambulance_states_sample', {})
    rescue_nashi_count = sum(1 for v in ems_sample.values() if v.get('name') == '救急隊なし')

    if rescue_nashi_count > 0:
        print(f"   サンプル5台中、{rescue_nashi_count}台が「救急隊なし」")
        print(f"   EMSEnvironmentは「救急隊なし」を含む{ems_data['total_ambulances']}台を使用")
        print(f"   ValidationSimulatorは「救急隊なし」を除外した{validation_data['total_ambulances']}台を使用")
        print(f"\n   ⚠️  これが台数不一致の原因です")

    # 解決策の提案
    print(f"\n5. 解決策:")
    print(f"   A. EMSEnvironmentでも「救急隊なし」を除外 (推奨)")
    print(f"      → 学習時も202台で学習し直す")
    print(f"   B. ValidationSimulatorでも「救急隊なし」を含める")
    print(f"      → 実際に出動しない救急車を含めることになる")
    print(f"   C. PPOStrategyでアクション範囲を制限")
    print(f"      → 202台に対応するアクションのみ選択可能にする")

    return {
        'validation_count': validation_data['total_ambulances'],
        'ems_count': ems_data['total_ambulances'],
        'diff': diff,
        'missing_actions': len(missing_actions) if missing_actions else 0
    }

if __name__ == "__main__":
    result = analyze_mapping_mismatch()
    
    print("\n" + "=" * 80)
    print("デバッグ完了")
    print("=" * 80)
    
    if result:
        print(f"\n推奨される対応:")
        if result['diff'] > 0:
            print(f"  1. config.yamlで「救急隊なし」を除外する設定を追加")
            print(f"  2. または、PPOStrategyでaction_dimを{result['validation_count']}に制限")
            print(f"  3. 根本的には、同じデータで再学習することを推奨")