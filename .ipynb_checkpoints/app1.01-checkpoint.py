import pandas as pd
import numpy as np
import pulp
import yaml
import os

# ====================================
# 各種設定 (PyCharm のコードからファイルパスを取得)
# ====================================
BASE_YML_PATH = "/Users/tkshsgw/crypto_trading/crypto_trading/Battery_Optimization/base.yml"
WHEELING_YML_PATH = "/Users/tkshsgw/crypto_trading/crypto_trading/Battery_Optimization/wheeling.yaml"
DATA_CSV_PATH = "/Users/tkshsgw/Desktop/Battery Optimization Project/JWAプライス予測サンプルデータシミュレーション用.csv"
OUTPUT_CSV_PATH = "optimal_transactions.csv"

# 最大連続スロット数 (EPRX1, EPRX3)
MAX_EPRX1_SLOTS = 7
MAX_EPRX3_SLOTS = 7

def main():
    # ------------------------------------------------
    # 1) 設定ファイル (base.yml, wheeling.yml) 読み込み
    # ------------------------------------------------
    if not os.path.exists(BASE_YML_PATH):
        print(f"ERROR: {BASE_YML_PATH} が見つかりません。")
        return
    if not os.path.exists(WHEELING_YML_PATH):
        print(f"ERROR: {WHEELING_YML_PATH} が見つかりません。")
        return

    with open(BASE_YML_PATH, "r", encoding="utf-8") as f:
        base_yaml = yaml.safe_load(f)
    with open(WHEELING_YML_PATH, "r", encoding="utf-8") as f:
        wheeling_yaml = yaml.safe_load(f)

    # base.yml からバッテリー設定を抽出
    battery_cfg = base_yaml.get("battery", {})
    battery_loss_rate = battery_cfg.get("loss_rate", 0.05)   # 放電時のバッテリー内部ロス
    battery_min_soc = battery_cfg.get("min_residual_soc", 0.1)
    battery_power_kW = battery_cfg.get("power_kW", 50)
    battery_capacity_kWh = battery_cfg.get("capacity_kWh", 200)

    # wheeling.yml から HV(loss_rate=0.03 など) を取得
    region_settings = wheeling_yaml.get("Kyushu", {})
    hv_settings = region_settings.get("HV", {})
    wheeling_loss_rate = hv_settings.get("loss_rate", 0.03)  # 充電時の託送ロス
    wheeling_base_charge = hv_settings.get("wheeling_base_charge", 1000)
    wheeling_usage_fee = hv_settings.get("wheeling_usage_fee", 3)

    # ------------------------------------------------
    # 2) Data.csv の読み込み
    # ------------------------------------------------
    if not os.path.exists(DATA_CSV_PATH):
        print(f"ERROR: {DATA_CSV_PATH} が見つかりません。")
        return

    df_all = pd.read_csv(DATA_CSV_PATH)

    # 必須カラムがあるかチェック
    required_cols = {
        "date", "slot",
        "JEPX_prediction", "JEPX_actual",
        "EPRX1_prediction", "EPRX1_actual",
        "EPRX3_prediction", "EPRX3_actual",
        "imbalance"
    }
    if not required_cols.issubset(df_all.columns):
        print(f"ERROR: CSVに必要な列が不足しています: {required_cols}")
        return

    # 予測価格が存在する日だけを対象にする
    df_all["prediction_available"] = (
        ~df_all["JEPX_prediction"].isna() |
        ~df_all["EPRX1_prediction"].isna() |
        ~df_all["EPRX3_prediction"].isna()
    )
    valid_dates = df_all.groupby("date")["prediction_available"].any()
    valid_dates = valid_dates[valid_dates].index.tolist()

    # 最適化結果を蓄積するリスト
    all_results = []

    # ------------------------------------------------
    # 3) 日毎に最適化
    # ------------------------------------------------
    for target_date in valid_dates:
        df_day = df_all[df_all["date"] == target_date].copy()
        df_day.sort_values(by="slot", inplace=True)
        df_day.reset_index(drop=True, inplace=True)

        num_slots = len(df_day)  # 通常48想定(1日30分刻み)
        # 30分あたりの最大充放電量
        half_power_kWh = battery_power_kW * 0.5

        # PuLP 問題定義
        prob = pulp.LpProblem(f"Battery_Optimization_{target_date}", pulp.LpMaximize)

        # バッテリー残量 (各スロット開始時)
        battery_soc = pulp.LpVariable.dicts(
            f"soc_{target_date}", range(num_slots + 1),
            lowBound=0, upBound=battery_capacity_kWh, cat=pulp.LpContinuous
        )

        # 通常の充電・放電 (binary)
        charge = pulp.LpVariable.dicts(
            f"charge_{target_date}", range(num_slots), cat=pulp.LpBinary)
        discharge = pulp.LpVariable.dicts(
            f"discharge_{target_date}", range(num_slots), cat=pulp.LpBinary)

        # EPRX1, EPRX3 の連続ブロック
        eprx1_blocks = []
        for s in range(num_slots):
            for d in range(1, MAX_EPRX1_SLOTS + 1):
                if s + d <= num_slots:
                    eprx1_blocks.append((s, d))
        eprx1_block = pulp.LpVariable.dicts(
            f"eprx1_{target_date}", eprx1_blocks, cat=pulp.LpBinary)

        eprx3_blocks = []
        for s in range(num_slots):
            for d in range(1, MAX_EPRX3_SLOTS + 1):
                if s + d <= num_slots:
                    eprx3_blocks.append((s, d))
        eprx3_block = pulp.LpVariable.dicts(
            f"eprx3_{target_date}", eprx3_blocks, cat=pulp.LpBinary)

        # 初期バッテリー残量 -> minSOC相当
        prob += battery_soc[0] == battery_capacity_kWh * battery_min_soc, f"InitBattery_{target_date}"

        # スロットごとの排他制約
        for i in range(num_slots):
            prob += charge[i] + discharge[i] <= 1
            eprx1_cover = [(s, d) for (s, d) in eprx1_blocks if s <= i < s + d]
            eprx3_cover = [(s, d) for (s, d) in eprx3_blocks if s <= i < s + d]
            prob += (pulp.lpSum([eprx1_block[bd] for bd in eprx1_cover])
                     + pulp.lpSum([eprx3_block[bd] for bd in eprx3_cover])
                     + charge[i] + discharge[i]) <= 1

        # バッテリーの遷移 (JEPX充放電のみ)
        for i in range(num_slots):
            # 充電時: バッテリーに half_power_kWh が加わる(託送ロスは購入電力量増)
            # 放電時: half_power_kWh 分だけ SOC が減る(売れる量は (1 - battery_loss_rate)倍)

            next_soc = battery_soc[i] + charge[i] * half_power_kWh - discharge[i] * half_power_kWh
            prob += battery_soc[i+1] == next_soc

        # EPRX1: (s,d) ブロック => 40~60% 範囲
        bigM = 1e6
        for (s, d) in eprx1_blocks:
            for k in range(s, s+d):
                prob += battery_soc[k] >= 0.4*battery_capacity_kWh - (1 - eprx1_block[(s,d)])*bigM
                prob += battery_soc[k] <= 0.6*battery_capacity_kWh + (1 - eprx1_block[(s,d)])*bigM

        # EPRX3: (s,d) ブロック => 連続放電
        for (s, d) in eprx3_blocks:
            block_discharge = d * half_power_kWh
            # 開始時点: block_discharge + minSOC 以上が必要
            prob += battery_soc[s] >= (block_discharge + battery_capacity_kWh*battery_min_soc) \
                    - (1 - eprx3_block[(s,d)])*bigM
            # 終了時: blockVar=1 => battery_soc[s+d] = battery_soc[s] - block_discharge
            prob += battery_soc[s + d] == battery_soc[s] - block_discharge*eprx3_block[(s,d)]

        # スロットごとの minSOC ~ capacity
        for i in range(num_slots+1):
            prob += battery_soc[i] >= battery_capacity_kWh * battery_min_soc
            prob += battery_soc[i] <= battery_capacity_kWh

        # 目的関数 (予測ベース)
        profit_terms = []

        for i in range(num_slots):
            jepx_pred = df_day.loc[i, "JEPX_prediction"]
            if pd.isna(jepx_pred):
                jepx_pred = 0.0

            # 充電時のコスト: wheeling_loss_rate のみ考慮
            # 「バッテリーに half_power_kWh 入れる」ために購入量 = half_power_kWh / (1 - wheeling_loss_rate)
            # コスト = jepx_pred * [上記購入量]
            cost_charge = jepx_pred * (half_power_kWh / (1 - wheeling_loss_rate)) * charge[i]

            # 放電時の収益: battery_loss_rate のみ考慮
            # 実際に売れる量 = half_power_kWh * (1 - battery_loss_rate)
            rev_discharge = jepx_pred * (half_power_kWh * (1 - battery_loss_rate)) * discharge[i]

            profit_terms.append(-cost_charge + rev_discharge)

        # EPRX1 収益
        for (s, d) in eprx1_blocks:
            block_profit = 0.0
            for k in range(s, s+d):
                val = df_day.loc[k, "EPRX1_prediction"]
                if pd.isna(val):
                    val = 0.0
                block_profit += val * battery_power_kW
            profit_terms.append(eprx1_block[(s,d)] * block_profit)

        # EPRX3 収益
        for (s, d) in eprx3_blocks:
            block_profit = 0.0
            imb_sum = 0.0
            for k in range(s, s+d):
                val3 = df_day.loc[k, "EPRX3_prediction"]
                if pd.isna(val3):
                    val3 = 0.0
                block_profit += val3 * battery_power_kW

                imb_val = df_day.loc[k, "imbalance"]
                if pd.isna(imb_val):
                    imb_val = 0.0
                imb_sum += imb_val

            # imbalance の平均
            if d > 0:
                imb_avg = imb_sum / d
            else:
                imb_avg = 0.0

            # 放電量 (dスロット合計) = d * half_power_kWh
            # ただし売れる量は d * half_power_kWh * (1 - battery_loss_rate)
            e3_sold_kWh = (d * half_power_kWh) * (1 - battery_loss_rate)
            block_profit += e3_sold_kWh * imb_avg

            profit_terms.append(eprx3_block[(s,d)] * block_profit)

        prob += pulp.lpSum(profit_terms), "Total_Profit"

        # ソルバー実行
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        status = pulp.LpStatus[prob.status]
        if status != "Optimal":
            print(f"Warning: {target_date} の最適解が見つかりませんでした (status={status})")
            continue

        # 実際の収益計算
        slot_actions = ["idle"] * num_slots

        # EPRX1 / EPRX3 ブロック
        for (s, d) in eprx1_blocks:
            if pulp.value(eprx1_block[(s,d)]) > 0.5:
                for k in range(s, s+d):
                    slot_actions[k] = f"EPRX1_{d}slots"
        for (s, d) in eprx3_blocks:
            if pulp.value(eprx3_block[(s,d)]) > 0.5:
                for k in range(s, s+d):
                    slot_actions[k] = f"EPRX3_{d}slots"

        for i in range(num_slots):
            if slot_actions[i] == "idle":
                if pulp.value(charge[i]) > 0.5:
                    slot_actions[i] = "charge"
                elif pulp.value(discharge[i]) > 0.5:
                    slot_actions[i] = "discharge"

        soc_vals = [pulp.value(battery_soc[i]) for i in range(num_slots+1)]

        day_profit = 0.0
        transactions = []
        for i in range(num_slots):
            tx = {
                "date": target_date,
                "slot": int(df_day.loc[i, "slot"]),
                "action": slot_actions[i],
                "battery_level_kWh": soc_vals[i+1],
                "JEPX_actual": df_day.loc[i, "JEPX_actual"],
                "EPRX1_actual": df_day.loc[i, "EPRX1_actual"],
                "EPRX3_actual": df_day.loc[i, "EPRX3_actual"],
                "imbalance": df_day.loc[i, "imbalance"]
            }
            transactions.append(tx)

        # 実価格での日次収益
        for tx in transactions:
            act = tx["action"]
            jepx_a = tx["JEPX_actual"] if not pd.isna(tx["JEPX_actual"]) else 0.0
            e1_a   = tx["EPRX1_actual"] if not pd.isna(tx["EPRX1_actual"]) else 0.0
            e3_a   = tx["EPRX3_actual"] if not pd.isna(tx["EPRX3_actual"]) else 0.0
            imb_a  = tx["imbalance"] if not pd.isna(tx["imbalance"]) else 0.0

            if act == "charge":
                # 託送ロスのみ
                cost = jepx_a * (half_power_kWh / (1 - wheeling_loss_rate))
                day_profit -= cost
            elif act == "discharge":
                # バッテリー損失のみ
                rev = jepx_a * (half_power_kWh * (1 - battery_loss_rate))
                day_profit += rev
            elif act.startswith("EPRX1_"):
                day_profit += e1_a * battery_power_kW
            elif act.startswith("EPRX3_"):
                # スロット単位の EPRX3_actual × power_kW
                day_profit += e3_a * battery_power_kW
            else:
                pass

        # EPRX3ブロック単位の imbalance 売電
        for (s, d) in eprx3_blocks:
            if pulp.value(eprx3_block[(s,d)]) > 0.5:
                imb_vals = []
                for k in range(s, s+d):
                    val = df_day.loc[k, "imbalance"]
                    if pd.isna(val):
                        val = 0.0
                    imb_vals.append(val)
                if d > 0:
                    imb_avg = sum(imb_vals)/d
                else:
                    imb_avg = 0.0
                # 放電量(バッテリー視点)= d*half_power_kWh
                # 実際に売れる量 = d*half_power_kWh * (1-battery_loss_rate)
                day_profit += (d*half_power_kWh*(1 - battery_loss_rate)) * imb_avg

        all_results.append({
            "date": target_date,
            "transactions": transactions,
            "day_profit": day_profit,
            "status": status
        })

    # 集計
    total_profit = 0.0
    final_txs = []
    for res in all_results:
        total_profit += res["day_profit"]
        final_txs.extend(res["transactions"])

    # wheeling費用 (仮)
    total_charge_kWh = 0.0
    total_discharge_kWh = 0.0
    for row in final_txs:
        if row["action"] == "charge":
            total_charge_kWh += half_power_kWh
        elif row["action"] == "discharge":
            total_discharge_kWh += half_power_kWh
        elif row["action"].startswith("EPRX3_"):
            total_discharge_kWh += half_power_kWh

    diff_kWh = max(0, total_charge_kWh - total_discharge_kWh)
    monthly_fee = wheeling_base_charge * battery_power_kW + wheeling_usage_fee * diff_kWh

    final_profit = total_profit - monthly_fee
    print("======================================")
    print(f"試算対象日の合計収益(実際価格ベース): {total_profit:.2f} 円")
    print(f"推定 wheeling 費用 (概算): {monthly_fee:.2f} 円")
    print(f"最終的な純収益: {final_profit:.2f} 円")
    print("======================================")

    # 出力CSV
    df_out = pd.DataFrame(final_txs)
    df_out.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8")
    print(f"最適化結果を {OUTPUT_CSV_PATH} に出力しました。")

if __name__ == "__main__":
    main()