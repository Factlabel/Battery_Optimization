import pandas as pd
import numpy as np
import pulp
import yaml
import os
from pathlib import Path  # 追加

## ====================================
# 各種設定
# ====================================

SCRIPT_DIR = Path(__file__).resolve().parent

# YAML
BASE_YML_PATH = SCRIPT_DIR.parent / "configs" / "base.yml"
WHEELING_YML_PATH = SCRIPT_DIR.parent / "configs" / "wheeling.yaml"

# CSV
DATA_CSV_PATH = SCRIPT_DIR.parent / "data" / "price_forecast_sample.csv"

# 出力ファイル
OUTPUT_CSV_PATH = SCRIPT_DIR / "optimal_transactions.csv"

def main():
    # 1) 設定ファイルの読み込み
    if not BASE_YML_PATH.exists():
        print(f"ERROR: {BASE_YML_PATH} が見つかりません。")
        return
    if not WHEELING_YML_PATH.exists():
        print(f"ERROR: {WHEELING_YML_PATH} が見つかりません。")
        return

    with BASE_YML_PATH.open("r", encoding="utf-8") as f:
        base_yaml = yaml.safe_load(f)
    with WHEELING_YML_PATH.open("r", encoding="utf-8") as f:
        wheeling_yaml = yaml.safe_load(f)

    battery_cfg = base_yaml.get("battery", {})
    battery_loss_rate = battery_cfg.get("loss_rate", 0.05)
    battery_power_kW = battery_cfg.get("power_kW", 50)
    battery_capacity_kWh = battery_cfg.get("capacity_kWh", 200)

    forecast_period = battery_cfg.get("forecast_period", 48)

    region_settings = wheeling_yaml.get("Kyushu", {})
    hv_settings = region_settings.get("HV", {})
    wheeling_loss_rate = hv_settings.get("loss_rate", 0.03)
    wheeling_base_charge = hv_settings.get("wheeling_base_charge", 1000)
    wheeling_usage_fee = hv_settings.get("wheeling_usage_fee", 3)

    # 2) CSVの読み込み
    if not DATA_CSV_PATH.exists():
        print(f"ERROR: {DATA_CSV_PATH} が見つかりません。")
        return
    df_all = pd.read_csv(DATA_CSV_PATH)

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

    # 時系列順にソート
    df_all.sort_values(by=["date", "slot"], inplace=True, ignore_index=True)

    total_slots = len(df_all)
    num_days = (total_slots + forecast_period - 1) // forecast_period

    print(f"全スロット: {total_slots}, forecast_period={forecast_period}, => {num_days}日分 として最適化")

    carry_over_soc = 0.0
    all_transactions = []
    total_profit = 0.0

    # 1スロットあたりの充放電量(kWh) (例: 50kWバッテリーで30分スロットなら25kWh)
    half_power_kWh = battery_power_kW * 0.5

    for day_idx in range(num_days):
        start_i = day_idx * forecast_period
        end_i   = start_i + forecast_period
        if start_i >= total_slots:
            break
        if end_i > total_slots:
            end_i = total_slots

        df_day = df_all.iloc[start_i:end_i].copy()
        df_day.reset_index(drop=True, inplace=True)
        day_slots = len(df_day)
        if day_slots == 0:
            break

        print(f"\n=== Day {day_idx+1}: スロット {start_i}~{end_i-1} ({day_slots}スロット) ===")

        prob = pulp.LpProblem(f"Battery_Optimization_Day{day_idx+1}", pulp.LpMaximize)

        battery_soc = pulp.LpVariable.dicts(f"soc_day{day_idx+1}", range(day_slots+1),
                                            lowBound=0, upBound=battery_capacity_kWh, cat=pulp.LpContinuous)

        charge    = pulp.LpVariable.dicts(f"charge_day{day_idx+1}", range(day_slots), cat=pulp.LpBinary)
        discharge = pulp.LpVariable.dicts(f"discharge_day{day_idx+1}", range(day_slots), cat=pulp.LpBinary)
        eprx1     = pulp.LpVariable.dicts(f"eprx1_day{day_idx+1}", range(day_slots), cat=pulp.LpBinary)
        eprx3     = pulp.LpVariable.dicts(f"eprx3_day{day_idx+1}", range(day_slots), cat=pulp.LpBinary)

        # (A) 排他制約
        for i in range(day_slots):
            prob += (charge[i] + discharge[i] + eprx1[i] + eprx3[i]) <= 1

        # (B) SOC遷移
        for i in range(day_slots):
            next_soc = (
                battery_soc[i]
                + charge[i] * half_power_kWh
                - discharge[i] * half_power_kWh
                - eprx3[i] * half_power_kWh
            )
            prob += battery_soc[i+1] == next_soc

        # 初期バッテリー量
        prob += battery_soc[0] == carry_over_soc, f"InitSOC_Day{day_idx+1}"

        # EPRX1 (1日合計6スロット) + 40~60% 制約
        bigM = 999999
        prob += pulp.lpSum(eprx1[i] for i in range(day_slots)) <= 6
        for i in range(day_slots):
            soc_i = battery_soc[i]
            prob += soc_i >= 0.4 * battery_capacity_kWh - (1 - eprx1[i]) * bigM
            prob += soc_i <= 0.6 * battery_capacity_kWh + (1 - eprx1[i]) * bigM

        # EPRX3 => SoC >= half_power_kWh
        for i in range(day_slots):
            prob += battery_soc[i] >= half_power_kWh - (1 - eprx3[i]) * bigM

        # 1日最大充電量 <= capacity_kWh
        prob += (pulp.lpSum([charge[i] for i in range(day_slots)]) * half_power_kWh) <= battery_capacity_kWh

        # prediction=0 => eprx1,eprx3=0
        for i in range(day_slots):
            e1pred = df_day.loc[i, "EPRX1_prediction"]
            e3pred = df_day.loc[i, "EPRX3_prediction"]
            if pd.isna(e1pred) or e1pred == 0:
                prob += eprx1[i] <= 0
            if pd.isna(e3pred) or e3pred == 0:
                prob += eprx3[i] <= 0

        # 目的関数(予測価格)
        profit_terms = []
        for i in range(day_slots):
            jpred  = df_day.loc[i, "JEPX_prediction"]  if not pd.isna(df_day.loc[i, "JEPX_prediction"]) else 0.0
            e1pred = df_day.loc[i, "EPRX1_prediction"] if not pd.isna(df_day.loc[i, "EPRX1_prediction"]) else 0.0
            e3pred = df_day.loc[i, "EPRX3_prediction"] if not pd.isna(df_day.loc[i, "EPRX3_prediction"]) else 0.0
            imb    = df_day.loc[i, "imbalance"]        if not pd.isna(df_day.loc[i, "imbalance"]) else 0.0

            # 充放電の費用/利益 (予測価格ベース)
            cost_c  = jpred * (half_power_kWh/(1 - wheeling_loss_rate)) * charge[i]
            rev_d   = jpred * (half_power_kWh * (1 - battery_loss_rate)) * discharge[i]
            rev_e1  = e1pred * battery_power_kW * eprx1[i]
            rev_e3  = e3pred * battery_power_kW * eprx3[i]
            # imbalanceのプラス分を放電収益に上乗せ (EPRX3のみ)
            rev_e3 += (half_power_kWh * (1 - battery_loss_rate)) * imb * eprx3[i]

            slot_profit = -cost_c + rev_d + rev_e1 + rev_e3
            profit_terms.append(slot_profit)

        prob += pulp.lpSum(profit_terms), f"TotalProfit_Day{day_idx+1}"

        # ソルバー実行
        solver = pulp.PULP_CBC_CMD(msg=0, threads=8)
        prob.solve(solver)

        status = pulp.LpStatus[prob.status]
        print(f"Day {day_idx+1} solve status: {status}")
        if status != "Optimal":
            print("Warning: Dayの最適解が見つかりませんでした (skip).")
            continue

        # 実際の収益計算 (日次)
        day_day_profit = 0.0
        day_transactions = []

        final_soc = pulp.value(battery_soc[day_slots])
        print(f"Day {day_idx+1} final SOC= {final_soc:.2f} kWh")
        carry_over_soc = final_soc

        for i in range(day_slots):
            c_val  = pulp.value(charge[i])
            d_val  = pulp.value(discharge[i])
            e1_val = pulp.value(eprx1[i])
            e3_val = pulp.value(eprx3[i])

            act = "idle"
            if e1_val > 0.5:
                act = "EPRX1"
            elif e3_val > 0.5:
                act = "EPRX3"
            elif c_val > 0.5:
                act = "charge"
            elif d_val > 0.5:
                act = "discharge"

            # 各スロットごとのPnL: 実際価格で再計算
            j_a  = df_day.loc[i, "JEPX_actual"]  if not pd.isna(df_day.loc[i, "JEPX_actual"]) else 0.0
            e1_a = df_day.loc[i, "EPRX1_actual"] if not pd.isna(df_day.loc[i, "EPRX1_actual"]) else 0.0
            e3_a = df_day.loc[i, "EPRX3_actual"] if not pd.isna(df_day.loc[i, "EPRX3_actual"]) else 0.0
            imb_a= df_day.loc[i, "imbalance"]    if not pd.isna(df_day.loc[i, "imbalance"]) else 0.0

            jepx_pnl = 0.0
            eprx1_pnl = 0.0
            eprx3_pnl = 0.0
            wheeling_cost = 0.0  # 個別スロットの wheeling費用はここでは計上していない

            if act == "charge":
                # 実際価格 * （battery_power_kW*0.5 / (1 - wheeling_loss_rate)）
                cost = j_a * (half_power_kWh / (1 - wheeling_loss_rate))
                jepx_pnl -= cost
            elif act == "discharge":
                # 実際価格 * バッテリー内部ロス控除後(= half_power_kWh*(1 - battery_loss_rate))
                rev = j_a * (half_power_kWh * (1 - battery_loss_rate))
                jepx_pnl += rev
            elif act == "EPRX1":
                eprx1_pnl += e1_a * battery_power_kW
            elif act == "EPRX3":
                rev = e3_a * battery_power_kW
                rev += (half_power_kWh*(1 - battery_loss_rate)) * imb_a
                eprx3_pnl += rev

            slot_total_pnl = jepx_pnl + eprx1_pnl + eprx3_pnl - wheeling_cost
            day_day_profit += slot_total_pnl

            row = {
                "date": df_day.loc[i, "date"],
                "slot": int(df_day.loc[i, "slot"]),
                "action": act,
                "battery_level_kWh": pulp.value(battery_soc[i+1]),
                "JEPX_prediction": df_day.loc[i, "JEPX_prediction"],
                "JEPX_actual": df_day.loc[i, "JEPX_actual"],
                "EPRX1_prediction": df_day.loc[i, "EPRX1_prediction"],
                "EPRX1_actual": df_day.loc[i, "EPRX1_actual"],
                "EPRX3_prediction": df_day.loc[i, "EPRX3_prediction"],
                "EPRX3_actual": df_day.loc[i, "EPRX3_actual"],
                "imbalance": df_day.loc[i, "imbalance"],

                "JEPX_PnL": jepx_pnl,
                "EPRX1_PnL": eprx1_pnl,
                "EPRX3_PnL": eprx3_pnl,
                "Wheeling_charges": wheeling_cost,
                "Total_Daily_PnL": slot_total_pnl
            }
            day_transactions.append(row)

        all_transactions.extend(day_transactions)
        total_profit += day_day_profit
        print(f"Day {day_idx+1} PL= {day_day_profit:.2f}, 累計= {total_profit:.2f}")

    # ===========================
    # wheeling費用(概算)の計算
    # ===========================
    # 1ヶ月のうち総充電量(送電ロス率は考慮せずbatteryに充電された量) -
    # 総放電量(バッテリー内部ロスを差し引いた実放電量) = 差分に従量料金をかける
    total_charge_kWh = 0.0
    total_discharge_kWh = 0.0

    for r in all_transactions:
        act = r["action"]
        if act == "charge":
            # 送電ロスは考慮せず、バッテリーに入った量だけを合計
            total_charge_kWh += (battery_power_kW * 0.5)

        elif act == "discharge":
            # バッテリー内部ロス後の実放電量
            actual_discharge_kWh = (battery_power_kW * 0.5) * (1 - battery_loss_rate)
            total_discharge_kWh += actual_discharge_kWh

        elif act == "EPRX3":
            # EPRX3 も放電を伴うので同様に
            actual_discharge_kWh = (battery_power_kW * 0.5) * (1 - battery_loss_rate)
            total_discharge_kWh += actual_discharge_kWh

    usage_fee_kWh = max(0, total_charge_kWh - total_discharge_kWh)

    # 月額基本料金 + 従量料金(差分kWh)
    monthly_fee = wheeling_base_charge * battery_power_kW + wheeling_usage_fee * usage_fee_kWh
    final_profit2 = total_profit - monthly_fee

    print("======================================")
    print(f"全期間(全日) 累計収益(実際価格) = {total_profit:.2f} 円")
    print(f"推定 wheeling費用(概算)      = {monthly_fee:.2f} 円")
    print(f"最終的な純収益               = {final_profit2:.2f} 円")
    print("======================================")

    # 結果をCSV出力
    df_out = pd.DataFrame(
        all_transactions,
        columns=[
            "date", "slot", "action", "battery_level_kWh",
            "JEPX_prediction", "JEPX_actual",
            "EPRX1_prediction", "EPRX1_actual",
            "EPRX3_prediction", "EPRX3_actual",
            "imbalance",
            "JEPX_PnL", "EPRX1_PnL", "EPRX3_PnL",
            "Wheeling_charges",
            "Total_Daily_PnL"
        ]
    )
    df_out.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8")
    print(f"最適化結果 (day-by-day) を {OUTPUT_CSV_PATH} に出力しました。")


if __name__ == "__main__":
    main()