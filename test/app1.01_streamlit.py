import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import yaml
import pulp
import io
import os

def run_optimization(base_yaml, wheeling_yaml, df_all):
  

    # ----------------------
    # 1) 設定ファイル読込
    # ----------------------
    battery_cfg = base_yaml.get("battery", {})
    battery_loss_rate = battery_cfg.get("loss_rate", 0.05)
    battery_power_kW = battery_cfg.get("power_kW", 50)
    battery_capacity_kWh = battery_cfg.get("capacity_kWh", 200)
    forecast_period = battery_cfg.get("forecast_period", 48)  # 1日のスロット数

    region_settings = wheeling_yaml.get("Kyushu", {})
    hv_settings = region_settings.get("HV", {})
    wheeling_loss_rate = hv_settings.get("loss_rate", 0.03)
    wheeling_base_charge = hv_settings.get("wheeling_base_charge", 1000)
    wheeling_usage_fee = hv_settings.get("wheeling_usage_fee", 3)

    required_cols = {
        "date", "slot",
        "JEPX_prediction", "JEPX_actual",
        "EPRX1_prediction", "EPRX1_actual",
        "EPRX3_prediction", "EPRX3_actual",
        "imbalance"
    }
    if not required_cols.issubset(df_all.columns):
        st.error(f"CSV is missing required columns: {required_cols}")
        return None, None, None

    # ----------------------
    # 2) CSV整形・日数特定
    # ----------------------
    df_all.sort_values(by=["date", "slot"], inplace=True, ignore_index=True)
    total_slots = len(df_all)
    # 日数 = スロット数 / forecast_period (端数切り上げ)
    num_days = (total_slots + forecast_period - 1) // forecast_period

    # ----------------------
    # 3) ループで日ごと最適化
    # ----------------------
    carry_over_soc = 0.0
    all_transactions = []
    total_profit = 0.0

    half_power_kWh = battery_power_kW * 0.5

    for day_idx in range(num_days):
        start_i = day_idx * forecast_period
        end_i = start_i + forecast_period
        if start_i >= total_slots:
            break
        if end_i > total_slots:
            end_i = total_slots

        # 当日のスロット切り出し
        df_day = df_all.iloc[start_i:end_i].copy()
        df_day.reset_index(drop=True, inplace=True)
        day_slots = len(df_day)
        if day_slots == 0:
            break

        # PuLP問題設定
        prob = pulp.LpProblem(f"Battery_Optimization_Day{day_idx+1}", pulp.LpMaximize)

        # 変数
        battery_soc = pulp.LpVariable.dicts(
            f"soc_day{day_idx+1}",
            range(day_slots+1),
            lowBound=0, upBound=battery_capacity_kWh,
            cat=pulp.LpContinuous
        )
        charge = pulp.LpVariable.dicts(f"charge_day{day_idx+1}", range(day_slots), cat=pulp.LpBinary)
        discharge = pulp.LpVariable.dicts(f"discharge_day{day_idx+1}", range(day_slots), cat=pulp.LpBinary)
        eprx1 = pulp.LpVariable.dicts(f"eprx1_day{day_idx+1}", range(day_slots), cat=pulp.LpBinary)
        eprx3 = pulp.LpVariable.dicts(f"eprx3_day{day_idx+1}", range(day_slots), cat=pulp.LpBinary)

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

        # (C) 初期SOC
        prob += battery_soc[0] == carry_over_soc, f"InitSOC_Day{day_idx+1}"

        # (D) EPRX1: 1日最大6スロット, SOC 40~60%時のみ
        bigM = 999999
        prob += pulp.lpSum(eprx1[i] for i in range(day_slots)) <= 6
        for i in range(day_slots):
            soc_i = battery_soc[i]
            prob += soc_i >= 0.4 * battery_capacity_kWh - (1 - eprx1[i]) * bigM
            prob += soc_i <= 0.6 * battery_capacity_kWh + (1 - eprx1[i]) * bigM

        # (E) EPRX3 => SOC >= half_power_kWh
        for i in range(day_slots):
            prob += battery_soc[i] >= half_power_kWh - (1 - eprx3[i]) * bigM

        # (F) 1日最大充電量 <= capacity
        prob += (
            pulp.lpSum(charge[i] for i in range(day_slots)) * half_power_kWh
        ) <= battery_capacity_kWh

        # (G) EPRXの予測値が0の場合 => 使えない
        for i in range(day_slots):
            e1pred = df_day.loc[i, "EPRX1_prediction"]
            e3pred = df_day.loc[i, "EPRX3_prediction"]
            if pd.isna(e1pred) or e1pred == 0:
                prob += eprx1[i] <= 0
            if pd.isna(e3pred) or e3pred == 0:
                prob += eprx3[i] <= 0

        # (H) 目的関数(予測価格での利益)
        profit_terms = []
        for i in range(day_slots):
            jpred = df_day.loc[i, "JEPX_prediction"] if not pd.isna(df_day.loc[i, "JEPX_prediction"]) else 0.0
            e1pred = df_day.loc[i, "EPRX1_prediction"] if not pd.isna(df_day.loc[i, "EPRX1_prediction"]) else 0.0
            e3pred = df_day.loc[i, "EPRX3_prediction"] if not pd.isna(df_day.loc[i, "EPRX3_prediction"]) else 0.0
            imb = df_day.loc[i, "imbalance"] if not pd.isna(df_day.loc[i, "imbalance"]) else 0.0

            cost_c = jpred * (half_power_kWh / (1 - wheeling_loss_rate)) * charge[i]
            rev_d = jpred * (half_power_kWh * (1 - battery_loss_rate)) * discharge[i]
            rev_e1 = e1pred * battery_power_kW * eprx1[i]
            rev_e3 = e3pred * battery_power_kW * eprx3[i]
            rev_e3 += (half_power_kWh * (1 - battery_loss_rate)) * imb * eprx3[i]

            slot_profit = -cost_c + rev_d + rev_e1 + rev_e3
            profit_terms.append(slot_profit)

        prob += pulp.lpSum(profit_terms), f"TotalProfit_Day{day_idx+1}"

        # ソルバー実行
        solver = pulp.PULP_CBC_CMD(msg=0, threads=4)
        prob.solve(solver)
        status = pulp.LpStatus[prob.status]
        if status != "Optimal":
            # 最適解が見つからなければ、その日の計算はスキップ
            continue

        # ----------------------
        # 4) 実際価格での利益計算
        # ----------------------
        day_profit = 0.0
        final_soc = pulp.value(battery_soc[day_slots])
        carry_over_soc = final_soc  # 翌日の初期SOCへ引き継ぐ

        # スロットごとの結果
        day_transactions = []
        for i in range(day_slots):
            c_val = pulp.value(charge[i])
            d_val = pulp.value(discharge[i])
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

            j_a = df_day.loc[i, "JEPX_actual"] if not pd.isna(df_day.loc[i, "JEPX_actual"]) else 0.0
            e1_a = df_day.loc[i, "EPRX1_actual"] if not pd.isna(df_day.loc[i, "EPRX1_actual"]) else 0.0
            e3_a = df_day.loc[i, "EPRX3_actual"] if not pd.isna(df_day.loc[i, "EPRX3_actual"]) else 0.0
            imb_a = df_day.loc[i, "imbalance"] if not pd.isna(df_day.loc[i, "imbalance"]) else 0.0

            slot_jepx_pnl = 0.0
            slot_eprx1_pnl = 0.0
            slot_eprx3_pnl = 0.0

            if act == "charge":
                cost = j_a * (half_power_kWh / (1 - wheeling_loss_rate))
                slot_jepx_pnl -= cost
            elif act == "discharge":
                rev = j_a * (half_power_kWh * (1 - battery_loss_rate))
                slot_jepx_pnl += rev
            elif act == "EPRX1":
                slot_eprx1_pnl += e1_a * battery_power_kW
            elif act == "EPRX3":
                rev = e3_a * battery_power_kW
                rev += (half_power_kWh * (1 - battery_loss_rate)) * imb_a
                slot_eprx3_pnl += rev

            slot_total_pnl = slot_jepx_pnl + slot_eprx1_pnl + slot_eprx3_pnl
            day_profit += slot_total_pnl

            row = {
                "date": df_day.loc[i, "date"],
                "slot": int(df_day.loc[i, "slot"]),
                "action": act,
                "battery_level_kWh": pulp.value(battery_soc[i+1]),
                "JEPX_prediction": df_day.loc[i, "JEPX_prediction"],
                "JEPX_actual": j_a,
                "EPRX1_prediction": df_day.loc[i, "EPRX1_prediction"],
                "EPRX1_actual": e1_a,
                "EPRX3_prediction": df_day.loc[i, "EPRX3_prediction"],
                "EPRX3_actual": e3_a,
                "imbalance": imb_a,
                "JEPX_PnL": slot_jepx_pnl,
                "EPRX1_PnL": slot_eprx1_pnl,
                "EPRX3_PnL": slot_eprx3_pnl,
                "Total_Daily_PnL": slot_total_pnl
            }
            day_transactions.append(row)

        all_transactions.extend(day_transactions)
        total_profit += day_profit

    # ----------------------
    # 5) Wheeling費用(概算)
    #    (総充電量 - 内部ロス後の放電量) に従量課金
    # ----------------------
    total_charge_kWh = 0.0
    total_discharge_kWh = 0.0
    for r in all_transactions:
        if r["action"] == "charge":
            # 送電ロスは考慮しない: バッテリーに入った量
            total_charge_kWh += (battery_power_kW * 0.5)
        elif r["action"] == "discharge":
            # バッテリー内部ロス後の実放電量
            actual_discharge_kWh = (battery_power_kW * 0.5) * (1 - battery_loss_rate)
            total_discharge_kWh += actual_discharge_kWh
        elif r["action"] == "EPRX3":
            # EPRX3 も放電を伴うので同様に
            actual_discharge_kWh = (battery_power_kW * 0.5) * (1 - battery_loss_rate)
            total_discharge_kWh += actual_discharge_kWh

    usage_fee_kWh = max(0, total_charge_kWh - total_discharge_kWh)
    monthly_fee = wheeling_base_charge * battery_power_kW + wheeling_usage_fee * usage_fee_kWh
    final_profit = total_profit - monthly_fee

    return all_transactions, total_profit, final_profit


############################
# Streamlit UI
############################

def main():
    st.title("Battery Optimizer 1.01 (Streamlit版)")

    base_file = st.file_uploader("BASE_YML", type=["yml", "yaml"])
    wheeling_file = st.file_uploader("WHEELING_YML", type=["yml", "yaml"])
    data_file = st.file_uploader("DATA_CSV", type=["csv"])

    # セッションステートで結果を保持
    if "calc_results" not in st.session_state:
        st.session_state["calc_results"] = None
        st.session_state["calc_day_profit"] = 0.0
        st.session_state["calc_final_profit"] = 0.0

    if base_file and wheeling_file and data_file:
        # ファイル読み込み
        with io.TextIOWrapper(base_file, encoding="utf-8") as bf:
            base_yaml = yaml.safe_load(bf)
        with io.TextIOWrapper(wheeling_file, encoding="utf-8") as wf:
            wheeling_yaml = yaml.safe_load(wf)
        df_all = pd.read_csv(data_file)

        # date列があればDatetimeに変換
        if "date" in df_all.columns:
            df_all["date"] = pd.to_datetime(df_all["date"], errors="coerce")

        # 「Calculate」ボタン
        if st.button("Calculate"):
            results, day_profit, final_profit = run_optimization(base_yaml, wheeling_yaml, df_all)

            if results is None:
                st.warning("No optimal solution found or missing columns.")
                return

            # 計算結果を session_state に保存しておく
            st.session_state["calc_results"] = results
            st.session_state["calc_day_profit"] = day_profit
            st.session_state["calc_final_profit"] = final_profit

    # 計算結果が session_state にあるかどうかで表示を分ける
    if st.session_state["calc_results"] is not None:
        results = st.session_state["calc_results"]
        day_profit = st.session_state["calc_day_profit"]
        final_profit = st.session_state["calc_final_profit"]

        st.success("Calculation Completed.")
        st.write(f"**Total Profit(Actual Price)**: {day_profit:,.2f} 円")
        st.write(f"**Final Profit (after wheeling fee)**: {final_profit:,.2f} 円")

        # 結果をDataFrame化
        df_res = pd.DataFrame(results)
        df_res.sort_values(by=["date", "slot"], inplace=True, ignore_index=True)

        # 結果テーブルの表示
        st.dataframe(df_res, height=600)

        # 日付範囲選択
        st.subheader("Select Date Range for Graph")
        min_date = df_res["date"].min()
        max_date = df_res["date"].max()

        # デフォルトで3日分: (start_date=最初の日, end_date=最初の日+2日)
        default_start = min_date
        default_end = min_date + pd.Timedelta(days=2)
        if default_end > max_date:
            default_end = max_date

        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=default_start,
                min_value=min_date,
                max_value=max_date
            )
        with col2:
            end_date = st.date_input(
                "End Date",
                value=default_end,
                min_value=min_date,
                max_value=max_date
            )

        if start_date > end_date:
            st.warning("Invalid date range.")
            return

        # グラフ描画用にフィルタ
        df_g = df_res[
            (df_res["date"] >= pd.to_datetime(start_date)) & (df_res["date"] <= pd.to_datetime(end_date))
        ].copy()
        df_g.reset_index(drop=True, inplace=True)

        if len(df_g) == 0:
            st.warning("No data in the selected date range.")
            return

        # バッテリー残量とJEPX価格を描画
        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax2 = ax1.twinx()
        x_vals = range(len(df_g))

        ax1.bar(x_vals, df_g["battery_level_kWh"], color="lightblue", label="Battery(kWh)")
        if "JEPX_actual" in df_g.columns:
            ax2.plot(x_vals, df_g["JEPX_actual"], color="red", label="JEPX(Actual)")

        ax1.set_ylabel("Battery Level (kWh)")
        ax2.set_ylabel("JEPX Price")
        ax1.legend(loc="upper left")
        ax2.legend(loc="upper right")
        st.pyplot(fig)

        # CSVダウンロード
        csv_data = df_res.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv_data,
            file_name="optimal_transactions.csv",
            mime="text/csv"
        )

    else:
        # 計算前または未アップロード状態の画面
        pass


if __name__ == "__main__":
    main()