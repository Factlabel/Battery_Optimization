import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pulp

# ----------------------------------------------
# wheeling.yaml の内容をコードに埋め込んだ辞書
# （託送料金は既に税込み想定）
# ----------------------------------------------
WHEELING_DATA = {
    "areas": {
        "Hokkaido": {
            "SHV": {
                "loss_rate": 0.02,
                "wheeling_base_charge": 503.80,
                "wheeling_usage_fee": 0.92
            },
            "HV": {
                "loss_rate": 0.047,
                "wheeling_base_charge": 792.00,
                "wheeling_usage_fee": 2.17
            },
            "LV": {
                "loss_rate": 0.079,
                "wheeling_base_charge": 618.20,
                "wheeling_usage_fee": 4.22
            }
        },
        "Tohoku": {
            "SHV": {
                "loss_rate": 0.019,
                "wheeling_base_charge": 630.30,
                "wheeling_usage_fee": 8.57
            },
            "HV": {
                "loss_rate": 0.052,
                "wheeling_base_charge": 706.20,
                "wheeling_usage_fee": 2.08
            },
            "LV": {
                "loss_rate": 0.085,
                "wheeling_base_charge": 456.50,
                "wheeling_usage_fee": 2.08
            }
        },
        "Tokyo": {
            "SHV": {
                "loss_rate": 0.013,
                "wheeling_base_charge": 423.39,
                "wheeling_usage_fee": 1.33
            },
            "HV": {
                "loss_rate": 0.037,
                "wheeling_base_charge": 653.87,
                "wheeling_usage_fee": 2.37
            },
            "LV": {
                "loss_rate": 0.069,
                "wheeling_base_charge": 461.14,
                "wheeling_usage_fee": 5.20
            }
        },
        "Chubu": {
            "SHV": {
                "loss_rate": 0.025,
                "wheeling_base_charge": 357.50,
                "wheeling_usage_fee": 0.88
            },
            "HV": {
                "loss_rate": 0.038,
                "wheeling_base_charge": 467.50,
                "wheeling_usage_fee": 2.21
            },
            "LV": {
                "loss_rate": 0.071,
                "wheeling_base_charge": 412.50,
                "wheeling_usage_fee": 6.07
            }
        },
        "Hokuriku": {
            "SHV": {
                "loss_rate": 0.013,
                "wheeling_base_charge": 572.00,
                "wheeling_usage_fee": 0.85
            },
            "HV": {
                "loss_rate": 0.034,
                "wheeling_base_charge": 748.00,
                "wheeling_usage_fee": 1.76
            },
            "LV": {
                "loss_rate": 0.078,
                "wheeling_base_charge": 396.00,
                "wheeling_usage_fee": 4.69
            }
        },
        "Kansai": {
            "SHV": {
                "loss_rate": 0.029,
                "wheeling_base_charge": 440.00,
                "wheeling_usage_fee": 0.84
            },
            "HV": {
                "loss_rate": 0.078,
                "wheeling_base_charge": 663.30,
                "wheeling_usage_fee": 2.29
            },
            "LV": {
                "loss_rate": 0.078,
                "wheeling_base_charge": 378.40,
                "wheeling_usage_fee": 4.69
            }
        },
        "Chugoku": {
            "SHV": {
                "loss_rate": 0.025,
                "wheeling_base_charge": 383.90,
                "wheeling_usage_fee": 0.70
            },
            "HV": {
                "loss_rate": 0.044,
                "wheeling_base_charge": 658.90,
                "wheeling_usage_fee": 2.43
            },
            "LV": {
                "loss_rate": 0.077,
                "wheeling_base_charge": 466.40,
                "wheeling_usage_fee": 6.07
            }
        },
        "Shikoku": {
            "SHV": {
                "loss_rate": 0.013,
                "wheeling_base_charge": 510.40,
                "wheeling_usage_fee": 0.77
            },
            "HV": {
                "loss_rate": 0.041,
                "wheeling_base_charge": 712.80,
                "wheeling_usage_fee": 2.01
            },
            "LV": {
                "loss_rate": 0.081,
                "wheeling_base_charge": 454.30,
                "wheeling_usage_fee": 5.97
            }
        },
        "Kyushu": {
            "SHV": {
                "loss_rate": 0.013,
                "wheeling_base_charge": 482.05,
                "wheeling_usage_fee": 1.27
            },
            "HV": {
                "loss_rate": 0.032,
                "wheeling_base_charge": 553.28,
                "wheeling_usage_fee": 2.61
            },
            "LV": {
                "loss_rate": 0.086,
                "wheeling_base_charge": 379.26,
                "wheeling_usage_fee": 5.58
            }
        }
    }
}

AREA_NUMBER_TO_NAME = {
    1: "Hokkaido",
    2: "Tohoku",
    3: "Tokyo",
    4: "Chubu",
    5: "Hokuriku",
    6: "Kansai",
    7: "Chugoku",
    8: "Shikoku",
    9: "Kyushu"
}


def run_optimization(
    target_area_name,
    voltage_type,
    battery_power_kW,
    battery_capacity_kWh,
    battery_loss_rate,
    daily_cycle_limit,
    yearly_cycle_limit,
    annual_degradation_rate,
    forecast_period,
    eprx1_block_size,       # EPRX1 は“このスロット数”連続で使用
    eprx1_block_cooldown,   # ブロック終了後、このスロット数は EPRX1 を開始できない
    max_daily_eprx1_slots,  # 1日あたり EPRX1 の使用スロット数の上限
    df_all
):
    """
      EPRX1:
         - M スロット連続で使う (ブロック)
         - ブロック終了後、C スロットは EPRX1ブロック開始不可
         - さらに EPRX1を使うスロット(SOCが 40~60%) かつ 1日あたりの EPRX1使用スロット数 <= max_daily_eprx1_slots
      EPRX3:
         - スロットごと (0~1連続値)
         - 予測値が 0 or NaN のスロットは使えない
    """

    wh = WHEELING_DATA["areas"].get(target_area_name, {}).get(voltage_type, {})
    wheeling_loss_rate = wh.get("loss_rate", 0.0)
    wheeling_base_charge = wh.get("wheeling_base_charge", 0.0)
    wheeling_usage_fee = wh.get("wheeling_usage_fee", 0.0)

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

    df_all.sort_values(by=["date", "slot"], inplace=True, ignore_index=True)
    total_slots = len(df_all)
    num_days = (total_slots + forecast_period - 1) // forecast_period

    carry_over_soc = 0.0
    all_transactions = []
    total_profit = 0.0

    half_power_kWh = battery_power_kW * 0.5  # スロット1時間仮定

    total_cycles_used = 0.0

    for day_idx in range(num_days):
        start_i = day_idx * forecast_period
        end_i = min(start_i + forecast_period, total_slots)
        if start_i >= total_slots:
            break

        df_day = df_all.iloc[start_i:end_i].copy()
        df_day.reset_index(drop=True, inplace=True)
        day_slots = len(df_day)
        if day_slots == 0:
            break

        prob = pulp.LpProblem(f"Battery_Optimization_Day{day_idx+1}", pulp.LpMaximize)

        # 充電/放電/EPRX3
        charge = pulp.LpVariable.dicts(
            f"charge_day{day_idx+1}",
            range(day_slots),
            lowBound=0, upBound=1,
            cat=pulp.LpContinuous
        )
        discharge = pulp.LpVariable.dicts(
            f"discharge_day{day_idx+1}",
            range(day_slots),
            lowBound=0, upBound=1,
            cat=pulp.LpContinuous
        )
        eprx3 = pulp.LpVariable.dicts(
            f"eprx3_day{day_idx+1}",
            range(day_slots),
            lowBound=0, upBound=1,
            cat=pulp.LpContinuous
        )

        # EPRX1 は「ブロック開始」を表すバイナリ変数
        block_start = pulp.LpVariable.dicts(
            f"block_start_day{day_idx+1}",
            range(day_slots),
            cat=pulp.LpBinary
        )

        # バッテリーSOC
        battery_soc = pulp.LpVariable.dicts(
            f"soc_day{day_idx+1}",
            range(day_slots+1),
            lowBound=0, upBound=battery_capacity_kWh,
            cat=pulp.LpContinuous
        )

        M = eprx1_block_size
        C = eprx1_block_cooldown
        bigM = 999999

        # (A) is_in_block[i]: スロット i が EPRX1ブロック中かどうか (0/1)
        #     i が x から x+M-1 の範囲内なら is_in_block[i] = 1
        is_in_block = {}
        for i in range(day_slots):
            is_in_block[i] = pulp.LpVariable(f"in_block_{day_idx+1}_slot{i}", cat=pulp.LpBinary)

            possible_starts = []
            for x in range(max(0, i - (M - 1)), i+1):
                if x + M - 1 >= i:
                    possible_starts.append(x)
            # i をカバーするブロック開始 x の合計が is_in_block[i]
            prob += is_in_block[i] == pulp.lpSum([block_start[x] for x in possible_starts])

        # (A') EPRX3: 予測値が 0 or NaN なら使えない
        for i in range(day_slots):
            e3pred = df_day.loc[i, "EPRX3_prediction"]
            if pd.isna(e3pred) or e3pred == 0:
                prob += eprx3[i] <= 0

        # (B) 同一スロットで EPRX1ブロック中なら (charge, discharge, eprx3=0)
        for i in range(day_slots):
            prob += (charge[i] + discharge[i] + eprx3[i]) <= (1 - is_in_block[i])

        # (C) SOC 遷移
        for i in range(day_slots):
            next_soc = (
                battery_soc[i]
                + charge[i] * half_power_kWh
                - discharge[i] * half_power_kWh
                - eprx3[i] * half_power_kWh
            )
            prob += battery_soc[i+1] == next_soc

        # (D) 初期SOC
        prob += battery_soc[0] == carry_over_soc

        # (E) EPRX1 のブロック連続制約 + クールダウン
        #     block_start[i] = 1 ⇒ そこから M スロット中は他の start 禁止
        #                         & さらに C スロットクールダウン
        for i in range(day_slots):
            # i + M + C - 1 < day_slots の場合だけ loop
            end_j = min(day_slots, i + M + C)
            for j in range(i+1, end_j):
                prob += block_start[i] + block_start[j] <= 1
            # i > day_slots - M なら残り枠不足でブロック開始できない
            if i > day_slots - M:
                prob += block_start[i] == 0

        # (F) 日次充電量制約
        prob += (
            pulp.lpSum(charge[i] for i in range(day_slots)) * half_power_kWh
        ) <= daily_cycle_limit * battery_capacity_kWh

        # (G) 予測値が 0 or NaN の場合は EPRX1ブロック開始不可
        #     (i..i+M-1 のいずれかが 0⇒開始不可)
        for i in range(day_slots):
            for slot_in_block in range(i, min(i+M, day_slots)):
                e1pred = df_day.loc[slot_in_block, "EPRX1_prediction"]
                if pd.isna(e1pred) or e1pred == 0:
                    prob += block_start[i] <= 0

        # (H) EPRX1 で使うスロットの SOC を 40～60% にする
        #     => is_in_block[i] =1 ⇒ battery_soc[i] in [0.4~0.6 * capacity]
        for i in range(day_slots):
            prob += battery_soc[i] >= 0.4*battery_capacity_kWh - (1 - is_in_block[i])*bigM
            prob += battery_soc[i] <= 0.6*battery_capacity_kWh + (1 - is_in_block[i])*bigM

        # (I) 1日あたり EPRX1 最大スロット数
        #     => sum(is_in_block[i]) <= max_daily_eprx1_slots
        if max_daily_eprx1_slots >= 0:
            prob += pulp.lpSum(is_in_block[i] for i in range(day_slots)) <= max_daily_eprx1_slots

        # (J) 目的関数(予測価格での収益最大化)
        profit_terms = []
        for i in range(day_slots):
            jpred = df_day.loc[i, "JEPX_prediction"] if not pd.isna(df_day.loc[i, "JEPX_prediction"]) else 0.0
            cost_c = jpred * (charge[i] * half_power_kWh / (1 - wheeling_loss_rate))
            rev_d = jpred * (discharge[i] * half_power_kWh * (1 - battery_loss_rate))

            e3pred = df_day.loc[i, "EPRX3_prediction"] if not pd.isna(df_day.loc[i, "EPRX3_prediction"]) else 0.0
            imb = df_day.loc[i, "imbalance"] if not pd.isna(df_day.loc[i, "imbalance"]) else 0.0
            # EPRX3
            rev_e3 = e3pred * battery_power_kW * eprx3[i]
            rev_e3 += (half_power_kWh * (1 - battery_loss_rate)) * imb * eprx3[i]

            slot_profit = -cost_c + rev_d + rev_e3

            # EPRX1 収益
            e1pred = df_day.loc[i, "EPRX1_prediction"] if not pd.isna(df_day.loc[i, "EPRX1_prediction"]) else 0.0
            slot_profit += e1pred * battery_power_kW * is_in_block[i]

            profit_terms.append(slot_profit)

        prob += pulp.lpSum(profit_terms)

        solver = pulp.PULP_CBC_CMD(msg=0, threads=4)
        prob.solve(solver)

        status = pulp.LpStatus[prob.status]
        if status != "Optimal":
            continue

        # ---- 実際価格での収益計算
        day_profit = 0.0
        final_soc = pulp.value(battery_soc[day_slots])
        carry_over_soc = final_soc

        day_transactions = []
        TAX = 1.1

        for i in range(day_slots):
            c_val = pulp.value(charge[i])
            d_val = pulp.value(discharge[i])
            e3_val = pulp.value(eprx3[i])
            in_b_val = pulp.value(is_in_block[i])  # 0 or 1

            act = "idle"
            if in_b_val > 0.5:
                act = "EPRX1"
            elif c_val > 0:
                act = "charge"
            elif d_val > 0:
                act = "discharge"
            elif e3_val > 0:
                act = "EPRX3"

            j_a = df_day.loc[i, "JEPX_actual"] if not pd.isna(df_day.loc[i, "JEPX_actual"]) else 0.0
            e1_a = df_day.loc[i, "EPRX1_actual"] if not pd.isna(df_day.loc[i, "EPRX1_actual"]) else 0.0
            e3_a = df_day.loc[i, "EPRX3_actual"] if not pd.isna(df_day.loc[i, "EPRX3_actual"]) else 0.0
            imb_a = df_day.loc[i, "imbalance"] if not pd.isna(df_day.loc[i, "imbalance"]) else 0.0

            c_kwh = c_val * half_power_kWh
            d_kwh = d_val * half_power_kWh
            e3_kwh = e3_val * half_power_kWh

            slot_jepx_pnl = 0.0
            slot_eprx1_pnl = 0.0
            slot_eprx3_pnl = 0.0

            if act == "charge":
                cost = j_a * TAX * (c_kwh / (1 - wheeling_loss_rate))
                slot_jepx_pnl -= cost
            elif act == "discharge":
                rev = j_a * TAX * (d_kwh * (1 - battery_loss_rate))
                slot_jepx_pnl += rev
            elif act == "EPRX3":
                rev = e3_a * TAX * battery_power_kW
                rev += TAX * (e3_kwh * (1 - battery_loss_rate)) * imb_a
                slot_eprx3_pnl += rev
            elif act == "EPRX1":
                rev = e1_a * TAX * battery_power_kW
                slot_eprx1_pnl += rev

            slot_total_pnl = slot_jepx_pnl + slot_eprx1_pnl + slot_eprx3_pnl
            day_profit += slot_total_pnl

            row = {
                "date": df_day.loc[i, "date"],
                "slot": int(df_day.loc[i, "slot"]),
                "action": act,
                "battery_level_kWh": pulp.value(battery_soc[i+1]),
                "charge_kWh": c_kwh,
                "discharge_kWh": d_kwh,
                "EPRX3_kWh": e3_kwh,
                "JEPX_actual": j_a,
                "EPRX1_actual": e1_a,
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

        day_charge_kWh = sum(pulp.value(charge[i]) for i in range(day_slots)) * half_power_kWh
        day_cycle_count = day_charge_kWh / battery_capacity_kWh
        total_cycles_used += day_cycle_count

        if yearly_cycle_limit > 0 and (total_cycles_used > yearly_cycle_limit):
            break

    # Wheeling費用(既に税込み想定)
    total_charge_kWh = 0.0
    total_discharge_kWh = 0.0
    for r in all_transactions:
        if r["action"] == "charge":
            total_charge_kWh += r["charge_kWh"]
        elif r["action"] == "discharge":
            total_discharge_kWh += r["discharge_kWh"] * (1 - battery_loss_rate)
        elif r["action"] == "EPRX3":
            total_discharge_kWh += r["EPRX3_kWh"] * (1 - battery_loss_rate)

    usage_fee_kWh = max(0, total_charge_kWh - total_discharge_kWh)
    monthly_fee = wheeling_base_charge * battery_power_kW + wheeling_usage_fee * usage_fee_kWh

    final_profit = total_profit - monthly_fee

    return all_transactions, total_profit, final_profit


############################
# Streamlit UI
############################

def main():
    st.title("Battery Optimizer 1.02")

    # 基本パラメータ
    st.header("基本パラメータ設定")
    selected_area_num = st.selectbox(
        "対象エリア (1-9)",
        options=list(AREA_NUMBER_TO_NAME.keys()),
        format_func=lambda x: f"{x}: {AREA_NUMBER_TO_NAME[x]}"
    )
    target_area_name = AREA_NUMBER_TO_NAME[selected_area_num]
    voltage_type = st.selectbox("電圧区分", ["SHV", "HV", "LV"], index=1)

    battery_power_kW = st.number_input("バッテリー出力(kW)", min_value=10, value=1000, step=100)
    battery_capacity_kWh = st.number_input("バッテリー容量(kWh) *使用可能容量", min_value=10, value=4000, step=100)
    battery_loss_rate = st.number_input("バッテリー損失率 (0.05=5%)", min_value=0.0, max_value=1.0, value=0.05, step=0.01)

    st.subheader("充放電サイクル上限")
    daily_cycle_limit = st.number_input("日次上限 (0=上限なし)", min_value=0, value=1, step=1)
    yearly_cycle_limit = st.number_input("年次上限 (0=上限なし)", min_value=0, value=365, step=1)
    annual_degradation_rate = st.number_input("バッテリー劣化率 (0.03=3%)", min_value=0.0, max_value=1.0, value=0.03, step=0.01)
    forecast_period = st.number_input("予測対象スロット数", min_value=48, value=48, step=48)

    # EPRX1ブロック設定
    st.subheader("EPRX1ブロック設定")
    eprx1_block_size = st.number_input("EPRX1 連続スロット数 (M)", min_value=1, value=3, step=1)
    eprx1_block_cooldown = st.number_input("EPRX1 ブロック終了後のクールダウンスロット数 (C)", min_value=0, value=2, step=1)

    # 1日あたりのEPRX1スロット上限
    st.subheader("EPRX1 1日の最大スロット数")
    max_daily_eprx1_slots = st.number_input("1日のEPRX1スロット最大数 (0=制限なし)", min_value=0, value=6, step=1)

    # CSVアップロード
    st.header("価格データ (CSV) アップロード")
    data_file = st.file_uploader("DATA_CSV", type=["csv"])

    if "calc_results" not in st.session_state:
        st.session_state["calc_results"] = None
        st.session_state["calc_day_profit"] = 0.0
        st.session_state["calc_final_profit"] = 0.0

    if data_file:
        df_all = pd.read_csv(data_file)
        if "date" in df_all.columns:
            df_all["date"] = pd.to_datetime(df_all["date"], errors="coerce")

        if st.button("Calculate"):
            results, day_profit, final_profit = run_optimization(
                target_area_name=target_area_name,
                voltage_type=voltage_type,
                battery_power_kW=battery_power_kW,
                battery_capacity_kWh=battery_capacity_kWh,
                battery_loss_rate=battery_loss_rate,
                daily_cycle_limit=daily_cycle_limit,
                yearly_cycle_limit=yearly_cycle_limit,
                annual_degradation_rate=annual_degradation_rate,
                forecast_period=forecast_period,
                eprx1_block_size=eprx1_block_size,
                eprx1_block_cooldown=eprx1_block_cooldown,
                max_daily_eprx1_slots=max_daily_eprx1_slots,
                df_all=df_all
            )

            if results is None:
                st.warning("No optimal solution found or missing columns.")
                return

            st.session_state["calc_results"] = results
            st.session_state["calc_day_profit"] = day_profit
            st.session_state["calc_final_profit"] = final_profit

    if st.session_state["calc_results"] is not None:
        results = st.session_state["calc_results"]
        day_profit = st.session_state["calc_day_profit"]
        final_profit = st.session_state["calc_final_profit"]

        st.success("Calculation Completed.")

        # 小数点以下を表示しない (整数化)
        st.write(f"**Total Profit(Actual Price,税込)**: {int(day_profit):,d} 円")
        st.write(f"**Final Profit (after wheeling fee)**: {int(final_profit):,d} 円")

        df_res = pd.DataFrame(results)
        df_res.sort_values(by=["date", "slot"], inplace=True, ignore_index=True)
        st.dataframe(df_res, height=600)

        st.subheader("バッテリー残量と JEPX実際価格 の推移")
        min_date = df_res["date"].min()
        max_date = df_res["date"].max()

        default_start = min_date
        default_end = min_date + pd.Timedelta(days=2)
        if default_end > max_date:
            default_end = max_date

        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", value=default_start, min_value=min_date, max_value=max_date)
        with col2:
            end_date = st.date_input("End Date", value=default_end, min_value=min_date, max_value=max_date)

        if start_date > end_date:
            st.warning("Invalid date range.")
            return

        df_g = df_res[
            (df_res["date"] >= pd.to_datetime(start_date)) & (df_res["date"] <= pd.to_datetime(end_date))
        ].copy()
        df_g.reset_index(drop=True, inplace=True)

        if len(df_g) == 0:
            st.warning("No data in the selected date range.")
            return

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

        csv_data = df_res.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv_data,
            file_name="optimal_transactions.csv",
            mime="text/csv"
        )
    else:
        st.write("ファイルをアップロード後、Calculate ボタンを押してください。")


if __name__ == "__main__":
    main()