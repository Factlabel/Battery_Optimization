import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import yaml
import pulp
import io
import os

def run_optimization(base_yaml, wheeling_yaml, df_all):
    """
    Existing logic from your old app1.01.py (unchanged in constraints/logic),
    only the file input is replaced and we produce 'results' for the entire dataset.
    """
    battery_cfg = base_yaml.get("battery", {})
    battery_loss_rate = battery_cfg.get("loss_rate", 0.05)
    battery_power_kW = battery_cfg.get("power_kW", 50)
    battery_capacity_kWh = battery_cfg.get("capacity_kWh", 200)

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
        st.error(f"CSV is missing columns: {required_cols}")
        return None, None, None

    # Sort by date, slot
    df_all.sort_values(by=["date","slot"], inplace=True, ignore_index=True)

    # Identify daily ranges
    unique_dates = df_all["date"].unique()
    day_indices = []
    current_start = 0
    current_date = df_all.loc[0,"date"]
    for i in range(len(df_all)):
        if df_all.loc[i,"date"] != current_date:
            day_indices.append((current_date, current_start, i-1))
            current_start = i
            current_date = df_all.loc[i,"date"]
    day_indices.append((current_date, current_start, len(df_all)-1))

    # Define PuLP problem
    prob = pulp.LpProblem("Battery_Optimization_MultiDay", pulp.LpMaximize)
    N = len(df_all)

    battery_soc = pulp.LpVariable.dicts("soc", range(N+1),
                                        lowBound=0, upBound=battery_capacity_kWh,
                                        cat=pulp.LpContinuous)
    charge = pulp.LpVariable.dicts("charge", range(N), cat=pulp.LpBinary)
    discharge = pulp.LpVariable.dicts("discharge", range(N), cat=pulp.LpBinary)
    eprx1 = pulp.LpVariable.dicts("eprx1", range(N), cat=pulp.LpBinary)
    eprx3 = pulp.LpVariable.dicts("eprx3", range(N), cat=pulp.LpBinary)

    half_power_kWh = battery_power_kW * 0.5

    # (A) Exclusive
    for i in range(N):
        prob += (charge[i] + discharge[i] + eprx1[i] + eprx3[i]) <= 1

    # (B) SOC transition
    for i in range(N):
        next_soc = (battery_soc[i]
                    + charge[i]*half_power_kWh
                    - discharge[i]*half_power_kWh
                    - eprx3[i]*half_power_kWh)
        prob += battery_soc[i+1] == next_soc

    # (C) Initial SOC=0
    prob += battery_soc[0] == 0

    # (D) EPRX1: daily sum <=6, SoC in [40%,60%]
    bigM = 999999
    for dateval, start_idx, end_idx in day_indices:
        prob += pulp.lpSum(eprx1[i] for i in range(start_idx, end_idx+1)) <= 6

    for i in range(N):
        soc_i = battery_soc[i]
        prob += soc_i >= 0.4*battery_capacity_kWh - (1 - eprx1[i])*bigM
        prob += soc_i <= 0.6*battery_capacity_kWh + (1 - eprx1[i])*bigM

    # (E) EPRX3 => soc[i]>=25kWh
    for i in range(N):
        prob += battery_soc[i] >= (half_power_kWh) - (1 - eprx3[i])*bigM

    # (F) 1日最大充電量 <= capacity
    for dateval, start_idx, end_idx in day_indices:
        prob += (
            pulp.lpSum(charge[i] for i in range(start_idx, end_idx+1)) * half_power_kWh
        ) <= battery_capacity_kWh

    # (G) If EPRX1_prediction=0 => eprx1=0; EPRX3_prediction=0 => eprx3=0
    for i in range(N):
        e1pred = df_all.loc[i,"EPRX1_prediction"]
        e3pred = df_all.loc[i,"EPRX3_prediction"]
        if pd.isna(e1pred) or e1pred==0:
            prob += eprx1[i] <= 0
        if pd.isna(e3pred) or e3pred==0:
            prob += eprx3[i] <= 0

    # Objective
    profit_terms = []
    for i in range(N):
        jpred = df_all.loc[i,"JEPX_prediction"] if not pd.isna(df_all.loc[i,"JEPX_prediction"]) else 0.0
        e1pred= df_all.loc[i,"EPRX1_prediction"] if not pd.isna(df_all.loc[i,"EPRX1_prediction"]) else 0.0
        e3pred= df_all.loc[i,"EPRX3_prediction"] if not pd.isna(df_all.loc[i,"EPRX3_prediction"]) else 0.0
        imb   = df_all.loc[i,"imbalance"]        if not pd.isna(df_all.loc[i,"imbalance"]) else 0.0

        cost_charge = jpred*(half_power_kWh/(1-wheeling_loss_rate))*charge[i]
        rev_discharge = jpred*(half_power_kWh*(1-battery_loss_rate))*discharge[i]
        rev_eprx1= e1pred*battery_power_kW* eprx1[i]
        rev_eprx3= e3pred*battery_power_kW* eprx3[i]
        rev_eprx3+= (half_power_kWh*(1-battery_loss_rate))* imb* eprx3[i]

        slot_profit= -cost_charge + rev_discharge + rev_eprx1 + rev_eprx3
        profit_terms.append(slot_profit)

    prob += pulp.lpSum(profit_terms), "TotalProfit"

    # Solve
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    status = pulp.LpStatus[prob.status]
    if status != "Optimal":
        return None, None, None

    # Actual price calculation
    day_profit = 0.0
    results = []
    for i in range(N):
        c_val= pulp.value(charge[i])
        d_val= pulp.value(discharge[i])
        e1_val= pulp.value(eprx1[i])
        e3_val= pulp.value(eprx3[i])
        act="idle"
        if e1_val>0.5: act="EPRX1"
        elif e3_val>0.5: act="EPRX3"
        elif c_val>0.5: act="charge"
        elif d_val>0.5: act="discharge"

        row = {
            "date": df_all.loc[i,"date"],
            "slot": int(df_all.loc[i,"slot"]),
            "action": act,
            "battery_level_kWh": pulp.value(battery_soc[i+1]),
            "JEPX_actual": df_all.loc[i,"JEPX_actual"],
            "EPRX1_actual": df_all.loc[i,"EPRX1_actual"],
            "EPRX3_actual": df_all.loc[i,"EPRX3_actual"],
            "imbalance": df_all.loc[i,"imbalance"]
        }
        results.append(row)

    half_kwh= half_power_kWh
    for r in results:
        j_a= r["JEPX_actual"] if not pd.isna(r["JEPX_actual"]) else 0.0
        e1_a= r["EPRX1_actual"] if not pd.isna(r["EPRX1_actual"]) else 0.0
        e3_a= r["EPRX3_actual"] if not pd.isna(r["EPRX3_actual"]) else 0.0
        imb_a= r["imbalance"] if not pd.isna(r["imbalance"]) else 0.0
        a= r["action"]
        if a=="charge":
            cost= j_a*(half_kwh/(1-wheeling_loss_rate))
            day_profit-=cost
        elif a=="discharge":
            rev= j_a*(half_kwh*(1-battery_loss_rate))
            day_profit+=rev
        elif a=="EPRX1":
            day_profit+= e1_a*battery_power_kW
        elif a=="EPRX3":
            rev= e3_a*battery_power_kW
            rev+= (half_kwh*(1-battery_loss_rate))*imb_a
            day_profit+= rev

    # Wheeling cost
    total_charge_kWh=0.0
    total_discharge_kWh=0.0
    for r in results:
        if r["action"]=="charge":
            total_charge_kWh += half_kwh
        elif r["action"]=="discharge":
            total_discharge_kWh += half_kwh
        elif r["action"]=="EPRX3":
            total_discharge_kWh += half_kwh

    diff_kWh= max(0, total_charge_kWh - total_discharge_kWh)
    monthly_fee= wheeling_base_charge*battery_power_kW + wheeling_usage_fee* diff_kWh
    final_profit= day_profit - monthly_fee

    return results, day_profit, final_profit

############################
# Streamlit UI
############################

def main():
    st.title("app1.01 with Streamlit")

    base_file = st.file_uploader("BASE_YML", type=["yml","yaml"])
    wheeling_file = st.file_uploader("WHEELING_YML", type=["yml","yaml"])
    data_file = st.file_uploader("DATA_CSV", type=["csv"])

    if base_file and wheeling_file and data_file:
        with io.TextIOWrapper(base_file, encoding="utf-8") as bf:
            base_yaml = yaml.safe_load(bf)
        with io.TextIOWrapper(wheeling_file, encoding="utf-8") as wf:
            wheeling_yaml = yaml.safe_load(wf)
        df_all = pd.read_csv(data_file)
        if "date" in df_all.columns:
            df_all["date"] = pd.to_datetime(df_all["date"], errors="coerce")

        results, day_profit, final_profit = run_optimization(base_yaml, wheeling_yaml, df_all)
        if results is None:
            st.warning("No optimal solution found (status != 'Optimal').")
            return

        st.success("Calculation Completed.")
        st.write(f"**Total Profit(Actual Price)**: {day_profit:.2f}")
        st.write(f"**Final Profit (after wheeling fee)**: {final_profit:.2f}")

        # Convert results to DataFrame
        df_res = pd.DataFrame(results)
        df_res.sort_values(by=["date","slot"], inplace=True, ignore_index=True)

        # "Scrollable" data table
        # height=600 for example (adjust as you like)
        st.dataframe(df_res, height=600)

        # User chooses date range for the graph
        # We take min/max from df_res["date"]
        min_date = df_res["date"].min()
        max_date = df_res["date"].max()

        # Let user pick a start/end date in that range
        st.subheader("Select Date Range for Graph")
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", value=min_date, min_value=min_date, max_value=max_date)
        with col2:
            end_date = st.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date)

        if start_date > end_date:
            st.warning("Invalid date range.")
            return

        # Filter df_res for that period
        df_g = df_res[(df_res["date"] >= pd.to_datetime(start_date)) & (df_res["date"] <= pd.to_datetime(end_date))].copy()
        df_g.reset_index(drop=True, inplace=True)

        if len(df_g)==0:
            st.warning("No data in the selected date range.")
            return

        # Plot bar(battery_level_kWh) + line(JEPX_actual)
        fig, ax1 = plt.subplots(figsize=(10,5))
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

        # CSV download
        csv_data = df_res.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv_data,
            file_name="optimal_transactions.csv",
            mime="text/csv"
        )

    else:
        st.info("Please upload BASE_YML, WHEELING_YML, and DATA_CSV.")


if __name__=="__main__":
    main()