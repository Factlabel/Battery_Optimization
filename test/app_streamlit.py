import streamlit as st
import pandas as pd
import pulp
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

# Set the page configuration
st.set_page_config(page_title="Battery Optimizer alpha", layout="wide")

# Title
st.title("Battery Optimizer alpha")

# Description
st.markdown("""
This application optimizes battery usage based on uploaded CSV data.
Upload your CSV file and click "Optimize" to see the results displayed as graphs.
""")

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file here", type=["csv"])

if uploaded_file is not None:
    try:
        # Read the CSV file
        df = pd.read_csv(uploaded_file)

        # Check for required columns
        required_columns = {'date', 'EPRX_slot', 'DA_prediction', 'IB_prediction', 'EPRX_prediction', 'DA_slot'}
        if not required_columns.issubset(df.columns):
            st.error(f"The uploaded CSV does not contain the required columns: {required_columns}")
        else:
            st.success("File uploaded successfully.")

            # Optimize button
            if st.button("Optimize"):
                # Execute optimization
                with st.spinner('Running optimization...'):
                    # Constants
                    CHARGE_CAPACITY = 1000  # Maximum battery capacity (kWh)
                    CHARGE_DISCHARGE_SPEED = 166  # Charge/discharge speed per slot (kWh)

                    # Group data by date
                    grouped = df.groupby('date')

                    # Initialize results list
                    results = []

                    for date, group in grouped:
                        group = group.reset_index(drop=True)
                        num_slots = len(group)

                        # Identify unique EPRX_slots and map each slot to the indices it covers
                        eprx_slots = group['EPRX_slot'].unique()
                        eprx_slot_dict = {slot: group[group['EPRX_slot'] == slot].index.tolist() for slot in eprx_slots}

                        # Define the optimization problem
                        prob = pulp.LpProblem(f"Battery_Optimization_{date}", pulp.LpMaximize)

                        # Define decision variables
                        charge = pulp.LpVariable.dicts("Charge", range(num_slots), cat='Binary')
                        discharge_da = pulp.LpVariable.dicts("Discharge_DA", range(num_slots), cat='Binary')
                        discharge_ib = pulp.LpVariable.dicts("Discharge_IB", range(num_slots), cat='Binary')
                        discharge_eprx = pulp.LpVariable.dicts("Discharge_EPRX", eprx_slots, cat='Binary')
                        battery = pulp.LpVariable.dicts("Battery", range(num_slots + 1), lowBound=0,
                                                        upBound=CHARGE_CAPACITY, cat='Continuous')

                        # Initial battery state
                        prob += battery[0] == 0, "Initial_Battery_State"

                        # Total charge constraint
                        total_charge = pulp.lpSum([charge[i] * CHARGE_DISCHARGE_SPEED for i in range(num_slots)])
                        prob += total_charge <= CHARGE_CAPACITY, "Total_Charge_Limit"

                        # Constraints for each slot
                        for i in range(num_slots):
                            # Ensure that a slot cannot have both charging and discharging
                            prob += charge[i] + discharge_da[i] + discharge_ib[
                                i] <= 1, f"Charge_Discharge_Exclusivity_Slot_{i}"

                            # Discharge amount cannot exceed the current battery level
                            prob += (discharge_da[i] + discharge_ib[i]) * CHARGE_DISCHARGE_SPEED <= battery[
                                i], f"Discharge_Limit_Slot_{i}"

                            # Battery level transitions
                            prob += battery[i + 1] == battery[i] + (CHARGE_DISCHARGE_SPEED * charge[i]) - (
                                        CHARGE_DISCHARGE_SPEED * (
                                            discharge_da[i] + discharge_ib[i])), f"Battery_Balance_Slot_{i}"

                            # Battery level constraints
                            prob += battery[i + 1] <= CHARGE_CAPACITY, f"Battery_Capacity_Slot_{i}"
                            prob += battery[i + 1] >= 0, f"Battery_Min_Slot_{i}"

                        # Constraints for EPRX_slot discharges
                        for s in eprx_slots:
                            slots_in_s = eprx_slot_dict[s]

                            for i in slots_in_s:
                                prob += discharge_ib[i] >= discharge_eprx[s], f"EPRX_Discharge_Start_Slot_{i}_Slot_{s}"
                                prob += discharge_ib[i] <= discharge_eprx[s], f"EPRX_Discharge_End_Slot_{i}_Slot_{s}"

                            first_slot_in_s = slots_in_s[0]
                            prob += battery[first_slot_in_s] >= 996 * discharge_eprx[
                                s], f"EPRX_Battery_Limit_Slot_{first_slot_in_s}_Slot_{s}"

                        # Objective function: Maximize profit
                        profit = pulp.lpSum([
                            discharge_da[i] * group.loc[i, 'DA_prediction'] * CHARGE_DISCHARGE_SPEED +
                            discharge_ib[i] * group.loc[i, 'IB_prediction'] * CHARGE_DISCHARGE_SPEED -
                            charge[i] * group.loc[i, 'DA_prediction'] * CHARGE_DISCHARGE_SPEED
                            for i in range(num_slots)
                        ]) + pulp.lpSum([
                            discharge_eprx[s] * pulp.lpSum([
                                (group.loc[i, 'EPRX_prediction'] + group.loc[
                                    i, 'IB_prediction']) * CHARGE_DISCHARGE_SPEED
                                for i in eprx_slot_dict[s]
                            ])
                            for s in eprx_slots
                        ])
                        prob += profit, "Total_Profit"

                        # Solve the optimization problem
                        prob.solve()

                        # Check if an optimal solution is found
                        if pulp.LpStatus[prob.status] != 'Optimal':
                            st.warning(f"No optimal solution found for date {date}.")
                            continue

                        # Record transaction details
                        transactions = []
                        battery_levels = [0]  # Initialize with initial battery level

                        for i in range(num_slots):
                            action = "idle"  # Default action is idle
                            if pulp.value(charge[i]) == 1:
                                action = "charge"
                            elif pulp.value(discharge_da[i]) == 1:
                                action = "discharge_DA"
                            elif pulp.value(discharge_ib[i]) == 1:
                                # Check if discharging for EPRX revenue
                                is_eprx = False
                                for s in eprx_slots:
                                    if i in eprx_slot_dict[s] and pulp.value(discharge_eprx[s]) == 1:
                                        is_eprx = True
                                        eprx_slot = s
                                        break
                                if is_eprx:
                                    action = "discharge_EPRX"
                                else:
                                    action = "discharge_IB"

                            # Track current battery level
                            current_battery = pulp.value(battery[i + 1])
                            battery_levels.append(current_battery)

                            # Record transaction details for this slot
                            txn = {
                                'slot': group.loc[i, 'DA_slot'],
                                'action': action,
                                'amount_kWh': CHARGE_DISCHARGE_SPEED if action != "idle" else 0,
                                'price_yen': 0,
                                'cost_yen': 0,
                                'revenue_yen': 0,
                                'eprx_revenue_yen': 0,
                                'battery_level_kWh': current_battery
                            }

                            # Update transaction details based on action type
                            if action == "charge":
                                txn['price_yen'] = group.loc[i, 'DA_prediction']
                                txn['cost_yen'] = CHARGE_DISCHARGE_SPEED * group.loc[i, 'DA_prediction']
                            elif action == "discharge_DA":
                                txn['price_yen'] = group.loc[i, 'DA_prediction']
                                txn['revenue_yen'] = CHARGE_DISCHARGE_SPEED * group.loc[i, 'DA_prediction']
                            elif action == "discharge_IB":
                                txn['price_yen'] = group.loc[i, 'IB_prediction']
                                txn['revenue_yen'] = CHARGE_DISCHARGE_SPEED * group.loc[i, 'IB_prediction']
                            elif action == "discharge_EPRX":
                                txn['price_yen'] = group.loc[i, 'IB_prediction']
                                txn['revenue_yen'] = CHARGE_DISCHARGE_SPEED * group.loc[i, 'IB_prediction']
                                txn['eprx_revenue_yen'] = CHARGE_DISCHARGE_SPEED * group.loc[i, 'EPRX_prediction']

                            transactions.append(txn)

                        # Summarize results
                        total_profit_value = pulp.value(prob.objective)
                        total_charge = sum([txn['amount_kWh'] for txn in transactions if txn['action'] == 'charge'])
                        total_discharge = sum(
                            [txn['amount_kWh'] for txn in transactions if 'discharge' in txn['action']])

                        # Add results to the list
                        results.append({
                            'date': date,
                            'Total_Profit_yen': total_profit_value,
                            'Total_Charge_kWh': total_charge,
                            'Total_Discharge_kWh': total_discharge,
                            'Transactions': transactions,
                            'Battery_Levels': battery_levels,
                            'Group': group  # Keep group for plotting prices
                        })

                    # Display results
                    total_strategy_profit = 0
                    for result in results:
                        st.subheader(f"=== Date: {result['date']} ===")
                        st.write(f"**Total Profit**: {result['Total_Profit_yen']:.2f} Yen")
                        st.write(f"**Total Charge**: {result['Total_Charge_kWh']} kWh")
                        st.write(f"**Total Discharge**: {result['Total_Discharge_kWh']} kWh")

                        # Transaction details as DataFrame
                        txn_df = pd.DataFrame(result['Transactions'])
                        st.write("**Transaction Details**")
                        st.dataframe(txn_df)

                        # Battery Level and Prices Graph
                        fig, ax1 = plt.subplots(figsize=(12, 6))

                        # Bar plot for Battery Level
                        slots = range(len(result['Battery_Levels']))
                        ax1.bar(slots, result['Battery_Levels'], color='skyblue', label='Battery Level (kWh)',
                                alpha=0.6)
                        ax1.set_xlabel('Slot')
                        ax1.set_ylabel('Battery Level (kWh)', color='skyblue')
                        ax1.tick_params(axis='y', labelcolor='skyblue')

                        # Line plots for DA, IB, and EPRX Prices
                        ax2 = ax1.twinx()
                        da_prices = result['Group']['DA_prediction'].tolist()
                        ib_prices = result['Group']['IB_prediction'].tolist()
                        eprx_prices = result['Group']['EPRX_prediction'].tolist()

                        ax2.plot(range(len(da_prices)), da_prices, color='red', marker='o',
                                 label='JEPX DA Price (Yen/kWh)')
                        ax2.plot(range(len(ib_prices)), ib_prices, color='green', marker='x', linestyle='--',
                                 label='Imbalance Price (Yen/kWh)')
                        ax2.plot(range(len(eprx_prices)), eprx_prices, color='purple', marker='^', linestyle='-.',
                                 label='EPRX Prediction (Yen/kWh)')

                        ax2.set_ylabel('Price (Yen/kWh)', color='black')
                        ax2.tick_params(axis='y', labelcolor='black')

                        # Titles and Legends
                        plt.title(f"Battery Level and Prices Over Slots ({result['date']})")
                        fig.tight_layout()

                        # Combine legends from both axes
                        lines_1, labels_1 = ax1.get_legend_handles_labels()
                        lines_2, labels_2 = ax2.get_legend_handles_labels()
                        ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

                        st.pyplot(fig)

                        total_strategy_profit += result['Total_Profit_yen']

                    st.markdown(f"## === Total Profit Across All Days: {total_strategy_profit:.2f} Yen ===")

                    # Optionally save transaction data to CSV
                    all_transactions = []
                    for result in results:
                        date = result['date']
                        for txn in result['Transactions']:
                            entry = {'date': date}
                            entry.update(txn)
                            all_transactions.append(entry)

                    optimal_df = pd.DataFrame(all_transactions)
                    csv = optimal_df.to_csv(index=False)
                    st.download_button(
                        label="Download Optimization Results as CSV",
                        data=csv,
                        file_name='optimal_transactions.csv',
                        mime='text/csv',
                    )

    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.info("Please upload a CSV file to perform optimization.")
