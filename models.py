import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests

# granger causality 
def granger_causality_analysis(data, max_lag=3):
    results = {}
    variables = data.columns[1:]  
    for var1 in variables:
        for var2 in variables:
            if var1 != var2:
                try:
                    test_result = grangercausalitytests(
                        data[[var1, var2]], max_lag, verbose=False
                    )
                    p_values = [round(test_result[lag][0]["ssr_ftest"][1], 4) for lag in range(1, max_lag + 1)]
                    results[f"{var1} causes {var2}"] = p_values
                except Exception as e:
                    results[f"{var1} causes {var2}"] = f"Error: {e}"
    return results

# monte carlo
def monte_carlo_simulation(data, variable, num_simulations=1000, forecast_period=10):
    historical_data = data[variable].dropna().values
    if len(historical_data) < 2:  # not enough data
        return None, None, None
    mean = np.mean(historical_data)
    std_dev = np.std(historical_data)
    
    simulations = np.zeros((num_simulations, forecast_period))
    for i in range(num_simulations):
        simulations[i, :] = np.random.normal(mean, std_dev, forecast_period)
    
    forecast_mean = np.mean(simulations, axis=0)
    forecast_std_dev = np.std(simulations, axis=0)
    return simulations, forecast_mean, forecast_std_dev

# data cleaning erm why is this file so gross
def robust_clean_data(df):
    df_cleaned = df.iloc[1:]
    df_cleaned.columns = df.iloc[0]
    df_cleaned = df_cleaned.dropna(axis=1, how="all")
    df_cleaned = df_cleaned.dropna(axis=0, how="any")
    df_cleaned = df_cleaned.rename(columns={df_cleaned.columns[0]: "Year"})
    df_cleaned["Year"] = pd.to_numeric(df_cleaned["Year"], errors="coerce")
    df_cleaned = df_cleaned.dropna(subset=["Year"])
    df_cleaned["Year"] = df_cleaned["Year"].astype(int)

    numeric_columns = ["Year"]
    for col in df_cleaned.columns[1:]:
        try:
            df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors="coerce")
            numeric_columns.append(col)
        except Exception:
            continue
    
    df_cleaned = df_cleaned[numeric_columns].dropna()
    return df_cleaned

# validation
def validate_data(df, min_rows=10):
    return df.dropna().shape[0] >= min_rows

# results
def print_results(results):
    print("=" * 60)
    print("ANALYSIS RESULTS".center(60))
    print("=" * 60)
    for sheet, data in results.items():
        print(f"\nSheet: {sheet}")
        print("-" * 60)
        if "Error" in data:
            print(f"  Error: {data['Error']}")
            continue

        # granger result
        if "Granger Causality" in data:
            print("\n  Granger Causality Tests:")
            for cause, p_values in data["Granger Causality"].items():
                print(f"    {cause}")
                if isinstance(p_values, str):
                    print(f"      Error: {p_values}")
                else:
                    for lag, p_value in enumerate(p_values, 1):
                        significance = " (significant)" if p_value < 0.05 else ""
                        print(f"      Lag {lag}: p = {p_value:.4f}{significance}")
        
        # monte carlo result
        if "Monte Carlo Simulation" in data:
            print("\n  Monte Carlo Simulations:")
            for variable, stats in data["Monte Carlo Simulation"].items():
                print(f"    Variable: {variable}")
                if stats == "Insufficient data for simulation":
                    print("      Insufficient data for simulation")
                else:
                    print(f"      Forecast Mean: {np.round(stats['Forecast Mean'], 2)}")
                    print(f"      Forecast Std Dev: {np.round(stats['Forecast Std Dev'], 2)}")
                    print(f"      Simulations Shape: {stats['Simulations'].shape}")

# all sheets
def analyze_sheets(sheets_data, key_variables, max_lag=3, num_simulations=1000, forecast_period=10):
    results = {}
    for sheet, df in sheets_data.items():
        cleaned_data = robust_clean_data(df)
        if not validate_data(cleaned_data):  # skip datasets with bad rows
            results[sheet] = {"Error": "Insufficient data for analysis"}
            continue

        # granger
        causality_results = granger_causality_analysis(cleaned_data, max_lag=max_lag)
        
        # monte
        simulation_results = {}
        for var in key_variables:
            if var in cleaned_data.columns:
                simulations, forecast_mean, forecast_std_dev = monte_carlo_simulation(
                    cleaned_data, var, num_simulations=num_simulations, forecast_period=forecast_period
                )
                if simulations is not None:
                    simulation_results[var] = {
                        "Simulations": simulations,
                        "Forecast Mean": forecast_mean,
                        "Forecast Std Dev": forecast_std_dev,
                    }
                else:
                    simulation_results[var] = "Insufficient data for simulation"
        
        results[sheet] = {
            "Granger Causality": causality_results,
            "Monte Carlo Simulation": simulation_results,
        }
    return results

# main
if __name__ == "__main__":
    # file stuff
    file_path = 'sheets_data.xlsx' 
    excel_data = pd.ExcelFile(file_path)
    
    sheets_data = {sheet: excel_data.parse(sheet) for sheet in excel_data.sheet_names}
   
    # idk which vars to use
    key_variables = ["Total housing units", "Occupied units"]  # choose whatever variables
    analysis_results = analyze_sheets(sheets_data, key_variables)
    print_results(analysis_results)
