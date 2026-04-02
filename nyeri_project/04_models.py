"""
CRISP-DM PHASE 4b Modeling: ARIMA, SARIMA, Facebook Prophet

04_models.py

PURPOSE:
Trains all three forecasting models on both datasets, saves forecast results
and model metadata to CSV files for use by the dashboard and 05_evaluate.py.

MODELS:
1. ARIMA  — statsmodels / pmdarima auto_arima
2. SARIMA — pmdarima auto_arima with seasonal=True, m=12
3. Prophet — Meta's open-source Bayesian additive regression model

DATASETS
Observed Annual  (n=9):  All 3 models applied. Demonstrates sparse-data limits.
Simulated Monthly (n=108): All 3 models applied. Proper train/test evaluation.

TRAIN/TEST SPLIT (Monthly):
80/20 chronological split:
  Train: rows  0–85  (Jan 2016 – Jun 2023, ~86 months)
  Test:  rows 86–107 (Jul 2023 – Dec 2024, ~22 months)
No shuffling — time series order must be preserved.

OUTPUTS:
  data/model_results_monthly.csv   — test set forecasts for all 3 models
  data/model_results_annual.csv    — annual fitted/forecast values
  data/model_metadata.csv          — order parameters selected per model

"""

import os
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
import pmdarima as pm
from prophet import Prophet

warnings.filterwarnings("ignore")

# UTILITIES
def load_data():
    df_annual  = pd.read_csv("data/01_county_annual.csv")
    df_monthly = pd.read_csv("data/03_simulated_monthly.csv", parse_dates=["date"])
    return df_annual, df_monthly


def check_stationarity(series: pd.Series) -> int:
    """
    Run ADF test. Return recommended differencing order d.
    d=0 if stationary, d=1 if unit root detected.
    """
    result = adfuller(series.dropna(), autolag="AIC")
    p_value = result[1]
    return 0 if p_value < 0.05 else 1


def train_test_split_monthly(df_monthly: pd.DataFrame, split_ratio: float = 0.80):
    """
    Chronological 80/20 split of monthly data.
    Returns train DataFrame, test DataFrame, and the split index.
    """
    n = len(df_monthly)
    split_idx = int(n * split_ratio)
    train = df_monthly.iloc[:split_idx].copy()
    test  = df_monthly.iloc[split_idx:].copy()
    return train, test, split_idx



# MODEL 1: ARIMA
def fit_arima_monthly(train_series: np.ndarray, d_order: int) -> pm.ARIMA:
    """
    Fit ARIMA on monthly series using pmdarima auto_arima.
    auto_arima selects (p, d, q) order by minimising AIC.

    Parameters
    train_series : np.ndarray
        Training data (monthly production values).
    d_order : int
        Differencing order from ADF test (0 or 1).

    Returns
    Fitted pmdarima ARIMA model.
    """
    print("  Fitting ARIMA (monthly simulated data)...")
    model = pm.auto_arima(
        train_series,
        d            = d_order,
        seasonal     = False,        # Non-seasonal ARIMA — SARIMA handles seasonal
        stepwise     = True,         # Faster search
        information_criterion = "aic",
        suppress_warnings = True,
        error_action  = "ignore",
        max_p = 5, max_q = 5,
    )
    print(f"    Best ARIMA order: {model.order}")
    return model


def fit_arima_annual(annual_series: np.ndarray) -> pm.ARIMA:
    """
    Fit ARIMA on the 9-point annual series.
    Orders will be minimal given the tiny sample size.
    Results demonstrate sparse-data limitations.
    """
    print("  Fitting ARIMA (observed annual data, n=9)...")
    model = pm.auto_arima(
        annual_series,
        seasonal     = False,
        stepwise     = True,
        information_criterion = "aic",
        suppress_warnings = True,
        error_action  = "ignore",
        max_p = 2, max_q = 2,   # Constrained: too few observations for higher orders
    )
    print(f"    Best ARIMA order: {model.order}")
    return model


# MODEL 2: SARIMA
def fit_sarima_monthly(train_series: np.ndarray, d_order: int) -> pm.ARIMA:
    """
    Fit SARIMA on monthly series with seasonal period m=12.
    auto_arima searches seasonal orders (P, D, Q) via AIC.

    SARIMA is NOT fitted on annual data: annual frequency has no
    sub-annual seasonal cycle to model — this inapplicability is
    itself a key comparative finding of the study.

    Parameters
    train_series : np.ndarray
        Training data (monthly production values).
    d_order : int
        Non-seasonal differencing order.

    Returns
    Fitted pmdarima ARIMA model with seasonal component.
    """
    print("  Fitting SARIMA (monthly simulated data)...")
    model = pm.auto_arima(
        train_series,
        d            = d_order,
        D            = 1,       # Seasonal differencing
        seasonal     = True,
        m            = 12,      # Monthly seasonality
        stepwise     = True,
        information_criterion = "aic",
        suppress_warnings = True,
        error_action  = "ignore",
        max_p = 3, max_q = 3,
        max_P = 2, max_Q = 2,
    )
    print(f"    Best SARIMA order: {model.order}  Seasonal: {model.seasonal_order}")
    return model


# MODEL 3: PROPHET
def fit_prophet_monthly(train_df: pd.DataFrame) -> Prophet:
    """
    Fit Prophet on monthly data.

    Prophet requires a DataFrame with:
      'ds' — datetime column
      'y'  — target values

    yearly_seasonality=True captures the bimodal East African dairy cycle.
    interval_width=0.95 gives 95% credible intervals.

    Parameters
    train_df : pd.DataFrame
        Must contain 'date' and 'monthly_production_litres' columns.

    Returns
    Fitted Prophet model.
    """
    print("  Fitting Prophet (monthly simulated data)...")
    prophet_train = pd.DataFrame({
        "ds": train_df["date"],
        "y":  train_df["monthly_production_litres"],
    })
    model = Prophet(
        yearly_seasonality  = True,
        weekly_seasonality  = False,   # Not applicable to monthly data
        daily_seasonality   = False,   # Not applicable to monthly data
        interval_width      = 0.95,
        seasonality_mode    = "multiplicative",  # Appropriate for growing trend
    )
    model.fit(prophet_train)
    print("    Prophet model fitted.")
    return model


def fit_prophet_annual(df_annual: pd.DataFrame) -> Prophet:
    """
    Fit Prophet on the 9-point annual series.
    Uses July 1st as representative annual timestamp.
    yearly_seasonality=False: no sub-annual seasonality at annual frequency.
    """
    print("  Fitting Prophet (observed annual data, n=9)...")
    prophet_annual = pd.DataFrame({
        "ds": pd.to_datetime([f"{int(y)}-07-01" for y in df_annual["year"]]),
        "y":  df_annual["total_milk_production_litres"].values,
    })
    model = Prophet(
        yearly_seasonality  = False,
        weekly_seasonality  = False,
        daily_seasonality   = False,
        interval_width      = 0.95,
    )
    model.fit(prophet_annual)
    print("    Prophet (annual) fitted.")
    return model

# GENERATE FORECASTS AND COLLECT RESULTS
def collect_monthly_results(
    test_df:      pd.DataFrame,
    arima_model:  pm.ARIMA,
    sarima_model: pm.ARIMA,
    prophet_model: Prophet,
) -> pd.DataFrame:
    """
    Generate test-set forecasts from all three monthly models.
    Returns a single DataFrame with actual and predicted columns.
    """
    n_test = len(test_df)

    # ARIMA forecast
    arima_fc, arima_ci = arima_model.predict(n_periods=n_test, return_conf_int=True)

    # SARIMA forecast
    sarima_fc, sarima_ci = sarima_model.predict(n_periods=n_test, return_conf_int=True)

    # Prophet forecast
    future = pd.DataFrame({"ds": test_df["date"].values})
    prophet_fc = prophet_model.predict(future)

    results = pd.DataFrame({
        "date":              test_df["date"].values,
        "actual":            test_df["monthly_production_litres"].values,

        "arima_forecast":    arima_fc,
        "arima_lower_95":    arima_ci[:, 0],
        "arima_upper_95":    arima_ci[:, 1],

        "sarima_forecast":   sarima_fc,
        "sarima_lower_95":   sarima_ci[:, 0],
        "sarima_upper_95":   sarima_ci[:, 1],

        "prophet_forecast":  prophet_fc["yhat"].values,
        "prophet_lower_95":  prophet_fc["yhat_lower"].values,
        "prophet_upper_95":  prophet_fc["yhat_upper"].values,
    })

    return results


def collect_annual_results(
    df_annual:      pd.DataFrame,
    arima_annual:   pm.ARIMA,
    prophet_annual: Prophet,
) -> pd.DataFrame:
    """
    Collect fitted values and a 3-year forward forecast from annual models.
    SARIMA is not included (not applicable to annual data).
    """
    n = len(df_annual)

    # ARIMA fitted values (in-sample)
    arima_fitted = arima_annual.predict_in_sample()

    # ARIMA 3-year forward forecast
    arima_future_fc, arima_future_ci = arima_annual.predict(n_periods=3, return_conf_int=True)
    future_years = [2025, 2026, 2027]
    future_labels = ["2025/2026", "2026/2027", "2027/2028"]

    # Prophet fitted + forecast
    prophet_df = pd.DataFrame({
        "ds": pd.to_datetime([f"{int(y)}-07-01" for y in df_annual["year"]]),
        "y":  df_annual["total_milk_production_litres"].values,
    })
    future_prophet = pd.DataFrame({
        "ds": pd.to_datetime([f"{y}-07-01" for y in future_years])
    })
    all_ds = pd.concat([prophet_df[["ds"]], future_prophet], ignore_index=True)
    prophet_forecast = prophet_annual.predict(all_ds)

    # In-sample portion
    observed_results = pd.DataFrame({
        "year":        df_annual["year"].values,
        "year_label":  df_annual["year_label"].values,
        "actual":      df_annual["total_milk_production_litres"].values,
        "arima_fitted":   arima_fitted,
        "prophet_fitted": prophet_forecast["yhat"].values[:n],
        "record_type": "observed",
    })

    # Future forecast portion
    future_results = pd.DataFrame({
        "year":        future_years,
        "year_label":  future_labels,
        "actual":      [None, None, None],
        "arima_fitted":      arima_future_fc,
        "prophet_fitted":    prophet_forecast["yhat"].values[n:],
        "record_type": "forecast",
    })

    annual_results = pd.concat([observed_results, future_results], ignore_index=True)
    return annual_results

# ENTRY POINT
if __name__ == "__main__":

    print("CRISP-DM PHASE 4: MODELING")

    #Load data 
    df_annual, df_monthly = load_data()
    print(f"\nLoaded {len(df_annual)} annual records and {len(df_monthly)} monthly records.")

    #Stationarity check
    monthly_series = df_monthly["monthly_production_litres"].values
    d = check_stationarity(pd.Series(monthly_series))
    print(f"\nADF stationarity test  -> recommended d = {d}")

    #Train/test split
    train_df, test_df, split_idx = train_test_split_monthly(df_monthly)
    print(f"\nTrain/test split (80/20 chronological):")
    print(f"  Train: {len(train_df)} months  ({train_df['date'].min().date()} to {train_df['date'].max().date()})")
    print(f"  Test:  {len(test_df)} months  ({test_df['date'].min().date()} to {test_df['date'].max().date()})")

    train_series = train_df["monthly_production_litres"].values
    annual_series = df_annual["total_milk_production_litres"].values

    #Fit all models
    print("\n Training on Simulated Monthly Data")
    arima_monthly  = fit_arima_monthly(train_series, d)
    sarima_monthly = fit_sarima_monthly(train_series, d)
    prophet_monthly = fit_prophet_monthly(train_df)

    print("\n Training on Observed Annual Data ")
    arima_annual   = fit_arima_annual(annual_series)
    prophet_annual = fit_prophet_annual(df_annual)

    #Generate forecasts
    print("\n Generating Forecasts ")
    monthly_results = collect_monthly_results(test_df, arima_monthly, sarima_monthly, prophet_monthly)
    annual_results  = collect_annual_results(df_annual, arima_annual, prophet_annual)

    #Save
    monthly_results.to_csv("data/model_results_monthly.csv", index=False)
    print(" Saved: data/model_results_monthly.csv")

    annual_results.to_csv("data/model_results_annual.csv", index=False)
    print(" Saved: data/model_results_annual.csv")

    #Save model metadata
    metadata = pd.DataFrame([
        {
            "model":          "ARIMA",
            "dataset":        "Monthly Simulated",
            "order":          str(arima_monthly.order),
            "seasonal_order": "N/A",
            "n_train":        len(train_df),
            "n_test":         len(test_df),
            "notes":          f"auto_arima, d={d}, AIC selection"
        },
        {
            "model":          "SARIMA",
            "dataset":        "Monthly Simulated",
            "order":          str(sarima_monthly.order),
            "seasonal_order": str(sarima_monthly.seasonal_order),
            "n_train":        len(train_df),
            "n_test":         len(test_df),
            "notes":          f"auto_arima, d={d}, D=1, m=12, AIC selection"
        },
        {
            "model":          "Prophet",
            "dataset":        "Monthly Simulated",
            "order":          "N/A (Bayesian)",
            "seasonal_order": "yearly_seasonality=True, multiplicative",
            "n_train":        len(train_df),
            "n_test":         len(test_df),
            "notes":          "interval_width=0.95"
        },
        {
            "model":          "ARIMA",
            "dataset":        "Annual Observed",
            "order":          str(arima_annual.order),
            "seasonal_order": "N/A",
            "n_train":        len(df_annual),
            "n_test":         "walk-forward",
            "notes":          "n=9, wide CIs expected — demonstrates data scarcity"
        },
        {
            "model":          "Prophet",
            "dataset":        "Annual Observed",
            "order":          "N/A (Bayesian)",
            "seasonal_order": "yearly_seasonality=False",
            "n_train":        len(df_annual),
            "n_test":         "walk-forward",
            "notes":          "n=9, high uncertainty — demonstrates data scarcity"
        },
        {
            "model":          "SARIMA",
            "dataset":        "Annual Observed",
            "order":          "NOT APPLICABLE",
            "seasonal_order": "NOT APPLICABLE",
            "n_train":        "N/A",
            "n_test":         "N/A",
            "notes":          "Annual frequency has no sub-annual seasonal component"
        },
    ])
    metadata.to_csv("data/model_metadata.csv", index=False)
    print(" Saved: data/model_metadata.csv")

    #Preview results
    print("\nTest set forecasts (first 5 rows):")
    preview_cols = ["date", "actual", "arima_forecast", "sarima_forecast", "prophet_forecast"]
    print(monthly_results[preview_cols].head().to_string(index=False))

    print("\n -> Run 05_evaluate.py next")
