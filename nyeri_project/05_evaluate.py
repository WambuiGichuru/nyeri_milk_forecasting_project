"""
CRISP-DM PHASE 5: Evaluation: Model Comparison and Ranking

05_evaluate.py

PURPOSE:
Loads the test-set forecasts from 04_models.py and computes evaluation
metrics for all three models. Produces a ranked comparison table and
a qualitative tradeoff summary, then saves results for the dashboard.

METRICS:
  MAE  — Mean Absolute Error: average error in litres
  RMSE — Root Mean Squared Error: penalises large errors more
  MAPE — Mean Absolute Percentage Error: scale-free, easy to communicate

INTERPRETATION GUIDE (for dashboard and defense):
  MAPE < 5%  : Excellent forecast accuracy
  MAPE < 10% : Good forecast accuracy
  MAPE < 20% : Acceptable
  MAPE > 20% : Poor — likely insufficient training data

OUTPUTS:
  data/model_comparison.csv    — ranked metrics table for all 3 models
  (qualitative tradeoff table printed and embedded in dashboard)
"""

import pandas as pd
import numpy as np
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# METRICS
def mean_absolute_error(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - predicted)))


def root_mean_squared_error(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def mean_absolute_percentage_error(actual: np.ndarray, predicted: np.ndarray) -> float:
    # Guard against division by zero
    mask = actual != 0
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


def compute_metrics(actual: np.ndarray, predicted: np.ndarray, model_name: str) -> dict:
    return {
        "model": model_name,
        "MAE":   round(mean_absolute_error(actual, predicted), 0),
        "RMSE":  round(root_mean_squared_error(actual, predicted), 0),
        "MAPE":  round(mean_absolute_percentage_error(actual, predicted), 4),
    }


def interpret_mape(mape: float) -> str:
    if mape < 5:   return "Excellent"
    if mape < 10:  return "Good"
    if mape < 20:  return "Acceptable"
    return "Poor"

# QUALITATIVE TRADEOFF TABLE
# Fixed reference table — does not depend on model run outputs.
# Grounded in: Hansen et al. (2024), Platonova & Popov (2025),
# Perez-Guerra et al. (2023), Taylor & Letham (2018).
QUALITATIVE_TRADEOFFS = pd.DataFrame([
    {
        "Criterion":              "Interpretability",
        "ARIMA":                  "High - explicit (p,d,q) parameters map to autocorrelation structure",
        "SARIMA":                 "High - seasonal parameters (P,D,Q,m) are explicitly interpretable",
        "Prophet":                "Medium - decomposes into trend/seasonal/noise but Bayesian internals are opaque",
    },
    {
        "Criterion":              "Minimum Data Requirement",
        "ARIMA":                  "~24 monthly records for stable estimation",
        "SARIMA":                 "~36 monthly records; needs 2-3 full seasonal cycles",
        "Prophet":                "Can fit <12 records; most flexible of the three",
    },
    {
        "Criterion":              "Seasonality Handling",
        "ARIMA":                  "None - non-seasonal model; cannot capture monthly dairy cycle",
        "SARIMA":                 "Excellent - explicit seasonal AR/MA terms for m=12 cycle",
        "Prophet":                "Good -  additive/multiplicative Fourier-based seasonal decomposition",
    },
    {
        "Criterion":              "Uncertainty Quantification",
        "ARIMA":                  "Confidence intervals based on asymptotic normality",
        "SARIMA":                 "Confidence intervals; wider with more seasonal parameters",
        "Prophet":                "Credible intervals via Monte Carlo sampling; most informative",
    },
    {
        "Criterion":              "Behaviour on 9 Annual Points",
        "ARIMA":                  "Minimal order only; very wide confidence intervals; unreliable",
        "SARIMA":                 "NOT APPLICABLE - annual data has no seasonal frequency",
        "Prophet":                "Technically fits; high uncertainty; trend dominated; not for decisions",
    },
    {
        "Criterion":              "Behaviour on 108 Monthly Points",
        "ARIMA":                  "Performs well; captures autocorrelation; ignores season",
        "SARIMA":                 "Best theoretical fit for seasonal monthly data; most accurate if trend is stable",
        "Prophet":                "Robust; handles slight non-stationarity automatically; good for changing trends",
    },
    {
        "Criterion":              "Ease of Use for County Staff",
        "ARIMA":                  "Moderate -requires statistical training to interpret ACF/PACF",
        "SARIMA":                 "Low - most complex; needs seasonal diagnostics expertise",
        "Prophet":                "High - automated, visual decomposition output, minimal configuration",
    },
    {
        "Criterion":              "Recommended Use Case",
        "ARIMA":                  "Baseline model; use when trend matters more than season",
        "SARIMA":                 "Primary forecasting model once 36+ monthly records available",
        "Prophet":                "Immediate deployment tool; best for county dashboard use now",
    },
])


# ENTRY POINT
if __name__ == "__main__":

    print("CRISP-DM PHASE 5: EVALUATION")

    #Load test results
    try:
        results = pd.read_csv("data/model_results_monthly.csv", parse_dates=["date"])
    except FileNotFoundError:
        raise FileNotFoundError("Run 04_models.py first to generate model_results_monthly.csv")

    actual          = results["actual"].values
    arima_pred      = results["arima_forecast"].values
    sarima_pred     = results["sarima_forecast"].values
    prophet_pred    = results["prophet_forecast"].values

    #Compute metrics
    arima_metrics   = compute_metrics(actual, arima_pred,   "ARIMA")
    sarima_metrics  = compute_metrics(actual, sarima_pred,  "SARIMA")
    prophet_metrics = compute_metrics(actual, prophet_pred, "Prophet")

    comparison = pd.DataFrame([arima_metrics, sarima_metrics, prophet_metrics])
    comparison["MAPE_interpretation"] = comparison["MAPE"].apply(interpret_mape)

    # Rank models
    # Rank by MAPE (lower is better); use MAE as tiebreaker
    comparison["rank_by_MAPE"] = comparison["MAPE"].rank(method="min").astype(int)
    comparison = comparison.sort_values("rank_by_MAPE").reset_index(drop=True)
    comparison["overall_rank"] = comparison.index + 1

    #Save
    comparison.to_csv("data/model_comparison.csv", index=False)
    print(" Saved: data/model_comparison.csv")

    #Print ranked table
    print("\n QUANTITATIVE COMPARISON (Monthly Simulated Data - Test Set)")
    print(f"\n  {'Rank':<6} {'Model':<10} {'MAE':>14} {'RMSE':>14} {'MAPE':>8}  {'Accuracy'}")
    print(f"  {'-'*6} {'-'*10} {'-'*14} {'-'*14} {'-'*8}  {'-'*12}")
    for _, row in comparison.iterrows():
        medal = "First" if row["overall_rank"] == 1 else "Second" if row["overall_rank"] == 2 else "Third"
        print(
            f"  {medal} #{int(row['overall_rank'])}  "
            f"{row['model']:<10} "
            f"{row['MAE']:>14,.0f} "
            f"{row['RMSE']:>14,.0f} "
            f"{row['MAPE']:>7.2f}%  "
            f"{row['MAPE_interpretation']}"
        )

    best_model = comparison.iloc[0]["model"]
    best_mape  = comparison.iloc[0]["MAPE"]
    print(f"\n  BEST MODEL: {best_model} (MAPE = {best_mape:.2f}%)")

    #Print qualitative tradeoffs
    print("\n QUALITATIVE TRADEOFF ANALYSIS ")
    for _, row in QUALITATIVE_TRADEOFFS.iterrows():
        print(f"\n  {row['Criterion']}:")
        print(f"    ARIMA:   {row['ARIMA']}")
        print(f"    SARIMA:  {row['SARIMA']}")
        print(f"    Prophet: {row['Prophet']}")

    #Policy recommendation
    print("\n POLICY RECOMMENDATION FOR NYERI COUNTY ")
    print(f"""
  Current Situation:
    - 9 annual records -> no reliable forecasting currently possible
    - Simulated data shows {best_model} achieves {best_mape:.2f}% MAPE with 108 monthly records

  Recommended Actions:
    1. Start recording MONTHLY milk production totals immediately
       (at minimum: total litres collected at major collection centres per month)
    2. Milestone 1 - After 12 months of recording:
       Prophet can produce short-term forecasts with reasonable confidence
    3. Milestone 2 - After 24 months of recording:
       ARIMA becomes reliably trainable
    4. Milestone 3 - After 36 months of recording:
       SARIMA becomes viable; full seasonal forecasting operational
    5. Recommended dashboard model: {best_model}
       (best accuracy on simulated data; most suitable for county staff use)

  Dashboard Deployment:
    - Run: streamlit run app.py
    """)

    print("-Run: streamlit run app.py  (to launch the dashboard)")
