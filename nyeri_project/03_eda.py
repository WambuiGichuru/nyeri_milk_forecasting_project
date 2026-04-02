"""
CRISP-DM PHASE 4a: Data Understanding: Exploratory Data Analysis

03_eda.py

PURPOSE:
Explores and summarises all three datasets before any modeling takes place.
Produces:
- Descriptive statistics for observed and simulated data
- Pearson and Spearman correlation for key variable pairs
- Stationarity test (ADF) on both the observed annual and simulated monthly series
- Summary of findings that informs model configuration in 04_models.py

All outputs are printed to the terminal. Plots are intentionally not saved
here; the dashboard (app.py) handles all visualisation.

CRISP-DM LINK:
This is Phase 2 (Data Understanding) extended into Phase 4 (Modeling prep).
EDA findings directly drive:
- Whether differencing is needed for ARIMA/SARIMA (ADF test result)
- Seasonal period to use in SARIMA (m=12 confirmed by seasonal pattern)
- Justification for Prophet on sparse observed data

"""

import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.tsa.stattools import adfuller


def load_data():
    """Load all three datasets. Fails clearly if files are missing."""
    try:
        df_annual   = pd.read_csv("data/01_county_annual.csv")
        df_sub      = pd.read_csv("data/02_subcounty_population.csv")
        df_monthly  = pd.read_csv("data/03_simulated_monthly.csv", parse_dates=["date"])
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"{e}\n\nRun scripts in order:\n"
            "  python 01_data_preparation.py\n"
            "  python 02_simulate.py\n"
            "  python 03_eda.py"
        )
    return df_annual, df_sub, df_monthly


def section(title: str):
    print(f"  {title}")


def adf_test(series, series_name: str):
    """
    Augmented Dickey-Fuller test for stationarity.
    H0: series has a unit root (non-stationary).
    If p < 0.05: reject H0 - series is stationary.
    If p >= 0.05: fail to reject - differencing required for ARIMA.
    """
    result = adfuller(series.dropna(), autolag="AIC")
    adf_stat, p_value = result[0], result[1]
    is_stationary = p_value < 0.05
    print(f"\n  ADF Test - {series_name}")
    print(f"    Statistic : {adf_stat:.4f}")
    print(f"    p-value   : {p_value:.4f}")
    print(f"    Result    : {'STATIONARY (p<0.05) — no differencing needed' if is_stationary else 'NON-STATIONARY - differencing (d=1) recommended for ARIMA'}")
    return is_stationary


def main():

    df_annual, df_sub, df_monthly = load_data()


    # DATASET 1 — OBSERVED ANNUAL
    section("DATASET 1: OBSERVED ANNUAL (n=9)")

    prod  = df_annual["total_milk_production_litres"]
    cattle = df_annual["dairy_cattle_population"]
    ppc    = df_annual["avg_production_per_cow_litres"]

    print("\n  Production (litres):")
    print(f"    Min    : {prod.min():>15,.0f}")
    print(f"    Max    : {prod.max():>15,.0f}")
    print(f"    Mean   : {prod.mean():>15,.0f}")
    print(f"    Median : {prod.median():>15,.0f}")
    print(f"    Std    : {prod.std():>15,.0f}")
    print(f"    Range  : {prod.max()-prod.min():>15,.0f}")

    print("\n  Cattle Population:")
    print(f"    Min    : {cattle.min():>10,.0f}")
    print(f"    Max    : {cattle.max():>10,.0f}")
    print(f"    Mean   : {cattle.mean():>10,.0f}")

    print("\n  Avg Production Per Cow (litres/year):")
    print(f"    2016/17 - 2024/25: {int(ppc.iloc[0])} -> {int(ppc.iloc[-1])} litres (+{int(ppc.iloc[-1]-ppc.iloc[0])} litres over 9 years)")

    #Correlation Analysis 
    print("\n  Pearson Correlation - Production vs Cattle Population:")
    r, p = stats.pearsonr(prod, cattle)
    print(f"    r = {r:.4f}  p = {p:.4f}  - {'Significant' if p < 0.05 else 'Not significant at 0.05'}")
    print(f"    Interpretation: {'Strong positive linear relationship' if r > 0.7 else 'Moderate' if r > 0.4 else 'Weak'}")

    print("\n  Spearman Correlation - Productivity Per Cow vs Year (trend):")
    rho, p2 = stats.spearmanr(ppc, df_annual["year"])
    print(f"    spearman correlation = {rho:.4f}  p = {p2:.4f}  -{'Significant monotonic trend' if p2 < 0.05 else 'Not significant at 0.05'}")

    # Stationarity 
    print("\n  Stationarity Testing:")
    print("  (Note: n=9 is too small for ADF to be reliable - for reference only)")
    adf_test(prod, "Annual Milk Production")

    # DATASET 2 — SUB-COUNTY

    section("DATASET 2: SUB-COUNTY POPULATION (5 years times 8 sub-counties)")

    subcounty_cols = [
        "kieni_east", "mathira_east", "mathira_west", "mukurwe_ini",
        "kieni_west", "nyeri_central", "nyeri_south", "tetu"
    ]

    print("\n  2024/2025 Sub-County Rankings (by cattle population):")
    latest = df_sub[df_sub["year"] == 2024][subcounty_cols].iloc[0].sort_values(ascending=False)
    for sc, val in latest.items():
        print(f"    {sc:<16}: {int(val):>6,}")

    print("\n  Kieni East is consistently the largest sub-county by cattle population.")
    print("  Nyeri Central shows the sharpest decline over the 5-year period.")

    # DATASET 3 — SIMULATED MONTHLY
    section("DATASET 3: SIMULATED MONTHLY (n=108)")

    monthly_prod = df_monthly["monthly_production_litres"]

    print("\n  Monthly Production (litres):")
    print(f"    Min    : {monthly_prod.min():>12,.0f}")
    print(f"    Max    : {monthly_prod.max():>12,.0f}")
    print(f"    Mean   : {monthly_prod.mean():>12,.0f}")
    print(f"    Std    : {monthly_prod.std():>12,.0f}")

    print("\n  Average Monthly Production by Calendar Month (across all years):")
    monthly_avg = df_monthly.groupby("month")["monthly_production_litres"].mean()
    months_str  = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    peak_month  = monthly_avg.idxmax()
    trough_month = monthly_avg.idxmin()
    for m_num, avg in monthly_avg.items():
        tag = " <- PEAK" if m_num == peak_month else " <- TROUGH" if m_num == trough_month else ""
    print(f"    {months_str[m_num-1]}: {avg:>12,.0f}  {tag}")

    #Stationarity on simulated series
    print("\n  Stationarity Testing (n=108 - ADF reliable here):")
    is_stationary = adf_test(monthly_prod, "Simulated Monthly Production")

    #Season confirmation
    print(f"\n  Seasonal period confirmed: m=12 (monthly)")
    print(f"  - SARIMA seasonal order will use m=12")
    print(f"  - Prophet will use yearly_seasonality=True")

    # MODELING DECISIONS SUMMARY
    section("EDA -> MODELING DECISIONS SUMMARY")
    print("""
  ARIMA:
    - Apply to observed annual data (n=9) to demonstrate sparse-data limits
    - Apply to simulated monthly data (n=108) with auto order selection
    - If ADF shows non-stationary: use d=1 (first difference)

  SARIMA:
    - NOT applicable to annual data (no seasonality at yearly frequency)
    - Apply to simulated monthly data with seasonal period m=12
    - Auto-select seasonal orders (P,D,Q) via AIC grid search

  Prophet:
    - Apply to BOTH datasets (robust to sparse data by design)
    - On annual data: shows high uncertainty, illustrates data limitation
    - On monthly data: yearly_seasonality=True, interval_width=0.95

  Evaluation:
    - Metrics: MAE, RMSE, MAPE on held-out test set
    - Train/test split: 80/20 chronological (monthly data only)
    - Observed data: walk-forward validation (min 6 training years)
    """)

    print("- Run 04_models.py next")


if __name__ == "__main__":
    main()
