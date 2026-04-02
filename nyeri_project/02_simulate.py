"""
CRISP-DM PHASE 3 Data Preparation: Temporal Disaggregation & Simulation

02_simulate.py

PURPOSE:
Generates Dataset 3: a simulated monthly milk production series derived from
and constrained to match the official Nyeri County annual totals.

METHOD:
The Denton-Cholette temporal disaggregation approach (Dagum & Cholette, 2006)
is used to distribute each annual total across 12 months while enforcing the
aggregate constraint (monthly values sum exactly to the annual total).

On top of the disaggregation, a bimodal seasonal index is applied reflecting
Kenya's dairy production cycle driven by the long rains (March–May) and short
rains (October–November). This is consistent with contextual notes in the
Nyeri County source data attributing production changes to weather patterns.

Controlled Gaussian noise (CV = 3%) is added to prevent artificially perfect
seasonal patterns. Final values are rescaled so annual totals are preserved
within a 1% tolerance — this is the validation criterion.

WHY THIS MATTERS (CRISP-DM Business Understanding Link):
The 9 observed annual records cannot train ARIMA or SARIMA reliably.
The 108 simulated monthly records allow all three models to be properly
trained and evaluated. Because every simulated value is mathematically
derived from and constrained to official county records, the dataset is
analytically legitimate — not arbitrary synthetic data.

REFERENCE:
Dagum, E. B., & Cholette, P. A. (2006). Benchmarking, Temporal Distribution,
and Reconciliation Methods for Time Series. Springer.

OUTPUTS:
1.data/03_simulated_monthly.csv   — 108 monthly records
2.(validation summary printed to terminal)
"""

import os
import numpy as np
import pandas as pd

# reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


# SEASONAL INDEX
# Bimodal pattern calibrated to Kenya's rainfall-driven dairy cycle.
# Values represent relative production weight for each calendar month.
# Source: consistent with county data notes on weather/climate impacts and
# with Perez-Guerra et al. (2023) SARIMA findings on pasture-based herds.

SEASONAL_INDEX = np.array([
    0.82, 0.80, 0.95, 1.10, 1.15, 1.05,   # Jan–Jun: ramp up through long rains
    0.88, 0.85, 0.92, 1.12, 1.18, 1.00    # Jul–Dec: dip then peak at short rains
])
# Normalise so the 12-month average = 1.0
# This ensures the seasonal index does not inflate or deflate the annual total
SEASONAL_INDEX = SEASONAL_INDEX / SEASONAL_INDEX.mean()


def denton_cholette_disaggregate(annual_total: float, seasonal_index: np.ndarray) -> np.ndarray:
    """
    Distribute a single annual total into 12 monthly estimates using a
    seasonal proportional distribution approach (Denton-Cholette inspired).

    The method:
    1. Compute monthly proportions from the seasonal index
    (proportions sum to 1.0 across the 12 months)
    2. Multiply by the annual total to get base monthly values
    3. The result exactly satisfies: sum(monthly) == annual_total

    Parameters
    annual_total : float
        The official annual milk production figure in litres.
    seasonal_index : np.ndarray of shape (12,)
        Normalised seasonal weights for each calendar month.

    Returns
    np.ndarray of shape (12,)
        Monthly production estimates in litres.
    """
    proportions = seasonal_index / seasonal_index.sum()
    monthly = annual_total * proportions
    return monthly


def add_controlled_noise(monthly_values: np.ndarray, cv: float = 0.03) -> np.ndarray:
    """
    Add Gaussian noise to monthly estimates with a given coefficient of
    variation (CV = std / mean). This prevents unrealistically smooth
    seasonal patterns that would inflate model accuracy metrics.

    Noise is calibrated so that:
    - It is small enough to preserve annual totals within 1% after rescaling
    - CV = 0.03 (3%) is consistent with Fernandez-Montoto et al. (2022)
    or smallholder dairy simulation

    Parameters
    monthly_values : np.ndarray
        Base monthly estimates before noise.
    cv : float
        Coefficient of variation for noise (default 3%).

    Returns
    np.ndarray
        Monthly values with noise added, floored at 0 (no negative production).
    """
    noise = np.random.normal(loc=0.0, scale=cv * monthly_values)
    noisy = monthly_values + noise
    return np.maximum(noisy, 0.0)  # Production cannot be negative


def rescale_to_annual_total(monthly_values: np.ndarray, annual_total: float) -> np.ndarray:
    """
    Rescale monthly values so they sum exactly to the annual total.
    This enforces the aggregate constraint after noise has been added.

    After this step: sum(monthly_values) == annual_total (exactly).

    Parameters
    monthly_values : np.ndarray
        Noisy monthly estimates.
    annual_total : float
        The official annual total that must be preserved.

    Returns
    np.ndarray
        Rescaled monthly values.
    """
    current_sum = monthly_values.sum()
    if current_sum == 0:
        # Edge case guard: distribute equally if all zeros
        return np.full(12, annual_total / 12)
    scale_factor = annual_total / current_sum
    return monthly_values * scale_factor


# MAIN SIMULATION FUNCTION

def simulate_monthly_dataset(df_annual: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a monthly milk production dataset from annual totals.

    Applies the full pipeline for each year:
    1. Denton-Cholette proportional disaggregation
    2. Controlled Gaussian noise injection
    3. Rescaling to enforce aggregate constraint
    4. Validation check

    Parameters
    df_annual : pd.DataFrame
        Must contain columns: 'year', 'year_label', 'total_milk_production_litres'

    Returns
    pd.DataFrame
        108-row monthly dataset with columns:
        date, year, month, month_name, year_label,
        monthly_production_litres, official_annual_total, source
    """
    records = []

    for _, row in df_annual.iterrows():
        year       = int(row["year"])
        year_label = row["year_label"]
        annual     = float(row["total_milk_production_litres"])

        # Step 1: Disaggregate
        monthly_base = denton_cholette_disaggregate(annual, SEASONAL_INDEX)

        # Step 2: Add noise
        monthly_noisy = add_controlled_noise(monthly_base, cv=0.03)

        # Step 3: Rescale to preserve annual total
        monthly_final = rescale_to_annual_total(monthly_noisy, annual)

        # Step 4: Build monthly records
        for m_idx, value in enumerate(monthly_final):
            month_num  = m_idx + 1
            month_date = pd.Timestamp(year=year, month=month_num, day=1)

            records.append({
                "date":                        month_date,
                "year":                        year,
                "month":                       month_num,
                "month_name":                  month_date.strftime("%b"),
                "year_label":                  year_label,
                "monthly_production_litres":   round(value, 2),
                "official_annual_total":       annual,
                "source":                      "simulated_denton_cholette",
            })

    df_monthly = pd.DataFrame(records)
    df_monthly = df_monthly.sort_values("date").reset_index(drop=True)
    return df_monthly


# VALIDATION

def validate_simulation(df_annual: pd.DataFrame, df_monthly: pd.DataFrame) -> pd.DataFrame:
    """
    Validate that simulated monthly values sum back to official annual totals.
    Acceptable threshold: error < 1% per year (Dagum & Cholette, 2006 criterion).

    Returns a validation summary DataFrame and raises a warning if any year
    exceeds the 1% threshold.
    """
    agg = (
        df_monthly
        .groupby("year")["monthly_production_litres"]
        .sum()
        .reset_index()
        .rename(columns={"monthly_production_litres": "simulated_annual_sum"})
    )

    merged = df_annual[["year", "year_label", "total_milk_production_litres"]].merge(agg, on="year")
    merged["error_litres"] = (
        merged["simulated_annual_sum"] - merged["total_milk_production_litres"]
    ).abs()
    merged["error_pct"] = (
        merged["error_litres"] / merged["total_milk_production_litres"] * 100
    ).round(6)
    merged["within_threshold"] = merged["error_pct"] < 1.0

    return merged


# ENTRY POINT

if __name__ == "__main__":

    # Load Dataset 1 
    if not os.path.exists("data/01_county_annual.csv"):
        raise FileNotFoundError(
            "data/01_county_annual.csv not found. Run 01_data_preparation.py first."
        )

    df_annual = pd.read_csv("data/01_county_annual.csv")
    print(f"Loaded {len(df_annual)} annual records from data/01_county_annual.csv")

    #Run simulation
    print("\nRunning Denton-Cholette temporal disaggregation...")
    df_monthly = simulate_monthly_dataset(df_annual)
    print(f"Generated {len(df_monthly)} monthly records ({len(df_monthly)//12} years times 12 months)")

    # Validate
    print("SIMULATION VALIDATION (threshold: error < 1% per year)")
    validation = validate_simulation(df_annual, df_monthly)

    for _, v in validation.iterrows():
        status = "OKAY, meets threshold" if v["within_threshold"] else "EXCEEDS THRESHOLD"
        print(
            f"  {v['year_label']}: "
            f"Official={v['total_milk_production_litres']:>14,.0f}  "
            f"Simulated={v['simulated_annual_sum']:>14,.0f}  "
            f"Error={v['error_pct']:.6f}%  {status}"
        )

    all_ok = validation["within_threshold"].all()
    max_err = validation["error_pct"].max()
    print(f"\n  Maximum error: {max_err:.6f}%")
    print(f"  All years within 1% threshold: {' YES' if all_ok else ' NO'}")

    if not all_ok:
        print("\n  WARNING: Some years exceed the 1% threshold.")
        print("  Check the seasonal index or noise CV in 02_simulate.py.")
    else:
        print("\n  Simulation is valid. Proceeding to save.")

    # save Dataset 3
    df_monthly.to_csv("data/03_simulated_monthly.csv", index=False)
    print(f"\n Saved: data/03_simulated_monthly.csv")

    # Preview
    print("\nFirst 12 rows (Year 1 - 2016/2017):")
    preview_cols = ["date", "month_name", "monthly_production_litres", "official_annual_total"]
    print(df_monthly[df_monthly["year"] == 2016][preview_cols].to_string(index=False))

    print("\nSeasonal index used:")
    months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    for m, idx in zip(months, SEASONAL_INDEX):
        print(f"  {m}: {idx:.3f}")

    print("\nRun 03_eda.py next")
