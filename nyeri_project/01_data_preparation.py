"""
Start of end of degree project
Faith Wambui Gichuru: SCT213-C002-0003/2022

CRISP-DM PHASE 1 & 2: Business Understanding + Data Understanding

01_data_preparation.py


PURPOSE:
This script hardcodes the raw data exactly as provided by the Nyeri County
Department of Livestock, structures it into two clean CSV files, runs basic
data quality checks, and prints a summary of what was received.

This is the single source of truth for all observed data in the project.

OUTPUTS:
1:data/01_county_annual.csv        - 9 annual county-level records
2:data/02_subcounty_population.csv - 5yr x 8 sub-county cattle population

"""

import os
import pandas as pd
import numpy as np

#data directory check 
os.makedirs("data", exist_ok=True)


# DATASET 1 — County-Level Annual Trends (Observed)
# Source: Nyeri County Department of Livestock production tables
# Variables: Year, Total Milk Production (litres), Dairy Cattle Population,
# Average Production Per Cow (litres/year)
# Coverage: 9 annual records, 2016/2017 to 2024/2025


county_annual_data = {
    "year": [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
    "year_label": [
        "2016/2017", "2017/2018", "2018/2019", "2019/2020",
        "2020/2021", "2021/2022", "2022/2023", "2023/2024", "2024/2025"
    ],
    "total_milk_production_litres": [
        102_600_000,   # 2016/2017
        93_826_201,   # 2017/2018  -slight dip, pre-COVID dry season
        112_934_263,   # 2018/2019
        115_617_604,   # 2019/2020
        111_727_403,   # 2020/2021  -slight drop attributed to COVID-19 impacts
        114_956_225,   # 2021/2022  -recovery, technology adoption
        115_338_802,   # 2022/2023
        116_144_542,   # 2023/2024
        121_031_160,   # 2024/2025  -attributed to TIMPs adoption + favourable weather
    ],
    "dairy_cattle_population": [
        169_118,   # 2016/2017
        171_000,   # 2017/2018
        171_288,   # 2018/2019
        178_753,   # 2019/2020
        192_483,   # 2020/2021
        193_310,   # 2021/2022
        188_500,   # 2022/2023  -drop attributed to very dry weather 2022/2023
        189_310,   # 2023/2024
        190_625,   # 2024/2025
    ],
    "avg_production_per_cow_litres": [
        5, 5, 5, 6, 6, 7, 7, 7, 8
    ],
    "context_note": [
        "",
        "",
        "",
        "",
        "Slight drop due to impacts of Covid-19",
        "Increase due to adoption of technologies despite effects of climate change",
        "Slight drop in cattle population attributed to very dry weather 2022/2023",
        "",
        "Attributed to adoption of TIMPs and relatively favourable weather",
    ]
}

df_annual = pd.DataFrame(county_annual_data)

#derived features 
df_annual["yoy_production_growth_pct"] = (
    df_annual["total_milk_production_litres"].pct_change() * 100
).round(2)

df_annual["yoy_cattle_growth_pct"] = (
    df_annual["dairy_cattle_population"].pct_change() * 100
).round(2)


df_annual.to_csv("data/01_county_annual.csv", index=False)
print(" Saved: data/01_county_annual.csv")



# DATASET 2 — Sub-County Dairy Cattle Population (Observed)
# Source: Nyeri County Department of Livestock production tables
# Variables: Year, Sub-County, Dairy Cattle Population
# Coverage: 5 annual records x 8 sub-counties, 2020/2021 to 2024/2025

subcounty_data = {
    "year": [2020, 2021, 2022, 2023, 2024],
    "year_label": ["2020/2021", "2021/2022", "2022/2023", "2023/2024", "2024/2025"],
    # Sub-county columns — cattle head count
    "kieni_east":    [35_385, 35_410, 34_250, 35_400, 35_610],
    "mathira_east":  [30_170, 30_250, 29_810, 30_080, 30_250],
    "mathira_west":  [18_005, 18_460, 17_975, 18_070, 18_105],
    "mukurwe_ini":   [21_251, 21_260, 20_975, 21_005, 21_260],
    "kieni_west":    [30_506, 30_520, 29_370, 30_205, 30_520],
    "nyeri_central": [10_451, 10_580,  9_935,  9_085,  9_150],
    "nyeri_south":   [19_705, 19_805, 19_395, 19_610, 19_830],
    "tetu":          [27_010, 27_025, 26_790, 25_165, 25_900],
    "total":         [192_483, 193_310, 188_500, 189_310, 190_625],
}

df_subcounty = pd.DataFrame(subcounty_data)

# validation: sub-county totals must match county totals for those years 
subcounty_cols = [
    "kieni_east", "mathira_east", "mathira_west", "mukurwe_ini",
    "kieni_west", "nyeri_central", "nyeri_south", "tetu"
]
df_subcounty["computed_total"] = df_subcounty[subcounty_cols].sum(axis=1)
df_subcounty["total_check_ok"] = (
    df_subcounty["computed_total"] == df_subcounty["total"]
)

# save
df_subcounty.to_csv("data/02_subcounty_population.csv", index=False)
print("Saved: data/02_subcounty_population.csv")


# DATA UNDERSTANDING SUMMARY:CRISP-DM Phase 2 Output
print("CRISP-DM PHASE 2: DATA UNDERSTANDING SUMMARY")

print("\n Dataset 1: County Annual (n=9) ")
print(df_annual[[
    "year_label",
    "total_milk_production_litres",
    "dairy_cattle_population",
    "avg_production_per_cow_litres"
]].to_string(index=False))

print(f"\nDescriptive statistics - Milk Production:")
prod = df_annual["total_milk_production_litres"]
print(f"  Min:    {prod.min():>15,.0f} litres")
print(f"  Max:    {prod.max():>15,.0f} litres")
print(f"  Mean:   {prod.mean():>15,.0f} litres")
print(f"  Median: {prod.median():>15,.0f} litres")
print(f"  Std:    {prod.std():>15,.0f} litres")

print(f"\nYear-over-Year Production Growth:")
for _, row in df_annual.dropna(subset=["yoy_production_growth_pct"]).iterrows():
    print(f"  {row['year_label']}: {row['yoy_production_growth_pct']:+.2f}%")

print(f"\n Dataset 2: Sub-County Population (n=5 years x 8 sub-counties) ")
print(df_subcounty[["year_label"] + subcounty_cols + ["total", "total_check_ok"]].to_string(index=False))

total_ok = df_subcounty["total_check_ok"].all()
print(f"\n  Sub-county totals match county totals: {'YES' if total_ok else ' NO - CHECK DATA'}")

print("\n Key Data Constraint (CRISP-DM Business Understanding)")
print(f"  Observed annual records: {len(df_annual)}")
print(f"  Minimum for ARIMA:       ~24 monthly records")
print(f"  Minimum for SARIMA:      ~36 monthly records")
print(f"  Simulation required to generate ~108 monthly records")
print(f"  Run 02_simulate.py next")

