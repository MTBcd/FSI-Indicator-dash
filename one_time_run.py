# import pandas as pd
# from pathlib import Path

# src = Path("cache-directory/Full_set_variables_brut.csv")
# dst = Path("seed-data/ice_bofa_history.csv")
# dst.parent.mkdir(parents=True, exist_ok=True)

# df = pd.read_csv(src).rename(columns={
#     "Unnamed: 0": "Date",
#     "US Corp OAS": "US IG OAS",
# })

# seed = df[["Date", "US IG OAS", "US HY OAS", "US BBB OAS"]].copy()
# seed["Date"] = pd.to_datetime(seed["Date"])
# seed = seed.sort_values("Date")

# seed.to_csv(dst, index=False)
# print(f"saved {dst}")