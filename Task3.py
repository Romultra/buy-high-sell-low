import pandas as pd
from UsefulFunctions import LoadData, PricesDK  

# Load the necessary data
df_prices, df_pro = LoadData()

# Compute Buy Prices using the given function
df_prices = PricesDK(df_prices)

# Keep only necessary columns
df_prices = df_prices[["HourDK", "Buy"]]
df_pro = df_pro[["HourDK", "Load"]]

# Merge both datasets on time
df = pd.merge(df_pro, df_prices, on="HourDK", how="inner")

# Convert time column to datetime format
df["HourDK"] = pd.to_datetime(df["HourDK"])

# Filter data for the years 2022 and 2023
df_filtered = df[(df["HourDK"].dt.year == 2022) | (df["HourDK"].dt.year == 2023)]

# Compute total cost for each year (Consumption * Buy Price per hour)
df_filtered["TotalCost"] = df_filtered["Load"] * df_filtered["Buy"]
annual_total_costs = df_filtered.groupby(df_filtered["HourDK"].dt.year)["TotalCost"].sum()

# Compute total yearly consumption and average Buy Price
annual_consumption = df_filtered.groupby(df_filtered["HourDK"].dt.year)["Load"].sum()
annual_avg_buy_price = df_filtered.groupby(df_filtered["HourDK"].dt.year)["Buy"].mean()

# Compute estimated cost using total consumption * average Buy Price
rough_costs = annual_consumption * annual_avg_buy_price

# 🔍 Analyze the difference
difference = abs(annual_total_costs - rough_costs)

# Compare the two methods
comparison_df = pd.DataFrame({
    "Exact Total Cost": annual_total_costs,
    "Total Consumption (kWh)": annual_consumption,
    "Average Buy Price (DKK/kWh)": annual_avg_buy_price,
    "Estimated Total Cost (Rough Calculation)": rough_costs,
    "Difference": difference
})

# Print the final comparison table
print(comparison_df)
