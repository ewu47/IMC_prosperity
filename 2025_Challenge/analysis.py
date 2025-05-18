import pandas as pd

def get_averages():
    # Load all 3 days
    df0 = pd.read_csv("round-1-island-data-bottle/prices_round_1_day_0.csv", delimiter=';')
    df1 = pd.read_csv("round-1-island-data-bottle/prices_round_1_day_-1.csv", delimiter=';')
    df2 = pd.read_csv("round-1-island-data-bottle/prices_round_1_day_-2.csv", delimiter=';')

    # Combine all days
    df_all = pd.concat([df0, df1, df2], ignore_index=True)

    # Drop rows without mid price
    df_all = df_all.dropna(subset=["mid_price"])

    # Target products and reference price
    products = ["SQUID_INK", "KELP", "RAINFOREST_RESIN"]
    reference_price= {"SQUID_INK" : 2000,
                        "KELP" : 2000,
                        "RAINFOREST_RESIN" : 10000}

    # Results dict
    results = {}

    for product in products:
        prices = df_all[df_all["product"] == product]["mid_price"]
        diffs = abs(prices - reference_price[product])
        mean_diff = diffs.mean()
        std_dev = diffs.std()

        results[product] = {
            "mean_diff_from_2000": float(mean_diff),
            "std_dev_from_2000": float(std_dev)
        }

    return results

# Example usage
print(get_averages())
