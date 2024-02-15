import numpy as np
import pandas as pd

np.random.seed(0)


def simulate_hvac(df, hvac):
    indoor_temp = 24 - 0.1 * hvac + \
        np.random.normal(0, 0.01, len(df))
    energy = 200 + 2 * hvac + 10 * indoor_temp + \
        np.random.normal(0, 0.01, len(df))

    return pd.DataFrame({
        "hvac": hvac,
        "indoor_temp": indoor_temp,
        "energy": energy
    })


if __name__ == "__main__":
    num_samples = 1000

    # Generate data
    hvac = np.random.normal(50, 7.5, num_samples)
    indoor_temp = 24 - 0.1 * hvac + \
        np.random.normal(0, 0.01, num_samples)
    energy = 200 + 2 * hvac + 10 * indoor_temp + \
        np.random.normal(0, 0.01, num_samples)

    df = pd.DataFrame({
        "hvac": hvac,
        "indoor_temp": indoor_temp,
        "energy": energy
    })

    print(indoor_temp.min(), indoor_temp.max())

    # Trim extreme values
    df = df[(df["indoor_temp"] > 13) & (df["indoor_temp"] < 27)]
    # Remove any negative values of energy
    df = df[df["energy"] > 0]

    df.to_csv("smart_building/data.csv", index=False)
