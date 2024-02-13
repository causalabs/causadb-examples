import numpy as np
import pandas as pd

np.random.seed(0)


def simulate_hvac(df, hvac_value):
    indoor_temp = 13 + hvac_value * 0.1 + np.random.normal(0, 0.01, len(df))
    energy = 20 * hvac_value + 10 * indoor_temp + \
        np.random.normal(0, 10, len(df))

    # Round
    indoor_temp = np.round(indoor_temp, 1)
    energy = np.round(energy)

    return pd.DataFrame({
        "hvac": hvac_value,
        "indoor_temp": indoor_temp,
        "energy": energy
    })


if __name__ == "__main__":
    num_samples = 1000

    # Generate data
    hvac = np.random.uniform(0, 100, num_samples)
    indoor_temp = 13 + hvac * 0.1 + np.random.normal(0, 0.01, num_samples)
    energy = 20 * hvac + 10 * indoor_temp + \
        np.random.normal(0, 10, num_samples)

    # Round
    hvac = np.round(hvac)
    indoor_temp = np.round(indoor_temp, 1)
    energy = np.round(energy)

    df = pd.DataFrame({
        "hvac": hvac,
        "indoor_temp": indoor_temp,
        "energy": energy
    })

    # Trim extreme values
    df = df[(df["indoor_temp"] > 13) & (df["indoor_temp"] < 27)]
    # Remove any negative values of energy
    df = df[df["energy"] > 0]

    df.to_csv("smart_building/data.csv", index=False)
