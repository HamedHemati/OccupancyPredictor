import pandas as pd


def load_dataset(file_path):
    """Load dataset from txt file and return a dataframe tuple (X, y)."""
    df = pd.read_csv(
        file_path,
        skiprows=1,
        names=["Index", "Date", "Temperature", "Humidity", "Light", "CO2", "HumidityRatio", "Occupancy"],
    )

    # Drop the Index and Date columns (We dont' need it for out model)
    df.drop(columns=["Index", "Date"], inplace=True)

    # Define features and labels
    X = df.drop(columns=["Occupancy"])
    y = df["Occupancy"]

    return X, y
