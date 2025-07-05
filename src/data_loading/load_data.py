import logging
import os
import time

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

logger = logging.getLogger("src.data_loading.load_data")


def fetch_data() -> pd.DataFrame:
    """Fetch the breast cancer dataset and convert to DataFrame.

    Returns:
        pd.DataFrame: DataFrame containing the breast cancer data with features and target
    """
    logger.info("Fetching data...")
    dataset = load_breast_cancer()
    
    # Features columns
    data = pd.DataFrame(data=dataset.data, columns=dataset.feature_names)
    
    # Introduce random NaN values
    np.random.seed(42)
    for col in data.columns:
        mask = np.random.random(len(data)) < 0.05  # 5% chance of NaN
        data.loc[mask, col] = np.nan
    
    # Target column
    data["target"] = dataset.target
    
    # Shuffle data to simulate changed data
    data = data.sample(frac=1, random_state=int(time.time())).reset_index(drop=True)

    return data


def save_data(data: pd.DataFrame) -> None:
    """Save the raw data to disk.

    Args:
        data (pd.DataFrame): Raw breast cancer dataset to save
    """
    output_path = "data/raw/raw.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    logger.info(f"Saving raw data to {output_path}")
    data.to_csv(output_path, index=False)


def main() -> None:
    """Main function to orchestrate the data loading process."""
    raw_data = fetch_data()
    save_data(raw_data)
    logger.info("Data loading completed")


if __name__ == "__main__":
    main()
    main()
