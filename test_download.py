import kagglehub
from kagglehub import KaggleDatasetAdapter
import sys
import pandas as pd # Import pandas to use its kwargs

print("Attempting to download dataset from KaggleHub...")

try:
    file_path = "IMDB Dataset.csv"

    # Load the latest version
    # Note: The warning about load_dataset vs dataset_load is fine, 
    # we can ignore it for now or fix it later.
    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "lakshmi25npathi/imdb-dataset-of-50k-movie-reviews",
        file_path,
        # This is the new line:
        # Pass the encoding argument directly to the underlying pandas reader
        pandas_kwargs={'encoding': 'latin-1'} 
    )
    
    print("\n--- Download Successful! ---")
    print("First 5 records:")
    print(df.head())
    print(f"\nTotal records loaded: {len(df)}")

except ImportError:
    print("\n--- Import Failed ---", file=sys.stderr)
    print("It looks like 'KaggleDatasetAdapter' still can't be imported.", file=sys.stderr)

except Exception as e:
    print(f"\n--- Download Failed ---", file=sys.stderr)
    print(f"An error occurred: {e}", file=sys.stderr)
    print("Please check your internet connection and Kaggle API credentials.", file=sys.stderr)