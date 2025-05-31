import pandas as pd

def load_email_data(filepath="combined_emails_with_natural_pii.csv"):
    """
    Loads the email dataset from a specified file path.
    """
    try:
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
        elif filepath.endswith('.json'):
            df = pd.read_json(filepath)
        else:
            print("Unsupported file format. Please provide a .csv or .json file.")
            return None
        return df
    except FileNotFoundError:
        print(f"Error: Dataset not found at {filepath}. Please ensure the file is in the correct location.")
        return None
    except Exception as e:
        print(f"An error occurred while loading the data: {e}")
        return None

if __name__ == "__main__":
    data_df = load_email_data()

    if data_df is not None:
        print("\n--- First 5 rows of the dataset ---")
        print(data_df.head())

        print("\n--- Dataset Information ---")
        data_df.info()

        print("\n--- Column Names ---")
        print(data_df.columns.tolist())

        print("\n--- Value Counts for Category Column ---")
        # Updated to directly use the 'type' column as identified from your data
        print(data_df['type'].value_counts())

        print("\n--- Check for missing values ---")
        print(data_df.isnull().sum())