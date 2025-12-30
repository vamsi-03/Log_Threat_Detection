import pandas as pd
import os

def prepare_balanced_dataset(input_csv, target_total=100000):
    """
    Standardizes and balances the 6-lakh raw dataset into a 1-lakh training set.
    
    Logic:
    1. Maps 'benign' to 0.
    2. Maps 'defacement', 'phishing', 'malware' to 1.
    3. Samples 50,000 from each class (total 100,000).
    4. Shuffles and saves for the BERT trainer.
    """
    if not os.path.exists(input_csv):
        print(f"Error: {input_csv} not found. Ensure the Kaggle file is in this directory.")
        return

    print(f"Loading raw dataset from {input_csv}...")
    # Using chunksize or high memory mode isn't necessary for 600k rows, 
    # but we'll read only the columns we need to be efficient.
    df = pd.read_csv(input_csv, usecols=['url', 'type'])
    
    # --- Transformation ---
    # Map classes to binary labels
    # Label 0: Benign
    # Label 1: Any threat (defacement, phishing, malware)
    df['label'] = df['type'].apply(lambda x: 0 if x == 'benign' else 1)
    df = df.rename(columns={'url': 'text'})
    
    # Cleaning: Remove duplicates and nulls which are common in real-world log dumps
    initial_count = len(df)
    df = df.dropna(subset=['text', 'label']).drop_duplicates(subset=['text'])
    print(f"Cleaned {initial_count - len(df)} duplicate/null entries.")

    # --- Balancing (The 1-Lakh Logic) ---
    print(f"Subsetting data to {target_total} balanced samples...")
    
    df_benign = df[df['label'] == 0]
    df_malicious = df[df['label'] == 1]
    
    half_target = target_total // 2
    
    # Validate we have enough samples in both categories
    if len(df_benign) < half_target or len(df_malicious) < half_target:
        print(f"Warning: Not enough samples to reach {target_total} balanced.")
        half_target = min(len(df_benign), len(df_malicious))
        print(f"Adjusting half-target to {half_target} per class.")

    # Randomly sample from each bucket
    df_0_sampled = df_benign.sample(n=half_target, random_state=42)
    df_1_sampled = df_malicious.sample(n=half_target, random_state=42)
    
    # Combine and Shuffle (Shuffling is critical so the model doesn't learn batch-level bias)
    df_final = pd.concat([df_0_sampled, df_1_sampled]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Final distribution check
    print("\n--- Processed Dataset Statistics ---")
    print(df_final['label'].value_counts())
    print(f"Total rows: {len(df_final)}")
    
    # Save the prepared file
    output_name = "cyber_logs.csv"
    df_final[['text', 'label']].to_csv(output_name, index=False)
    print(f"\nSUCCESS: Balanced 1-lakh dataset saved as '{output_name}'.")

if __name__ == "__main__":
    # Ensure the input filename matches your downloaded Kaggle file exactly
    # Common names: 'malicious_urls.csv' or 'urldata.csv'
    input_filename = "malicious_phish.csv" 
    prepare_balanced_dataset(input_filename, target_total=30000)