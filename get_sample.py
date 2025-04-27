import pandas as pd
import os

def get_dataset_sample(data_dir="data", sample_size=5, output_file="worldrep_sample.csv"):
    """
    Extract and save a small sample from the WORLDREP dataset files
    
    Args:
        data_dir (str): Directory containing the data files
        sample_size (int): Number of rows to sample from each file
        output_file (str): File to save the combined sample
    """
    # List all CSV files in the data directory
    all_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) 
                if f.endswith('.csv')]
    
    print(f"Found {len(all_files)} CSV files in {data_dir}")
    
    # Initialize empty list to store samples
    samples = []
    
    # Read a sample from each file
    for file in all_files:
        try:
            # First try with utf-8 encoding
            df = pd.read_csv(file, encoding='utf-8', nrows=2)
            # Print column names
            print(f"\nFile: {os.path.basename(file)}")
            print(f"Columns: {df.columns.tolist()}")
            # Get sample
            sample = pd.read_csv(file, encoding='utf-8', nrows=sample_size)
            samples.append(sample)
        except UnicodeDecodeError:
            # Try with latin1 encoding if utf-8 fails
            try:
                df = pd.read_csv(file, encoding='latin1', nrows=2)
                print(f"\nFile: {os.path.basename(file)}")
                print(f"Columns: {df.columns.tolist()}")
                sample = pd.read_csv(file, encoding='latin1', nrows=sample_size)
                samples.append(sample)
            except Exception as e:
                print(f"Error reading {file}: {e}")
    
    if not samples:
        print("No samples were extracted. Check file paths and formats.")
        return None
    
    # Combine all samples
    combined_sample = pd.concat(samples, ignore_index=True)
    
    # Save the combined sample
    combined_sample.to_csv(output_file, index=False)
    print(f"\nSaved {len(combined_sample)} sample rows to {output_file}")
    
    return combined_sample

if __name__ == "__main__":
    sample = get_dataset_sample()
    if sample is not None:
        # Display basic info about the sample
        print("\nSample shape:", sample.shape)
        print("\nSample data types:")
        print(sample.dtypes)
        
        # Display a few rows
        print("\nSample data (first 3 rows):")
        pd.set_option('display.max_columns', None)  # Show all columns
        pd.set_option('display.max_colwidth', 30)   # Truncate column content
        print(sample.head(3))