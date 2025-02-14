import pandas as pd
import hashlib
from pathlib import Path

def get_file_hash(file_path):
    """Calculate MD5 hash of a file"""
    with open(file_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def compare_csv_files(directory='.'):
    """Compare all submission CSV files in the directory"""
    # Get all submission files
    files = list(Path(directory).glob('submission*.csv'))
    
    # Store file hashes and dataframes
    file_hashes = {}
    dataframes = {}
    
    print("Analyzing files...")
    print("-" * 80)
    
    # First pass: Calculate hashes and load dataframes
    for file_path in files:
        file_name = file_path.name
        file_hash = get_file_hash(file_path)
        df = pd.read_csv(file_path)
        
        file_hashes[file_name] = file_hash
        dataframes[file_name] = df
        
        print(f"File: {file_name}")
        print(f"Size: {file_path.stat().st_size / 1024:.2f} KB")
        print(f"Hash: {file_hash}")
        print(f"Columns: {list(df.columns)}")
        print("-" * 80)
    
    # Second pass: Compare files
    print("\nFile Comparisons:")
    print("-" * 80)
    
    for i, (file1, hash1) in enumerate(file_hashes.items()):
        for file2, hash2 in list(file_hashes.items())[i+1:]:
            print(f"\nComparing {file1} with {file2}:")
            if hash1 == hash2:
                print("  ✓ Files are IDENTICAL (same binary content)")
            else:
                df1 = dataframes[file1]
                df2 = dataframes[file2]
                
                # Compare columns
                cols1 = set(df1.columns)
                cols2 = set(df2.columns)
                common_cols = cols1.intersection(cols2)
                
                if cols1 != cols2:
                    print(f"  ⚠ Different columns:")
                    if cols1 - cols2:
                        print(f"    - Only in {file1}: {cols1 - cols2}")
                    if cols2 - cols1:
                        print(f"    - Only in {file2}: {cols2 - cols1}")
                
                # Compare content for common columns
                if len(common_cols) > 0:
                    different_cols = []
                    for col in common_cols:
                        if not df1[col].equals(df2[col]):
                            different_cols.append(col)
                    
                    if different_cols:
                        print(f"  ⚠ Different values in columns: {different_cols}")
                    else:
                        print("  ✓ All common columns have identical values")

if __name__ == "__main__":
    compare_csv_files() 