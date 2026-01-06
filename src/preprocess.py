"""
Task 1: Data Preprocessing Script

This script filters and cleans the CFPB complaints dataset for the RAG pipeline.
It produces data/filtered_complaints.csv with only the 5 target product categories
and cleaned narrative text.

Usage:
    python src/preprocess.py
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path
from tqdm import tqdm

# Paths
RAW_DATA_PATH = Path('data/raw/complaints.csv')
OUTPUT_PATH = Path('data/filtered_complaints.csv')

# Product mapping: CFPB product names -> our standardized categories
PRODUCT_MAPPING = {
    # Credit Card
    'Credit card': 'credit_card',
    'Credit card or prepaid card': 'credit_card',
    
    # Personal Loan
    'Consumer Loan': 'personal_loan',
    'Payday loan': 'personal_loan',
    'Payday loan, title loan, or personal loan': 'personal_loan',
    'Payday loan, title loan, personal loan, or advance app': 'personal_loan',
    
    # Savings Account
    'Bank account or service': 'savings_account',
    'Checking or savings account': 'savings_account',
    
    # Money Transfers
    'Money transfers': 'money_transfer',
    'Money transfer, virtual currency, or money service': 'money_transfer',
}

# Columns to load from raw data
USECOLS = [
    'Complaint ID',
    'Product',
    'Sub-product',
    'Issue',
    'Sub-issue',
    'Consumer complaint narrative',
    'Company',
    'Date received'
]


def clean_narrative(text: str) -> str:
    """Clean complaint narrative text.
    
    Args:
        text: Raw narrative text
        
    Returns:
        Cleaned text
    """
    if pd.isna(text) or not isinstance(text, str):
        return ''
    
    # Lowercase
    text = text.lower()
    
    # Remove XXXX placeholders (CFPB redaction pattern)
    text = re.sub(r'x{2,}', '[REDACTED]', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def load_raw_data(path: Path) -> pd.DataFrame:
    """Load raw CFPB complaints data.
    
    Args:
        path: Path to raw CSV file
        
    Returns:
        DataFrame with selected columns
    """
    print(f"Loading data from {path}...")
    df = pd.read_csv(
        path,
        usecols=USECOLS,
        dtype={
            'Product': 'string',
            'Sub-product': 'string',
            'Consumer complaint narrative': 'string',
            'Issue': 'string',
            'Company': 'string'
        }
    )
    print(f"Loaded {len(df):,} rows")
    return df


def filter_to_target_products(df: pd.DataFrame) -> pd.DataFrame:
    """Filter dataset to target product categories.
    
    Args:
        df: Raw DataFrame
        
    Returns:
        Filtered DataFrame with product_category column
    """
    target_products = list(PRODUCT_MAPPING.keys())
    df_filtered = df[df['Product'].isin(target_products)].copy()
    print(f"After filtering to target products: {len(df_filtered):,} rows")
    
    # Map to standardized names
    df_filtered['product_category'] = df_filtered['Product'].map(PRODUCT_MAPPING)
    
    return df_filtered


def remove_empty_narratives(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows with empty or missing narratives.
    
    Args:
        df: DataFrame with narratives
        
    Returns:
        DataFrame with only non-empty narratives
    """
    df_clean = df[
        df['Consumer complaint narrative'].fillna('').str.strip() != ''
    ].copy()
    print(f"After removing empty narratives: {len(df_clean):,} rows")
    return df_clean


def apply_text_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """Apply text cleaning to narratives.
    
    Args:
        df: DataFrame with narratives
        
    Returns:
        DataFrame with cleaned narrative_clean column
    """
    print("Cleaning narratives...")
    tqdm.pandas(desc="Cleaning")
    df['narrative_clean'] = df['Consumer complaint narrative'].progress_apply(clean_narrative)
    return df


def prepare_final_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare final dataset with renamed columns.
    
    Args:
        df: Processed DataFrame
        
    Returns:
        Final DataFrame ready for saving
    """
    df_final = df[[
        'Complaint ID',
        'product_category',
        'Product',
        'Sub-product',
        'Issue',
        'Sub-issue',
        'narrative_clean',
        'Company',
        'Date received'
    ]].copy()
    
    df_final.columns = [
        'complaint_id',
        'product',
        'product_original',
        'sub_product',
        'issue',
        'sub_issue',
        'narrative',
        'company',
        'date_received'
    ]
    
    return df_final


def main():
    """Main preprocessing pipeline."""
    print("=" * 60)
    print("CFPB Complaints Preprocessing Pipeline")
    print("=" * 60)
    
    # Load data
    df = load_raw_data(RAW_DATA_PATH)
    original_count = len(df)
    
    # Filter to target products
    df = filter_to_target_products(df)
    
    # Remove empty narratives
    df = remove_empty_narratives(df)
    
    # Clean text
    df = apply_text_cleaning(df)
    
    # Prepare final dataset
    df_final = prepare_final_dataset(df)
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Original complaints: {original_count:,}")
    print(f"Final complaints: {len(df_final):,}")
    print(f"Reduction: {(1 - len(df_final)/original_count)*100:.1f}%")
    print(f"\nDistribution by product category:")
    for cat, count in df_final['product'].value_counts().items():
        pct = count / len(df_final) * 100
        print(f"  {cat}: {count:,} ({pct:.1f}%)")
    
    # Save
    print(f"\nSaving to {OUTPUT_PATH}...")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(OUTPUT_PATH, index=False)
    print(f"Done! File size: {OUTPUT_PATH.stat().st_size / 1e6:.2f} MB")
    
    return df_final


if __name__ == '__main__':
    main()
