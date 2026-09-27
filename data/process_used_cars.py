import os
import re
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_PATH = os.path.join(BASE_DIR, 'used_cars.csv')
CLEAN_PATH = os.path.join(BASE_DIR, 'used_cars_clean.csv')
REFERENCE_YEAR = 2026


def clean_mileage(value):
    if pd.isna(value):
        return np.nan
    text = str(value).replace(',', '')
    match = re.search(r'[\d.]+', text)
    if not match:
        return np.nan
    return float(match.group())


def clean_price(value):
    if pd.isna(value):
        return np.nan
    text = str(value).replace('$', '').replace(',', '')
    match = re.search(r'[\d.]+', text)
    if not match:
        return np.nan
    return float(match.group())


def process_dataset():
    if not os.path.exists(RAW_PATH):
        raise FileNotFoundError(
            'Missing raw dataset at data/used_cars.csv. '
            'Download from: https://www.kaggle.com/datasets/taeefnajib/used-car-price-prediction-dataset'
        )

    df = pd.read_csv(RAW_PATH)
    original_records = len(df)
    print('Columns found:', list(df.columns))

    mileage_col = 'milage' if 'milage' in df.columns else 'mileage'
    df['mileage_numeric'] = df[mileage_col].apply(clean_mileage)
    df['model_year_numeric'] = pd.to_numeric(df['model_year'], errors='coerce')
    df['price_numeric'] = df['price'].apply(clean_price)
    df['vehicle_age'] = REFERENCE_YEAR - df['model_year_numeric']

    df = df.dropna(subset=['vehicle_age', 'mileage_numeric'])
    df = df[df['vehicle_age'] >= 0]
    df = df[df['vehicle_age'] <= 60]
    df = df[df['mileage_numeric'] >= 0]
    df = df[df['mileage_numeric'] <= 500000]
    df = df.drop_duplicates()

    valid_records = len(df)
    removed_records = original_records - valid_records

    output = df[[
        'brand', 'model', 'model_year_numeric', 'vehicle_age',
        'mileage_numeric', 'price_numeric', 'fuel_type', 'transmission'
    ]].rename(columns={
        'model_year_numeric': 'model_year',
        'mileage_numeric': 'mileage',
        'price_numeric': 'price'
    })

    output.to_csv(CLEAN_PATH, index=False)

    print('Used-car dataset processed successfully.')
    print('Original records:', original_records)
    print('Valid records:', valid_records)
    print('Removed records:', removed_records)
    print('Saved to:', CLEAN_PATH)


if __name__ == '__main__':
    process_dataset()