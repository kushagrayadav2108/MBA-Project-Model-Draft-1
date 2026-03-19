"""
Data Processor Module for Seed Demand Prediction
Handles Excel parsing, data cleaning, and feature extraction
"""
import pandas as pd
import numpy as np
from pathlib import Path


def load_and_process_data(excel_path: str) -> pd.DataFrame:
    """
    Load Excel data and process it into a clean DataFrame.
    
    Args:
        excel_path: Path to the Excel file
        
    Returns:
        Cleaned DataFrame with proper column names
    """
    # Read Excel - skip first 3 rows as they contain merged headers
    df = pd.read_excel(excel_path, header=None)
    
    # Extract actual data starting from row 3 (0-indexed row 3 contains column names)
    column_names = [
        'Unnamed', 'Year', 'Max_Temp', 'Min_Temp', 
        'Annual_Rainfall', 'Pre_Monsoon_Rainfall', 'Monsoon_Rainfall', 'Post_Monsoon_Rainfall',
        'Monsoon_Duration', 'Total_Area_Hectare', 'Total_Production_Tonne', 'Yield_Tonne_Hectare',
        'Pb_1121_Share', 'Pb_1718_Share', 'Pb_1885_Share', 'Pb_1509_Share', 'Pb_1692_Share', 'Pb_1847_Share', 'Others_Share',
        'Pb_1121_Price', 'Pb_1718_Price', 'Pb_1885_Price', 'Pb_1509_Price', 'Pb_1692_Price', 'Pb_1847_Price'
    ]
    
    # Get data rows (skip header rows)
    data_df = df.iloc[4:].copy()  # Start from row 4 (actual data)
    data_df.columns = column_names
    
    # Drop the first unnamed column
    data_df = data_df.drop('Unnamed', axis=1)
    
    # Convert Year to numeric
    data_df['Year'] = pd.to_numeric(data_df['Year'], errors='coerce')
    
    # Drop rows where Year is NaN
    data_df = data_df.dropna(subset=['Year'])
    data_df['Year'] = data_df['Year'].astype(int)
    
    # Sort by year (ascending)
    data_df = data_df.sort_values('Year').reset_index(drop=True)
    
    return data_df


def parse_price_range(price_str) -> tuple:
    """
    Parse price range string like '3400-4200' into (min, max) tuple.
    Returns (NaN, NaN) if parsing fails.
    """
    if pd.isna(price_str) or price_str == '-' or str(price_str).strip() == '':
        return np.nan, np.nan
    
    try:
        price_str = str(price_str).strip()
        if '-' in price_str:
            parts = price_str.split('-')
            if len(parts) == 2:
                min_price = float(parts[0].strip())
                max_price = float(parts[1].strip())
                return min_price, max_price
        return np.nan, np.nan
    except (ValueError, AttributeError):
        return np.nan, np.nan


def process_prices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process price ranges into min/max/mid columns for each variety.
    """
    varieties = ['Pb_1121', 'Pb_1718', 'Pb_1885', 'Pb_1509', 'Pb_1692', 'Pb_1847']
    
    for variety in varieties:
        price_col = f'{variety}_Price'
        if price_col in df.columns:
            # Parse price ranges
            price_ranges = df[price_col].apply(parse_price_range)
            df[f'{variety}_Price_Min'] = price_ranges.apply(lambda x: x[0])
            df[f'{variety}_Price_Max'] = price_ranges.apply(lambda x: x[1])
            df[f'{variety}_Price_Mid'] = (df[f'{variety}_Price_Min'] + df[f'{variety}_Price_Max']) / 2
    
    return df


def convert_shares_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert variety share percentages to numeric, handling '-' and NaN values.
    """
    share_cols = ['Pb_1121_Share', 'Pb_1718_Share', 'Pb_1885_Share', 
                  'Pb_1509_Share', 'Pb_1692_Share', 'Pb_1847_Share', 'Others_Share']
    
    for col in share_cols:
        if col in df.columns:
            df[col] = df[col].replace('-', np.nan)
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df


def calculate_variety_areas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate estimated area for each variety based on total area and share percentage.
    """
    varieties = ['Pb_1121', 'Pb_1718', 'Pb_1885', 'Pb_1509', 'Pb_1692', 'Pb_1847']
    
    for variety in varieties:
        share_col = f'{variety}_Share'
        if share_col in df.columns:
            df[f'{variety}_Area'] = (df['Total_Area_Hectare'] * df[share_col] / 100)
    
    return df


def calculate_seed_demand(df: pd.DataFrame, seeding_rate_kg_per_ha: float = 15.0) -> pd.DataFrame:
    """
    Calculate seed demand for each variety.
    Seed demand (quintals) = Area (hectares) * Seeding rate (kg/ha) / 100
    
    Args:
        df: DataFrame with variety areas
        seeding_rate_kg_per_ha: Seeding rate in kg per hectare (default 15 kg/ha = ~6 kg/acre)
    """
    varieties = ['Pb_1121', 'Pb_1718', 'Pb_1885', 'Pb_1509', 'Pb_1692', 'Pb_1847']
    
    for variety in varieties:
        area_col = f'{variety}_Area'
        if area_col in df.columns:
            # Seed demand in quintals (1 quintal = 100 kg)
            df[f'{variety}_Seed_Demand_Qtl'] = df[area_col] * seeding_rate_kg_per_ha / 100
    
    return df


def convert_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert all numeric columns to proper numeric types.
    """
    numeric_cols = [
        'Max_Temp', 'Min_Temp', 'Annual_Rainfall', 'Pre_Monsoon_Rainfall', 
        'Monsoon_Rainfall', 'Post_Monsoon_Rainfall', 'Monsoon_Duration',
        'Total_Area_Hectare', 'Total_Production_Tonne', 'Yield_Tonne_Hectare'
    ]
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df


def prepare_data_for_modeling(excel_path: str) -> pd.DataFrame:
    """
    Main function to prepare data for modeling.
    
    Args:
        excel_path: Path to the Excel file
        
    Returns:
        Fully processed DataFrame ready for modeling
    """
    # Load and process data
    df = load_and_process_data(excel_path)
    
    # Convert numeric columns
    df = convert_numeric_columns(df)
    
    # Convert share percentages
    df = convert_shares_to_numeric(df)
    
    # Process price ranges
    df = process_prices(df)
    
    # Calculate variety areas
    df = calculate_variety_areas(df)
    
    # Calculate seed demand
    df = calculate_seed_demand(df)
    
    return df


if __name__ == "__main__":
    # Test the data processor
    excel_path = "Mba BA project.xlsx"
    df = prepare_data_for_modeling(excel_path)
    
    print("Data loaded successfully!")
    print(f"Shape: {df.shape}")
    print(f"\nYears: {df['Year'].min()} - {df['Year'].max()}")
    print(f"\nColumns:\n{df.columns.tolist()}")
    print(f"\nSample data (last 5 years):")
    print(df.tail().to_string())
