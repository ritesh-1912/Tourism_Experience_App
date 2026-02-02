"""
Data Preprocessing Module for Tourism Analytics Project
This module handles data loading, cleaning, and preprocessing
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class TourismDataProcessor:
    """Class to handle all data preprocessing tasks"""
    
    def __init__(self, data_path=None):
        """Initialize with data path"""
        # Resolve data path: prefer 'data/' subdir if present, otherwise use script directory
        if data_path is None:
            default_data_dir = Path(__file__).parent / 'data'
            if default_data_dir.exists():
                self.data_path = str(default_data_dir) + '/'
            else:
                self.data_path = str(Path(__file__).parent) + '/'
        else:
            self.data_path = data_path if str(data_path).endswith('/') else str(data_path) + '/'
        self.encoders = {}
        self.scaler = StandardScaler()
        
    def load_data(self):
        """Load all data files"""
        print("Loading data files...")
        
        # Load all datasets
        self.transaction = pd.read_excel(f'{self.data_path}Transaction.xlsx')
        self.user = pd.read_excel(f'{self.data_path}User.xlsx')
        self.city = pd.read_excel(f'{self.data_path}City.xlsx')
        self.continent = pd.read_excel(f'{self.data_path}Continent.xlsx')
        self.country = pd.read_excel(f'{self.data_path}Country.xlsx')
        self.region = pd.read_excel(f'{self.data_path}Region.xlsx')
        self.item = pd.read_excel(f'{self.data_path}Item.xlsx')
        self.visit_mode = pd.read_excel(f'{self.data_path}Mode.xlsx')
        self.attraction_type = pd.read_excel(f'{self.data_path}Type.xlsx')
        
        print("Data loaded successfully!")
        self._print_data_info()
        
    def _print_data_info(self):
        """Print information about loaded datasets"""
        print("\n" + "="*80)
        print("DATA OVERVIEW")
        print("="*80)
        datasets = {
            'Transaction': self.transaction,
            'User': self.user,
            'City': self.city,
            'Continent': self.continent,
            'Country': self.country,
            'Region': self.region,
            'Item': self.item,
            'VisitMode': self.visit_mode,
            'AttractionType': self.attraction_type
        }
        
        for name, df in datasets.items():
            print(f"\n{name}: {df.shape[0]} rows, {df.shape[1]} columns")
            
    def clean_data(self):
        """Clean all datasets"""
        print("\n" + "="*80)
        print("CLEANING DATA")
        print("="*80)
        
        # Clean City data - handle missing CityName
        missing_city = self.city['CityName'].isnull().sum()
        if missing_city > 0:
            print(f"Filling {missing_city} missing city names with 'Unknown'")
            self.city['CityName'].fillna('Unknown', inplace=True)
        
        # Clean User data - handle missing CityId
        missing_user_city = self.user['CityId'].isnull().sum()
        if missing_user_city > 0:
            print(f"Filling {missing_user_city} missing user CityId with 0 (Unknown)")
            self.user['CityId'].fillna(0, inplace=True)
            self.user['CityId'] = self.user['CityId'].astype(int)
        
        # Remove rows with ID = 0 or '-' (unknown/missing values) from lookup tables
        print("\nRemoving unknown/placeholder entries from lookup tables...")
        self.city = self.city[self.city['CityId'] != 0].reset_index(drop=True)
        self.continent = self.continent[self.continent['ContinentId'] != 0].reset_index(drop=True)
        self.country = self.country[self.country['CountryId'] != 0].reset_index(drop=True)
        self.region = self.region[self.region['RegionId'] != 0].reset_index(drop=True)
        self.visit_mode = self.visit_mode[self.visit_mode['VisitModeId'] != 0].reset_index(drop=True)
        
        print("Data cleaning completed!")
        
    def merge_data(self):
        """Merge all datasets to create consolidated dataframe"""
        print("\n" + "="*80)
        print("MERGING DATASETS")
        print("="*80)
        
        # Start with transaction data
        df = self.transaction.copy()
        print(f"Starting with Transaction data: {df.shape}")
        
        # Merge with User data
        df = df.merge(self.user, on='UserId', how='left')
        print(f"After merging User data: {df.shape}")
        
        # Merge with Item (Attraction) data
        df = df.merge(self.item, on='AttractionId', how='left')
        print(f"After merging Item data: {df.shape}")
        
        # Merge with City data for user
        df = df.merge(
            self.city.rename(columns={'CityId': 'CityId', 'CityName': 'UserCityName', 'CountryId': 'UserCountryId_temp'}),
            on='CityId',
            how='left'
        )
        df.drop('UserCountryId_temp', axis=1, inplace=True)
        print(f"After merging User City data: {df.shape}")
        
        # Merge with City data for attraction
        df = df.merge(
            self.city.rename(columns={'CityId': 'AttractionCityId', 'CityName': 'AttractionCityName', 'CountryId': 'AttractionCountryId_temp'}),
            on='AttractionCityId',
            how='left'
        )
        df.drop('AttractionCountryId_temp', axis=1, inplace=True)
        print(f"After merging Attraction City data: {df.shape}")
        
        # Merge with Country data for user
        df = df.merge(
            self.country.rename(columns={'CountryId': 'CountryId', 'Country': 'UserCountry', 'RegionId': 'UserRegionId_temp'}),
            on='CountryId',
            how='left'
        )
        df.drop('UserRegionId_temp', axis=1, inplace=True)
        print(f"After merging User Country data: {df.shape}")
        
        # Merge with Region data for user
        df = df.merge(
            self.region.rename(columns={'RegionId': 'RegionId', 'Region': 'UserRegion', 'ContinentId': 'UserContinentId_temp'}),
            on='RegionId',
            how='left'
        )
        df.drop('UserContinentId_temp', axis=1, inplace=True)
        print(f"After merging User Region data: {df.shape}")
        
        # Merge with Continent data
        df = df.merge(
            self.continent.rename(columns={'ContinentId': 'ContinentId', 'Continent': 'UserContinent'}),
            on='ContinentId',
            how='left'
        )
        print(f"After merging Continent data: {df.shape}")
        
        # Merge with VisitMode data
        df = df.merge(
            self.visit_mode.rename(columns={'VisitMode': 'VisitModeName'}),
            left_on='VisitMode',
            right_on='VisitModeId',
            how='left'
        )
        df.drop('VisitModeId', axis=1, inplace=True)
        print(f"After merging VisitMode data: {df.shape}")
        
        # Merge with AttractionType data
        df = df.merge(self.attraction_type, on='AttractionTypeId', how='left')
        print(f"After merging AttractionType data: {df.shape}")
        
        self.merged_df = df
        print(f"\nFinal merged dataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        return df
    
    def engineer_features(self, df):
        """Create new features for modeling"""
        print("\n" + "="*80)
        print("FEATURE ENGINEERING")
        print("="*80)
        
        df = df.copy()
        
        # User-level aggregated features
        print("Creating user-level features...")
        user_stats = df.groupby('UserId').agg({
            'Rating': ['mean', 'std', 'count'],
            'AttractionId': 'nunique',
            'VisitMode': lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0]
        }).reset_index()
        
        user_stats.columns = ['UserId', 'UserAvgRating', 'UserRatingStd', 'UserTotalVisits', 
                              'UserUniqueAttractions', 'UserMostFrequentMode']
        
        # Fill NaN in std with 0 (for users with only 1 visit)
        user_stats['UserRatingStd'].fillna(0, inplace=True)
        
        df = df.merge(user_stats, on='UserId', how='left')
        
        # Attraction-level aggregated features
        print("Creating attraction-level features...")
        attraction_stats = df.groupby('AttractionId').agg({
            'Rating': ['mean', 'std', 'count'],
            'UserId': 'nunique'
        }).reset_index()
        
        attraction_stats.columns = ['AttractionId', 'AttractionAvgRating', 'AttractionRatingStd',
                                   'AttractionTotalVisits', 'AttractionUniqueVisitors']
        
        attraction_stats['AttractionRatingStd'].fillna(0, inplace=True)
        
        df = df.merge(attraction_stats, on='AttractionId', how='left')
        
        # Temporal features
        print("Creating temporal features...")
        df['VisitQuarter'] = df['VisitMonth'].apply(lambda x: (x-1)//3 + 1)
        df['IsSummerMonth'] = df['VisitMonth'].apply(lambda x: 1 if x in [6, 7, 8] else 0)
        df['IsWinterMonth'] = df['VisitMonth'].apply(lambda x: 1 if x in [12, 1, 2] else 0)
        
        # Visit mode popularity by attraction type
        print("Creating visit mode - attraction type features...")
        mode_type_stats = df.groupby(['AttractionTypeId', 'VisitMode']).size().reset_index(name='ModeTypeCount')
        df = df.merge(mode_type_stats, on=['AttractionTypeId', 'VisitMode'], how='left')
        
        print(f"Feature engineering completed! New shape: {df.shape}")
        
        return df
    
    def prepare_for_modeling(self, df, target_column=None, encode_target=False):
        """Prepare data for modeling with encoding"""
        print("\n" + "="*80)
        print("PREPARING DATA FOR MODELING")
        print("="*80)
        
        df = df.copy()
        
        # Identify categorical columns to encode
        categorical_cols = [
            'UserCityName', 'UserCountry', 'UserRegion', 'UserContinent',
            'AttractionCityName', 'AttractionType', 'VisitModeName', 'Attraction'
        ]
        
        # Keep only columns that exist in the dataframe
        categorical_cols = [col for col in categorical_cols if col in df.columns]
        
        print(f"Encoding {len(categorical_cols)} categorical columns...")
        
        # Label encode categorical columns
        for col in categorical_cols:
            if col not in self.encoders:
                self.encoders[col] = LabelEncoder()
                df[f'{col}_encoded'] = self.encoders[col].fit_transform(df[col].astype(str))
            else:
                df[f'{col}_encoded'] = self.encoders[col].transform(df[col].astype(str))
        
        # Encode target if specified
        if target_column and encode_target:
            if target_column not in self.encoders:
                self.encoders[target_column] = LabelEncoder()
                df[f'{target_column}_encoded'] = self.encoders[target_column].fit_transform(df[target_column])
            else:
                df[f'{target_column}_encoded'] = self.encoders[target_column].transform(df[target_column])
        
        print("Encoding completed!")
        
        return df
    
    def get_model_features(self, df, task='regression'):
        """Get appropriate features for different modeling tasks"""
        
        # Common features
        base_features = [
            'UserId', 'AttractionId', 'VisitYear', 'VisitMonth', 'VisitQuarter',
            'ContinentId', 'RegionId', 'CountryId', 'CityId',
            'AttractionCityId', 'AttractionTypeId',
            'UserAvgRating', 'UserRatingStd', 'UserTotalVisits', 'UserUniqueAttractions',
            'AttractionAvgRating', 'AttractionRatingStd', 'AttractionTotalVisits', 
            'AttractionUniqueVisitors',
            'IsSummerMonth', 'IsWinterMonth', 'ModeTypeCount'
        ]
        
        # Add encoded categorical features
        encoded_features = [col for col in df.columns if col.endswith('_encoded')]
        
        all_features = base_features + encoded_features
        
        # Filter only existing columns
        available_features = [f for f in all_features if f in df.columns]
        
        if task == 'classification':
            # Remove VisitMode from features as it's the target
            available_features = [f for f in available_features if 'VisitMode' not in f]
        
        return available_features
    
    def save_processed_data(self, df, filename='processed_data.csv'):
        """Save processed data"""
        filepath = f'{self.data_path}{filename}'
        df.to_csv(filepath, index=False)
        print(f"\nProcessed data saved to {filepath}")


def main():
    """Main function to run preprocessing"""
    print("="*80)
    print("TOURISM DATA PREPROCESSING PIPELINE")
    print("="*80)
    
    # Initialize processor
    processor = TourismDataProcessor()
    
    # Load data
    processor.load_data()
    
    # Clean data
    processor.clean_data()
    
    # Merge data
    merged_df = processor.merge_data()
    
    # Engineer features
    featured_df = processor.engineer_features(merged_df)
    
    # Prepare for modeling
    final_df = processor.prepare_for_modeling(featured_df)
    
    # Save processed data
    processor.save_processed_data(final_df)
    
    print("\n" + "="*80)
    print("PREPROCESSING COMPLETED SUCCESSFULLY!")
    print("="*80)
    
    return processor, final_df


if __name__ == "__main__":
    processor, df = main()
