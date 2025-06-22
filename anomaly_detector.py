# anomaly_detector.py

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import DBSCAN
import joblib
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class ProcurementAnomalyDetector:
    def __init__(self):
        self.price_scaler = RobustScaler()
        self.feature_scaler = StandardScaler()
        self.price_model = IsolationForest(
            contamination=0.1,
            random_state=42,
            n_estimators=200,
            max_samples='auto'
        )
        self.risk_thresholds = {
            'price_percentile': 95,
            'concentration_threshold': 0.5,
            'splitting_contracts': 5,
            'competition_ratio': 0.2
        }
        
    def load_and_validate(self, filepath: str) -> pd.DataFrame:
        """Load CSV and validate data types"""
        df = pd.read_csv(filepath)
        
        # Convert data types
        df['lot_id'] = df['lot_id'].astype(str)
        df['provider_bin'] = pd.to_numeric(df['provider_bin'], errors='coerce')
        df['customer_bin'] = df['customer_bin'].astype(str)
        df['contract_sum'] = pd.to_numeric(df['contract_sum'], errors='coerce').fillna(0)
        df['paid_sum'] = pd.to_numeric(df['paid_sum'], errors='coerce').fillna(0)
        df['accept_date'] = pd.to_datetime(df['accept_date'], errors='coerce')
        df['order_method_id'] = pd.to_numeric(df['order_method_id'], errors='coerce')
        
        # Remove invalid rows
        df = df.dropna(subset=['accept_date', 'provider_bin', 'customer_bin'])
        df = df[df['contract_sum'] > 0]
        
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Feature engineering for anomaly detection"""
        df = df.copy()
        
        # Time features
        df['year'] = df['accept_date'].dt.year
        df['month'] = df['accept_date'].dt.month
        df['quarter'] = df['accept_date'].dt.quarter
        df['day_of_week'] = df['accept_date'].dt.dayofweek
        
        # Price features
        df['payment_ratio'] = df['paid_sum'] / (df['contract_sum'] + 1e-6)
        df['log_sum'] = np.log1p(df['contract_sum'])
        df['price_category'] = pd.qcut(df['contract_sum'], q=10, labels=False, duplicates='drop')
        
        # Provider analytics
        provider_agg = df.groupby('provider_bin').agg({
            'lot_id': 'count',
            'contract_sum': ['sum', 'mean', 'std', 'median'],
            'customer_bin': 'nunique',
            'order_method_id': lambda x: x.mode()[0] if len(x) > 0 else 0
        }).reset_index()
        provider_agg.columns = ['provider_bin', 'provider_contracts', 'provider_total_value',
                               'provider_avg_value', 'provider_std_value', 'provider_median_value',
                               'provider_unique_customers', 'provider_common_method']
        
        # Customer analytics
        customer_agg = df.groupby('customer_bin').agg({
            'lot_id': 'count',
            'contract_sum': ['sum', 'mean', 'median'],
            'provider_bin': 'nunique',
            'order_method_id': 'nunique'
        }).reset_index()
        customer_agg.columns = ['customer_bin', 'customer_contracts', 'customer_total_value',
                               'customer_avg_value', 'customer_median_value',
                               'customer_unique_providers', 'customer_unique_methods']
        
        # Customer-Provider relationships
        cp_agg = df.groupby(['customer_bin', 'provider_bin']).agg({
            'lot_id': 'count',
            'contract_sum': ['sum', 'mean'],
            'accept_date': ['min', 'max']
        }).reset_index()
        cp_agg.columns = ['customer_bin', 'provider_bin', 'cp_contracts', 'cp_total_value',
                         'cp_avg_value', 'cp_first_date', 'cp_last_date']
        cp_agg['cp_duration_days'] = (cp_agg['cp_last_date'] - cp_agg['cp_first_date']).dt.days
        
        # Method analytics
        method_agg = df.groupby('order_method_id').agg({
            'lot_id': 'count',
            'provider_bin': 'nunique',
            'contract_sum': ['mean', 'std']
        }).reset_index()
        method_agg.columns = ['order_method_id', 'method_contracts', 'method_unique_providers',
                             'method_avg_value', 'method_std_value']
        method_agg['method_competition_ratio'] = method_agg['method_unique_providers'] / method_agg['method_contracts']
        
        # Merge all features
        df = df.merge(provider_agg, on='provider_bin', how='left')
        df = df.merge(customer_agg, on='customer_bin', how='left')
        df = df.merge(cp_agg, on=['customer_bin', 'provider_bin'], how='left')
        df = df.merge(method_agg, on='order_method_id', how='left')
        
        # Calculate concentration metrics
        df['provider_concentration'] = df['cp_contracts'] / df['customer_contracts']
        df['value_concentration'] = df['cp_total_value'] / df['customer_total_value']
        df['provider_dependency'] = df['cp_contracts'] / df['provider_contracts']
        df['customer_loyalty'] = df['cp_contracts'] / df['provider_contracts']
        
        # Market share
        total_market_value = df['contract_sum'].sum()
        provider_market_share = df.groupby('provider_bin')['contract_sum'].sum() / total_market_value
        df['provider_market_share'] = df['provider_bin'].map(provider_market_share)
        
        return df
    
    def detect_price_anomalies(self, df: pd.DataFrame) -> np.ndarray:
        """Advanced price anomaly detection"""
        # Group by method and calculate statistics
        method_price_stats = df.groupby('order_method_id')['contract_sum'].agg([
            'mean', 'std', 'median',
            lambda x: np.percentile(x, 25),
            lambda x: np.percentile(x, 75)
        ]).reset_index()
        method_price_stats.columns = ['order_method_id', 'mean', 'std', 'median', 'q1', 'q3']
        method_price_stats['iqr'] = method_price_stats['q3'] - method_price_stats['q1']
        
        # Statistical outlier detection
        price_scores = np.zeros(len(df))
        
        for idx, row in df.iterrows():
            method_stats = method_price_stats[method_price_stats['order_method_id'] == row['order_method_id']]
            if len(method_stats) > 0:
                stats = method_stats.iloc[0]
                
                # Modified Z-score
                if stats['std'] > 0:
                    z_score = abs(row['contract_sum'] - stats['median']) / stats['std']
                    price_scores[idx] = min(z_score / 3, 1.0)
                
                # IQR method
                upper_bound = stats['q3'] + 1.5 * stats['iqr']
                if row['contract_sum'] > upper_bound:
                    price_scores[idx] = max(price_scores[idx], 0.7)
        
        # ML-based anomaly detection
        features = ['log_sum', 'payment_ratio', 'provider_avg_value', 
                   'customer_avg_value', 'method_avg_value', 'price_category']
        
        X = df[features].fillna(0)
        X_scaled = self.price_scaler.fit_transform(X)
        
        ml_scores = self.price_model.fit_predict(X_scaled)
        ml_proba = self.price_model.score_samples(X_scaled)
        ml_risk = 1 - (ml_proba - ml_proba.min()) / (ml_proba.max() - ml_proba.min() + 1e-6)
        
        # Combine statistical and ML approaches
        combined_scores = 0.5 * price_scores + 0.5 * ml_risk
        
        # Additional checks for extreme cases
        extreme_high = df['contract_sum'] > df['contract_sum'].quantile(0.99)
        combined_scores[extreme_high] = np.maximum(combined_scores[extreme_high], 0.8)
        
        return combined_scores
    
    def detect_supplier_concentration(self, df: pd.DataFrame) -> np.ndarray:
        """Detect suspicious supplier-customer relationships"""
        concentration_scores = np.zeros(len(df))
        
        # Provider concentration with customer
        high_concentration = df['provider_concentration'] > self.risk_thresholds['concentration_threshold']
        concentration_scores[high_concentration] = df.loc[high_concentration, 'provider_concentration']
        
        # Single customer dependency
        single_customer_dependency = (
            (df['provider_unique_customers'] <= 2) & 
            (df['provider_contracts'] > 10)
        )
        concentration_scores[single_customer_dependency] = np.maximum(
            concentration_scores[single_customer_dependency], 0.85
        )
        
        # Value concentration
        high_value_concentration = df['value_concentration'] > 0.7
        concentration_scores[high_value_concentration] = np.maximum(
            concentration_scores[high_value_concentration],
            df.loc[high_value_concentration, 'value_concentration']
        )
        
        # Long-term exclusive relationships
        exclusive_long_term = (
            (df['cp_duration_days'] > 365) & 
            (df['provider_concentration'] > 0.8)
        )
        concentration_scores[exclusive_long_term] = 0.9
        
        # Market dominance
        market_dominant = df['provider_market_share'] > 0.2
        concentration_scores[market_dominant] = np.maximum(
            concentration_scores[market_dominant], 0.6
        )
        
        return concentration_scores
    
    def detect_contract_splitting(self, df: pd.DataFrame) -> np.ndarray:
        """Detect artificial contract splitting patterns"""
        splitting_scores = np.zeros(len(df))
        
        # Sort by date for temporal analysis
        df_sorted = df.sort_values(['customer_bin', 'provider_bin', 'accept_date']).copy()
        df_sorted['prev_date'] = df_sorted.groupby(['customer_bin', 'provider_bin'])['accept_date'].shift(1)
        df_sorted['days_since_last'] = (df_sorted['accept_date'] - df_sorted['prev_date']).dt.days
        
        for idx, row in df_sorted.iterrows():
            # Find contracts within 30-day window
            window_mask = (
                (df_sorted['customer_bin'] == row['customer_bin']) &
                (df_sorted['provider_bin'] == row['provider_bin']) &
                (df_sorted['accept_date'] >= row['accept_date'] - timedelta(days=30)) &
                (df_sorted['accept_date'] <= row['accept_date'] + timedelta(days=30))
            )
            
            window_contracts = df_sorted[window_mask]
            
            if len(window_contracts) >= self.risk_thresholds['splitting_contracts']:
                # Check for similar amounts
                amounts = window_contracts['contract_sum'].values
                cv = np.std(amounts) / (np.mean(amounts) + 1e-6)
                
                if cv < 0.2:  # Very similar amounts
                    splitting_scores[idx] = min(0.95, len(window_contracts) / 8)
                elif cv < 0.4:
                    splitting_scores[idx] = min(0.7, len(window_contracts) / 12)
                
                # Check for round numbers
                round_numbers = window_contracts['contract_sum'].apply(
                    lambda x: x % 1000 == 0 or x % 10000 == 0
                ).sum()
                if round_numbers / len(window_contracts) > 0.8:
                    splitting_scores[idx] = max(splitting_scores[idx], 0.7)
            
            # Sequential small contracts
            if row['days_since_last'] is not pd.NaT and row['days_since_last'] < 7:
                recent_contracts = df_sorted[
                    (df_sorted['customer_bin'] == row['customer_bin']) &
                    (df_sorted['provider_bin'] == row['provider_bin']) &
                    (df_sorted['accept_date'] >= row['accept_date'] - timedelta(days=30))
                ]
                if len(recent_contracts) > 5:
                    splitting_scores[idx] = max(splitting_scores[idx], 0.6)
        
        return splitting_scores
    
    def detect_low_competition(self, df: pd.DataFrame) -> np.ndarray:
        """Detect low competition patterns"""
        competition_scores = np.zeros(len(df))
        
        # Method-based competition
        low_competition_methods = df['method_competition_ratio'] < self.risk_thresholds['competition_ratio']
        competition_scores[low_competition_methods] = 0.7
        
        # Single source methods (typically method_id 6)
        single_source_methods = df['order_method_id'].isin([6, 7])  # Adjust based on actual method IDs
        competition_scores[single_source_methods] = np.maximum(
            competition_scores[single_source_methods], 0.5
        )
        
        # Customer with limited providers
        limited_providers = (
            (df['customer_unique_providers'] <= 3) & 
            (df['customer_contracts'] > 20)
        )
        competition_scores[limited_providers] = np.maximum(
            competition_scores[limited_providers], 0.8
        )
        
        # Provider winning rate
        df['provider_win_rate'] = df['provider_contracts'] / df['method_contracts']
        high_win_rate = df['provider_win_rate'] > 0.5
        competition_scores[high_win_rate] = np.maximum(
            competition_scores[high_win_rate], 0.6
        )
        
        # Closed market indicators
        closed_market = (
            (df['method_unique_providers'] < 5) & 
            (df['method_contracts'] > 50)
        )
        competition_scores[closed_market] = 0.9
        
        return competition_scores
    
    def calculate_risk_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive risk scores"""
        # Feature engineering
        df_features = self.engineer_features(df)
        
        # Calculate individual risk components
        price_risk = self.detect_price_anomalies(df_features)
        concentration_risk = self.detect_supplier_concentration(df_features)
        splitting_risk = self.detect_contract_splitting(df_features)
        competition_risk = self.detect_low_competition(df_features)
        
        # Weighted risk calculation
        weights = {
            'price': 0.25,
            'concentration': 0.30,
            'splitting': 0.20,
            'competition': 0.25
        }
        
        overall_risk = (
            weights['price'] * price_risk +
            weights['concentration'] * concentration_risk +
            weights['splitting'] * splitting_risk +
            weights['competition'] * competition_risk
        )
        
        # Risk categorization
        risk_levels = pd.cut(
            overall_risk,
            bins=[-0.01, 0.30, 0.65, 1.01],
            labels=['GREEN', 'YELLOW', 'RED']
        )
        
        # Prepare output
        results = pd.DataFrame({
            'lot_id': df_features['lot_id'],
            'provider_bin': df_features['provider_bin'],
            'customer_bin': df_features['customer_bin'],
            'contract_sum': df_features['contract_sum'],
            'paid_sum': df_features['paid_sum'],
            'accept_date': df_features['accept_date'],
            'order_method_id': df_features['order_method_id'],
            'price_anomaly': np.round(price_risk, 3),
            'supplier_concentration': np.round(concentration_risk, 3),
            'contract_splitting': np.round(splitting_risk, 3),
            'low_competition': np.round(competition_risk, 3),
            'risk_score': np.round(overall_risk, 3),
            'risk_level': risk_levels
        })
        
        return results





