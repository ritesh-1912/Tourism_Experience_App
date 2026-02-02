"""
Machine Learning Models Module for Tourism Analytics Project
Includes Regression, Classification, and Recommendation Systems
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import (mean_squared_error, r2_score, mean_absolute_error,
                            accuracy_score, precision_score, recall_score, f1_score, 
                            classification_report, confusion_matrix)
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')


class RatingPredictor:
    """Class for predicting attraction ratings (Regression Task)"""
    
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def prepare_data(self, df, features, target='Rating', test_size=0.2, random_state=42):
        """Prepare data for regression"""
        print("Preparing data for Rating Prediction (Regression)...")
        
        # Select features that exist in dataframe
        available_features = [f for f in features if f in df.columns]
        self.feature_names = available_features
        
        X = df[available_features].copy()
        y = df[target].copy()
        
        # Handle any missing values
        X.fillna(X.mean(), inplace=True)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=available_features, index=X_train.index)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=available_features, index=X_test.index)
        
        print(f"Training set: {X_train_scaled.shape}, Test set: {X_test_scaled.shape}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_models(self, X_train, y_train):
        """Train multiple regression models"""
        print("\n" + "="*80)
        print("TRAINING REGRESSION MODELS")
        print("="*80)
        
        # Define models
        self.models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0, random_state=42),
            'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42),
            'Decision Tree': DecisionTreeRegressor(max_depth=10, random_state=42)
        }
        
        # Train and evaluate each model
        results = {}
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            model.fit(X_train, y_train)
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, 
                                       scoring='neg_mean_squared_error', n_jobs=-1)
            cv_rmse = np.sqrt(-cv_scores.mean())
            
            results[name] = {
                'model': model,
                'cv_rmse': cv_rmse
            }
            
            print(f"{name} - CV RMSE: {cv_rmse:.4f}")
        
        # Select best model based on CV score
        self.best_model_name = min(results, key=lambda x: results[x]['cv_rmse'])
        self.best_model = results[self.best_model_name]['model']
        
        print(f"\n{'='*80}")
        print(f"Best Model: {self.best_model_name}")
        print(f"{'='*80}")
        
        return results
    
    def evaluate(self, X_test, y_test):
        """Evaluate the best model"""
        print("\n" + "="*80)
        print("MODEL EVALUATION - REGRESSION")
        print("="*80)
        
        y_pred = self.best_model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"\nBest Model: {self.best_model_name}")
        print(f"{'='*80}")
        print(f"R² Score:  {r2:.4f}")
        print(f"RMSE:      {rmse:.4f}")
        print(f"MAE:       {mae:.4f}")
        print(f"MSE:       {mse:.4f}")
        
        return {
            'model_name': self.best_model_name,
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'mse': mse,
            'predictions': y_pred,
            'actuals': y_test
        }
    
    def get_feature_importance(self, top_n=15):
        """Get feature importance for tree-based models"""
        if hasattr(self.best_model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.best_model.feature_importances_
            }).sort_values('importance', ascending=False).head(top_n)
            
            return importance_df
        else:
            print("Feature importance not available for this model.")
            return None
    
    def save_model(self, filepath='models/rating_predictor.pkl'):
        """Save the trained model"""
        joblib.dump({
            'model': self.best_model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'model_name': self.best_model_name
        }, filepath)
        print(f"\nModel saved to {filepath}")


class VisitModeClassifier:
    """Class for predicting visit mode (Classification Task)"""
    
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def prepare_data(self, df, features, target='VisitMode', test_size=0.2, random_state=42):
        """Prepare data for classification"""
        print("Preparing data for Visit Mode Classification...")
        
        # Select features that exist in dataframe
        available_features = [f for f in features if f in df.columns]
        self.feature_names = available_features
        
        X = df[available_features].copy()
        y = df[target].copy()
        
        # Handle any missing values
        X.fillna(X.mean(), inplace=True)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=available_features, index=X_train.index)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=available_features, index=X_test.index)
        
        print(f"Training set: {X_train_scaled.shape}, Test set: {X_test_scaled.shape}")
        print(f"Class distribution in training set:\n{y_train.value_counts()}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_models(self, X_train, y_train):
        """Train multiple classification models"""
        print("\n" + "="*80)
        print("TRAINING CLASSIFICATION MODELS")
        print("="*80)
        
        # Define models
        self.models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1),
            'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
        }
        
        # Train and evaluate each model
        results = {}
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            model.fit(X_train, y_train)
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, 
                                       scoring='accuracy', n_jobs=-1)
            cv_accuracy = cv_scores.mean()
            
            results[name] = {
                'model': model,
                'cv_accuracy': cv_accuracy
            }
            
            print(f"{name} - CV Accuracy: {cv_accuracy:.4f}")
        
        # Select best model based on CV score
        self.best_model_name = max(results, key=lambda x: results[x]['cv_accuracy'])
        self.best_model = results[self.best_model_name]['model']
        
        print(f"\n{'='*80}")
        print(f"Best Model: {self.best_model_name}")
        print(f"{'='*80}")
        
        return results
    
    def evaluate(self, X_test, y_test):
        """Evaluate the best model"""
        print("\n" + "="*80)
        print("MODEL EVALUATION - CLASSIFICATION")
        print("="*80)
        
        y_pred = self.best_model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        print(f"\nBest Model: {self.best_model_name}")
        print(f"{'='*80}")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        
        print(f"\n{'='*80}")
        print("Classification Report:")
        print(f"{'='*80}")
        print(classification_report(y_test, y_pred))
        
        return {
            'model_name': self.best_model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'predictions': y_pred,
            'actuals': y_test,
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
    
    def get_feature_importance(self, top_n=15):
        """Get feature importance for tree-based models"""
        if hasattr(self.best_model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.best_model.feature_importances_
            }).sort_values('importance', ascending=False).head(top_n)
            
            return importance_df
        else:
            print("Feature importance not available for this model.")
            return None
    
    def save_model(self, filepath='models/visitmode_classifier.pkl'):
        """Save the trained model"""
        joblib.dump({
            'model': self.best_model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'model_name': self.best_model_name
        }, filepath)
        print(f"\nModel saved to {filepath}")


class AttractionRecommender:
    """Class for recommending attractions (Recommendation System)"""
    
    def __init__(self):
        self.user_item_matrix = None
        self.similarity_matrix = None
        self.attraction_features = None
        
    def prepare_data(self, df):
        """Prepare data for recommendation system"""
        print("\n" + "="*80)
        print("PREPARING RECOMMENDATION SYSTEM")
        print("="*80)
        
        # Create user-item rating matrix
        self.user_item_matrix = df.pivot_table(
            index='UserId',
            columns='AttractionId',
            values='Rating',
            fill_value=0
        )
        
        print(f"User-Item Matrix shape: {self.user_item_matrix.shape}")
        print(f"Users: {self.user_item_matrix.shape[0]}, Attractions: {self.user_item_matrix.shape[1]}")
        
        # Store attraction features for content-based filtering
        self.attraction_features = df[['AttractionId', 'Attraction', 'AttractionType', 
                                       'AttractionCityName', 'AttractionAvgRating']].drop_duplicates()
        
        return self.user_item_matrix
    
    def build_collaborative_filtering(self):
        """Build collaborative filtering model using user-user similarity"""
        print("\nBuilding Collaborative Filtering Model...")
        
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Calculate user-user similarity
        self.user_similarity = cosine_similarity(self.user_item_matrix)
        self.user_similarity_df = pd.DataFrame(
            self.user_similarity,
            index=self.user_item_matrix.index,
            columns=self.user_item_matrix.index
        )
        
        print("Collaborative filtering model built successfully!")
        
    def recommend_collaborative(self, user_id, top_n=10):
        """Recommend attractions using collaborative filtering"""
        
        if user_id not in self.user_item_matrix.index:
            print(f"User {user_id} not found in the system.")
            return None
        
        # Get similar users
        similar_users = self.user_similarity_df[user_id].sort_values(ascending=False)[1:11]
        
        # Get attractions rated by similar users but not by target user
        user_rated = self.user_item_matrix.loc[user_id]
        user_unrated = user_rated[user_rated == 0].index
        
        # Calculate weighted ratings for unrated attractions
        recommendations = {}
        for attraction in user_unrated:
            weighted_sum = 0
            similarity_sum = 0
            
            for similar_user, similarity in similar_users.items():
                if self.user_item_matrix.loc[similar_user, attraction] > 0:
                    weighted_sum += similarity * self.user_item_matrix.loc[similar_user, attraction]
                    similarity_sum += similarity
            
            if similarity_sum > 0:
                recommendations[attraction] = weighted_sum / similarity_sum
        
        # Sort and return top N
        sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        # Get attraction details
        recommended_attractions = []
        for attraction_id, predicted_rating in sorted_recommendations:
            attraction_info = self.attraction_features[
                self.attraction_features['AttractionId'] == attraction_id
            ].iloc[0]
            
            recommended_attractions.append({
                'AttractionId': attraction_id,
                'Attraction': attraction_info['Attraction'],
                'Type': attraction_info['AttractionType'],
                'City': attraction_info['AttractionCityName'],
                'AvgRating': attraction_info['AttractionAvgRating'],
                'PredictedRating': predicted_rating
            })
        
        return pd.DataFrame(recommended_attractions)
    
    def recommend_content_based(self, user_id, top_n=10):
        """Recommend attractions using content-based filtering"""
        
        if user_id not in self.user_item_matrix.index:
            print(f"User {user_id} not found in the system.")
            return None
        
        # Get user's past ratings
        user_ratings = self.user_item_matrix.loc[user_id]
        user_rated_attractions = user_ratings[user_ratings > 0].index.tolist()
        
        if len(user_rated_attractions) == 0:
            print(f"User {user_id} has no ratings. Cannot provide content-based recommendations.")
            return None
        
        # Get preferred attraction types
        user_preferences = self.attraction_features[
            self.attraction_features['AttractionId'].isin(user_rated_attractions)
        ]
        
        preferred_types = user_preferences['AttractionType'].value_counts().head(3).index.tolist()
        
        # Find similar attractions not yet visited
        unrated_attractions = self.user_item_matrix.loc[user_id][
            self.user_item_matrix.loc[user_id] == 0
        ].index.tolist()
        
        similar_attractions = self.attraction_features[
            (self.attraction_features['AttractionId'].isin(unrated_attractions)) &
            (self.attraction_features['AttractionType'].isin(preferred_types))
        ].sort_values('AttractionAvgRating', ascending=False).head(top_n)
        
        return similar_attractions[['AttractionId', 'Attraction', 'AttractionType', 
                                   'AttractionCityName', 'AttractionAvgRating']]
    
    def recommend_hybrid(self, user_id, top_n=10):
        """Hybrid recommendation combining collaborative and content-based"""
        
        collaborative_recs = self.recommend_collaborative(user_id, top_n=top_n)
        content_recs = self.recommend_content_based(user_id, top_n=top_n)
        
        if collaborative_recs is None and content_recs is None:
            return None
        elif collaborative_recs is None:
            return content_recs
        elif content_recs is None:
            return collaborative_recs
        
        # Combine both recommendations
        all_recommendations = pd.concat([
            collaborative_recs[['AttractionId', 'Attraction', 'Type', 'City', 'AvgRating']],
            content_recs.rename(columns={'AttractionType': 'Type', 'AttractionCityName': 'City'})
        ]).drop_duplicates(subset='AttractionId').head(top_n)
        
        return all_recommendations
    
    def save_model(self, filepath='models/recommender.pkl'):
        """Save the recommendation model"""
        joblib.dump({
            'user_item_matrix': self.user_item_matrix,
            'user_similarity': self.user_similarity_df,
            'attraction_features': self.attraction_features
        }, filepath)
        print(f"\nRecommendation model saved to {filepath}")


def main():
    """Main function to train all models"""
    print("="*80)
    print("TOURISM ANALYTICS - MODEL TRAINING PIPELINE")
    print("="*80)
    
    # This would be called after preprocessing
    # Load processed data
    # df = pd.read_csv('data/processed_data.csv')
    
    print("\nThis module contains all model training functions.")
    print("Import and use the classes as needed:")
    print("  - RatingPredictor: For rating prediction (regression)")
    print("  - VisitModeClassifier: For visit mode prediction (classification)")
    print("  - AttractionRecommender: For attraction recommendations")


if __name__ == "__main__":
    main()
