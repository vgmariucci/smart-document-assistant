# src/models/classical_ml.py
import pickle
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from src.utils.config import MODEL_DIR, MODEL_PARAMS
from src.utils.logger import setup_logger

logger = setup_logger("classical_ml")

class SalesPredictor:
    """Class to train and predict sales using classical ML models"""
    
    def __init__(self, model_type='random_forest'):
        """Initialize the model"""
        logger.info(f"Initializing {model_type} model")
        self.model_type = model_type
        self.model = None
        self.preprocessor = None
        
        # Define model parameters from config
        self.model_params = MODEL_PARAMS.get(model_type, {})
    
    def _prepare_features(self, df):
        """
        Prepare features for modeling:
        - Select relevant features
        - Handle categorical variables
        """
        logger.info("Preparing features for modeling")
        
        # Define features and target
        numeric_features = ['unit_price', 'quantity', 'warranty_years']
        categorical_features = ['product_category', 'product_brand', 'customer_type', 'payment_method']
        date_features = ['purchase_month', 'purchase_dayofweek', 'purchase_quarter']
        
        # Combine all features
        all_features = numeric_features + categorical_features + date_features
        
        return all_features
    
    def train(self, df, target_column='total_price', test_size=0.2, random_state=42):
        """Train the model using the provided DataFrame"""
        logger.info(f"Training {self.model_type} model to predict {target_column}")
        
        # Prepare features
        features = self._prepare_features(df)
        X = df[features]
        y = df[target_column]
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        logger.info(f"Train set size: {X_train.shape}, Test set size: {X_test.shape}")
        
        # Define preprocessing for numeric and categorical features
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        numeric_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])
        
        # Create and train the pipeline
        if self.model_type == 'random_forest':
            model = RandomForestRegressor(**self.model_params)
        elif self.model_type == 'gradient_boosting':
            model = GradientBoostingRegressor(**self.model_params)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        # Create pipeline
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        # Fit model
        logger.info("Fitting model...")
        pipeline.fit(X_train, y_train)
        
        # Evaluate
        logger.info("Evaluating model...")
        y_pred = pipeline.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        logger.info(f"Model evaluation metrics:")
        logger.info(f"RMSE: {rmse:.2f}")
        logger.info(f"MAE: {mae:.2f}")
        logger.info(f"R²: {r2:.2f}")
        
        # Save model and preprocessor
        self.model = pipeline
        self.preprocessor = preprocessor
        
        return {
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
    
    def save_model(self, filename=None):
        """Save the trained model to disk"""
        if self.model is None:
            logger.error("No trained model to save")
            return False
        
        if filename is None:
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.model_type}_{timestamp}.pkl"
        
        model_path = os.path.join(MODEL_DIR, filename)
        logger.info(f"Saving model to {model_path}")
        
        try:
            with open(model_path, 'wb') as f:
                pickle.dump(self.model, f)
            logger.info("Model saved successfully")
            return True
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return False
    
    def load_model(self, filename):
        """Load a trained model from disk"""
        model_path = os.path.join(MODEL_DIR, filename)
        logger.info(f"Loading model from {model_path}")
        
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            logger.info("Model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def predict(self, df):
        """Make predictions using the trained model"""
        if self.model is None:
            logger.error("No trained model available for prediction")
            return None
        
        logger.info(f"Making predictions on {len(df)} records")
        
        # Prepare features
        features = self._prepare_features(df)
        X = df[features]
        
        # Make predictions
        try:
            predictions = self.model.predict(X)
            logger.info("Predictions completed successfully")
            return predictions
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            return None