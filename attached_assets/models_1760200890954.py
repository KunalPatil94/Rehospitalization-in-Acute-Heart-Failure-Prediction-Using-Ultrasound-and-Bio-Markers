import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.metrics import precision_recall_fscore_support
import xgboost as xgb
import pickle
import os
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

class AHFPredictionModels:
    """Machine Learning models for AHF rehospitalization prediction."""
    
    def __init__(self):
        """Initialize the prediction models."""
        self.logistic_model = None
        self.xgboost_model = None
        self.scaler = StandardScaler()
        self.feature_names = [
            'age', 'gender', 'weight', 'nt_probnp', 'creatinine',
            'b_line_score', 'ivc_collapsibility', 'ejection_fraction',
            'systolic_bp', 'heart_rate', 'diabetes', 'hypertension', 'ckd', 'afib'
        ]
        self.performance_metrics = {}
        self.trained = False
        
        # Try to load pre-trained models
        self.load_models()
    
    def prepare_features(self, data):
        """Prepare features for model training/prediction."""
        if isinstance(data, dict):
            # Single prediction
            features = np.array([[data[feature] for feature in self.feature_names]])
        else:
            # DataFrame for training
            features = data[self.feature_names].values
        
        return features
    
    def train_models(self, training_data):
        """Train both logistic regression and XGBoost models."""
        print("Training AHF prediction models...")
        
        # Prepare features and target
        X = self.prepare_features(training_data)
        y = training_data['readmission_30d'].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Apply SMOTE to handle class imbalance
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        
        # Scale features for logistic regression
        X_train_scaled = self.scaler.fit_transform(X_train_balanced)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Logistic Regression
        print("Training Logistic Regression...")
        self.logistic_model = LogisticRegression(
            random_state=42,
            max_iter=1000,
            class_weight='balanced'
        )
        self.logistic_model.fit(X_train_scaled, y_train_balanced)
        
        # Train XGBoost
        print("Training XGBoost...")
        self.xgboost_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss'
        )
        self.xgboost_model.fit(X_train_balanced, y_train_balanced)
        
        # Evaluate models
        self.evaluate_models(X_test_scaled, X_test, y_test)
        
        # Save models
        self.save_models()
        
        self.trained = True
        print("Model training completed!")
    
    def evaluate_models(self, X_test_scaled, X_test_raw, y_test):
        """Evaluate both models and store performance metrics."""
        # Logistic Regression predictions
        lr_pred = self.logistic_model.predict(X_test_scaled)
        lr_pred_proba = self.logistic_model.predict_proba(X_test_scaled)[:, 1]
        
        # XGBoost predictions
        xgb_pred = self.xgboost_model.predict(X_test_raw)
        xgb_pred_proba = self.xgboost_model.predict_proba(X_test_raw)[:, 1]
        
        # Calculate metrics for Logistic Regression
        lr_metrics = self.calculate_metrics(y_test, lr_pred, lr_pred_proba)
        
        # Calculate metrics for XGBoost
        xgb_metrics = self.calculate_metrics(y_test, xgb_pred, xgb_pred_proba)
        
        self.performance_metrics = {
            'logistic_regression': lr_metrics,
            'xgboost': xgb_metrics
        }
    
    def calculate_metrics(self, y_true, y_pred, y_pred_proba):
        """Calculate comprehensive performance metrics."""
        accuracy = accuracy_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_pred_proba)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Calculate sensitivity and specificity
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Precision, recall, f1
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary'
        )
        
        return {
            'accuracy': accuracy,
            'auc': auc,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'true_labels': y_true
        }
    
    def predict_risk(self, patient_data):
        """Predict 30-day readmission risk for a patient."""
        if not self.trained:
            raise ValueError("Models must be trained before making predictions")
        
        features = self.prepare_features(patient_data)
        features_scaled = self.scaler.transform(features)
        
        # Logistic Regression prediction
        lr_prob = self.logistic_model.predict_proba(features_scaled)[0, 1]
        lr_risk = "High Risk" if lr_prob > 0.5 else "Low Risk"
        
        # XGBoost prediction
        xgb_prob = self.xgboost_model.predict_proba(features)[0, 1]
        xgb_risk = "High Risk" if xgb_prob > 0.5 else "Low Risk"
        
        return {
            'logistic_regression': {
                'probability': lr_prob,
                'risk_level': lr_risk
            },
            'xgboost': {
                'probability': xgb_prob,
                'risk_level': xgb_risk
            }
        }
    
    def get_feature_importance(self):
        """Get feature importance from XGBoost model."""
        if self.xgboost_model is None:
            return None
        
        importance_scores = self.xgboost_model.feature_importances_
        importance_dict = dict(zip(self.feature_names, importance_scores))
        
        # Sort by importance
        sorted_importance = dict(sorted(importance_dict.items(), 
                                      key=lambda x: x[1], reverse=True))
        
        return sorted_importance
    
    def get_performance_metrics(self):
        """Get stored performance metrics."""
        return self.performance_metrics
    
    def models_trained(self):
        """Check if models are trained."""
        return self.trained and self.logistic_model is not None and self.xgboost_model is not None
    
    def save_models(self):
        """Save trained models to disk."""
        try:
            # Save logistic regression model
            with open('logistic_model.pkl', 'wb') as f:
                pickle.dump(self.logistic_model, f)
            
            # Save XGBoost model
            with open('xgboost_model.pkl', 'wb') as f:
                pickle.dump(self.xgboost_model, f)
            
            # Save scaler
            with open('scaler.pkl', 'wb') as f:
                pickle.dump(self.scaler, f)
            
            # Save performance metrics
            with open('performance_metrics.pkl', 'wb') as f:
                pickle.dump(self.performance_metrics, f)
            
            print("Models saved successfully!")
        except Exception as e:
            print(f"Error saving models: {e}")
    
    def load_models(self):
        """Load pre-trained models from disk."""
        try:
            if (os.path.exists('logistic_model.pkl') and 
                os.path.exists('xgboost_model.pkl') and 
                os.path.exists('scaler.pkl')):
                
                # Load logistic regression model
                with open('logistic_model.pkl', 'rb') as f:
                    self.logistic_model = pickle.load(f)
                
                # Load XGBoost model
                with open('xgboost_model.pkl', 'rb') as f:
                    self.xgboost_model = pickle.load(f)
                
                # Load scaler
                with open('scaler.pkl', 'rb') as f:
                    self.scaler = pickle.load(f)
                
                # Load performance metrics
                if os.path.exists('performance_metrics.pkl'):
                    with open('performance_metrics.pkl', 'rb') as f:
                        self.performance_metrics = pickle.load(f)
                
                self.trained = True
                print("Pre-trained models loaded successfully!")
                return True
        except Exception as e:
            print(f"Error loading models: {e}")
            self.trained = False
        
        return False
    
    def get_model_summary(self):
        """Get summary of trained models."""
        if not self.trained:
            return "Models not trained"
        
        summary = {
            'logistic_regression': {
                'type': 'Logistic Regression',
                'features': len(self.feature_names),
                'coefficients': self.logistic_model.coef_[0] if self.logistic_model else None
            },
            'xgboost': {
                'type': 'XGBoost Classifier',
                'features': len(self.feature_names),
                'n_estimators': self.xgboost_model.n_estimators if self.xgboost_model else None,
                'max_depth': self.xgboost_model.max_depth if self.xgboost_model else None
            }
        }
        
        return summary
