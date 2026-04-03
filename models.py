import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.metrics import precision_recall_fscore_support, roc_curve
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
        self.model_version = "1.0"
        self.training_date = None
        
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
    
    def train_models(self, training_data, validation_data=None):
        """Train both logistic regression and XGBoost models."""
        print("Training AHF prediction models...")
        
        # Prepare features and target
        X = self.prepare_features(training_data)
        y = training_data['readmission_30d'].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training set size: {len(X_train)}")
        print(f"Test set size: {len(X_test)}")
        print(f"Readmission rate in training: {y_train.mean():.1%}")
        
        # Apply SMOTE to handle class imbalance
        smote = SMOTE(random_state=42, k_neighbors=3)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        
        print(f"After SMOTE - Training set size: {len(X_train_balanced)}")
        print(f"After SMOTE - Readmission rate: {y_train_balanced.mean():.1%}")
        
        # Scale features for logistic regression
        X_train_scaled = self.scaler.fit_transform(X_train_balanced)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Logistic Regression
        print("Training Logistic Regression...")
        self.logistic_model = LogisticRegression(
            random_state=42,
            max_iter=2000,
            class_weight='balanced',
            solver='liblinear',
            penalty='l2',
            C=1.0
        )
        self.logistic_model.fit(X_train_scaled, y_train_balanced)
        
        # Train XGBoost
        print("Training XGBoost...")
        
        # Calculate scale_pos_weight for XGBoost
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        
        self.xgboost_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss',
            scale_pos_weight=scale_pos_weight,
            early_stopping_rounds=20,
            verbosity=0
        )
        
        # Train with early stopping using validation set
        eval_set = [(X_test, y_test)]
        self.xgboost_model.fit(
            X_train_balanced, 
            y_train_balanced,
            eval_set=eval_set,
            verbose=False
        )
        
        # Evaluate models
        self.evaluate_models(X_test_scaled, X_test, y_test)
        
        # Additional validation if provided
        if validation_data is not None:
            print("Evaluating on separate validation set...")
            self._evaluate_on_validation_set(validation_data)
        
        # Save models
        self.save_models()
        
        self.trained = True
        self.training_date = pd.Timestamp.now().isoformat()
        print("Model training completed!")
        
        # Print training summary
        self._print_training_summary()
    
    def evaluate_models(self, X_test_scaled, X_test_raw, y_test):
        """Evaluate both models and store performance metrics."""
        print("Evaluating model performance...")
        
        # Logistic Regression predictions
        lr_pred = self.logistic_model.predict(X_test_scaled)
        lr_pred_proba = self.logistic_model.predict_proba(X_test_scaled)[:, 1]
        
        # XGBoost predictions
        xgb_pred = self.xgboost_model.predict(X_test_raw)
        xgb_pred_proba = self.xgboost_model.predict_proba(X_test_raw)[:, 1]
        
        # Calculate metrics for Logistic Regression
        lr_metrics = self.calculate_metrics(y_test, lr_pred, lr_pred_proba)
        lr_metrics['model_name'] = 'Logistic Regression'
        
        # Calculate metrics for XGBoost
        xgb_metrics = self.calculate_metrics(y_test, xgb_pred, xgb_pred_proba)
        xgb_metrics['model_name'] = 'XGBoost'
        
        # Ensemble predictions
        ensemble_pred_proba = (lr_pred_proba + xgb_pred_proba) / 2
        ensemble_pred = (ensemble_pred_proba > 0.5).astype(int)
        ensemble_metrics = self.calculate_metrics(y_test, ensemble_pred, ensemble_pred_proba)
        ensemble_metrics['model_name'] = 'Ensemble'
        
        self.performance_metrics = {
            'logistic_regression': lr_metrics,
            'xgboost': xgb_metrics,
            'ensemble': ensemble_metrics
        }
        
        print(f"Logistic Regression - AUC: {lr_metrics['auc']:.3f}, Accuracy: {lr_metrics['accuracy']:.3f}")
        print(f"XGBoost - AUC: {xgb_metrics['auc']:.3f}, Accuracy: {xgb_metrics['accuracy']:.3f}")
        print(f"Ensemble - AUC: {ensemble_metrics['auc']:.3f}, Accuracy: {ensemble_metrics['accuracy']:.3f}")
    
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
            y_true, y_pred, average='binary', zero_division=0
        )
        
        # Additional metrics
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # Positive Predictive Value
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
        
        # ROC curve data
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        
        return {
            'accuracy': accuracy,
            'auc': auc,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'ppv': ppv,
            'npv': npv,
            'confusion_matrix': cm,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'true_labels': y_true,
            'roc_curve': {'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds},
            'tp': tp,
            'tn': tn,
            'fp': fp,
            'fn': fn
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
        
        # Ensemble prediction
        ensemble_prob = (lr_prob + xgb_prob) / 2
        ensemble_risk = "High Risk" if ensemble_prob > 0.5 else "Low Risk"
        
        return {
            'logistic_regression': {
                'probability': lr_prob,
                'risk_level': lr_risk
            },
            'xgboost': {
                'probability': xgb_prob,
                'risk_level': xgb_risk
            },
            'ensemble': {
                'probability': ensemble_prob,
                'risk_level': ensemble_risk
            }
        }
    
    def predict_batch(self, patient_data_list):
        """Predict risks for multiple patients."""
        if not self.trained:
            raise ValueError("Models must be trained before making predictions")
        
        predictions = []
        for patient_data in patient_data_list:
            try:
                pred = self.predict_risk(patient_data)
                pred['patient_id'] = patient_data.get('patient_id', 'Unknown')
                pred['prediction_timestamp'] = pd.Timestamp.now().isoformat()
                predictions.append(pred)
            except Exception as e:
                print(f"Error predicting for patient {patient_data.get('patient_id', 'Unknown')}: {str(e)}")
                
        return predictions
    
    def get_feature_importance(self, model_type='xgboost'):
        """Get feature importance from specified model."""
        if model_type == 'xgboost' and self.xgboost_model is not None:
            importance_scores = self.xgboost_model.feature_importances_
        elif model_type == 'logistic' and self.logistic_model is not None:
            # For logistic regression, use absolute coefficients
            importance_scores = np.abs(self.logistic_model.coef_[0])
        else:
            return None
        
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
            
            # Save model metadata
            metadata = {
                'model_version': self.model_version,
                'training_date': self.training_date,
                'feature_names': self.feature_names,
                'trained': self.trained
            }
            
            with open('model_metadata.pkl', 'wb') as f:
                pickle.dump(metadata, f)
            
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
                
                # Load metadata
                if os.path.exists('model_metadata.pkl'):
                    with open('model_metadata.pkl', 'rb') as f:
                        metadata = pickle.load(f)
                        self.model_version = metadata.get('model_version', '1.0')
                        self.training_date = metadata.get('training_date')
                        self.feature_names = metadata.get('feature_names', self.feature_names)
                
                self.trained = True
                print("Pre-trained models loaded successfully!")
                return True
        except Exception as e:
            print(f"Error loading models: {e}")
            self.trained = False
        
        return False
    
    def _evaluate_on_validation_set(self, validation_data):
        """Evaluate models on separate validation set."""
        X_val = self.prepare_features(validation_data)
        y_val = validation_data['readmission_30d'].values
        X_val_scaled = self.scaler.transform(X_val)
        
        # Predictions
        lr_pred_val = self.logistic_model.predict(X_val_scaled)
        lr_pred_proba_val = self.logistic_model.predict_proba(X_val_scaled)[:, 1]
        
        xgb_pred_val = self.xgboost_model.predict(X_val)
        xgb_pred_proba_val = self.xgboost_model.predict_proba(X_val)[:, 1]
        
        # Calculate validation metrics
        lr_val_metrics = self.calculate_metrics(y_val, lr_pred_val, lr_pred_proba_val)
        xgb_val_metrics = self.calculate_metrics(y_val, xgb_pred_val, xgb_pred_proba_val)
        
        print(f"Validation Set Performance:")
        print(f"Logistic Regression - AUC: {lr_val_metrics['auc']:.3f}, Accuracy: {lr_val_metrics['accuracy']:.3f}")
        print(f"XGBoost - AUC: {xgb_val_metrics['auc']:.3f}, Accuracy: {xgb_val_metrics['accuracy']:.3f}")
        
        # Store validation metrics
        self.performance_metrics['logistic_regression']['validation'] = lr_val_metrics
        self.performance_metrics['xgboost']['validation'] = xgb_val_metrics
    
    def _print_training_summary(self):
        """Print comprehensive training summary."""
        print("\n" + "="*60)
        print("MODEL TRAINING SUMMARY")
        print("="*60)
        
        if self.performance_metrics:
            for model_name, metrics in self.performance_metrics.items():
                if model_name in ['logistic_regression', 'xgboost', 'ensemble']:
                    print(f"\n{metrics.get('model_name', model_name).upper()}:")
                    print(f"  Accuracy:    {metrics['accuracy']:.3f}")
                    print(f"  AUC:         {metrics['auc']:.3f}")
                    print(f"  Sensitivity: {metrics['sensitivity']:.3f}")
                    print(f"  Specificity: {metrics['specificity']:.3f}")
                    print(f"  Precision:   {metrics['precision']:.3f}")
                    print(f"  F1-Score:    {metrics['f1']:.3f}")
        
        # Feature importance summary
        importance = self.get_feature_importance('xgboost')
        if importance:
            print(f"\nTOP 5 MOST IMPORTANT FEATURES (XGBoost):")
            for i, (feature, score) in enumerate(list(importance.items())[:5]):
                print(f"  {i+1}. {feature}: {score:.3f}")
        
        print("="*60 + "\n")
    
    def get_model_summary(self):
        """Get summary of trained models."""
        if not self.trained:
            return "Models not trained"
        
        summary = {
            'model_version': self.model_version,
            'training_date': self.training_date,
            'feature_count': len(self.feature_names),
            'logistic_regression': {
                'type': 'Logistic Regression',
                'features': len(self.feature_names),
                'coefficients': self.logistic_model.coef_[0].tolist() if self.logistic_model else None,
                'intercept': float(self.logistic_model.intercept_[0]) if self.logistic_model else None
            },
            'xgboost': {
                'type': 'XGBoost Classifier',
                'features': len(self.feature_names),
                'n_estimators': self.xgboost_model.n_estimators if self.xgboost_model else None,
                'max_depth': self.xgboost_model.max_depth if self.xgboost_model else None,
                'learning_rate': self.xgboost_model.learning_rate if self.xgboost_model else None
            },
            'performance_summary': {}
        }
        
        # Add performance summary
        if self.performance_metrics:
            for model_name, metrics in self.performance_metrics.items():
                if model_name in ['logistic_regression', 'xgboost', 'ensemble']:
                    summary['performance_summary'][model_name] = {
                        'auc': metrics.get('auc'),
                        'accuracy': metrics.get('accuracy'),
                        'sensitivity': metrics.get('sensitivity'),
                        'specificity': metrics.get('specificity')
                    }
        
        return summary
    
    def calibrate_probability_thresholds(self, validation_data=None):
        """Calibrate probability thresholds for optimal performance."""
        if not self.trained:
            return None
        
        if validation_data is None:
            print("No validation data provided for threshold calibration")
            return None
        
        X_val = self.prepare_features(validation_data)
        y_val = validation_data['readmission_30d'].values
        X_val_scaled = self.scaler.transform(X_val)
        
        # Get predicted probabilities
        lr_proba = self.logistic_model.predict_proba(X_val_scaled)[:, 1]
        xgb_proba = self.xgboost_model.predict_proba(X_val)[:, 1]
        ensemble_proba = (lr_proba + xgb_proba) / 2
        
        # Find optimal thresholds
        thresholds = np.arange(0.3, 0.8, 0.05)
        optimal_thresholds = {}
        
        for model_name, probabilities in [
            ('logistic_regression', lr_proba),
            ('xgboost', xgb_proba),
            ('ensemble', ensemble_proba)
        ]:
            best_f1 = 0
            best_threshold = 0.5
            
            for threshold in thresholds:
                predictions = (probabilities > threshold).astype(int)
                _, _, f1, _ = precision_recall_fscore_support(y_val, predictions, average='binary', zero_division=0)
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
            
            optimal_thresholds[model_name] = {
                'threshold': best_threshold,
                'f1_score': best_f1
            }
        
        print("Optimal probability thresholds:")
        for model, data in optimal_thresholds.items():
            print(f"  {model}: {data['threshold']:.2f} (F1: {data['f1_score']:.3f})")
        
        return optimal_thresholds

