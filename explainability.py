import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from models import AHFPredictionModels

class ExplainabilityManager:
    """Manages model explainability for AHF predictions using native model capabilities."""
    
    def __init__(self):
        """Initialize explainability manager."""
        self.feature_names = [
            'age', 'gender', 'weight', 'nt_probnp', 'creatinine',
            'b_line_score', 'ivc_collapsibility', 'ejection_fraction',
            'systolic_bp', 'heart_rate', 'diabetes', 'hypertension', 'ckd', 'afib'
        ]
        
        self.feature_descriptions = {
            'age': 'Patient Age (years)',
            'gender': 'Gender (0=Female, 1=Male)',
            'weight': 'Body Weight (kg)',
            'nt_probnp': 'NT-proBNP Level (pg/mL)',
            'creatinine': 'Serum Creatinine (mg/dL)',
            'b_line_score': 'Ultrasound B-line Score',
            'ivc_collapsibility': 'IVC Collapsibility Index (%)',
            'ejection_fraction': 'Left Ventricular Ejection Fraction (%)',
            'systolic_bp': 'Systolic Blood Pressure (mmHg)',
            'heart_rate': 'Heart Rate (bpm)',
            'diabetes': 'Diabetes Mellitus (0=No, 1=Yes)',
            'hypertension': 'Hypertension (0=No, 1=Yes)',
            'ckd': 'Chronic Kidney Disease (0=No, 1=Yes)',
            'afib': 'Atrial Fibrillation (0=No, 1=Yes)'
        }
        
        self.background_data = None
    
    def initialize_explainer(self, models, background_data=None):
        """Initialize explainer with background data."""
        try:
            if background_data is None:
                from data_generator import SyntheticDataGenerator
                generator = SyntheticDataGenerator()
                bg_data = generator.generate_training_dataset(100)
                self.background_data = models.prepare_features(bg_data)
            else:
                self.background_data = background_data
            
            return True
            
        except Exception as e:
            print(f"Error initializing explainer: {e}")
            return False
    
    def _get_xgboost_contributions(self, features, models):
        """Get feature contributions from XGBoost using pred_contribs."""
        try:
            import xgboost as xgb
            dmatrix = xgb.DMatrix(features)
            contribs = models.xgboost_model.predict(dmatrix, pred_contribs=True)
            
            # pred_contribs returns [features..., bias]
            # We want just the feature contributions
            if len(contribs.shape) == 2:
                bias = contribs[0, -1]
                feature_contribs = contribs[0, :-1]
            else:
                bias = 0
                feature_contribs = contribs[:-1]
            
            return feature_contribs, bias
            
        except Exception as e:
            print(f"Error getting XGBoost contributions: {e}")
            return None, None
    
    def _get_logistic_contributions(self, features, models):
        """Get feature contributions from Logistic Regression using coefficients."""
        try:
            # Scale features
            features_scaled = models.scaler.transform(features)
            
            # Get coefficients
            coefficients = models.logistic_model.coef_[0]
            intercept = models.logistic_model.intercept_[0]
            
            # Calculate contributions: coef * scaled_feature
            contributions = coefficients * features_scaled[0]
            
            return contributions, intercept
            
        except Exception as e:
            print(f"Error getting logistic contributions: {e}")
            return None, None
    
    def explain_prediction(self, patient_data, models=None):
        """Generate explanation for a single prediction."""
        try:
            if models is None:
                from models import AHFPredictionModels
                models = AHFPredictionModels()
            
            if not models.models_trained():
                return None
            
            # Prepare patient features
            if isinstance(patient_data, dict):
                features = np.array([[patient_data[feature] for feature in self.feature_names]])
            else:
                features = patient_data
            
            # Get contributions based on available model
            if models.xgboost_model is not None:
                contributions, base_value = self._get_xgboost_contributions(features, models)
            elif models.logistic_model is not None:
                contributions, base_value = self._get_logistic_contributions(features, models)
            else:
                return None
            
            if contributions is None:
                return None
            
            # Get feature values for the prediction
            feature_values = features[0] if len(features.shape) > 1 else features
            
            # Create explanation plot
            explanation_plot = self.create_explanation_plot(
                contributions, feature_values, patient_data
            )
            
            # Get top contributing factors
            top_factors = self.get_top_factors(contributions, feature_values)
            
            return {
                'contributions': contributions,
                'feature_values': feature_values,
                'plot': explanation_plot,
                'top_factors': top_factors,
                'base_value': base_value
            }
            
        except Exception as e:
            print(f"Error generating explanation: {e}")
            return None
    
    def create_explanation_plot(self, contributions, feature_values, patient_data):
        """Create interactive feature contribution plot."""
        try:
            # Create feature impact data
            impact_data = []
            for i, (feature, contrib, feature_val) in enumerate(zip(self.feature_names, contributions, feature_values)):
                impact_data.append({
                    'feature': self.feature_descriptions.get(feature, feature),
                    'feature_name': feature,
                    'contribution': contrib,
                    'feature_value': feature_val,
                    'abs_contribution': abs(contrib)
                })
            
            # Sort by absolute contribution
            impact_data.sort(key=lambda x: x['abs_contribution'], reverse=True)
            
            # Take top 10 features
            top_impact = impact_data[:10]
            
            # Create horizontal bar chart
            colors = ['red' if x['contribution'] > 0 else 'blue' for x in top_impact]
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                y=[x['feature'] for x in top_impact],
                x=[x['contribution'] for x in top_impact],
                orientation='h',
                marker_color=colors,
                text=[f"Value: {x['feature_value']:.2f}" for x in top_impact],
                textposition='outside',
                hovertemplate=(
                    "<b>%{y}</b><br>" +
                    "Contribution: %{x:.4f}<br>" +
                    "Feature Value: %{text}<br>" +
                    "<extra></extra>"
                )
            ))
            
            fig.update_layout(
                title="Feature Impact on Prediction (Model Contributions)",
                xaxis_title="Impact on Prediction",
                yaxis_title="Clinical Features",
                height=500,
                template="plotly_white",
                annotations=[
                    dict(
                        x=0.02, y=0.98,
                        xref="paper", yref="paper",
                        text="← Decreases Risk | Increases Risk →",
                        showarrow=False,
                        font=dict(size=12)
                    )
                ]
            )
            
            # Add vertical line at x=0
            fig.add_vline(x=0, line_width=2, line_color="black", opacity=0.5)
            
            return fig
            
        except Exception as e:
            print(f"Error creating explanation plot: {e}")
            return None
    
    def get_top_factors(self, contributions, feature_values, n_top=5):
        """Get top contributing factors with descriptions."""
        try:
            # Create factor impact dictionary
            factor_impacts = {}
            
            for feature, contrib, feature_val in zip(self.feature_names, contributions, feature_values):
                if abs(contrib) > 0.001:  # Only include meaningful contributions
                    description = self.feature_descriptions.get(feature, feature)
                    factor_impacts[f"{description} ({feature_val:.2f})"] = contrib
            
            # Sort by absolute impact and return top factors
            sorted_factors = dict(sorted(factor_impacts.items(), 
                                       key=lambda x: abs(x[1]), reverse=True))
            
            return dict(list(sorted_factors.items())[:n_top])
            
        except Exception as e:
            print(f"Error getting top factors: {e}")
            return {}
    
    def create_feature_importance_summary(self, models=None):
        """Create overall feature importance summary."""
        try:
            if models is None:
                from models import AHFPredictionModels
                models = AHFPredictionModels()
            
            if not models.models_trained():
                return None
            
            # Get feature importance from available model
            if models.xgboost_model is not None:
                # XGBoost has built-in feature importance
                importance_scores = models.xgboost_model.feature_importances_
            elif models.logistic_model is not None:
                # Use absolute coefficients for logistic regression
                importance_scores = np.abs(models.logistic_model.coef_[0])
            else:
                return None
            
            # Create summary plot
            summary_data = pd.DataFrame({
                'feature': [self.feature_descriptions.get(f, f) for f in self.feature_names],
                'importance': importance_scores
            }).sort_values('importance', ascending=True)
            
            fig = px.bar(
                summary_data.tail(10),  # Top 10 features
                x='importance',
                y='feature',
                orientation='h',
                title="Overall Feature Importance",
                labels={'importance': 'Importance Score', 'feature': 'Clinical Features'}
            )
            
            fig.update_layout(height=500, template="plotly_white")
            
            return {
                'plot': fig,
                'importance_scores': dict(zip(self.feature_names, importance_scores))
            }
            
        except Exception as e:
            print(f"Error creating feature importance summary: {e}")
            return None
    
    def explain_model_behavior(self, models=None, sample_size=100):
        """Generate comprehensive model behavior analysis."""
        try:
            if models is None:
                from models import AHFPredictionModels
                models = AHFPredictionModels()
            
            if not models.models_trained():
                return None
            
            # Generate sample data for analysis
            from data_generator import SyntheticDataGenerator
            generator = SyntheticDataGenerator()
            sample_data = generator.generate_training_dataset(sample_size)
            sample_features = models.prepare_features(sample_data)
            
            # Calculate contributions for all samples
            all_contributions = []
            
            for i in range(len(sample_features)):
                features = sample_features[i:i+1]
                
                if models.xgboost_model is not None:
                    contribs, _ = self._get_xgboost_contributions(features, models)
                elif models.logistic_model is not None:
                    contribs, _ = self._get_logistic_contributions(features, models)
                else:
                    return None
                
                if contribs is not None:
                    all_contributions.append(contribs)
            
            all_contributions = np.array(all_contributions)
            
            # Create summary plots
            plots = {}
            
            # 1. Feature importance (mean absolute contributions)
            mean_abs_contrib = np.mean(np.abs(all_contributions), axis=0)
            importance_df = pd.DataFrame({
                'feature': [self.feature_descriptions.get(f, f) for f in self.feature_names],
                'importance': mean_abs_contrib
            }).sort_values('importance', ascending=True)
            
            plots['importance'] = px.bar(
                importance_df.tail(10),
                x='importance',
                y='feature',
                orientation='h',
                title="Feature Importance"
            )
            
            # 2. Feature effects distribution
            contrib_df = pd.DataFrame(all_contributions, columns=self.feature_names)
            feature_effects = {}
            
            for feature in self.feature_names[:6]:  # Top 6 features
                feature_effects[feature] = {
                    'mean_effect': contrib_df[feature].mean(),
                    'std_effect': contrib_df[feature].std(),
                    'positive_impact': (contrib_df[feature] > 0).sum(),
                    'negative_impact': (contrib_df[feature] < 0).sum()
                }
            
            return {
                'plots': plots,
                'feature_effects': feature_effects,
                'contributions': all_contributions,
                'sample_features': sample_features
            }
            
        except Exception as e:
            print(f"Error in model behavior analysis: {e}")
            return None
