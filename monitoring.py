import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sqlite3
from sklearn.metrics import roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

class ModelMonitor:
    """Monitors model performance and data drift."""
    
    def __init__(self, db_manager):
        """Initialize model monitor."""
        self.db_manager = db_manager
        
    def create_roc_comparison(self, metrics):
        """Create ROC curve comparison plot."""
        try:
            fig = go.Figure()
            
            # Logistic Regression ROC
            lr_metrics = metrics['logistic_regression']
            if 'true_labels' in lr_metrics and 'probabilities' in lr_metrics:
                lr_fpr, lr_tpr, _ = roc_curve(lr_metrics['true_labels'], lr_metrics['probabilities'])
                lr_auc = auc(lr_fpr, lr_tpr)
                
                fig.add_trace(go.Scatter(
                    x=lr_fpr, y=lr_tpr,
                    mode='lines',
                    name=f'Logistic Regression (AUC = {lr_auc:.3f})',
                    line=dict(color='blue', width=2)
                ))
            
            # XGBoost ROC
            xgb_metrics = metrics['xgboost']
            if 'true_labels' in xgb_metrics and 'probabilities' in xgb_metrics:
                xgb_fpr, xgb_tpr, _ = roc_curve(xgb_metrics['true_labels'], xgb_metrics['probabilities'])
                xgb_auc = auc(xgb_fpr, xgb_tpr)
                
                fig.add_trace(go.Scatter(
                    x=xgb_fpr, y=xgb_tpr,
                    mode='lines',
                    name=f'XGBoost (AUC = {xgb_auc:.3f})',
                    line=dict(color='red', width=2)
                ))
            
            # Random classifier line
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode='lines',
                name='Random Classifier',
                line=dict(color='gray', width=1, dash='dash')
            ))
            
            fig.update_layout(
                title='ROC Curve Comparison',
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate',
                template='plotly_white',
                width=600,
                height=500,
                legend=dict(
                    yanchor="bottom",
                    y=0.01,
                    xanchor="right",
                    x=0.99
                )
            )
            
            return fig
            
        except Exception as e:
            print(f"Error creating ROC comparison: {e}")
            return None
    
    def create_confusion_matrix_plot(self, confusion_matrix, model_name):
        """Create confusion matrix heatmap."""
        try:
            tn, fp, fn, tp = confusion_matrix.ravel()
            
            # Create confusion matrix data
            cm_data = np.array([[tn, fp], [fn, tp]])
            
            # Create annotations
            annotations = []
            for i in range(2):
                for j in range(2):
                    annotations.append(
                        dict(
                            x=j, y=i,
                            text=str(cm_data[i, j]),
                            showarrow=False,
                            font=dict(color="white" if cm_data[i, j] > cm_data.max()/2 else "black", size=16)
                        )
                    )
            
            fig = go.Figure(data=go.Heatmap(
                z=cm_data,
                x=['Predicted Negative', 'Predicted Positive'],
                y=['Actual Negative', 'Actual Positive'],
                colorscale='Blues',
                showscale=True
            ))
            
            fig.update_layout(
                title=f'Confusion Matrix - {model_name}',
                annotations=annotations,
                xaxis_title='Predicted',
                yaxis_title='Actual',
                template='plotly_white',
                width=400,
                height=400
            )
            
            return fig
            
        except Exception as e:
            print(f"Error creating confusion matrix plot: {e}")
            return None
    
    def check_model_drift(self, lookback_days=30):
        """Monitor for model performance drift over time."""
        try:
            # Get recent assessments
            assessments = self.db_manager.get_all_assessments()
            
            if not assessments or len(assessments) < 50:
                return None
            
            df = pd.DataFrame(assessments)
            df['assessment_date'] = pd.to_datetime(df['assessment_date'])
            
            # Filter to recent data
            cutoff_date = datetime.now() - timedelta(days=lookback_days)
            recent_df = df[df['assessment_date'] >= cutoff_date].copy()
            
            if len(recent_df) < 20:
                return None
            
            # Group by day and calculate daily metrics
            recent_df['date'] = recent_df['assessment_date'].dt.date
            daily_stats = recent_df.groupby('date').agg({
                'ensemble_probability': ['mean', 'std', 'count'],
                'lr_probability': 'mean',
                'xgb_probability': 'mean'
            }).reset_index()
            
            # Flatten column names
            daily_stats.columns = ['date', 'avg_risk', 'risk_std', 'count', 'avg_lr', 'avg_xgb']
            
            # Create drift monitoring plot
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Average Risk Score Trend', 'Risk Score Distribution',
                               'Model Agreement', 'Daily Assessment Volume'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Average risk trend
            fig.add_trace(
                go.Scatter(
                    x=daily_stats['date'],
                    y=daily_stats['avg_risk'],
                    mode='lines+markers',
                    name='Average Risk',
                    line=dict(color='blue')
                ),
                row=1, col=1
            )
            
            # Risk distribution
            fig.add_trace(
                go.Histogram(
                    x=recent_df['ensemble_probability'],
                    nbinsx=20,
                    name='Risk Distribution',
                    showlegend=False
                ),
                row=1, col=2
            )
            
            # Model agreement
            fig.add_trace(
                go.Scatter(
                    x=daily_stats['avg_lr'],
                    y=daily_stats['avg_xgb'],
                    mode='markers',
                    name='LR vs XGB',
                    showlegend=False
                ),
                row=2, col=1
            )
            
            # Add diagonal line for perfect agreement
            min_val = min(daily_stats['avg_lr'].min(), daily_stats['avg_xgb'].min())
            max_val = max(daily_stats['avg_lr'].max(), daily_stats['avg_xgb'].max())
            fig.add_trace(
                go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    line=dict(color='red', dash='dash'),
                    name='Perfect Agreement',
                    showlegend=False
                ),
                row=2, col=1
            )
            
            # Daily volume
            fig.add_trace(
                go.Bar(
                    x=daily_stats['date'],
                    y=daily_stats['count'],
                    name='Daily Assessments',
                    showlegend=False
                ),
                row=2, col=2
            )
            
            fig.update_layout(
                title=f'Model Performance Monitoring (Last {lookback_days} Days)',
                height=600,
                template='plotly_white'
            )
            
            # Calculate drift metrics
            drift_metrics = {
                'avg_risk_change': daily_stats['avg_risk'].iloc[-7:].mean() - daily_stats['avg_risk'].iloc[:7].mean(),
                'risk_volatility': daily_stats['avg_risk'].std(),
                'model_agreement': np.corrcoef(daily_stats['avg_lr'], daily_stats['avg_xgb'])[0, 1],
                'recent_volume': daily_stats['count'].iloc[-7:].mean()
            }
            
            return {
                'plot': fig,
                'metrics': drift_metrics,
                'daily_stats': daily_stats
            }
            
        except Exception as e:
            print(f"Error checking model drift: {e}")
            return None
    
    def create_performance_dashboard(self):
        """Create comprehensive performance monitoring dashboard."""
        try:
            # Get model performance data
            assessments = self.db_manager.get_all_assessments()
            
            if not assessments:
                return None
            
            df = pd.DataFrame(assessments)
            df['assessment_date'] = pd.to_datetime(df['assessment_date'])
            
            # Create dashboard with multiple metrics
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Risk Level Distribution', 'Risk Scores Over Time',
                               'Model Probability Comparison', 'Biomarker Correlation'),
                specs=[[{"type": "pie"}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Risk level distribution
            risk_counts = df['risk_level'].value_counts()
            colors = ['green', 'orange', 'red']
            
            fig.add_trace(
                go.Pie(
                    labels=risk_counts.index,
                    values=risk_counts.values,
                    name="Risk Distribution",
                    marker_colors=colors[:len(risk_counts)]
                ),
                row=1, col=1
            )
            
            # Risk scores over time
            df_sorted = df.sort_values('assessment_date')
            fig.add_trace(
                go.Scatter(
                    x=df_sorted['assessment_date'],
                    y=df_sorted['ensemble_probability'],
                    mode='markers',
                    name='Ensemble Risk',
                    marker=dict(
                        color=df_sorted['ensemble_probability'],
                        colorscale='RdYlGn_r',
                        showscale=False
                    )
                ),
                row=1, col=2
            )
            
            # Model comparison
            fig.add_trace(
                go.Scatter(
                    x=df['lr_probability'],
                    y=df['xgb_probability'],
                    mode='markers',
                    name='LR vs XGB',
                    marker=dict(
                        color=df['ensemble_probability'],
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="Ensemble Risk")
                    )
                ),
                row=2, col=1
            )
            
            # Biomarker correlation (NT-proBNP vs Risk)
            fig.add_trace(
                go.Scatter(
                    x=df['nt_probnp'],
                    y=df['ensemble_probability'],
                    mode='markers',
                    name='NT-proBNP vs Risk',
                    showlegend=False
                ),
                row=2, col=2
            )
            
            fig.update_layout(
                title='Model Performance Dashboard',
                height=700,
                template='plotly_white'
            )
            
            # Update axis labels
            fig.update_xaxes(title_text="Time", row=1, col=2)
            fig.update_yaxes(title_text="Risk Score", row=1, col=2)
            fig.update_xaxes(title_text="Logistic Regression Probability", row=2, col=1)
            fig.update_yaxes(title_text="XGBoost Probability", row=2, col=1)
            fig.update_xaxes(title_text="NT-proBNP (pg/mL)", row=2, col=2)
            fig.update_yaxes(title_text="Risk Score", row=2, col=2)
            
            return fig
            
        except Exception as e:
            print(f"Error creating performance dashboard: {e}")
            return None
    
    def generate_performance_report(self):
        """Generate detailed performance metrics report."""
        try:
            assessments = self.db_manager.get_all_assessments()
            
            if not assessments:
                return None
            
            df = pd.DataFrame(assessments)
            
            # Calculate performance metrics
            report = {
                'total_assessments': len(df),
                'unique_patients': df['patient_id'].nunique(),
                'risk_distribution': df['risk_level'].value_counts().to_dict(),
                'average_risk_score': df['ensemble_probability'].mean(),
                'risk_score_std': df['ensemble_probability'].std(),
                'high_risk_percentage': (df['risk_level'] == 'High Risk').mean() * 100,
                'model_correlation': np.corrcoef(df['lr_probability'], df['xgb_probability'])[0, 1],
                'date_range': {
                    'from': df['assessment_date'].min(),
                    'to': df['assessment_date'].max()
                }
            }
            
            # Biomarker statistics
            biomarkers = ['nt_probnp', 'weight', 'creatinine', 'b_line_score', 'ejection_fraction']
            biomarker_stats = {}
            
            for biomarker in biomarkers:
                if biomarker in df.columns:
                    biomarker_stats[biomarker] = {
                        'mean': df[biomarker].mean(),
                        'std': df[biomarker].std(),
                        'correlation_with_risk': np.corrcoef(df[biomarker], df['ensemble_probability'])[0, 1]
                    }
            
            report['biomarker_stats'] = biomarker_stats
            
            # Recent trends (last 7 days)
            df['assessment_date'] = pd.to_datetime(df['assessment_date'])
            recent_cutoff = datetime.now() - timedelta(days=7)
            recent_df = df[df['assessment_date'] >= recent_cutoff]
            
            if len(recent_df) > 0:
                report['recent_trends'] = {
                    'recent_assessments': len(recent_df),
                    'recent_avg_risk': recent_df['ensemble_probability'].mean(),
                    'recent_high_risk_count': (recent_df['risk_level'] == 'High Risk').sum()
                }
            
            return report
            
        except Exception as e:
            print(f"Error generating performance report: {e}")
            return None
