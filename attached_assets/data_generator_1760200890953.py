import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class SyntheticDataGenerator:
    """Generate synthetic clinical data for AHF patients based on medical literature."""
    
    def __init__(self, random_seed=42):
        """Initialize the synthetic data generator."""
        np.random.seed(random_seed)
        self.random_seed = random_seed
        
        # Define realistic parameter distributions based on clinical literature
        self.parameter_distributions = {
            'age': {'mean': 72, 'std': 12, 'min': 18, 'max': 95},
            'weight': {'mean': 78, 'std': 18, 'min': 40, 'max': 150},
            'nt_probnp_log': {'mean': 7.8, 'std': 1.5},  # log-transformed
            'creatinine': {'mean': 1.4, 'std': 0.8, 'min': 0.5, 'max': 5.0},
            'b_line_score': {'mean': 12, 'std': 8, 'min': 0, 'max': 28},
            'ivc_collapsibility': {'mean': 35, 'std': 20, 'min': 0, 'max': 100},
            'ejection_fraction': {'mean': 38, 'std': 15, 'min': 10, 'max': 75},
            'systolic_bp': {'mean': 125, 'std': 25, 'min': 80, 'max': 200},
            'heart_rate': {'mean': 85, 'std': 20, 'min': 50, 'max': 150}
        }
        
        # Comorbidity probabilities
        self.comorbidity_probs = {
            'diabetes': 0.45,
            'hypertension': 0.75,
            'ckd': 0.40,
            'afib': 0.35
        }
    
    def generate_training_dataset(self, n_samples=2000):
        """Generate synthetic training dataset with realistic distributions."""
        np.random.seed(self.random_seed)
        
        print(f"Generating {n_samples} synthetic AHF patient records...")
        
        # Generate basic demographics
        data = {}
        
        # Age (normal distribution, clipped)
        data['age'] = np.clip(
            np.random.normal(
                self.parameter_distributions['age']['mean'],
                self.parameter_distributions['age']['std'],
                n_samples
            ),
            self.parameter_distributions['age']['min'],
            self.parameter_distributions['age']['max']
        ).astype(int)
        
        # Gender (binary, slightly more males in HF population)
        data['gender'] = np.random.choice([0, 1], n_samples, p=[0.45, 0.55])
        
        # Weight (normal distribution, clipped)
        data['weight'] = np.clip(
            np.random.normal(
                self.parameter_distributions['weight']['mean'],
                self.parameter_distributions['weight']['std'],
                n_samples
            ),
            self.parameter_distributions['weight']['min'],
            self.parameter_distributions['weight']['max']
        )
        
        # NT-proBNP (log-normal distribution)
        nt_probnp_log = np.random.normal(
            self.parameter_distributions['nt_probnp_log']['mean'],
            self.parameter_distributions['nt_probnp_log']['std'],
            n_samples
        )
        data['nt_probnp'] = np.exp(nt_probnp_log)
        data['nt_probnp'] = np.clip(data['nt_probnp'], 50, 50000)
        
        # Creatinine (log-normal distribution)
        data['creatinine'] = np.clip(
            np.random.lognormal(
                np.log(self.parameter_distributions['creatinine']['mean']),
                0.5,
                n_samples
            ),
            self.parameter_distributions['creatinine']['min'],
            self.parameter_distributions['creatinine']['max']
        )
        
        # B-line score (gamma distribution)
        data['b_line_score'] = np.clip(
            np.random.gamma(2, 6, n_samples).astype(int),
            self.parameter_distributions['b_line_score']['min'],
            self.parameter_distributions['b_line_score']['max']
        )
        
        # IVC collapsibility (beta distribution, scaled)
        data['ivc_collapsibility'] = np.random.beta(2, 3, n_samples) * 100
        
        # Ejection fraction (normal distribution, clipped)
        data['ejection_fraction'] = np.clip(
            np.random.normal(
                self.parameter_distributions['ejection_fraction']['mean'],
                self.parameter_distributions['ejection_fraction']['std'],
                n_samples
            ),
            self.parameter_distributions['ejection_fraction']['min'],
            self.parameter_distributions['ejection_fraction']['max']
        )
        
        # Systolic BP (normal distribution, clipped)
        data['systolic_bp'] = np.clip(
            np.random.normal(
                self.parameter_distributions['systolic_bp']['mean'],
                self.parameter_distributions['systolic_bp']['std'],
                n_samples
            ),
            self.parameter_distributions['systolic_bp']['min'],
            self.parameter_distributions['systolic_bp']['max']
        ).astype(int)
        
        # Heart rate (normal distribution, clipped)
        data['heart_rate'] = np.clip(
            np.random.normal(
                self.parameter_distributions['heart_rate']['mean'],
                self.parameter_distributions['heart_rate']['std'],
                n_samples
            ),
            self.parameter_distributions['heart_rate']['min'],
            self.parameter_distributions['heart_rate']['max']
        ).astype(int)
        
        # Comorbidities (binary)
        for condition, prob in self.comorbidity_probs.items():
            data[condition] = np.random.choice([0, 1], n_samples, p=[1-prob, prob])
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Generate target variable (30-day readmission) based on clinical risk factors
        readmission_prob = self._calculate_readmission_probability(df)
        df['readmission_30d'] = np.random.binomial(1, readmission_prob)
        
        # Add some noise and interactions
        df = self._add_clinical_interactions(df)
        
        print(f"Generated dataset shape: {df.shape}")
        print(f"Readmission rate: {df['readmission_30d'].mean():.1%}")
        
        return df
    
    def _calculate_readmission_probability(self, df):
        """Calculate readmission probability based on clinical risk factors."""
        # Base probability
        base_prob = 0.15  # 15% baseline readmission rate
        
        # Risk factors with weights based on clinical literature
        risk_score = (
            # NT-proBNP (log scale, higher = higher risk)
            0.08 * np.log(df['nt_probnp'] + 1) +
            
            # Age (higher = higher risk)
            0.015 * (df['age'] - 65) +
            
            # B-line score (higher = higher risk)
            0.02 * df['b_line_score'] +
            
            # Lower IVC collapsibility = higher risk
            -0.01 * (df['ivc_collapsibility'] - 50) +
            
            # Lower EF = higher risk
            -0.015 * (df['ejection_fraction'] - 40) +
            
            # Higher creatinine = higher risk
            0.15 * (df['creatinine'] - 1.0) +
            
            # Comorbidities
            0.3 * df['diabetes'] +
            0.2 * df['ckd'] +
            0.25 * df['afib'] +
            0.1 * df['hypertension'] +
            
            # Age interactions
            0.005 * df['age'] * df['diabetes'] +
            
            # Gender effect (males slightly higher risk)
            0.1 * df['gender']
        )
        
        # Convert to probability using sigmoid function
        probability = base_prob + 0.6 / (1 + np.exp(-risk_score))
        
        # Ensure probabilities are between 0 and 1
        probability = np.clip(probability, 0.02, 0.85)
        
        return probability
    
    def _add_clinical_interactions(self, df):
        """Add realistic clinical interactions and correlations."""
        # Age-related adjustments
        older_patients = df['age'] > 75
        df.loc[older_patients, 'creatinine'] *= 1.2  # Older patients tend to have higher creatinine
        df.loc[older_patients, 'nt_probnp'] *= 1.3   # Higher NT-proBNP in elderly
        
        # CKD patients adjustments
        ckd_patients = df['ckd'] == 1
        df.loc[ckd_patients, 'creatinine'] *= 2.0    # CKD patients have higher creatinine
        df.loc[ckd_patients, 'nt_probnp'] *= 1.4     # Higher NT-proBNP in CKD
        
        # Heart failure severity correlations
        severe_hf = df['ejection_fraction'] < 30
        df.loc[severe_hf, 'nt_probnp'] *= 1.6        # Higher NT-proBNP in severe HF
        df.loc[severe_hf, 'b_line_score'] += 3       # More B-lines in severe HF
        
        # Fluid overload correlations
        fluid_overload = df['b_line_score'] > 15
        df.loc[fluid_overload, 'weight'] += np.random.normal(3, 2, sum(fluid_overload))  # Higher weight
        df.loc[fluid_overload, 'ivc_collapsibility'] -= 10  # Reduced IVC collapsibility
        
        # Ensure values stay within realistic ranges
        df['nt_probnp'] = np.clip(df['nt_probnp'], 50, 50000)
        df['creatinine'] = np.clip(df['creatinine'], 0.5, 5.0)
        df['b_line_score'] = np.clip(df['b_line_score'], 0, 28)
        df['ivc_collapsibility'] = np.clip(df['ivc_collapsibility'], 0, 100)
        df['weight'] = np.clip(df['weight'], 40, 150)
        
        return df
    
    def generate_validation_dataset(self, n_samples=500):
        """Generate a separate validation dataset."""
        # Use different random seed for validation data
        original_seed = self.random_seed
        self.random_seed = original_seed + 1000
        
        validation_data = self.generate_training_dataset(n_samples)
        
        # Restore original seed
        self.random_seed = original_seed
        
        return validation_data
    
    def get_dataset_summary(self, df):
        """Get summary statistics of the generated dataset."""
        summary = {
            'total_samples': len(df),
            'readmission_rate': df['readmission_30d'].mean(),
            'demographics': {
                'mean_age': df['age'].mean(),
                'percent_male': (df['gender'] == 1).mean(),
                'mean_weight': df['weight'].mean()
            },
            'biomarkers': {
                'median_nt_probnp': df['nt_probnp'].median(),
                'mean_creatinine': df['creatinine'].mean(),
                'mean_ejection_fraction': df['ejection_fraction'].mean()
            },
            'ultrasound': {
                'mean_b_line_score': df['b_line_score'].mean(),
                'mean_ivc_collapsibility': df['ivc_collapsibility'].mean()
            },
            'comorbidities': {
                'diabetes_rate': df['diabetes'].mean(),
                'hypertension_rate': df['hypertension'].mean(),
                'ckd_rate': df['ckd'].mean(),
                'afib_rate': df['afib'].mean()
            }
        }
        
        return summary
    
    def export_dataset(self, df, filename=None):
        """Export dataset to CSV file."""
        if filename is None:
            filename = f"synthetic_ahf_dataset_{len(df)}_samples.csv"
        
        df.to_csv(filename, index=False)
        print(f"Dataset exported to {filename}")
        
        return filename
