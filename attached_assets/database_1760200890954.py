import sqlite3
import pandas as pd
import os
from datetime import datetime
import json

class DatabaseManager:
    """Manages SQLite database operations for patient records and predictions."""
    
    def __init__(self, db_path="ahf_predictions.db"):
        """Initialize database manager."""
        self.db_path = db_path
        self.initialize_database()
    
    def initialize_database(self):
        """Create database tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create assessments table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS assessments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT NOT NULL,
                assessment_date TEXT NOT NULL,
                age INTEGER,
                gender TEXT,
                weight REAL,
                nt_probnp REAL,
                creatinine REAL,
                b_line_score INTEGER,
                ivc_collapsibility REAL,
                ejection_fraction REAL,
                systolic_bp INTEGER,
                heart_rate INTEGER,
                diabetes INTEGER,
                hypertension INTEGER,
                ckd INTEGER,
                afib INTEGER,
                lr_probability REAL,
                xgb_probability REAL,
                ensemble_probability REAL,
                risk_level TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create model_performance table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                accuracy REAL,
                auc REAL,
                sensitivity REAL,
                specificity REAL,
                precision_score REAL,
                recall REAL,
                f1_score REAL,
                training_date TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        conn.close()
    
    def save_assessment(self, record_data):
        """Save a patient assessment record."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        columns = list(record_data.keys())
        placeholders = ', '.join(['?' for _ in columns])
        values = list(record_data.values())
        
        query = f"""
            INSERT INTO assessments ({', '.join(columns)})
            VALUES ({placeholders})
        """
        
        try:
            cursor.execute(query, values)
            conn.commit()
            return cursor.lastrowid
        except sqlite3.Error as e:
            print(f"Error saving assessment: {e}")
            return None
        finally:
            conn.close()
    
    def get_all_assessments(self):
        """Retrieve all assessment records."""
        conn = sqlite3.connect(self.db_path)
        try:
            df = pd.read_sql_query("SELECT * FROM assessments ORDER BY assessment_date DESC", conn)
            return df.to_dict('records')
        except sqlite3.Error as e:
            print(f"Error retrieving assessments: {e}")
            return []
        finally:
            conn.close()
    
    def get_assessment_by_patient_id(self, patient_id):
        """Retrieve assessments for a specific patient."""
        conn = sqlite3.connect(self.db_path)
        try:
            query = "SELECT * FROM assessments WHERE patient_id = ? ORDER BY assessment_date DESC"
            df = pd.read_sql_query(query, conn, params=[patient_id])
            return df.to_dict('records')
        except sqlite3.Error as e:
            print(f"Error retrieving patient assessments: {e}")
            return []
        finally:
            conn.close()
    
    def save_model_performance(self, model_name, metrics):
        """Save model performance metrics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = """
            INSERT INTO model_performance 
            (model_name, accuracy, auc, sensitivity, specificity, precision_score, recall, f1_score, training_date)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        values = [
            model_name,
            metrics.get('accuracy'),
            metrics.get('auc'),
            metrics.get('sensitivity'),
            metrics.get('specificity'),
            metrics.get('precision'),
            metrics.get('recall'),
            metrics.get('f1'),
            datetime.now().isoformat()
        ]
        
        try:
            cursor.execute(query, values)
            conn.commit()
            return cursor.lastrowid
        except sqlite3.Error as e:
            print(f"Error saving model performance: {e}")
            return None
        finally:
            conn.close()
    
    def get_latest_model_performance(self, model_name):
        """Get latest performance metrics for a model."""
        conn = sqlite3.connect(self.db_path)
        try:
            query = """
                SELECT * FROM model_performance 
                WHERE model_name = ? 
                ORDER BY created_at DESC 
                LIMIT 1
            """
            df = pd.read_sql_query(query, conn, params=[model_name])
            if not df.empty:
                return df.iloc[0].to_dict()
            return None
        except sqlite3.Error as e:
            print(f"Error retrieving model performance: {e}")
            return None
        finally:
            conn.close()
    
    def get_database_stats(self):
        """Get database statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Count total records
            cursor.execute("SELECT COUNT(*) FROM assessments")
            total_records = cursor.fetchone()[0]
            
            # Get database file size
            db_size_bytes = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
            db_size_mb = db_size_bytes / (1024 * 1024)
            
            # Get last update
            cursor.execute("SELECT MAX(created_at) FROM assessments")
            last_update = cursor.fetchone()[0]
            
            return {
                'total_records': total_records,
                'db_size_mb': db_size_mb,
                'last_update': last_update if last_update else 'Never'
            }
        except sqlite3.Error as e:
            print(f"Error getting database stats: {e}")
            return {
                'total_records': 0,
                'db_size_mb': 0,
                'last_update': 'Error'
            }
        finally:
            conn.close()
    
    def clear_all_records(self):
        """Clear all assessment records."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("DELETE FROM assessments")
            cursor.execute("DELETE FROM model_performance")
            conn.commit()
            return True
        except sqlite3.Error as e:
            print(f"Error clearing records: {e}")
            return False
        finally:
            conn.close()
    
    def reset_database(self):
        """Reset the entire database."""
        try:
            if os.path.exists(self.db_path):
                os.remove(self.db_path)
            self.initialize_database()
            return True
        except Exception as e:
            print(f"Error resetting database: {e}")
            return False
    
    def export_to_csv(self, filename=None):
        """Export all assessments to CSV."""
        if filename is None:
            filename = f"ahf_assessments_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        conn = sqlite3.connect(self.db_path)
        try:
            df = pd.read_sql_query("SELECT * FROM assessments", conn)
            df.to_csv(filename, index=False)
            return filename
        except sqlite3.Error as e:
            print(f"Error exporting to CSV: {e}")
            return None
        finally:
            conn.close()
