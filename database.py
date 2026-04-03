import sqlite3
import pandas as pd
import os
from datetime import datetime, timedelta
import json
import logging

class DatabaseManager:
    """Manages SQLite database operations for patient records and predictions."""
    
    def __init__(self, db_path="ahf_predictions.db"):
        """Initialize database manager."""
        self.db_path = db_path
        self.initialize_database()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
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
                validation_status TEXT DEFAULT 'valid',
                validation_warnings TEXT,
                prediction_confidence REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create model_performance table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                model_version TEXT,
                accuracy REAL,
                auc REAL,
                sensitivity REAL,
                specificity REAL,
                precision_score REAL,
                recall REAL,
                f1_score REAL,
                ppv REAL,
                npv REAL,
                training_date TEXT,
                validation_auc REAL,
                validation_accuracy REAL,
                feature_importance TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create patient_history table for tracking patient over time
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS patient_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT NOT NULL,
                first_assessment TEXT,
                last_assessment TEXT,
                total_assessments INTEGER DEFAULT 1,
                highest_risk_score REAL,
                latest_risk_level TEXT,
                alert_count INTEGER DEFAULT 0,
                last_alert_date TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(patient_id)
            )
        """)
        
        # Create system_logs table for audit trail
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS system_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                log_level TEXT NOT NULL,
                module TEXT NOT NULL,
                message TEXT NOT NULL,
                user_id TEXT,
                patient_id TEXT,
                additional_data TEXT,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create data_quality_metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS data_quality_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date_calculated TEXT NOT NULL,
                total_records INTEGER,
                valid_records INTEGER,
                records_with_warnings INTEGER,
                completeness_percentage REAL,
                data_drift_score REAL,
                anomaly_count INTEGER,
                quality_score REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indexes for better performance
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_assessments_patient_id 
            ON assessments(patient_id)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_assessments_date 
            ON assessments(assessment_date)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_assessments_risk_level 
            ON assessments(risk_level)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_system_logs_timestamp 
            ON system_logs(timestamp)
        """)
        
        conn.commit()
        conn.close()
    
    def save_assessment(self, record_data):
        """Save a patient assessment record with enhanced data tracking."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Add timestamp if not provided
            if 'created_at' not in record_data:
                record_data['created_at'] = datetime.now().isoformat()
            
            record_data['updated_at'] = datetime.now().isoformat()
            
            # Calculate prediction confidence if probabilities are available
            if 'lr_probability' in record_data and 'xgb_probability' in record_data:
                prob_diff = abs(record_data['lr_probability'] - record_data['xgb_probability'])
                record_data['prediction_confidence'] = 1.0 - prob_diff
            
            columns = list(record_data.keys())
            placeholders = ', '.join(['?' for _ in columns])
            values = list(record_data.values())
            
            query = f"""
                INSERT INTO assessments ({', '.join(columns)})
                VALUES ({placeholders})
            """
            
            cursor.execute(query, values)
            assessment_id = cursor.lastrowid
            
            # Update or create patient history
            self._update_patient_history(cursor, record_data)
            
            conn.commit()
            
            # Log the assessment
            self.log_system_event(
                'INFO', 
                'assessment', 
                f"Assessment saved for patient {record_data.get('patient_id')}", 
                patient_id=record_data.get('patient_id')
            )
            
            return assessment_id
            
        except sqlite3.Error as e:
            self.logger.error(f"Error saving assessment: {e}")
            conn.rollback()
            return None
        finally:
            conn.close()
    
    def _update_patient_history(self, cursor, record_data):
        """Update patient history tracking."""
        patient_id = record_data.get('patient_id')
        assessment_date = record_data.get('assessment_date', datetime.now().isoformat())
        risk_score = record_data.get('ensemble_probability', 0)
        risk_level = record_data.get('risk_level', 'Unknown')
        
        # Check if patient history exists
        cursor.execute("""
            SELECT id, total_assessments, highest_risk_score, first_assessment
            FROM patient_history 
            WHERE patient_id = ?
        """, (patient_id,))
        
        existing = cursor.fetchone()
        
        if existing:
            # Update existing record
            history_id, total_assessments, highest_risk, first_assessment = existing
            new_total = total_assessments + 1
            new_highest = max(highest_risk or 0, risk_score or 0)
            
            cursor.execute("""
                UPDATE patient_history 
                SET last_assessment = ?, 
                    total_assessments = ?,
                    highest_risk_score = ?,
                    latest_risk_level = ?,
                    updated_at = ?
                WHERE patient_id = ?
            """, (assessment_date, new_total, new_highest, risk_level, 
                  datetime.now().isoformat(), patient_id))
        else:
            # Create new patient history record
            cursor.execute("""
                INSERT INTO patient_history 
                (patient_id, first_assessment, last_assessment, total_assessments, 
                 highest_risk_score, latest_risk_level, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (patient_id, assessment_date, assessment_date, 1, 
                  risk_score, risk_level, datetime.now().isoformat(), 
                  datetime.now().isoformat()))
    
    def get_all_assessments(self):
        """Retrieve all assessment records with enhanced filtering."""
        conn = sqlite3.connect(self.db_path)
        try:
            query = """
                SELECT * FROM assessments 
                WHERE validation_status != 'invalid'
                ORDER BY assessment_date DESC
            """
            df = pd.read_sql_query(query, conn)
            return df.to_dict('records') if not df.empty else []
        except sqlite3.Error as e:
            self.logger.error(f"Error retrieving assessments: {e}")
            return []
        finally:
            conn.close()
    
    def get_assessments_by_date_range(self, start_date, end_date):
        """Get assessments within date range."""
        conn = sqlite3.connect(self.db_path)
        try:
            query = """
                SELECT * FROM assessments 
                WHERE assessment_date BETWEEN ? AND ?
                AND validation_status != 'invalid'
                ORDER BY assessment_date DESC
            """
            df = pd.read_sql_query(query, conn, params=[start_date, end_date])
            return df.to_dict('records') if not df.empty else []
        except sqlite3.Error as e:
            self.logger.error(f"Error retrieving assessments by date range: {e}")
            return []
        finally:
            conn.close()
    
    def get_assessment_by_patient_id(self, patient_id):
        """Retrieve assessments for a specific patient."""
        conn = sqlite3.connect(self.db_path)
        try:
            query = """
                SELECT * FROM assessments 
                WHERE patient_id = ? 
                AND validation_status != 'invalid'
                ORDER BY assessment_date DESC
            """
            df = pd.read_sql_query(query, conn, params=[patient_id])
            return df.to_dict('records') if not df.empty else []
        except sqlite3.Error as e:
            self.logger.error(f"Error retrieving patient assessments: {e}")
            return []
        finally:
            conn.close()
    
    def get_high_risk_patients(self, risk_threshold=0.7, hours=24):
        """Get high-risk patients within specified timeframe."""
        conn = sqlite3.connect(self.db_path)
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            query = """
                SELECT * FROM assessments 
                WHERE ensemble_probability >= ? 
                AND assessment_date >= ?
                AND validation_status != 'invalid'
                ORDER BY ensemble_probability DESC
            """
            df = pd.read_sql_query(query, conn, params=[risk_threshold, cutoff_time.isoformat()])
            return df.to_dict('records') if not df.empty else []
        except sqlite3.Error as e:
            self.logger.error(f"Error retrieving high-risk patients: {e}")
            return []
        finally:
            conn.close()
    
    def get_patient_history_summary(self, patient_id):
        """Get comprehensive patient history."""
        conn = sqlite3.connect(self.db_path)
        try:
            # Get patient history record
            history_query = """
                SELECT * FROM patient_history WHERE patient_id = ?
            """
            history_df = pd.read_sql_query(history_query, conn, params=[patient_id])
            
            # Get recent assessments
            assessments_query = """
                SELECT assessment_date, ensemble_probability, risk_level, 
                       nt_probnp, weight, b_line_score
                FROM assessments 
                WHERE patient_id = ? 
                AND validation_status != 'invalid'
                ORDER BY assessment_date DESC
                LIMIT 10
            """
            assessments_df = pd.read_sql_query(assessments_query, conn, params=[patient_id])
            
            return {
                'history': history_df.to_dict('records')[0] if not history_df.empty else None,
                'recent_assessments': assessments_df.to_dict('records') if not assessments_df.empty else []
            }
        except sqlite3.Error as e:
            self.logger.error(f"Error retrieving patient history: {e}")
            return {'history': None, 'recent_assessments': []}
        finally:
            conn.close()
    
    def save_model_performance(self, model_name, metrics, model_version="1.0"):
        """Save enhanced model performance metrics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Serialize feature importance if available
            feature_importance_json = None
            if 'feature_importance' in metrics:
                feature_importance_json = json.dumps(metrics['feature_importance'])
            
            query = """
                INSERT INTO model_performance 
                (model_name, model_version, accuracy, auc, sensitivity, specificity, 
                 precision_score, recall, f1_score, ppv, npv, training_date,
                 validation_auc, validation_accuracy, feature_importance)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            
            values = [
                model_name,
                model_version,
                metrics.get('accuracy'),
                metrics.get('auc'),
                metrics.get('sensitivity'),
                metrics.get('specificity'),
                metrics.get('precision'),
                metrics.get('recall'),
                metrics.get('f1'),
                metrics.get('ppv'),
                metrics.get('npv'),
                datetime.now().isoformat(),
                metrics.get('validation', {}).get('auc') if 'validation' in metrics else None,
                metrics.get('validation', {}).get('accuracy') if 'validation' in metrics else None,
                feature_importance_json
            ]
            
            cursor.execute(query, values)
            conn.commit()
            
            self.log_system_event(
                'INFO', 
                'model_performance', 
                f"Performance metrics saved for {model_name}"
            )
            
            return cursor.lastrowid
            
        except sqlite3.Error as e:
            self.logger.error(f"Error saving model performance: {e}")
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
                result = df.iloc[0].to_dict()
                # Parse feature importance JSON if present
                if result.get('feature_importance'):
                    try:
                        result['feature_importance'] = json.loads(result['feature_importance'])
                    except json.JSONDecodeError:
                        result['feature_importance'] = None
                return result
            return None
        except sqlite3.Error as e:
            self.logger.error(f"Error retrieving model performance: {e}")
            return None
        finally:
            conn.close()
    
    def get_model_performance_trends(self, model_name, days=30):
        """Get model performance trends over time."""
        conn = sqlite3.connect(self.db_path)
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            query = """
                SELECT training_date, accuracy, auc, sensitivity, specificity, f1_score
                FROM model_performance 
                WHERE model_name = ? AND created_at >= ?
                ORDER BY created_at ASC
            """
            df = pd.read_sql_query(query, conn, params=[model_name, cutoff_date.isoformat()])
            return df.to_dict('records') if not df.empty else []
        except sqlite3.Error as e:
            self.logger.error(f"Error retrieving model performance trends: {e}")
            return []
        finally:
            conn.close()
    
    def log_system_event(self, log_level, module, message, user_id=None, patient_id=None, additional_data=None):
        """Log system events for audit trail."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            additional_data_json = json.dumps(additional_data) if additional_data else None
            
            cursor.execute("""
                INSERT INTO system_logs 
                (log_level, module, message, user_id, patient_id, additional_data, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (log_level, module, message, user_id, patient_id, 
                  additional_data_json, datetime.now().isoformat()))
            
            conn.commit()
        except sqlite3.Error as e:
            self.logger.error(f"Error logging system event: {e}")
        finally:
            conn.close()
    
    def get_system_logs(self, hours=24, log_level=None):
        """Retrieve system logs."""
        conn = sqlite3.connect(self.db_path)
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            if log_level:
                query = """
                    SELECT * FROM system_logs 
                    WHERE timestamp >= ? AND log_level = ?
                    ORDER BY timestamp DESC
                    LIMIT 1000
                """
                params = [cutoff_time.isoformat(), log_level]
            else:
                query = """
                    SELECT * FROM system_logs 
                    WHERE timestamp >= ?
                    ORDER BY timestamp DESC
                    LIMIT 1000
                """
                params = [cutoff_time.isoformat()]
            
            df = pd.read_sql_query(query, conn, params=params)
            return df.to_dict('records') if not df.empty else []
        except sqlite3.Error as e:
            self.logger.error(f"Error retrieving system logs: {e}")
            return []
        finally:
            conn.close()
    
    def calculate_data_quality_metrics(self):
        """Calculate and store data quality metrics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Get recent assessments for quality analysis
            cutoff_date = datetime.now() - timedelta(days=7)
            
            cursor.execute("""
                SELECT COUNT(*) as total,
                       SUM(CASE WHEN validation_status = 'valid' THEN 1 ELSE 0 END) as valid,
                       SUM(CASE WHEN validation_warnings IS NOT NULL AND validation_warnings != '' THEN 1 ELSE 0 END) as with_warnings
                FROM assessments 
                WHERE created_at >= ?
            """, (cutoff_date.isoformat(),))
            
            counts = cursor.fetchone()
            total_records, valid_records, records_with_warnings = counts
            
            if total_records > 0:
                completeness_percentage = (valid_records / total_records) * 100
                quality_score = max(0, (completeness_percentage - (records_with_warnings / total_records * 10)))
                
                # Store quality metrics
                cursor.execute("""
                    INSERT INTO data_quality_metrics 
                    (date_calculated, total_records, valid_records, records_with_warnings, 
                     completeness_percentage, quality_score)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (datetime.now().isoformat(), total_records, valid_records, 
                      records_with_warnings, completeness_percentage, quality_score))
                
                conn.commit()
                
                return {
                    'total_records': total_records,
                    'valid_records': valid_records,
                    'records_with_warnings': records_with_warnings,
                    'completeness_percentage': completeness_percentage,
                    'quality_score': quality_score
                }
            
            return None
            
        except sqlite3.Error as e:
            self.logger.error(f"Error calculating data quality metrics: {e}")
            return None
        finally:
            conn.close()
    
    def get_database_stats(self):
        """Get comprehensive database statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            stats = {}
            
            # Basic counts
            cursor.execute("SELECT COUNT(*) FROM assessments")
            stats['total_assessments'] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(DISTINCT patient_id) FROM assessments")
            stats['unique_patients'] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM model_performance")
            stats['model_trainings'] = cursor.fetchone()[0]
            
            # Database file size
            if os.path.exists(self.db_path):
                db_size_bytes = os.path.getsize(self.db_path)
                stats['db_size_mb'] = db_size_bytes / (1024 * 1024)
            else:
                stats['db_size_mb'] = 0
            
            # Last update
            cursor.execute("SELECT MAX(created_at) FROM assessments")
            stats['last_assessment'] = cursor.fetchone()[0] or 'Never'
            
            # Risk distribution
            cursor.execute("""
                SELECT risk_level, COUNT(*) 
                FROM assessments 
                WHERE validation_status != 'invalid'
                GROUP BY risk_level
            """)
            
            risk_distribution = dict(cursor.fetchall())
            stats['risk_distribution'] = risk_distribution
            
            # Recent activity (last 24 hours)
            cutoff_24h = datetime.now() - timedelta(hours=24)
            cursor.execute("""
                SELECT COUNT(*) FROM assessments 
                WHERE created_at >= ?
            """, (cutoff_24h.isoformat(),))
            
            stats['assessments_24h'] = cursor.fetchone()[0]
            
            # Data quality summary
            quality_metrics = self.calculate_data_quality_metrics()
            if quality_metrics:
                stats['data_quality'] = quality_metrics
            
            return stats
            
        except sqlite3.Error as e:
            self.logger.error(f"Error getting database stats: {e}")
            return {
                'total_assessments': 0,
                'unique_patients': 0,
                'db_size_mb': 0,
                'last_assessment': 'Error',
                'error': str(e)
            }
        finally:
            conn.close()
    
    def clear_all_records(self):
        """Clear all assessment records while preserving structure."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("DELETE FROM assessments")
            cursor.execute("DELETE FROM patient_history")
            cursor.execute("DELETE FROM model_performance")
            cursor.execute("DELETE FROM data_quality_metrics")
            # Keep system_logs for audit trail
            
            conn.commit()
            
            self.log_system_event(
                'WARNING', 
                'database', 
                'All records cleared from database'
            )
            
            return True
        except sqlite3.Error as e:
            self.logger.error(f"Error clearing records: {e}")
            return False
        finally:
            conn.close()
    
    def export_to_csv(self, filename=None, table='assessments'):
        """Export specified table to CSV."""
        if filename is None:
            filename = f"{table}_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        conn = sqlite3.connect(self.db_path)
        try:
            if table == 'assessments':
                # Export only valid assessments
                query = "SELECT * FROM assessments WHERE validation_status != 'invalid'"
            else:
                query = f"SELECT * FROM {table}"
            
            df = pd.read_sql_query(query, conn)
            df.to_csv(filename, index=False)
            
            self.log_system_event(
                'INFO', 
                'export', 
                f"Data exported to {filename}"
            )
            
            return filename
        except (sqlite3.Error, Exception) as e:
            self.logger.error(f"Error exporting to CSV: {e}")
            return None
        finally:
            conn.close()
    
    def backup_database(self, backup_path=None):
        """Create database backup."""
        if backup_path is None:
            backup_path = f"ahf_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
        
        try:
            source_conn = sqlite3.connect(self.db_path)
            backup_conn = sqlite3.connect(backup_path)
            
            source_conn.backup(backup_conn)
            
            backup_conn.close()
            source_conn.close()
            
            self.log_system_event(
                'INFO', 
                'backup', 
                f"Database backed up to {backup_path}"
            )
            
            return backup_path
        except sqlite3.Error as e:
            self.logger.error(f"Error creating database backup: {e}")
            return None

