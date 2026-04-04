import os
import psycopg2
import psycopg2.extras
import pandas as pd
from datetime import datetime, timedelta
import json
import logging

# ─── Connection ───────────────────────────────────────────────────────────────
DATABASE_URL = os.environ.get('DATABASE_URL')

def get_connection():
    """Get a PostgreSQL connection."""
    if not DATABASE_URL:
        raise ValueError("DATABASE_URL environment variable not set.")
    conn = psycopg2.connect(DATABASE_URL)
    return conn


# ─── DatabaseManager ─────────────────────────────────────────────────────────
class DatabaseManager:
    """Manages PostgreSQL database operations for patient records and predictions."""

    def __init__(self, db_path=None):
        """Initialize database manager (db_path ignored — uses DATABASE_URL)."""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.initialize_database()

    # ── Schema ────────────────────────────────────────────────────────────────
    def initialize_database(self):
        """Create all tables and indexes if they don't exist."""
        conn = get_connection()
        cursor = conn.cursor()

        try:
            # assessments table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS assessments (
                    id SERIAL PRIMARY KEY,
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

            # model_performance table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_performance (
                    id SERIAL PRIMARY KEY,
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

            # patient_history table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS patient_history (
                    id SERIAL PRIMARY KEY,
                    patient_id TEXT NOT NULL UNIQUE,
                    first_assessment TEXT,
                    last_assessment TEXT,
                    total_assessments INTEGER DEFAULT 1,
                    highest_risk_score REAL,
                    latest_risk_level TEXT,
                    alert_count INTEGER DEFAULT 0,
                    last_alert_date TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # system_logs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS system_logs (
                    id SERIAL PRIMARY KEY,
                    log_level TEXT NOT NULL,
                    module TEXT NOT NULL,
                    message TEXT NOT NULL,
                    user_id TEXT,
                    patient_id TEXT,
                    additional_data TEXT,
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # data_quality_metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS data_quality_metrics (
                    id SERIAL PRIMARY KEY,
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

            # Indexes
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
            self.logger.info("Database initialized successfully.")

        except Exception as e:
            self.logger.error(f"Error initializing database: {e}")
            conn.rollback()
        finally:
            conn.close()

    # ── Assessments ───────────────────────────────────────────────────────────
    def save_assessment(self, record_data):
        """Save a patient assessment record."""
        conn = get_connection()
        cursor = conn.cursor()

        try:
            record_data['created_at'] = datetime.now().isoformat()
            record_data['updated_at'] = datetime.now().isoformat()

            # Calculate prediction confidence
            if 'lr_probability' in record_data and 'xgb_probability' in record_data:
                prob_diff = abs(record_data['lr_probability'] - record_data['xgb_probability'])
                record_data['prediction_confidence'] = 1.0 - prob_diff

            columns = list(record_data.keys())
            placeholders = ', '.join(['%s'] * len(columns))
            values = list(record_data.values())

            query = f"""
                INSERT INTO assessments ({', '.join(columns)})
                VALUES ({placeholders})
                RETURNING id
            """

            cursor.execute(query, values)
            assessment_id = cursor.fetchone()[0]

            self._update_patient_history(cursor, record_data)

            conn.commit()

            self.log_system_event(
                'INFO', 'assessment',
                f"Assessment saved for patient {record_data.get('patient_id')}",
                patient_id=record_data.get('patient_id')
            )

            return assessment_id

        except Exception as e:
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

        cursor.execute("""
            SELECT id, total_assessments, highest_risk_score
            FROM patient_history WHERE patient_id = %s
        """, (patient_id,))

        existing = cursor.fetchone()

        if existing:
            history_id, total_assessments, highest_risk = existing
            new_total = total_assessments + 1
            new_highest = max(highest_risk or 0, risk_score or 0)

            cursor.execute("""
                UPDATE patient_history
                SET last_assessment = %s,
                    total_assessments = %s,
                    highest_risk_score = %s,
                    latest_risk_level = %s,
                    updated_at = %s
                WHERE patient_id = %s
            """, (assessment_date, new_total, new_highest, risk_level,
                  datetime.now().isoformat(), patient_id))
        else:
            cursor.execute("""
                INSERT INTO patient_history
                (patient_id, first_assessment, last_assessment, total_assessments,
                 highest_risk_score, latest_risk_level, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, (patient_id, assessment_date, assessment_date, 1,
                  risk_score, risk_level,
                  datetime.now().isoformat(), datetime.now().isoformat()))

    def get_all_assessments(self):
        """Retrieve all assessment records."""
        conn = get_connection()
        try:
            query = """
                SELECT * FROM assessments
                WHERE validation_status != 'invalid'
                ORDER BY assessment_date DESC
            """
            df = pd.read_sql_query(query, conn)
            return df.to_dict('records') if not df.empty else []
        except Exception as e:
            self.logger.error(f"Error retrieving assessments: {e}")
            return []
        finally:
            conn.close()

    def get_assessments_by_date_range(self, start_date, end_date):
        """Get assessments within date range."""
        conn = get_connection()
        try:
            query = """
                SELECT * FROM assessments
                WHERE assessment_date BETWEEN %s AND %s
                AND validation_status != 'invalid'
                ORDER BY assessment_date DESC
            """
            df = pd.read_sql_query(query, conn, params=[start_date, end_date])
            return df.to_dict('records') if not df.empty else []
        except Exception as e:
            self.logger.error(f"Error retrieving assessments by date range: {e}")
            return []
        finally:
            conn.close()

    def get_assessment_by_patient_id(self, patient_id):
        """Retrieve assessments for a specific patient."""
        conn = get_connection()
        try:
            query = """
                SELECT * FROM assessments
                WHERE patient_id = %s
                AND validation_status != 'invalid'
                ORDER BY assessment_date DESC
            """
            df = pd.read_sql_query(query, conn, params=[patient_id])
            return df.to_dict('records') if not df.empty else []
        except Exception as e:
            self.logger.error(f"Error retrieving patient assessments: {e}")
            return []
        finally:
            conn.close()

    def get_high_risk_patients(self, risk_threshold=0.7, hours=24):
        """Get high-risk patients within specified timeframe."""
        conn = get_connection()
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            query = """
                SELECT * FROM assessments
                WHERE ensemble_probability >= %s
                AND assessment_date >= %s
                AND validation_status != 'invalid'
                ORDER BY ensemble_probability DESC
            """
            df = pd.read_sql_query(query, conn, params=[risk_threshold, cutoff_time.isoformat()])
            return df.to_dict('records') if not df.empty else []
        except Exception as e:
            self.logger.error(f"Error retrieving high-risk patients: {e}")
            return []
        finally:
            conn.close()

    def get_patient_history_summary(self, patient_id):
        """Get comprehensive patient history."""
        conn = get_connection()
        try:
            history_df = pd.read_sql_query(
                "SELECT * FROM patient_history WHERE patient_id = %s",
                conn, params=[patient_id]
            )
            assessments_df = pd.read_sql_query("""
                SELECT assessment_date, ensemble_probability, risk_level,
                       nt_probnp, weight, b_line_score
                FROM assessments
                WHERE patient_id = %s AND validation_status != 'invalid'
                ORDER BY assessment_date DESC LIMIT 10
            """, conn, params=[patient_id])

            return {
                'history': history_df.to_dict('records')[0] if not history_df.empty else None,
                'recent_assessments': assessments_df.to_dict('records') if not assessments_df.empty else []
            }
        except Exception as e:
            self.logger.error(f"Error retrieving patient history: {e}")
            return {'history': None, 'recent_assessments': []}
        finally:
            conn.close()

    # ── Model Performance ─────────────────────────────────────────────────────
    def save_model_performance(self, model_name, metrics, model_version="1.0"):
        """Save model performance metrics."""
        conn = get_connection()
        cursor = conn.cursor()

        try:
            feature_importance_json = None
            if 'feature_importance' in metrics:
                feature_importance_json = json.dumps(metrics['feature_importance'])

            cursor.execute("""
                INSERT INTO model_performance
                (model_name, model_version, accuracy, auc, sensitivity, specificity,
                 precision_score, recall, f1_score, ppv, npv, training_date,
                 validation_auc, validation_accuracy, feature_importance)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, [
                model_name, model_version,
                metrics.get('accuracy'), metrics.get('auc'),
                metrics.get('sensitivity'), metrics.get('specificity'),
                metrics.get('precision'), metrics.get('recall'),
                metrics.get('f1'), metrics.get('ppv'), metrics.get('npv'),
                datetime.now().isoformat(),
                metrics.get('validation', {}).get('auc') if 'validation' in metrics else None,
                metrics.get('validation', {}).get('accuracy') if 'validation' in metrics else None,
                feature_importance_json
            ])

            conn.commit()
            return True

        except Exception as e:
            self.logger.error(f"Error saving model performance: {e}")
            conn.rollback()
            return None
        finally:
            conn.close()

    def get_latest_model_performance(self, model_name):
        """Get latest performance metrics for a model."""
        conn = get_connection()
        try:
            df = pd.read_sql_query("""
                SELECT * FROM model_performance
                WHERE model_name = %s
                ORDER BY created_at DESC LIMIT 1
            """, conn, params=[model_name])

            if not df.empty:
                result = df.iloc[0].to_dict()
                if result.get('feature_importance'):
                    try:
                        result['feature_importance'] = json.loads(result['feature_importance'])
                    except json.JSONDecodeError:
                        result['feature_importance'] = None
                return result
            return None
        except Exception as e:
            self.logger.error(f"Error retrieving model performance: {e}")
            return None
        finally:
            conn.close()

    def get_model_performance_trends(self, model_name, days=30):
        """Get model performance trends over time."""
        conn = get_connection()
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            df = pd.read_sql_query("""
                SELECT training_date, accuracy, auc, sensitivity, specificity, f1_score
                FROM model_performance
                WHERE model_name = %s AND created_at >= %s
                ORDER BY created_at ASC
            """, conn, params=[model_name, cutoff_date.isoformat()])
            return df.to_dict('records') if not df.empty else []
        except Exception as e:
            self.logger.error(f"Error retrieving model performance trends: {e}")
            return []
        finally:
            conn.close()

    # ── Logging ───────────────────────────────────────────────────────────────
    def log_system_event(self, log_level, module, message, user_id=None, patient_id=None, additional_data=None):
        """Log system events for audit trail."""
        conn = get_connection()
        cursor = conn.cursor()
        try:
            additional_data_json = json.dumps(additional_data) if additional_data else None
            cursor.execute("""
                INSERT INTO system_logs
                (log_level, module, message, user_id, patient_id, additional_data, timestamp)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (log_level, module, message, user_id, patient_id,
                  additional_data_json, datetime.now().isoformat()))
            conn.commit()
        except Exception as e:
            self.logger.error(f"Error logging system event: {e}")
        finally:
            conn.close()

    def get_system_logs(self, hours=24, log_level=None):
        """Retrieve system logs."""
        conn = get_connection()
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            if log_level:
                df = pd.read_sql_query("""
                    SELECT * FROM system_logs
                    WHERE timestamp >= %s AND log_level = %s
                    ORDER BY timestamp DESC LIMIT 1000
                """, conn, params=[cutoff_time.isoformat(), log_level])
            else:
                df = pd.read_sql_query("""
                    SELECT * FROM system_logs
                    WHERE timestamp >= %s
                    ORDER BY timestamp DESC LIMIT 1000
                """, conn, params=[cutoff_time.isoformat()])
            return df.to_dict('records') if not df.empty else []
        except Exception as e:
            self.logger.error(f"Error retrieving system logs: {e}")
            return []
        finally:
            conn.close()

    # ── Stats & Quality ───────────────────────────────────────────────────────
    def calculate_data_quality_metrics(self):
        """Calculate and store data quality metrics."""
        conn = get_connection()
        cursor = conn.cursor()

        try:
            cutoff_date = datetime.now() - timedelta(days=7)
            cursor.execute("""
                SELECT COUNT(*) as total,
                       SUM(CASE WHEN validation_status = 'valid' THEN 1 ELSE 0 END) as valid,
                       SUM(CASE WHEN validation_warnings IS NOT NULL AND validation_warnings != '' THEN 1 ELSE 0 END) as with_warnings
                FROM assessments WHERE created_at >= %s
            """, (cutoff_date.isoformat(),))

            counts = cursor.fetchone()
            total_records, valid_records, records_with_warnings = counts

            if total_records and total_records > 0:
                completeness_percentage = (valid_records / total_records) * 100
                quality_score = max(0, completeness_percentage - ((records_with_warnings or 0) / total_records * 10))

                cursor.execute("""
                    INSERT INTO data_quality_metrics
                    (date_calculated, total_records, valid_records, records_with_warnings,
                     completeness_percentage, quality_score)
                    VALUES (%s, %s, %s, %s, %s, %s)
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

        except Exception as e:
            self.logger.error(f"Error calculating data quality metrics: {e}")
            return None
        finally:
            conn.close()

    def get_database_stats(self):
        """Get comprehensive database statistics."""
        conn = get_connection()
        cursor = conn.cursor()

        try:
            stats = {}

            cursor.execute("SELECT COUNT(*) FROM assessments")
            stats['total_assessments'] = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(DISTINCT patient_id) FROM assessments")
            stats['unique_patients'] = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM model_performance")
            stats['model_trainings'] = cursor.fetchone()[0]

            stats['db_size_mb'] = 0  # N/A for cloud DB

            cursor.execute("SELECT MAX(created_at) FROM assessments")
            stats['last_assessment'] = cursor.fetchone()[0] or 'Never'

            cursor.execute("""
                SELECT risk_level, COUNT(*)
                FROM assessments WHERE validation_status != 'invalid'
                GROUP BY risk_level
            """)
            stats['risk_distribution'] = dict(cursor.fetchall())

            cutoff_24h = datetime.now() - timedelta(hours=24)
            cursor.execute("""
                SELECT COUNT(*) FROM assessments WHERE created_at >= %s
            """, (cutoff_24h.isoformat(),))
            stats['assessments_24h'] = cursor.fetchone()[0]

            quality_metrics = self.calculate_data_quality_metrics()
            if quality_metrics:
                stats['data_quality'] = quality_metrics

            return stats

        except Exception as e:
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

    # ── Utilities ─────────────────────────────────────────────────────────────
    def clear_all_records(self):
        """Clear all assessment records while preserving structure."""
        conn = get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("DELETE FROM assessments")
            cursor.execute("DELETE FROM patient_history")
            cursor.execute("DELETE FROM model_performance")
            cursor.execute("DELETE FROM data_quality_metrics")
            conn.commit()
            self.log_system_event('WARNING', 'database', 'All records cleared from database')
            return True
        except Exception as e:
            self.logger.error(f"Error clearing records: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()

    def export_to_csv(self, filename=None, table='assessments'):
        """Export specified table to CSV."""
        if filename is None:
            filename = f"{table}_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        conn = get_connection()
        try:
            if table == 'assessments':
                query = "SELECT * FROM assessments WHERE validation_status != 'invalid'"
            else:
                query = f"SELECT * FROM {table}"
            df = pd.read_sql_query(query, conn)
            df.to_csv(filename, index=False)
            self.log_system_event('INFO', 'export', f"Data exported to {filename}")
            return filename
        except Exception as e:
            self.logger.error(f"Error exporting to CSV: {e}")
            return None
        finally:
            conn.close()

    def backup_database(self, backup_path=None):
        """For PostgreSQL — export all data to CSV as backup."""
        return self.export_to_csv(
            filename=backup_path or f"ahf_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )


# ─── Backward-compatible helper ───────────────────────────────────────────────
def get_db_manager():
    return DatabaseManager()
