import os
import pandas as pd
from datetime import datetime, timedelta
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATABASE_URL = os.environ.get('DATABASE_URL')

# ── Detect which DB to use ────────────────────────────────────────────────────
USE_POSTGRES = False
if DATABASE_URL:
    try:
        import psycopg2
        import psycopg2.extras
        test_conn = psycopg2.connect(DATABASE_URL, connect_timeout=5)
        test_conn.close()
        USE_POSTGRES = True
        logger.info("✅ PostgreSQL connection successful.")
    except Exception as e:
        logger.warning(f"⚠️ PostgreSQL failed ({e}). Falling back to SQLite.")
        USE_POSTGRES = False
else:
    logger.info("No DATABASE_URL found. Using SQLite.")

# ── Connection helpers ────────────────────────────────────────────────────────
def get_connection():
    if USE_POSTGRES:
        import psycopg2
        return psycopg2.connect(DATABASE_URL)
    else:
        import sqlite3
        return sqlite3.connect("ahf_predictions.db", check_same_thread=False)

def placeholder():
    """Return correct SQL placeholder for current DB."""
    return "%s" if USE_POSTGRES else "?"

def serial_pk():
    return "SERIAL PRIMARY KEY" if USE_POSTGRES else "INTEGER PRIMARY KEY AUTOINCREMENT"


# ── DatabaseManager ───────────────────────────────────────────────────────────
class DatabaseManager:
    """Manages database operations — works with both PostgreSQL and SQLite."""

    def __init__(self, db_path=None):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.initialize_database()

    def initialize_database(self):
        conn = get_connection()
        cursor = conn.cursor()
        pk = serial_pk()

        try:
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS assessments (
                    id {pk},
                    patient_id TEXT NOT NULL,
                    assessment_date TEXT NOT NULL,
                    age INTEGER, gender TEXT, weight REAL,
                    nt_probnp REAL, creatinine REAL, b_line_score INTEGER,
                    ivc_collapsibility REAL, ejection_fraction REAL,
                    systolic_bp INTEGER, heart_rate INTEGER,
                    diabetes INTEGER, hypertension INTEGER, ckd INTEGER, afib INTEGER,
                    lr_probability REAL, xgb_probability REAL, ensemble_probability REAL,
                    risk_level TEXT,
                    validation_status TEXT DEFAULT 'valid',
                    validation_warnings TEXT,
                    prediction_confidence REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS model_performance (
                    id {pk},
                    model_name TEXT NOT NULL, model_version TEXT,
                    accuracy REAL, auc REAL, sensitivity REAL, specificity REAL,
                    precision_score REAL, recall REAL, f1_score REAL,
                    ppv REAL, npv REAL, training_date TEXT,
                    validation_auc REAL, validation_accuracy REAL,
                    feature_importance TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS patient_history (
                    id {pk},
                    patient_id TEXT NOT NULL UNIQUE,
                    first_assessment TEXT, last_assessment TEXT,
                    total_assessments INTEGER DEFAULT 1,
                    highest_risk_score REAL, latest_risk_level TEXT,
                    alert_count INTEGER DEFAULT 0, last_alert_date TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS system_logs (
                    id {pk},
                    log_level TEXT NOT NULL, module TEXT NOT NULL,
                    message TEXT NOT NULL, user_id TEXT, patient_id TEXT,
                    additional_data TEXT,
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS data_quality_metrics (
                    id {pk},
                    date_calculated TEXT NOT NULL,
                    total_records INTEGER, valid_records INTEGER,
                    records_with_warnings INTEGER,
                    completeness_percentage REAL, data_drift_score REAL,
                    anomaly_count INTEGER, quality_score REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Indexes (ignore if already exist)
            for idx_sql in [
                "CREATE INDEX IF NOT EXISTS idx_assessments_patient_id ON assessments(patient_id)",
                "CREATE INDEX IF NOT EXISTS idx_assessments_date ON assessments(assessment_date)",
                "CREATE INDEX IF NOT EXISTS idx_assessments_risk_level ON assessments(risk_level)",
                "CREATE INDEX IF NOT EXISTS idx_system_logs_timestamp ON system_logs(timestamp)",
            ]:
                try:
                    cursor.execute(idx_sql)
                except Exception:
                    pass

            conn.commit()
            self.logger.info("Database initialized successfully.")

        except Exception as e:
            self.logger.error(f"Error initializing database: {e}")
            conn.rollback()
        finally:
            conn.close()

    def _ph(self):
        return placeholder()

    def save_assessment(self, record_data):
        conn = get_connection()
        cursor = conn.cursor()
        ph = self._ph()

        try:
            record_data['created_at'] = datetime.now().isoformat()
            record_data['updated_at'] = datetime.now().isoformat()

            if 'lr_probability' in record_data and 'xgb_probability' in record_data:
                prob_diff = abs(record_data['lr_probability'] - record_data['xgb_probability'])
                record_data['prediction_confidence'] = 1.0 - prob_diff

            columns = list(record_data.keys())
            placeholders = ', '.join([ph] * len(columns))
            values = list(record_data.values())

            if USE_POSTGRES:
                query = f"INSERT INTO assessments ({', '.join(columns)}) VALUES ({placeholders}) RETURNING id"
                cursor.execute(query, values)
                assessment_id = cursor.fetchone()[0]
            else:
                query = f"INSERT INTO assessments ({', '.join(columns)}) VALUES ({placeholders})"
                cursor.execute(query, values)
                assessment_id = cursor.lastrowid

            self._update_patient_history(cursor, record_data)
            conn.commit()
            return assessment_id

        except Exception as e:
            self.logger.error(f"Error saving assessment: {e}")
            conn.rollback()
            return None
        finally:
            conn.close()

    def _update_patient_history(self, cursor, record_data):
        ph = self._ph()
        patient_id = record_data.get('patient_id')
        assessment_date = record_data.get('assessment_date', datetime.now().isoformat())
        risk_score = record_data.get('ensemble_probability', 0)
        risk_level = record_data.get('risk_level', 'Unknown')

        cursor.execute(
            f"SELECT id, total_assessments, highest_risk_score FROM patient_history WHERE patient_id = {ph}",
            (patient_id,)
        )
        existing = cursor.fetchone()

        if existing:
            history_id, total_assessments, highest_risk = existing
            cursor.execute(f"""
                UPDATE patient_history
                SET last_assessment={ph}, total_assessments={ph},
                    highest_risk_score={ph}, latest_risk_level={ph}, updated_at={ph}
                WHERE patient_id={ph}
            """, (assessment_date, total_assessments + 1,
                  max(highest_risk or 0, risk_score or 0),
                  risk_level, datetime.now().isoformat(), patient_id))
        else:
            now = datetime.now().isoformat()
            cursor.execute(f"""
                INSERT INTO patient_history
                (patient_id, first_assessment, last_assessment, total_assessments,
                 highest_risk_score, latest_risk_level, created_at, updated_at)
                VALUES ({ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph})
            """, (patient_id, assessment_date, assessment_date, 1,
                  risk_score, risk_level, now, now))

    def get_all_assessments(self):
        conn = get_connection()
        try:
            df = pd.read_sql_query(
                "SELECT * FROM assessments WHERE validation_status != 'invalid' ORDER BY assessment_date DESC",
                conn
            )
            return df.to_dict('records') if not df.empty else []
        except Exception as e:
            self.logger.error(f"Error retrieving assessments: {e}")
            return []
        finally:
            conn.close()

    def get_assessments_by_date_range(self, start_date, end_date):
        conn = get_connection()
        ph = self._ph()
        try:
            df = pd.read_sql_query(
                f"SELECT * FROM assessments WHERE assessment_date BETWEEN {ph} AND {ph} AND validation_status != 'invalid' ORDER BY assessment_date DESC",
                conn, params=[start_date, end_date]
            )
            return df.to_dict('records') if not df.empty else []
        except Exception as e:
            self.logger.error(f"Error: {e}")
            return []
        finally:
            conn.close()

    def get_assessment_by_patient_id(self, patient_id):
        conn = get_connection()
        ph = self._ph()
        try:
            df = pd.read_sql_query(
                f"SELECT * FROM assessments WHERE patient_id = {ph} AND validation_status != 'invalid' ORDER BY assessment_date DESC",
                conn, params=[patient_id]
            )
            return df.to_dict('records') if not df.empty else []
        except Exception as e:
            self.logger.error(f"Error: {e}")
            return []
        finally:
            conn.close()

    def get_high_risk_patients(self, risk_threshold=0.7, hours=24):
        conn = get_connection()
        ph = self._ph()
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            df = pd.read_sql_query(
                f"SELECT * FROM assessments WHERE ensemble_probability >= {ph} AND assessment_date >= {ph} AND validation_status != 'invalid' ORDER BY ensemble_probability DESC",
                conn, params=[risk_threshold, cutoff_time.isoformat()]
            )
            return df.to_dict('records') if not df.empty else []
        except Exception as e:
            self.logger.error(f"Error: {e}")
            return []
        finally:
            conn.close()

    def get_patient_history_summary(self, patient_id):
        conn = get_connection()
        ph = self._ph()
        try:
            history_df = pd.read_sql_query(
                f"SELECT * FROM patient_history WHERE patient_id = {ph}",
                conn, params=[patient_id]
            )
            assessments_df = pd.read_sql_query(
                f"SELECT assessment_date, ensemble_probability, risk_level, nt_probnp, weight, b_line_score FROM assessments WHERE patient_id = {ph} AND validation_status != 'invalid' ORDER BY assessment_date DESC LIMIT 10",
                conn, params=[patient_id]
            )
            return {
                'history': history_df.to_dict('records')[0] if not history_df.empty else None,
                'recent_assessments': assessments_df.to_dict('records') if not assessments_df.empty else []
            }
        except Exception as e:
            self.logger.error(f"Error: {e}")
            return {'history': None, 'recent_assessments': []}
        finally:
            conn.close()

    def save_model_performance(self, model_name, metrics, model_version="1.0"):
        conn = get_connection()
        cursor = conn.cursor()
        ph = self._ph()
        try:
            feature_importance_json = json.dumps(metrics.get('feature_importance')) if 'feature_importance' in metrics else None
            cursor.execute(f"""
                INSERT INTO model_performance
                (model_name, model_version, accuracy, auc, sensitivity, specificity,
                 precision_score, recall, f1_score, ppv, npv, training_date,
                 validation_auc, validation_accuracy, feature_importance)
                VALUES ({ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph})
            """, [model_name, model_version,
                  metrics.get('accuracy'), metrics.get('auc'),
                  metrics.get('sensitivity'), metrics.get('specificity'),
                  metrics.get('precision'), metrics.get('recall'),
                  metrics.get('f1'), metrics.get('ppv'), metrics.get('npv'),
                  datetime.now().isoformat(),
                  metrics.get('validation', {}).get('auc') if 'validation' in metrics else None,
                  metrics.get('validation', {}).get('accuracy') if 'validation' in metrics else None,
                  feature_importance_json])
            conn.commit()
            return True
        except Exception as e:
            self.logger.error(f"Error saving model performance: {e}")
            conn.rollback()
            return None
        finally:
            conn.close()

    def get_latest_model_performance(self, model_name):
        conn = get_connection()
        ph = self._ph()
        try:
            df = pd.read_sql_query(
                f"SELECT * FROM model_performance WHERE model_name = {ph} ORDER BY created_at DESC LIMIT 1",
                conn, params=[model_name]
            )
            if not df.empty:
                result = df.iloc[0].to_dict()
                if result.get('feature_importance'):
                    try:
                        result['feature_importance'] = json.loads(result['feature_importance'])
                    except Exception:
                        result['feature_importance'] = None
                return result
            return None
        except Exception as e:
            self.logger.error(f"Error: {e}")
            return None
        finally:
            conn.close()

    def get_model_performance_trends(self, model_name, days=30):
        conn = get_connection()
        ph = self._ph()
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            df = pd.read_sql_query(
                f"SELECT training_date, accuracy, auc, sensitivity, specificity, f1_score FROM model_performance WHERE model_name = {ph} AND created_at >= {ph} ORDER BY created_at ASC",
                conn, params=[model_name, cutoff_date.isoformat()]
            )
            return df.to_dict('records') if not df.empty else []
        except Exception as e:
            self.logger.error(f"Error: {e}")
            return []
        finally:
            conn.close()

    def log_system_event(self, log_level, module, message, user_id=None, patient_id=None, additional_data=None):
        conn = get_connection()
        cursor = conn.cursor()
        ph = self._ph()
        try:
            additional_data_json = json.dumps(additional_data) if additional_data else None
            cursor.execute(
                f"INSERT INTO system_logs (log_level, module, message, user_id, patient_id, additional_data, timestamp) VALUES ({ph},{ph},{ph},{ph},{ph},{ph},{ph})",
                (log_level, module, message, user_id, patient_id, additional_data_json, datetime.now().isoformat())
            )
            conn.commit()
        except Exception as e:
            self.logger.error(f"Error logging: {e}")
        finally:
            conn.close()

    def get_system_logs(self, hours=24, log_level=None):
        conn = get_connection()
        ph = self._ph()
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            if log_level:
                df = pd.read_sql_query(
                    f"SELECT * FROM system_logs WHERE timestamp >= {ph} AND log_level = {ph} ORDER BY timestamp DESC LIMIT 1000",
                    conn, params=[cutoff_time.isoformat(), log_level]
                )
            else:
                df = pd.read_sql_query(
                    f"SELECT * FROM system_logs WHERE timestamp >= {ph} ORDER BY timestamp DESC LIMIT 1000",
                    conn, params=[cutoff_time.isoformat()]
                )
            return df.to_dict('records') if not df.empty else []
        except Exception as e:
            self.logger.error(f"Error: {e}")
            return []
        finally:
            conn.close()

    def calculate_data_quality_metrics(self):
        conn = get_connection()
        cursor = conn.cursor()
        ph = self._ph()
        try:
            cutoff_date = datetime.now() - timedelta(days=7)
            cursor.execute(f"""
                SELECT COUNT(*),
                       SUM(CASE WHEN validation_status = 'valid' THEN 1 ELSE 0 END),
                       SUM(CASE WHEN validation_warnings IS NOT NULL AND validation_warnings != '' THEN 1 ELSE 0 END)
                FROM assessments WHERE created_at >= {ph}
            """, (cutoff_date.isoformat(),))
            counts = cursor.fetchone()
            total_records, valid_records, records_with_warnings = counts
            if total_records and total_records > 0:
                completeness_percentage = (valid_records / total_records) * 100
                quality_score = max(0, completeness_percentage - ((records_with_warnings or 0) / total_records * 10))
                cursor.execute(f"""
                    INSERT INTO data_quality_metrics
                    (date_calculated, total_records, valid_records, records_with_warnings, completeness_percentage, quality_score)
                    VALUES ({ph},{ph},{ph},{ph},{ph},{ph})
                """, (datetime.now().isoformat(), total_records, valid_records, records_with_warnings, completeness_percentage, quality_score))
                conn.commit()
                return {'total_records': total_records, 'valid_records': valid_records,
                        'records_with_warnings': records_with_warnings,
                        'completeness_percentage': completeness_percentage, 'quality_score': quality_score}
            return None
        except Exception as e:
            self.logger.error(f"Error: {e}")
            return None
        finally:
            conn.close()

    def get_database_stats(self):
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
            stats['db_size_mb'] = 0
            cursor.execute("SELECT MAX(created_at) FROM assessments")
            stats['last_assessment'] = cursor.fetchone()[0] or 'Never'
            cursor.execute("SELECT risk_level, COUNT(*) FROM assessments WHERE validation_status != 'invalid' GROUP BY risk_level")
            stats['risk_distribution'] = dict(cursor.fetchall())
            cutoff_24h = datetime.now() - timedelta(hours=24)
            ph = self._ph()
            cursor.execute(f"SELECT COUNT(*) FROM assessments WHERE created_at >= {ph}", (cutoff_24h.isoformat(),))
            stats['assessments_24h'] = cursor.fetchone()[0]
            quality_metrics = self.calculate_data_quality_metrics()
            if quality_metrics:
                stats['data_quality'] = quality_metrics
            return stats
        except Exception as e:
            self.logger.error(f"Error getting database stats: {e}")
            return {'total_assessments': 0, 'unique_patients': 0, 'db_size_mb': 0, 'last_assessment': 'Error', 'error': str(e)}
        finally:
            conn.close()

    def clear_all_records(self):
        conn = get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("DELETE FROM assessments")
            cursor.execute("DELETE FROM patient_history")
            cursor.execute("DELETE FROM model_performance")
            cursor.execute("DELETE FROM data_quality_metrics")
            conn.commit()
            return True
        except Exception as e:
            self.logger.error(f"Error clearing records: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()

    def export_to_csv(self, filename=None, table='assessments'):
        if filename is None:
            filename = f"{table}_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        conn = get_connection()
        try:
            query = f"SELECT * FROM {table}" if table != 'assessments' else "SELECT * FROM assessments WHERE validation_status != 'invalid'"
            df = pd.read_sql_query(query, conn)
            df.to_csv(filename, index=False)
            return filename
        except Exception as e:
            self.logger.error(f"Error exporting: {e}")
            return None
        finally:
            conn.close()

    def backup_database(self, backup_path=None):
        return self.export_to_csv(filename=backup_path)


def get_db_manager():
    return DatabaseManager()
