import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import threading
import time
import os
import json
import logging

class AlertSystem:
    """Real-time alert system for high-risk patients with configurable thresholds."""
    
    def __init__(self, db_manager, notification_manager, alert_db_path="alerts.db"):
        """Initialize alert system."""
        self.db_manager = db_manager
        self.notification_manager = notification_manager
        self.alert_db_path = alert_db_path
        
        # Default alert thresholds
        self.alert_thresholds = {
            'high_risk': 0.7,
            'medium_risk': 0.5,
            'critical_risk': 0.85
        }
        
        # Alert configuration
        self.alert_config = {
            'enabled': True,
            'email_alerts': True,
            'immediate_alerts': True,
            'batch_alerts': False,
            'alert_cooldown_hours': 6,  # Prevent spam alerts for same patient
            'max_alerts_per_hour': 10
        }
        
        # Initialize alert database
        self.initialize_alert_database()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Track alert counts for rate limiting
        self.alert_counts = {}
        
    def initialize_alert_database(self):
        """Create alert tracking database."""
        conn = sqlite3.connect(self.alert_db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT NOT NULL,
                alert_type TEXT NOT NULL,
                risk_score REAL NOT NULL,
                alert_timestamp TEXT NOT NULL,
                alert_message TEXT,
                notification_sent INTEGER DEFAULT 0,
                acknowledged INTEGER DEFAULT 0,
                acknowledged_by TEXT,
                acknowledged_at TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS alert_config (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                config_key TEXT UNIQUE NOT NULL,
                config_value TEXT NOT NULL,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS alert_statistics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                total_alerts INTEGER DEFAULT 0,
                high_risk_alerts INTEGER DEFAULT 0,
                critical_alerts INTEGER DEFAULT 0,
                notifications_sent INTEGER DEFAULT 0,
                response_rate REAL DEFAULT 0.0,
                avg_response_time_minutes REAL DEFAULT 0.0
            )
        """)
        
        conn.commit()
        conn.close()
    
    def update_alert_thresholds(self, thresholds):
        """Update alert threshold configuration."""
        self.alert_thresholds.update(thresholds)
        
        # Save to database
        conn = sqlite3.connect(self.alert_db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO alert_config (config_key, config_value, updated_at)
            VALUES (?, ?, ?)
        """, ('alert_thresholds', json.dumps(self.alert_thresholds), datetime.now().isoformat()))
        
        conn.commit()
        conn.close()
        
        self.logger.info(f"Alert thresholds updated: {self.alert_thresholds}")
    
    def update_alert_config(self, config):
        """Update alert system configuration."""
        self.alert_config.update(config)
        
        # Save to database
        conn = sqlite3.connect(self.alert_db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO alert_config (config_key, config_value, updated_at)
            VALUES (?, ?, ?)
        """, ('alert_config', json.dumps(self.alert_config), datetime.now().isoformat()))
        
        conn.commit()
        conn.close()
        
        self.logger.info(f"Alert configuration updated: {self.alert_config}")
    
    def check_and_send_alerts(self, patient_data, risk_score, risk_level):
        """Check if alerts should be sent and send them."""
        if not self.alert_config['enabled']:
            return False
        
        patient_id = patient_data.get('patient_id', 'UNKNOWN')
        
        # Check rate limiting
        if not self._check_rate_limits(patient_id):
            self.logger.warning(f"Rate limit exceeded for patient {patient_id}")
            return False
        
        # Check cooldown period
        if not self._check_cooldown_period(patient_id):
            self.logger.info(f"Patient {patient_id} still in cooldown period")
            return False
        
        # Determine alert type based on risk score
        alert_type = self._determine_alert_type(risk_score)
        
        if alert_type:
            # Create alert record
            alert_id = self._create_alert_record(patient_id, alert_type, risk_score, patient_data)
            
            # Send notifications if configured
            if self.alert_config['email_alerts']:
                notification_sent = self._send_alert_notification(patient_data, risk_score, risk_level, alert_type)
                self._update_alert_notification_status(alert_id, notification_sent)
            
            self.logger.info(f"Alert triggered for patient {patient_id}: {alert_type} (Risk: {risk_score:.1%})")
            return True
        
        return False
    
    def _determine_alert_type(self, risk_score):
        """Determine alert type based on risk score."""
        if risk_score >= self.alert_thresholds['critical_risk']:
            return 'CRITICAL'
        elif risk_score >= self.alert_thresholds['high_risk']:
            return 'HIGH_RISK'
        elif risk_score >= self.alert_thresholds['medium_risk']:
            return 'MEDIUM_RISK'
        return None
    
    def _check_rate_limits(self, patient_id):
        """Check if rate limits are exceeded."""
        current_hour = datetime.now().strftime('%Y-%m-%d %H')
        key = f"{patient_id}_{current_hour}"
        
        if key not in self.alert_counts:
            self.alert_counts[key] = 0
        
        if self.alert_counts[key] >= self.alert_config['max_alerts_per_hour']:
            return False
        
        self.alert_counts[key] += 1
        
        # Clean up old entries
        cutoff_time = datetime.now() - timedelta(hours=2)
        cutoff_str = cutoff_time.strftime('%Y-%m-%d %H')
        
        keys_to_remove = [k for k in self.alert_counts.keys() if k.split('_')[-1] < cutoff_str]
        for k in keys_to_remove:
            del self.alert_counts[k]
        
        return True
    
    def _check_cooldown_period(self, patient_id):
        """Check if patient is still in cooldown period."""
        conn = sqlite3.connect(self.alert_db_path)
        cursor = conn.cursor()
        
        cooldown_time = datetime.now() - timedelta(hours=self.alert_config['alert_cooldown_hours'])
        
        cursor.execute("""
            SELECT COUNT(*) FROM alerts 
            WHERE patient_id = ? AND alert_timestamp > ?
        """, (patient_id, cooldown_time.isoformat()))
        
        recent_alerts = cursor.fetchone()[0]
        conn.close()
        
        return recent_alerts == 0
    
    def _create_alert_record(self, patient_id, alert_type, risk_score, patient_data):
        """Create alert record in database."""
        conn = sqlite3.connect(self.alert_db_path)
        cursor = conn.cursor()
        
        alert_message = self._generate_alert_message(patient_data, risk_score, alert_type)
        
        cursor.execute("""
            INSERT INTO alerts (patient_id, alert_type, risk_score, alert_timestamp, alert_message)
            VALUES (?, ?, ?, ?, ?)
        """, (patient_id, alert_type, risk_score, datetime.now().isoformat(), alert_message))
        
        alert_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return alert_id
    
    def _generate_alert_message(self, patient_data, risk_score, alert_type):
        """Generate alert message based on patient data."""
        patient_id = patient_data.get('patient_id', 'Unknown')
        age = patient_data.get('age', 'Unknown')
        nt_probnp = patient_data.get('nt_probnp', 'Unknown')
        
        message = f"{alert_type} RISK ALERT: Patient {patient_id} (Age: {age}) "
        message += f"has {risk_score:.1%} probability of 30-day readmission. "
        
        # Add key clinical indicators
        key_factors = []
        if isinstance(nt_probnp, (int, float)) and nt_probnp > 5000:
            key_factors.append(f"Elevated NT-proBNP ({nt_probnp:.0f})")
        
        if patient_data.get('b_line_score', 0) > 15:
            key_factors.append(f"High B-line score ({patient_data.get('b_line_score')})")
        
        if patient_data.get('ejection_fraction', 100) < 30:
            key_factors.append(f"Reduced EF ({patient_data.get('ejection_fraction')}%)")
        
        if key_factors:
            message += f"Key factors: {', '.join(key_factors)}. "
        
        message += "Immediate clinical review recommended."
        
        return message
    
    def _send_alert_notification(self, patient_data, risk_score, risk_level, alert_type):
        """Send alert notification via configured channels."""
        try:
            if alert_type == 'CRITICAL':
                success = self.notification_manager.send_high_risk_alert(patient_data, risk_score, risk_level)
            else:
                # For non-critical alerts, send standard notification
                success = self.notification_manager.send_high_risk_alert(patient_data, risk_score, risk_level)
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error sending alert notification: {str(e)}")
            return False
    
    def _update_alert_notification_status(self, alert_id, sent):
        """Update notification sent status for alert."""
        conn = sqlite3.connect(self.alert_db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE alerts 
            SET notification_sent = ?
            WHERE id = ?
        """, (1 if sent else 0, alert_id))
        
        conn.commit()
        conn.close()
    
    def acknowledge_alert(self, alert_id, acknowledged_by):
        """Acknowledge an alert."""
        conn = sqlite3.connect(self.alert_db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE alerts 
            SET acknowledged = 1, acknowledged_by = ?, acknowledged_at = ?
            WHERE id = ?
        """, (acknowledged_by, datetime.now().isoformat(), alert_id))
        
        success = cursor.rowcount > 0
        conn.commit()
        conn.close()
        
        if success:
            self.logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")
        
        return success
    
    def get_recent_alerts(self, hours=24):
        """Get recent alerts within specified hours."""
        conn = sqlite3.connect(self.alert_db_path)
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        try:
            query = """
                SELECT id, patient_id, alert_type, risk_score, alert_timestamp, 
                       alert_message, notification_sent, acknowledged, acknowledged_by
                FROM alerts 
                WHERE alert_timestamp > ?
                ORDER BY alert_timestamp DESC
            """
            
            df = pd.read_sql_query(query, conn, params=[cutoff_time.isoformat()])
            return df.to_dict('records')
            
        except Exception as e:
            self.logger.error(f"Error getting recent alerts: {str(e)}")
            return []
        finally:
            conn.close()
    
    def get_active_alerts(self):
        """Get unacknowledged alerts."""
        conn = sqlite3.connect(self.alert_db_path)
        
        try:
            query = """
                SELECT id, patient_id, alert_type, risk_score, alert_timestamp, 
                       alert_message, notification_sent
                FROM alerts 
                WHERE acknowledged = 0
                ORDER BY alert_timestamp DESC
            """
            
            df = pd.read_sql_query(query, conn)
            return df.to_dict('records')
            
        except Exception as e:
            self.logger.error(f"Error getting active alerts: {str(e)}")
            return []
        finally:
            conn.close()
    
    def get_alert_statistics(self, days=7):
        """Get alert statistics for specified period."""
        conn = sqlite3.connect(self.alert_db_path)
        
        try:
            cutoff_time = datetime.now() - timedelta(days=days)
            
            # Get alert counts by type
            query_counts = """
                SELECT 
                    COUNT(*) as total_alerts,
                    SUM(CASE WHEN alert_type = 'HIGH_RISK' THEN 1 ELSE 0 END) as high_risk_alerts,
                    SUM(CASE WHEN alert_type = 'CRITICAL' THEN 1 ELSE 0 END) as critical_alerts,
                    SUM(CASE WHEN notification_sent = 1 THEN 1 ELSE 0 END) as notifications_sent,
                    SUM(CASE WHEN acknowledged = 1 THEN 1 ELSE 0 END) as acknowledged_alerts
                FROM alerts 
                WHERE alert_timestamp > ?
            """
            
            cursor = conn.cursor()
            cursor.execute(query_counts, [cutoff_time.isoformat()])
            stats = cursor.fetchone()
            
            if stats and stats[0] > 0:
                total, high_risk, critical, sent, acked = stats
                response_rate = (acked / total) * 100 if total > 0 else 0
                
                # Get 24-hour counts
                cutoff_24h = datetime.now() - timedelta(hours=24)
                cursor.execute("""
                    SELECT COUNT(*) FROM alerts 
                    WHERE alert_timestamp > ? AND alert_type IN ('HIGH_RISK', 'CRITICAL')
                """, [cutoff_24h.isoformat()])
                
                alerts_24h = cursor.fetchone()[0]
                
                return {
                    'total_alerts': total,
                    'high_risk_alerts': high_risk,
                    'critical_alerts': critical,
                    'notifications_sent': sent,
                    'response_rate': response_rate / 100,  # As decimal
                    'alerts_24h': alerts_24h,
                    'high_risk_7d': high_risk + critical
                }
            else:
                return {
                    'total_alerts': 0,
                    'high_risk_alerts': 0,
                    'critical_alerts': 0,
                    'notifications_sent': 0,
                    'response_rate': 0.0,
                    'alerts_24h': 0,
                    'high_risk_7d': 0
                }
                
        except Exception as e:
            self.logger.error(f"Error getting alert statistics: {str(e)}")
            return {
                'total_alerts': 0,
                'high_risk_alerts': 0,
                'critical_alerts': 0,
                'notifications_sent': 0,
                'response_rate': 0.0,
                'alerts_24h': 0,
                'high_risk_7d': 0
            }
        finally:
            conn.close()
    
    def get_patient_alert_history(self, patient_id):
        """Get alert history for specific patient."""
        conn = sqlite3.connect(self.alert_db_path)
        
        try:
            query = """
                SELECT id, alert_type, risk_score, alert_timestamp, alert_message,
                       notification_sent, acknowledged, acknowledged_by, acknowledged_at
                FROM alerts 
                WHERE patient_id = ?
                ORDER BY alert_timestamp DESC
            """
            
            df = pd.read_sql_query(query, conn, params=[patient_id])
            return df.to_dict('records')
            
        except Exception as e:
            self.logger.error(f"Error getting patient alert history: {str(e)}")
            return []
        finally:
            conn.close()
    
    def clear_old_alerts(self, days=30):
        """Clear alerts older than specified days (Admin function)."""
        conn = sqlite3.connect(self.alert_db_path)
        cursor = conn.cursor()
        
        try:
            cutoff_time = datetime.now() - timedelta(days=days)
            
            cursor.execute("""
                DELETE FROM alerts 
                WHERE alert_timestamp < ? AND acknowledged = 1
            """, [cutoff_time.isoformat()])
            
            deleted_count = cursor.rowcount
            conn.commit()
            
            self.logger.info(f"Cleared {deleted_count} old alerts")
            return deleted_count
            
        except Exception as e:
            self.logger.error(f"Error clearing old alerts: {str(e)}")
            return 0
        finally:
            conn.close()
    
    def get_alert_trends(self, days=30):
        """Get alert trends over time."""
        conn = sqlite3.connect(self.alert_db_path)
        
        try:
            cutoff_time = datetime.now() - timedelta(days=days)
            
            query = """
                SELECT 
                    DATE(alert_timestamp) as alert_date,
                    COUNT(*) as total_alerts,
                    SUM(CASE WHEN alert_type = 'HIGH_RISK' THEN 1 ELSE 0 END) as high_risk_count,
                    SUM(CASE WHEN alert_type = 'CRITICAL' THEN 1 ELSE 0 END) as critical_count,
                    AVG(risk_score) as avg_risk_score
                FROM alerts 
                WHERE alert_timestamp > ?
                GROUP BY DATE(alert_timestamp)
                ORDER BY alert_date
            """
            
            df = pd.read_sql_query(query, conn, params=[cutoff_time.isoformat()])
            
            if not df.empty:
                df['alert_date'] = pd.to_datetime(df['alert_date'])
                return df.to_dict('records')
            else:
                return []
                
        except Exception as e:
            self.logger.error(f"Error getting alert trends: {str(e)}")
            return []
        finally:
            conn.close()
    
    def test_alert_system(self):
        """Test alert system functionality."""
        test_results = {
            'database_connection': False,
            'notification_service': False,
            'alert_creation': False,
            'rate_limiting': False
        }
        
        try:
            # Test database connection
            conn = sqlite3.connect(self.alert_db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM alerts")
            test_results['database_connection'] = True
            conn.close()
            
            # Test notification service
            test_results['notification_service'] = hasattr(self.notification_manager, 'send_test_email')
            
            # Test alert creation with dummy data
            test_patient = {
                'patient_id': 'TEST_PATIENT',
                'age': 75,
                'nt_probnp': 8000,
                'b_line_score': 20,
                'ejection_fraction': 25
            }
            
            alert_id = self._create_alert_record('TEST_PATIENT', 'HIGH_RISK', 0.8, test_patient)
            if alert_id:
                test_results['alert_creation'] = True
                # Clean up test alert
                conn = sqlite3.connect(self.alert_db_path)
                cursor = conn.cursor()
                cursor.execute("DELETE FROM alerts WHERE id = ?", [alert_id])
                conn.commit()
                conn.close()
            
            # Test rate limiting
            test_results['rate_limiting'] = self._check_rate_limits('TEST_PATIENT_RATE_LIMIT')
            
        except Exception as e:
            self.logger.error(f"Error testing alert system: {str(e)}")
        
        return test_results

