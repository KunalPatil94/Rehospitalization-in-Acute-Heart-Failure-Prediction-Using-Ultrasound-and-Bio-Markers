import os
import json
import smtplib
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import streamlit as st

class NotificationManager:
    """Manages email notifications and alerts."""
    
    def __init__(self):
        """Initialize notification manager."""
        self.resend_api_key = os.getenv("RESEND_API_KEY")
        self.sendgrid_api_key = os.getenv("SENDGRID_API_KEY")
        self.notification_settings = {
            'enabled': True,
            'recipients': ['admin@hospital.com'],
            'from_email': 'noreply@hospital.com',
            'from_name': 'AHF Risk Predictor'
        }
    
    def update_notification_settings(self, settings):
        """Update notification configuration."""
        self.notification_settings.update(settings)
    
    def send_email_via_resend(self, to_email, subject, html_content, text_content=None):
        """Send email using Resend API."""
        if not self.resend_api_key:
            print("Resend API key not configured")
            return False
        
        url = "https://api.resend.com/emails"
        headers = {
            "Authorization": f"Bearer {self.resend_api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "from": f"{self.notification_settings['from_name']} <{self.notification_settings['from_email']}>",
            "to": [to_email],
            "subject": subject,
            "html": html_content
        }
        
        if text_content:
            payload["text"] = text_content
        
        try:
            response = requests.post(url, json=payload, headers=headers)
            return response.status_code == 200
        except Exception as e:
            print(f"Error sending email via Resend: {e}")
            return False
    
    def send_email_via_sendgrid(self, to_email, subject, html_content, text_content=None):
        """Send email using SendGrid API."""
        if not self.sendgrid_api_key:
            print("SendGrid API key not configured")
            return False
        
        url = "https://api.sendgrid.com/v3/mail/send"
        headers = {
            "Authorization": f"Bearer {self.sendgrid_api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "personalizations": [{"to": [{"email": to_email}]}],
            "from": {
                "email": self.notification_settings['from_email'],
                "name": self.notification_settings['from_name']
            },
            "subject": subject,
            "content": [{"type": "text/html", "value": html_content}]
        }
        
        if text_content:
            payload["content"].append({"type": "text/plain", "value": text_content})
        
        try:
            response = requests.post(url, json=payload, headers=headers)
            return response.status_code == 202
        except Exception as e:
            print(f"Error sending email via SendGrid: {e}")
            return False
    
    def send_email(self, to_email, subject, html_content, text_content=None):
        """Send email using available service."""
        # Try Resend first, fall back to SendGrid
        if self.resend_api_key:
            success = self.send_email_via_resend(to_email, subject, html_content, text_content)
            if success:
                return True
        
        if self.sendgrid_api_key:
            return self.send_email_via_sendgrid(to_email, subject, html_content, text_content)
        
        print("No email service configured")
        return False
    
    def send_high_risk_alert(self, patient_data, risk_score, risk_level):
        """Send high-risk patient alert."""
        if not self.notification_settings['enabled']:
            return False
        
        subject = f"🚨 HIGH RISK ALERT - Patient {patient_data.get('patient_id', 'Unknown')}"
        
        html_content = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; }}
                .header {{ background-color: #dc3545; color: white; padding: 20px; text-align: center; }}
                .content {{ padding: 20px; }}
                .patient-info {{ background-color: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 5px; }}
                .risk-score {{ font-size: 24px; font-weight: bold; color: #dc3545; }}
                .footer {{ background-color: #f8f9fa; padding: 10px; text-align: center; font-size: 12px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🚨 HIGH RISK PATIENT ALERT</h1>
            </div>
            
            <div class="content">
                <p>A patient has been identified as <strong>HIGH RISK</strong> for 30-day readmission.</p>
                
                <div class="patient-info">
                    <h3>Patient Information:</h3>
                    <p><strong>Patient ID:</strong> {patient_data.get('patient_id', 'Unknown')}</p>
                    <p><strong>Age:</strong> {patient_data.get('age', 'Unknown')} years</p>
                    <p><strong>Assessment Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    <p><strong>Risk Score:</strong> <span class="risk-score">{risk_score:.1%}</span></p>
                    <p><strong>Risk Level:</strong> {risk_level}</p>
                </div>
                
                <div class="patient-info">
                    <h3>Key Clinical Parameters:</h3>
                    <p><strong>NT-proBNP:</strong> {patient_data.get('nt_probnp', 'Unknown')} pg/mL</p>
                    <p><strong>Weight:</strong> {patient_data.get('weight', 'Unknown')} kg</p>
                    <p><strong>Creatinine:</strong> {patient_data.get('creatinine', 'Unknown')} mg/dL</p>
                    <p><strong>Ejection Fraction:</strong> {patient_data.get('ejection_fraction', 'Unknown')}%</p>
                    <p><strong>B-line Score:</strong> {patient_data.get('b_line_score', 'Unknown')}</p>
                </div>
                
                <p><strong>Recommended Action:</strong> Immediate clinical review and consideration of enhanced monitoring protocols.</p>
            </div>
            
            <div class="footer">
                <p>This alert was generated by the AHF Rehospitalization Prediction System</p>
                <p>Generated on: {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')}</p>
            </div>
        </body>
        </html>
        """
        
        text_content = f"""
        HIGH RISK PATIENT ALERT
        
        Patient ID: {patient_data.get('patient_id', 'Unknown')}
        Risk Score: {risk_score:.1%}
        Risk Level: {risk_level}
        
        Key Parameters:
        - NT-proBNP: {patient_data.get('nt_probnp', 'Unknown')} pg/mL
        - Age: {patient_data.get('age', 'Unknown')} years
        - Weight: {patient_data.get('weight', 'Unknown')} kg
        
        Immediate clinical review recommended.
        
        Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        success_count = 0
        for recipient in self.notification_settings['recipients']:
            if self.send_email(recipient, subject, html_content, text_content):
                success_count += 1
        
        return success_count > 0
    
    def send_daily_summary(self, summary_data):
        """Send daily summary report."""
        if not self.notification_settings['enabled']:
            return False
        
        subject = f"📊 Daily AHF Risk Summary - {datetime.now().strftime('%Y-%m-%d')}"
        
        html_content = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; }}
                .header {{ background-color: #007bff; color: white; padding: 20px; text-align: center; }}
                .content {{ padding: 20px; }}
                .summary-box {{ background-color: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 5px; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: white; border-radius: 5px; text-align: center; }}
                .high-risk {{ color: #dc3545; font-weight: bold; }}
                .moderate-risk {{ color: #ffc107; font-weight: bold; }}
                .low-risk {{ color: #28a745; font-weight: bold; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>📊 Daily AHF Risk Summary</h1>
                <p>{datetime.now().strftime('%Y-%m-%d')}</p>
            </div>
            
            <div class="content">
                <div class="summary-box">
                    <h3>Assessment Summary</h3>
                    <div class="metric">
                        <h4>Total Assessments</h4>
                        <p style="font-size: 24px; margin: 0;">{summary_data.get('total_assessments', 0)}</p>
                    </div>
                    <div class="metric">
                        <h4 class="high-risk">High Risk</h4>
                        <p style="font-size: 24px; margin: 0;">{summary_data.get('high_risk_count', 0)}</p>
                    </div>
                    <div class="metric">
                        <h4 class="moderate-risk">Moderate Risk</h4>
                        <p style="font-size: 24px; margin: 0;">{summary_data.get('moderate_risk_count', 0)}</p>
                    </div>
                    <div class="metric">
                        <h4 class="low-risk">Low Risk</h4>
                        <p style="font-size: 24px; margin: 0;">{summary_data.get('low_risk_count', 0)}</p>
                    </div>
                </div>
                
                <div class="summary-box">
                    <h3>Key Metrics</h3>
                    <p><strong>Average Risk Score:</strong> {summary_data.get('avg_risk_score', 0):.1%}</p>
                    <p><strong>Unique Patients:</strong> {summary_data.get('unique_patients', 0)}</p>
                    <p><strong>Alerts Sent:</strong> {summary_data.get('alerts_sent', 0)}</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        success_count = 0
        for recipient in self.notification_settings['recipients']:
            if self.send_email(recipient, subject, html_content):
                success_count += 1
        
        return success_count > 0
    
    def send_test_email(self, test_email):
        """Send test email to verify configuration."""
        subject = "🧪 Test Email - AHF Risk Predictor"
        
        html_content = f"""
        <html>
        <body>
            <h2>Test Email Successful! ✅</h2>
            <p>This is a test email from the AHF Rehospitalization Prediction System.</p>
            <p>If you received this email, your notification system is working correctly.</p>
            <p><strong>Sent at:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </body>
        </html>
        """
        
        text_content = f"""
        Test Email Successful!
        
        This is a test email from the AHF Rehospitalization Prediction System.
        If you received this email, your notification system is working correctly.
        
        Sent at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        return self.send_email(test_email, subject, html_content, text_content)
    
    def send_weekly_report(self, report_data):
        """Send weekly performance report."""
        if not self.notification_settings['enabled']:
            return False
        
        subject = f"📈 Weekly AHF Performance Report - Week of {datetime.now().strftime('%Y-%m-%d')}"
        
        html_content = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; }}
                .header {{ background-color: #28a745; color: white; padding: 20px; text-align: center; }}
                .content {{ padding: 20px; }}
                .report-section {{ background-color: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>📈 Weekly Performance Report</h1>
                <p>Week of {datetime.now().strftime('%Y-%m-%d')}</p>
            </div>
            
            <div class="content">
                <div class="report-section">
                    <h3>Assessment Statistics</h3>
                    <p><strong>Total Assessments:</strong> {report_data.get('total_assessments', 0)}</p>
                    <p><strong>Unique Patients:</strong> {report_data.get('unique_patients', 0)}</p>
                    <p><strong>High Risk Patients:</strong> {report_data.get('high_risk_patients', 0)}</p>
                </div>
                
                <div class="report-section">
                    <h3>Model Performance</h3>
                    <p><strong>Average Prediction Accuracy:</strong> {report_data.get('avg_accuracy', 0):.1%}</p>
                    <p><strong>Alert Response Rate:</strong> {report_data.get('response_rate', 0):.1%}</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        success_count = 0
        for recipient in self.notification_settings['recipients']:
            if self.send_email(recipient, subject, html_content):
                success_count += 1
        
        return success_count > 0
