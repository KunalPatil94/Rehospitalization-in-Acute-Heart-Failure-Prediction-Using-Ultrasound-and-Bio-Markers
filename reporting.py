import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import io
import os
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import plotly.graph_objects as go
import plotly.io as pio
import sqlite3

class ReportGenerator:
    """Generates comprehensive reports for AHF prediction system."""
    
    def __init__(self, db_manager):
        """Initialize report generator."""
        self.db_manager = db_manager
        self.styles = getSampleStyleSheet()
        
        # Create custom styles
        self.title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=18,
            spaceAfter=30,
            textColor=colors.HexColor('#1f77b4')
        )
        
        self.heading_style = ParagraphStyle(
            'CustomHeading',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceBefore=20,
            spaceAfter=10,
            textColor=colors.HexColor('#2c3e50')
        )
    
    def generate_report(self, report_type, date_from, date_to, format_type='pdf'):
        """Generate report based on type and format."""
        try:
            # Get data for report
            data = self.get_report_data(date_from, date_to)
            
            if not data:
                return None
            
            if report_type == 'daily_summary':
                return self.generate_daily_summary(data, format_type)
            elif report_type == 'weekly_summary':
                return self.generate_weekly_summary(data, format_type)
            elif report_type == 'monthly_summary':
                return self.generate_monthly_summary(data, format_type)
            elif report_type == 'high_risk_patients':
                return self.generate_high_risk_report(data, format_type)
            elif report_type == 'model_performance':
                return self.generate_model_performance_report(data, format_type)
            else:
                return None
                
        except Exception as e:
            print(f"Error generating report: {e}")
            return None
    
    def get_report_data(self, date_from, date_to):
        """Fetch data for report generation."""
        try:
            assessments = self.db_manager.get_all_assessments()
            
            if not assessments:
                return None
            
            df = pd.DataFrame(assessments)
            df['assessment_date'] = pd.to_datetime(df['assessment_date'])
            
            # Filter by date range
            if date_from:
                df = df[df['assessment_date'].dt.date >= date_from]
            if date_to:
                df = df[df['assessment_date'].dt.date <= date_to]
            
            return df
            
        except Exception as e:
            print(f"Error fetching report data: {e}")
            return None
    
    def generate_daily_summary(self, data, format_type):
        """Generate daily summary report."""
        try:
            # Calculate summary statistics
            summary_stats = {
                'total_assessments': len(data),
                'unique_patients': data['patient_id'].nunique(),
                'high_risk_count': len(data[data['risk_level'] == 'High Risk']),
                'moderate_risk_count': len(data[data['risk_level'] == 'Moderate Risk']),
                'low_risk_count': len(data[data['risk_level'] == 'Low Risk']),
                'avg_risk_score': data['ensemble_probability'].mean(),
                'avg_nt_probnp': data['nt_probnp'].mean(),
                'avg_age': data['age'].mean(),
                'male_percentage': (data['gender'] == 'Male').mean() * 100
            }
            
            if format_type == 'pdf':
                return self.create_pdf_daily_summary(summary_stats, data)
            elif format_type == 'csv':
                return self.create_csv_summary(data)
            elif format_type == 'excel':
                return self.create_excel_summary(data)
                
        except Exception as e:
            print(f"Error generating daily summary: {e}")
            return None
    
    def create_pdf_daily_summary(self, summary_stats, data):
        """Create PDF daily summary report."""
        try:
            filename = f"daily_summary_{datetime.now().strftime('%Y%m%d')}.pdf"
            doc = SimpleDocTemplate(filename, pagesize=A4)
            story = []
            
            # Title
            title = Paragraph(f"AHF Risk Assessment Daily Summary", self.title_style)
            story.append(title)
            story.append(Spacer(1, 20))
            
            # Date and summary info
            date_info = Paragraph(f"Report Date: {datetime.now().strftime('%Y-%m-%d')}", self.styles['Normal'])
            story.append(date_info)
            story.append(Spacer(1, 12))
            
            # Summary statistics table
            summary_data = [
                ['Metric', 'Value'],
                ['Total Assessments', str(summary_stats['total_assessments'])],
                ['Unique Patients', str(summary_stats['unique_patients'])],
                ['High Risk Patients', str(summary_stats['high_risk_count'])],
                ['Moderate Risk Patients', str(summary_stats['moderate_risk_count'])],
                ['Low Risk Patients', str(summary_stats['low_risk_count'])],
                ['Average Risk Score', f"{summary_stats['avg_risk_score']:.1%}"],
                ['Average NT-proBNP', f"{summary_stats['avg_nt_probnp']:.0f} pg/mL"],
                ['Average Age', f"{summary_stats['avg_age']:.1f} years"],
                ['Male Percentage', f"{summary_stats['male_percentage']:.1f}%"]
            ]
            
            summary_table = Table(summary_data, colWidths=[3*inch, 2*inch])
            summary_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(Paragraph("Summary Statistics", self.heading_style))
            story.append(summary_table)
            story.append(Spacer(1, 20))
            
            # High risk patients table if any
            high_risk_patients = data[data['risk_level'] == 'High Risk']
            if len(high_risk_patients) > 0:
                story.append(Paragraph("High Risk Patients", self.heading_style))
                
                high_risk_data = [['Patient ID', 'Age', 'Risk Score', 'NT-proBNP', 'Assessment Time']]
                
                for _, patient in high_risk_patients.head(10).iterrows():  # Limit to top 10
                    high_risk_data.append([
                        str(patient['patient_id']),
                        str(patient['age']),
                        f"{patient['ensemble_probability']:.1%}",
                        f"{patient['nt_probnp']:.0f}",
                        patient['assessment_date'].strftime('%H:%M')
                    ])
                
                high_risk_table = Table(high_risk_data, colWidths=[1.5*inch, 0.8*inch, 1*inch, 1*inch, 1*inch])
                high_risk_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e74c3c')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 10),
                    ('FONTSIZE', (0, 1), (-1, -1), 8),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#fadbd8')),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                
                story.append(high_risk_table)
            
            # Footer
            story.append(Spacer(1, 40))
            footer = Paragraph("Generated by AHF Rehospitalization Prediction System", self.styles['Normal'])
            story.append(footer)
            
            doc.build(story)
            
            return {
                'filename': filename,
                'mime_type': 'application/pdf'
            }
            
        except Exception as e:
            print(f"Error creating PDF daily summary: {e}")
            return None
    
    def create_csv_summary(self, data):
        """Create CSV summary report."""
        try:
            filename = f"assessment_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            
            # Select relevant columns
            export_columns = [
                'assessment_date', 'patient_id', 'age', 'gender', 'weight',
                'nt_probnp', 'creatinine', 'b_line_score', 'ejection_fraction',
                'ensemble_probability', 'risk_level'
            ]
            
            export_data = data[export_columns].copy()
            export_data['assessment_date'] = export_data['assessment_date'].dt.strftime('%Y-%m-%d %H:%M:%S')
            export_data['ensemble_probability'] = export_data['ensemble_probability'].apply(lambda x: f"{x:.3f}")
            
            csv_buffer = io.StringIO()
            export_data.to_csv(csv_buffer, index=False)
            csv_content = csv_buffer.getvalue()
            
            return {
                'filename': filename,
                'data': csv_content,
                'mime_type': 'text/csv'
            }
            
        except Exception as e:
            print(f"Error creating CSV summary: {e}")
            return None
    
    def create_excel_summary(self, data):
        """Create Excel summary report."""
        try:
            filename = f"ahf_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            
            # Create Excel buffer
            excel_buffer = io.BytesIO()
            
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                # Summary sheet
                summary_stats = {
                    'Metric': ['Total Assessments', 'Unique Patients', 'High Risk Count', 
                              'Average Risk Score', 'Average Age', 'Average NT-proBNP'],
                    'Value': [
                        len(data),
                        data['patient_id'].nunique(),
                        len(data[data['risk_level'] == 'High Risk']),
                        f"{data['ensemble_probability'].mean():.1%}",
                        f"{data['age'].mean():.1f}",
                        f"{data['nt_probnp'].mean():.0f}"
                    ]
                }
                
                summary_df = pd.DataFrame(summary_stats)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # Full data sheet
                export_columns = [
                    'assessment_date', 'patient_id', 'age', 'gender', 'weight',
                    'nt_probnp', 'creatinine', 'b_line_score', 'ejection_fraction',
                    'ensemble_probability', 'risk_level'
                ]
                
                export_data = data[export_columns].copy()
                export_data.to_excel(writer, sheet_name='Detailed Data', index=False)
                
                # High risk patients sheet
                high_risk_data = data[data['risk_level'] == 'High Risk'][export_columns]
                if len(high_risk_data) > 0:
                    high_risk_data.to_excel(writer, sheet_name='High Risk Patients', index=False)
            
            excel_content = excel_buffer.getvalue()
            
            return {
                'filename': filename,
                'data': excel_content,
                'mime_type': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            }
            
        except Exception as e:
            print(f"Error creating Excel summary: {e}")
            return None
    
    def generate_high_risk_report(self, data, format_type):
        """Generate high-risk patients report."""
        try:
            high_risk_data = data[data['risk_level'] == 'High Risk'].copy()
            
            if len(high_risk_data) == 0:
                return None
            
            # Sort by risk score descending
            high_risk_data = high_risk_data.sort_values('ensemble_probability', ascending=False)
            
            if format_type == 'pdf':
                return self.create_high_risk_pdf(high_risk_data)
            elif format_type == 'csv':
                return self.create_csv_summary(high_risk_data)
            elif format_type == 'excel':
                return self.create_excel_summary(high_risk_data)
                
        except Exception as e:
            print(f"Error generating high risk report: {e}")
            return None
    
    def create_high_risk_pdf(self, high_risk_data):
        """Create PDF report for high-risk patients."""
        try:
            filename = f"high_risk_patients_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            doc = SimpleDocTemplate(filename, pagesize=A4)
            story = []
            
            # Title
            title = Paragraph("High Risk Patients Report", self.title_style)
            story.append(title)
            story.append(Spacer(1, 20))
            
            # Summary
            summary_text = f"""
            This report contains {len(high_risk_data)} patients identified as high risk for 30-day readmission.
            Average risk score: {high_risk_data['ensemble_probability'].mean():.1%}
            Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            """
            
            summary_para = Paragraph(summary_text, self.styles['Normal'])
            story.append(summary_para)
            story.append(Spacer(1, 20))
            
            # Patient details table
            patient_data = [['Patient ID', 'Age', 'Gender', 'Risk Score', 'NT-proBNP', 'Weight', 'EF%']]
            
            for _, patient in high_risk_data.head(20).iterrows():  # Limit to top 20
                patient_data.append([
                    str(patient['patient_id']),
                    str(patient['age']),
                    str(patient.get('gender', 'Unknown')),
                    f"{patient['ensemble_probability']:.1%}",
                    f"{patient['nt_probnp']:.0f}",
                    f"{patient['weight']:.1f}",
                    f"{patient['ejection_fraction']:.0f}"
                ])
            
            patient_table = Table(patient_data, colWidths=[1.2*inch, 0.6*inch, 0.8*inch, 0.8*inch, 0.9*inch, 0.8*inch, 0.6*inch])
            patient_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e74c3c')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 9),
                ('FONTSIZE', (0, 1), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#fadbd8')),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(Paragraph("Patient Details", self.heading_style))
            story.append(patient_table)
            
            doc.build(story)
            
            return {
                'filename': filename,
                'mime_type': 'application/pdf'
            }
            
        except Exception as e:
            print(f"Error creating high risk PDF: {e}")
            return None
    
    def generate_weekly_summary(self, data, format_type):
        """Generate weekly summary report."""
        try:
            # Group data by day
            data['date'] = data['assessment_date'].dt.date
            daily_stats = data.groupby('date').agg({
                'patient_id': 'count',
                'ensemble_probability': ['mean', 'std'],
                'risk_level': lambda x: (x == 'High Risk').sum()
            }).reset_index()
            
            # Flatten columns
            daily_stats.columns = ['date', 'total_assessments', 'avg_risk', 'risk_std', 'high_risk_count']
            
            if format_type == 'pdf':
                return self.create_weekly_pdf(daily_stats, data)
            else:
                return self.create_csv_summary(data)
                
        except Exception as e:
            print(f"Error generating weekly summary: {e}")
            return None
    
    def create_weekly_pdf(self, daily_stats, data):
        """Create PDF weekly summary."""
        try:
            filename = f"weekly_summary_{datetime.now().strftime('%Y%m%d')}.pdf"
            doc = SimpleDocTemplate(filename, pagesize=A4)
            story = []
            
            # Title
            title = Paragraph("Weekly AHF Risk Assessment Summary", self.title_style)
            story.append(title)
            story.append(Spacer(1, 20))
            
            # Weekly overview
            overview_text = f"""
            Report Period: {daily_stats['date'].min()} to {daily_stats['date'].max()}
            Total Assessments: {len(data)}
            Total High Risk Patients: {len(data[data['risk_level'] == 'High Risk'])}
            Average Daily Assessments: {daily_stats['total_assessments'].mean():.1f}
            """
            
            overview_para = Paragraph(overview_text, self.styles['Normal'])
            story.append(overview_para)
            story.append(Spacer(1, 20))
            
            # Daily breakdown table
            daily_data = [['Date', 'Assessments', 'Avg Risk', 'High Risk Count']]
            
            for _, day in daily_stats.iterrows():
                daily_data.append([
                    str(day['date']),
                    str(day['total_assessments']),
                    f"{day['avg_risk']:.1%}",
                    str(day['high_risk_count'])
                ])
            
            daily_table = Table(daily_data, colWidths=[1.5*inch, 1.2*inch, 1.2*inch, 1.2*inch])
            daily_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('FONTSIZE', (0, 1), (-1, -1), 9),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(Paragraph("Daily Breakdown", self.heading_style))
            story.append(daily_table)
            
            doc.build(story)
            
            return {
                'filename': filename,
                'mime_type': 'application/pdf'
            }
            
        except Exception as e:
            print(f"Error creating weekly PDF: {e}")
            return None
    
    def get_recent_reports(self):
        """Get list of recently generated reports."""
        # This would typically query a reports database table
        # For now, return empty list
        return []
