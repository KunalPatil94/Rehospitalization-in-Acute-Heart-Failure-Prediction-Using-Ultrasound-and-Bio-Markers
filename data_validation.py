import pandas as pd
import numpy as np
from datetime import datetime
import logging
import os
import re

class DataValidator:
    """Validates patient data and handles missing/inconsistent data with error logging."""
    
    def __init__(self, log_file="validation_errors.log"):
        """Initialize data validator with error logging."""
        self.log_file = log_file
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Define validation rules
        self.validation_rules = {
            'age': {'min': 18, 'max': 100, 'type': 'int', 'required': True},
            'gender': {'values': [0, 1], 'type': 'int', 'required': True},
            'weight': {'min': 30.0, 'max': 300.0, 'type': 'float', 'required': True},
            'nt_probnp': {'min': 50.0, 'max': 100000.0, 'type': 'float', 'required': True},
            'creatinine': {'min': 0.3, 'max': 10.0, 'type': 'float', 'required': True},
            'b_line_score': {'min': 0, 'max': 28, 'type': 'int', 'required': True},
            'ivc_collapsibility': {'min': 0.0, 'max': 100.0, 'type': 'float', 'required': True},
            'ejection_fraction': {'min': 5, 'max': 80, 'type': 'int', 'required': True},
            'systolic_bp': {'min': 60, 'max': 250, 'type': 'int', 'required': True},
            'heart_rate': {'min': 30, 'max': 200, 'type': 'int', 'required': True},
            'diabetes': {'values': [0, 1], 'type': 'int', 'required': True},
            'hypertension': {'values': [0, 1], 'type': 'int', 'required': True},
            'ckd': {'values': [0, 1], 'type': 'int', 'required': True},
            'afib': {'values': [0, 1], 'type': 'int', 'required': True},
            'patient_id': {'type': 'string', 'required': True, 'min_length': 3, 'max_length': 50}
        }
        
        # Clinical correlation rules
        self.correlation_rules = [
            {
                'name': 'CKD_Creatinine_Check',
                'condition': lambda data: data.get('ckd') == 1 and data.get('creatinine', 0) < 1.2,
                'warning': 'Patient has CKD but creatinine level is unexpectedly low (<1.2 mg/dL)'
            },
            {
                'name': 'Severe_HF_EF_Check',
                'condition': lambda data: data.get('ejection_fraction', 100) > 50 and data.get('nt_probnp', 0) > 10000,
                'warning': 'High NT-proBNP (>10000) with normal EF (>50%) - verify measurements'
            },
            {
                'name': 'Age_Weight_Check',
                'condition': lambda data: data.get('age', 0) > 80 and data.get('weight', 0) > 120,
                'warning': 'Elderly patient (>80) with high weight (>120kg) - verify accuracy'
            },
            {
                'name': 'B_Line_IVC_Correlation',
                'condition': lambda data: data.get('b_line_score', 0) > 20 and data.get('ivc_collapsibility', 0) > 70,
                'warning': 'High B-line score with high IVC collapsibility - clinical inconsistency'
            }
        ]
    
    def validate_patient_data(self, patient_data):
        """Comprehensive validation of patient data."""
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'cleaned_data': patient_data.copy() if isinstance(patient_data, dict) else {},
            'validation_summary': {}
        }
        
        try:
            # Basic field validation
            field_validation = self._validate_fields(patient_data)
            validation_result['errors'].extend(field_validation['errors'])
            validation_result['warnings'].extend(field_validation['warnings'])
            validation_result['cleaned_data'].update(field_validation['cleaned_data'])
            
            # Clinical correlation checks
            correlation_validation = self._check_clinical_correlations(validation_result['cleaned_data'])
            validation_result['warnings'].extend(correlation_validation['warnings'])
            
            # Data completeness check
            completeness_check = self._check_data_completeness(patient_data)
            validation_result['validation_summary']['completeness'] = completeness_check
            
            # Set overall validation status
            validation_result['valid'] = len(validation_result['errors']) == 0
            
            # Log validation results
            self._log_validation_results(patient_data.get('patient_id', 'UNKNOWN'), validation_result)
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"Error during validation: {str(e)}")
            validation_result['valid'] = False
            validation_result['errors'].append(f"Validation system error: {str(e)}")
            return validation_result
    
    def _validate_fields(self, data):
        """Validate individual fields according to rules."""
        result = {
            'errors': [],
            'warnings': [],
            'cleaned_data': {}
        }
        
        for field, rules in self.validation_rules.items():
            value = data.get(field)
            
            # Check if required field is missing
            if rules.get('required', False) and (value is None or value == ''):
                result['errors'].append(f"Required field '{field}' is missing or empty")
                continue
            
            if value is None:
                continue
                
            # Type validation and conversion
            try:
                if rules['type'] == 'int':
                    cleaned_value = int(float(value))  # Handle string numbers
                elif rules['type'] == 'float':
                    cleaned_value = float(value)
                elif rules['type'] == 'string':
                    cleaned_value = str(value).strip()
                else:
                    cleaned_value = value
                    
                result['cleaned_data'][field] = cleaned_value
                
            except (ValueError, TypeError):
                result['errors'].append(f"Field '{field}' has invalid type. Expected {rules['type']}, got {type(value).__name__}")
                continue
            
            # Range validation
            if 'min' in rules and cleaned_value < rules['min']:
                result['errors'].append(f"Field '{field}' value {cleaned_value} is below minimum {rules['min']}")
            
            if 'max' in rules and cleaned_value > rules['max']:
                result['errors'].append(f"Field '{field}' value {cleaned_value} is above maximum {rules['max']}")
            
            # Value validation for categorical fields
            if 'values' in rules and cleaned_value not in rules['values']:
                result['errors'].append(f"Field '{field}' value {cleaned_value} is not in allowed values {rules['values']}")
            
            # String length validation
            if rules['type'] == 'string':
                if 'min_length' in rules and len(cleaned_value) < rules['min_length']:
                    result['errors'].append(f"Field '{field}' is too short. Minimum length: {rules['min_length']}")
                if 'max_length' in rules and len(cleaned_value) > rules['max_length']:
                    result['errors'].append(f"Field '{field}' is too long. Maximum length: {rules['max_length']}")
                
                # Patient ID format validation
                if field == 'patient_id':
                    if not re.match(r'^[A-Za-z0-9_-]+$', cleaned_value):
                        result['errors'].append("Patient ID contains invalid characters. Use only letters, numbers, hyphens, and underscores")
        
        return result
    
    def _check_clinical_correlations(self, data):
        """Check for clinical inconsistencies and correlations."""
        result = {
            'warnings': []
        }
        
        for rule in self.correlation_rules:
            try:
                if rule['condition'](data):
                    result['warnings'].append(f"Clinical Warning ({rule['name']}): {rule['warning']}")
                    self.logger.warning(f"Clinical correlation warning for patient {data.get('patient_id', 'UNKNOWN')}: {rule['warning']}")
            except Exception as e:
                self.logger.error(f"Error checking correlation rule {rule['name']}: {str(e)}")
        
        return result
    
    def _check_data_completeness(self, data):
        """Analyze data completeness."""
        total_fields = len(self.validation_rules)
        provided_fields = sum(1 for field in self.validation_rules.keys() if data.get(field) is not None)
        required_fields = sum(1 for rules in self.validation_rules.values() if rules.get('required', False))
        provided_required = sum(1 for field, rules in self.validation_rules.items() 
                               if rules.get('required', False) and data.get(field) is not None)
        
        return {
            'total_fields': total_fields,
            'provided_fields': provided_fields,
            'completeness_percentage': (provided_fields / total_fields) * 100,
            'required_fields': required_fields,
            'provided_required_fields': provided_required,
            'required_completeness_percentage': (provided_required / required_fields) * 100 if required_fields > 0 else 100
        }
    
    def _log_validation_results(self, patient_id, validation_result):
        """Log validation results for audit trail."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'patient_id': patient_id,
            'validation_status': 'PASSED' if validation_result['valid'] else 'FAILED',
            'error_count': len(validation_result['errors']),
            'warning_count': len(validation_result['warnings']),
            'completeness': validation_result['validation_summary']['completeness']['completeness_percentage']
        }
        
        if validation_result['errors']:
            self.logger.error(f"Validation FAILED for patient {patient_id}: {', '.join(validation_result['errors'])}")
        
        if validation_result['warnings']:
            self.logger.warning(f"Validation warnings for patient {patient_id}: {', '.join(validation_result['warnings'])}")
        
        if validation_result['valid'] and not validation_result['warnings']:
            self.logger.info(f"Validation PASSED for patient {patient_id} - {log_entry['completeness']:.1f}% complete")
    
    def validate_batch_data(self, data_list):
        """Validate a batch of patient records."""
        batch_results = {
            'total_records': len(data_list),
            'valid_records': 0,
            'invalid_records': 0,
            'records_with_warnings': 0,
            'validation_results': [],
            'summary_errors': {},
            'summary_warnings': {}
        }
        
        self.logger.info(f"Starting batch validation of {len(data_list)} records")
        
        for i, patient_data in enumerate(data_list):
            result = self.validate_patient_data(patient_data)
            batch_results['validation_results'].append(result)
            
            if result['valid']:
                batch_results['valid_records'] += 1
            else:
                batch_results['invalid_records'] += 1
            
            if result['warnings']:
                batch_results['records_with_warnings'] += 1
            
            # Aggregate error types
            for error in result['errors']:
                error_type = error.split(':')[0] if ':' in error else error
                batch_results['summary_errors'][error_type] = batch_results['summary_errors'].get(error_type, 0) + 1
            
            # Aggregate warning types
            for warning in result['warnings']:
                warning_type = warning.split(':')[0] if ':' in warning else warning
                batch_results['summary_warnings'][warning_type] = batch_results['summary_warnings'].get(warning_type, 0) + 1
        
        # Log batch summary
        self.logger.info(f"Batch validation completed: {batch_results['valid_records']}/{batch_results['total_records']} records valid")
        
        return batch_results
    
    def get_validation_statistics(self):
        """Get validation error statistics from log file."""
        try:
            if not os.path.exists(self.log_file):
                return {'error': 'No validation log file found'}
            
            with open(self.log_file, 'r') as f:
                log_lines = f.readlines()
            
            stats = {
                'total_validations': 0,
                'failed_validations': 0,
                'warnings_issued': 0,
                'common_errors': {},
                'recent_activity': []
            }
            
            for line in log_lines[-1000:]:  # Last 1000 log entries
                if 'Validation FAILED' in line:
                    stats['failed_validations'] += 1
                    stats['total_validations'] += 1
                elif 'Validation PASSED' in line:
                    stats['total_validations'] += 1
                elif 'Validation warnings' in line:
                    stats['warnings_issued'] += 1
                
                # Extract recent activity
                if any(keyword in line for keyword in ['FAILED', 'PASSED', 'WARNING']):
                    stats['recent_activity'].append({
                        'timestamp': line.split(' - ')[0] if ' - ' in line else 'Unknown',
                        'message': line.strip()
                    })
            
            # Keep only recent 50 activities
            stats['recent_activity'] = stats['recent_activity'][-50:]
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting validation statistics: {str(e)}")
            return {'error': f'Error reading validation statistics: {str(e)}'}
    
    def clear_validation_logs(self):
        """Clear validation log file (Admin only)."""
        try:
            if os.path.exists(self.log_file):
                with open(self.log_file, 'w') as f:
                    f.write(f"# Validation log cleared on {datetime.now().isoformat()}\n")
                self.logger.info("Validation logs cleared")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error clearing validation logs: {str(e)}")
            return False
    
    def suggest_data_corrections(self, patient_data, validation_result):
        """Suggest corrections for invalid data."""
        suggestions = []
        
        if not validation_result['valid']:
            for error in validation_result['errors']:
                field = None
                suggestion = None
                
                # Parse error message to extract field and suggest correction
                if 'is below minimum' in error:
                    field = error.split("'")[1]
                    min_val = self.validation_rules[field]['min']
                    suggestion = f"Increase {field} to at least {min_val}"
                
                elif 'is above maximum' in error:
                    field = error.split("'")[1]
                    max_val = self.validation_rules[field]['max']
                    suggestion = f"Reduce {field} to at most {max_val}"
                
                elif 'missing or empty' in error:
                    field = error.split("'")[1]
                    if field in ['age', 'weight']:
                        suggestion = f"Please provide patient's {field}"
                    elif field == 'patient_id':
                        suggestion = "Generate or provide a unique patient identifier"
                    else:
                        suggestion = f"Obtain {field} measurement from clinical assessment"
                
                elif 'invalid type' in error:
                    field = error.split("'")[1]
                    expected_type = self.validation_rules[field]['type']
                    suggestion = f"Ensure {field} is provided as a {expected_type}"
                
                if suggestion:
                    suggestions.append({
                        'field': field,
                        'error': error,
                        'suggestion': suggestion,
                        'current_value': patient_data.get(field, 'Missing')
                    })
        
        return suggestions

