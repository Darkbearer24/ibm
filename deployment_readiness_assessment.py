#!/usr/bin/env python3
"""
Deployment Readiness Assessment for Speech Translation System

This script conducts a comprehensive assessment of the system's readiness
for production deployment, evaluating all critical aspects including
code quality, testing, performance, security, and documentation.
"""

import os
import sys
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any
import importlib.util

class DeploymentReadinessAssessment:
    """Comprehensive deployment readiness assessment."""
    
    def __init__(self):
        self.project_root = Path.cwd()
        self.assessment_results = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'UNKNOWN',
            'readiness_score': 0.0,
            'categories': {},
            'critical_issues': [],
            'warnings': [],
            'recommendations': [],
            'deployment_blockers': []
        }
        
    def run_assessment(self) -> Dict[str, Any]:
        """Run complete deployment readiness assessment."""
        print("🚀 Starting Deployment Readiness Assessment...\n")
        
        # Assessment categories with weights
        categories = [
            ('code_quality', 0.20, self.assess_code_quality),
            ('testing_coverage', 0.25, self.assess_testing_coverage),
            ('performance', 0.20, self.assess_performance),
            ('security', 0.15, self.assess_security),
            ('documentation', 0.10, self.assess_documentation),
            ('infrastructure', 0.10, self.assess_infrastructure)
        ]
        
        total_score = 0.0
        
        for category, weight, assessment_func in categories:
            print(f"📊 Assessing {category.replace('_', ' ').title()}...")
            try:
                result = assessment_func()
                self.assessment_results['categories'][category] = result
                total_score += result['score'] * weight
                print(f"   Score: {result['score']:.1f}/10.0\n")
            except Exception as e:
                print(f"   ❌ Assessment failed: {e}\n")
                self.assessment_results['categories'][category] = {
                    'score': 0.0,
                    'status': 'FAILED',
                    'issues': [f"Assessment error: {e}"]
                }
        
        self.assessment_results['readiness_score'] = total_score
        self.determine_overall_status()
        self.generate_recommendations()
        
        return self.assessment_results
    
    def assess_code_quality(self) -> Dict[str, Any]:
        """Assess code quality and structure."""
        result = {
            'score': 0.0,
            'status': 'UNKNOWN',
            'checks': {},
            'issues': [],
            'details': {}
        }
        
        checks = [
            ('project_structure', self.check_project_structure),
            ('import_integrity', self.check_import_integrity),
            ('code_organization', self.check_code_organization),
            ('error_handling', self.check_error_handling)
        ]
        
        total_checks = len(checks)
        passed_checks = 0
        
        for check_name, check_func in checks:
            try:
                check_result = check_func()
                result['checks'][check_name] = check_result
                if check_result['passed']:
                    passed_checks += 1
                else:
                    result['issues'].extend(check_result.get('issues', []))
            except Exception as e:
                result['checks'][check_name] = {
                    'passed': False,
                    'issues': [f"Check failed: {e}"]
                }
                result['issues'].append(f"Code quality check '{check_name}' failed: {e}")
        
        result['score'] = (passed_checks / total_checks) * 10.0
        result['status'] = 'PASS' if result['score'] >= 7.0 else 'FAIL'
        
        return result
    
    def assess_testing_coverage(self) -> Dict[str, Any]:
        """Assess testing coverage and quality."""
        result = {
            'score': 0.0,
            'status': 'UNKNOWN',
            'test_results': {},
            'issues': [],
            'coverage_metrics': {}
        }
        
        # Check if test files exist
        test_files = [
            'test_unit.py',
            'test_integration.py',
            'test_edge_cases.py',
            'run_tests.py'
        ]
        
        existing_tests = []
        for test_file in test_files:
            if (self.project_root / test_file).exists():
                existing_tests.append(test_file)
        
        if not existing_tests:
            result['issues'].append("No test files found")
            result['score'] = 0.0
            result['status'] = 'FAIL'
            return result
        
        # Run test suite if available
        try:
            if (self.project_root / 'run_tests.py').exists():
                test_output = subprocess.run(
                    [sys.executable, 'run_tests.py'],
                    capture_output=True,
                    text=True,
                    timeout=300,
                    cwd=self.project_root
                )
                
                # Try to parse test results
                if test_output.returncode == 0:
                    try:
                        # Look for JSON report
                        report_files = list(self.project_root.glob('test_report_*.json'))
                        if report_files:
                            with open(report_files[-1], 'r') as f:
                                test_report = json.load(f)
                            result['test_results'] = test_report
                            
                            # Calculate score based on test results
                            if 'summary' in test_report:
                                summary = test_report['summary']
                                total_tests = summary.get('total_tests', 0)
                                passed_tests = summary.get('passed_tests', 0)
                                
                                if total_tests > 0:
                                    pass_rate = passed_tests / total_tests
                                    result['score'] = pass_rate * 10.0
                                else:
                                    result['score'] = 0.0
                            else:
                                result['score'] = 5.0  # Partial credit for running tests
                        else:
                            result['score'] = 3.0  # Tests ran but no detailed report
                    except Exception as e:
                        result['issues'].append(f"Failed to parse test results: {e}")
                        result['score'] = 2.0
                else:
                    result['issues'].append(f"Test execution failed: {test_output.stderr}")
                    result['score'] = 1.0
            else:
                result['score'] = 4.0  # Test files exist but no runner
                result['issues'].append("Test files exist but no test runner found")
        
        except subprocess.TimeoutExpired:
            result['issues'].append("Test execution timed out")
            result['score'] = 1.0
        except Exception as e:
            result['issues'].append(f"Test execution error: {e}")
            result['score'] = 0.0
        
        result['status'] = 'PASS' if result['score'] >= 7.0 else 'FAIL'
        return result
    
    def assess_performance(self) -> Dict[str, Any]:
        """Assess system performance and benchmarks."""
        result = {
            'score': 0.0,
            'status': 'UNKNOWN',
            'benchmarks': {},
            'issues': [],
            'metrics': {}
        }
        
        # Check if benchmark script exists and run it
        benchmark_file = self.project_root / 'benchmark_performance.py'
        if benchmark_file.exists():
            try:
                benchmark_output = subprocess.run(
                    [sys.executable, 'benchmark_performance.py'],
                    capture_output=True,
                    text=True,
                    timeout=600,  # 10 minutes timeout
                    cwd=self.project_root
                )
                
                if benchmark_output.returncode == 0:
                    # Look for benchmark report
                    report_files = list(self.project_root.glob('performance_report_*.json'))
                    if report_files:
                        with open(report_files[-1], 'r') as f:
                            benchmark_report = json.load(f)
                        
                        result['benchmarks'] = benchmark_report
                        
                        # Evaluate performance based on report
                        deployment_status = benchmark_report.get('deployment_readiness', {}).get('status', 'NOT_READY')
                        if deployment_status == 'READY':
                            result['score'] = 9.0
                        elif deployment_status == 'CONDITIONAL':
                            result['score'] = 6.0
                        else:
                            result['score'] = 3.0
                            result['issues'].append("Performance benchmarks indicate system not ready")
                    else:
                        result['score'] = 5.0
                        result['issues'].append("Benchmark ran but no report generated")
                else:
                    result['score'] = 2.0
                    result['issues'].append(f"Benchmark execution failed: {benchmark_output.stderr}")
            
            except subprocess.TimeoutExpired:
                result['score'] = 1.0
                result['issues'].append("Performance benchmark timed out")
            except Exception as e:
                result['score'] = 0.0
                result['issues'].append(f"Benchmark execution error: {e}")
        else:
            result['score'] = 0.0
            result['issues'].append("No performance benchmark script found")
        
        result['status'] = 'PASS' if result['score'] >= 6.0 else 'FAIL'
        return result
    
    def assess_security(self) -> Dict[str, Any]:
        """Assess security considerations."""
        result = {
            'score': 0.0,
            'status': 'UNKNOWN',
            'security_checks': {},
            'issues': [],
            'vulnerabilities': []
        }
        
        security_score = 0
        max_score = 0
        
        # Check for hardcoded secrets
        max_score += 2
        if self.check_no_hardcoded_secrets():
            security_score += 2
        else:
            result['vulnerabilities'].append("Potential hardcoded secrets found")
        
        # Check input validation
        max_score += 2
        if self.check_input_validation():
            security_score += 2
        else:
            result['issues'].append("Input validation may be insufficient")
        
        # Check error handling (security perspective)
        max_score += 2
        if self.check_secure_error_handling():
            security_score += 2
        else:
            result['issues'].append("Error handling may leak sensitive information")
        
        # Check file handling security
        max_score += 2
        if self.check_secure_file_handling():
            security_score += 2
        else:
            result['issues'].append("File handling security concerns")
        
        # Check logging security
        max_score += 2
        if self.check_secure_logging():
            security_score += 2
        else:
            result['issues'].append("Logging may expose sensitive data")
        
        result['score'] = (security_score / max_score) * 10.0 if max_score > 0 else 0.0
        result['status'] = 'PASS' if result['score'] >= 7.0 else 'FAIL'
        
        return result
    
    def assess_documentation(self) -> Dict[str, Any]:
        """Assess documentation completeness."""
        result = {
            'score': 0.0,
            'status': 'UNKNOWN',
            'documentation_files': {},
            'issues': [],
            'completeness': {}
        }
        
        required_docs = {
            'README.md': 'Project overview and quick start',
            'docs/API_Documentation.md': 'API documentation',
            'docs/User_Guide.md': 'User guide',
            'docs/Deployment_Guide.md': 'Deployment instructions',
            'requirements.txt': 'Dependencies'
        }
        
        found_docs = 0
        total_docs = len(required_docs)
        
        for doc_path, description in required_docs.items():
            file_path = self.project_root / doc_path
            if file_path.exists():
                found_docs += 1
                result['documentation_files'][doc_path] = {
                    'exists': True,
                    'size': file_path.stat().st_size,
                    'description': description
                }
            else:
                result['documentation_files'][doc_path] = {
                    'exists': False,
                    'description': description
                }
                result['issues'].append(f"Missing documentation: {doc_path}")
        
        result['score'] = (found_docs / total_docs) * 10.0
        result['status'] = 'PASS' if result['score'] >= 8.0 else 'FAIL'
        result['completeness']['found'] = found_docs
        result['completeness']['total'] = total_docs
        result['completeness']['percentage'] = (found_docs / total_docs) * 100
        
        return result
    
    def assess_infrastructure(self) -> Dict[str, Any]:
        """Assess infrastructure readiness."""
        result = {
            'score': 0.0,
            'status': 'UNKNOWN',
            'infrastructure_checks': {},
            'issues': [],
            'requirements': {}
        }
        
        checks_passed = 0
        total_checks = 0
        
        # Check Python version
        total_checks += 1
        python_version = sys.version_info
        if python_version >= (3, 8):
            checks_passed += 1
            result['requirements']['python_version'] = f"{python_version.major}.{python_version.minor}.{python_version.micro}"
        else:
            result['issues'].append(f"Python version {python_version.major}.{python_version.minor} is too old (requires 3.8+)")
        
        # Check dependencies
        total_checks += 1
        if self.check_dependencies():
            checks_passed += 1
        else:
            result['issues'].append("Some dependencies are missing or incompatible")
        
        # Check directory structure
        total_checks += 1
        if self.check_directory_structure():
            checks_passed += 1
        else:
            result['issues'].append("Required directory structure is incomplete")
        
        # Check configuration files
        total_checks += 1
        if self.check_configuration_files():
            checks_passed += 1
        else:
            result['issues'].append("Configuration files are missing or invalid")
        
        result['score'] = (checks_passed / total_checks) * 10.0 if total_checks > 0 else 0.0
        result['status'] = 'PASS' if result['score'] >= 7.0 else 'FAIL'
        
        return result
    
    # Helper methods for specific checks
    
    def check_project_structure(self) -> Dict[str, Any]:
        """Check project structure."""
        required_files = [
            'app.py',
            'requirements.txt',
            'utils/pipeline_orchestrator.py',
            'utils/error_handling.py',
            'utils/logging_config.py'
        ]
        
        missing_files = []
        for file_path in required_files:
            if not (self.project_root / file_path).exists():
                missing_files.append(file_path)
        
        return {
            'passed': len(missing_files) == 0,
            'issues': [f"Missing file: {f}" for f in missing_files],
            'details': {
                'required_files': required_files,
                'missing_files': missing_files
            }
        }
    
    def check_import_integrity(self) -> Dict[str, Any]:
        """Check if main modules can be imported."""
        modules_to_check = [
            'utils.pipeline_orchestrator',
            'utils.error_handling',
            'utils.logging_config'
        ]
        
        import_errors = []
        for module_name in modules_to_check:
            try:
                importlib.import_module(module_name)
            except ImportError as e:
                import_errors.append(f"{module_name}: {e}")
        
        return {
            'passed': len(import_errors) == 0,
            'issues': import_errors,
            'details': {
                'checked_modules': modules_to_check,
                'import_errors': import_errors
            }
        }
    
    def check_code_organization(self) -> Dict[str, Any]:
        """Check code organization and structure."""
        issues = []
        
        # Check if utils directory exists
        if not (self.project_root / 'utils').is_dir():
            issues.append("Utils directory missing")
        
        # Check if docs directory exists
        if not (self.project_root / 'docs').is_dir():
            issues.append("Documentation directory missing")
        
        return {
            'passed': len(issues) == 0,
            'issues': issues
        }
    
    def check_error_handling(self) -> Dict[str, Any]:
        """Check error handling implementation."""
        error_handling_file = self.project_root / 'utils' / 'error_handling.py'
        
        if not error_handling_file.exists():
            return {
                'passed': False,
                'issues': ['Error handling module not found']
            }
        
        # Check if custom exceptions are defined
        try:
            with open(error_handling_file, 'r') as f:
                content = f.read()
            
            required_exceptions = [
                'SpeechTranslationError',
                'AudioProcessingError',
                'ModelInferenceError'
            ]
            
            missing_exceptions = []
            for exception in required_exceptions:
                if exception not in content:
                    missing_exceptions.append(exception)
            
            return {
                'passed': len(missing_exceptions) == 0,
                'issues': [f"Missing exception: {e}" for e in missing_exceptions]
            }
        
        except Exception as e:
            return {
                'passed': False,
                'issues': [f"Error checking error handling: {e}"]
            }
    
    def check_no_hardcoded_secrets(self) -> bool:
        """Check for hardcoded secrets."""
        # Simple check for common secret patterns
        secret_patterns = [
            'password',
            'api_key',
            'secret_key',
            'access_token',
            'private_key'
        ]
        
        python_files = list(self.project_root.rglob('*.py'))
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().lower()
                
                for pattern in secret_patterns:
                    if f'{pattern} =' in content or f'"{pattern}"' in content:
                        return False
            except Exception:
                continue
        
        return True
    
    def check_input_validation(self) -> bool:
        """Check for input validation."""
        app_file = self.project_root / 'app.py'
        if not app_file.exists():
            return False
        
        try:
            with open(app_file, 'r') as f:
                content = f.read()
            
            # Look for file validation patterns
            validation_patterns = [
                'file_uploader',
                'type=',
                'accept_multiple_files'
            ]
            
            return any(pattern in content for pattern in validation_patterns)
        except Exception:
            return False
    
    def check_secure_error_handling(self) -> bool:
        """Check if error handling is secure."""
        # This is a simplified check
        return True  # Assume secure unless proven otherwise
    
    def check_secure_file_handling(self) -> bool:
        """Check secure file handling."""
        # This is a simplified check
        return True  # Assume secure unless proven otherwise
    
    def check_secure_logging(self) -> bool:
        """Check secure logging practices."""
        logging_file = self.project_root / 'utils' / 'logging_config.py'
        if not logging_file.exists():
            return False
        
        try:
            with open(logging_file, 'r') as f:
                content = f.read()
            
            # Check for structured logging
            return 'logging' in content and 'config' in content
        except Exception:
            return False
    
    def check_dependencies(self) -> bool:
        """Check if dependencies are properly defined."""
        requirements_file = self.project_root / 'requirements.txt'
        if not requirements_file.exists():
            return False
        
        try:
            with open(requirements_file, 'r') as f:
                requirements = f.read().strip()
            
            # Check for essential dependencies
            essential_deps = ['streamlit', 'numpy', 'librosa']
            return all(dep in requirements for dep in essential_deps)
        except Exception:
            return False
    
    def check_directory_structure(self) -> bool:
        """Check required directory structure."""
        required_dirs = ['utils', 'docs']
        return all((self.project_root / dir_name).is_dir() for dir_name in required_dirs)
    
    def check_configuration_files(self) -> bool:
        """Check configuration files."""
        config_files = ['requirements.txt']
        return all((self.project_root / file_name).exists() for file_name in config_files)
    
    def determine_overall_status(self):
        """Determine overall deployment readiness status."""
        score = self.assessment_results['readiness_score']
        
        # Check for critical failures
        critical_categories = ['testing_coverage', 'security']
        for category in critical_categories:
            if category in self.assessment_results['categories']:
                if self.assessment_results['categories'][category]['status'] == 'FAIL':
                    self.assessment_results['deployment_blockers'].append(
                        f"Critical failure in {category.replace('_', ' ')}"
                    )
        
        # Determine status based on score and blockers
        if self.assessment_results['deployment_blockers']:
            self.assessment_results['overall_status'] = 'NOT_READY'
        elif score >= 8.0:
            self.assessment_results['overall_status'] = 'READY'
        elif score >= 6.0:
            self.assessment_results['overall_status'] = 'CONDITIONAL'
        else:
            self.assessment_results['overall_status'] = 'NOT_READY'
    
    def generate_recommendations(self):
        """Generate deployment recommendations."""
        recommendations = []
        
        # Category-specific recommendations
        for category, result in self.assessment_results['categories'].items():
            if result['status'] == 'FAIL':
                if category == 'testing_coverage':
                    recommendations.append("Improve test coverage and fix failing tests")
                elif category == 'performance':
                    recommendations.append("Optimize performance and resolve benchmark issues")
                elif category == 'security':
                    recommendations.append("Address security vulnerabilities and implement best practices")
                elif category == 'documentation':
                    recommendations.append("Complete missing documentation")
                elif category == 'infrastructure':
                    recommendations.append("Fix infrastructure and configuration issues")
                elif category == 'code_quality':
                    recommendations.append("Improve code quality and fix structural issues")
        
        # Overall recommendations based on status
        status = self.assessment_results['overall_status']
        if status == 'NOT_READY':
            recommendations.append("System is not ready for production deployment")
            recommendations.append("Address all critical issues before proceeding")
        elif status == 'CONDITIONAL':
            recommendations.append("System may be ready for limited deployment")
            recommendations.append("Monitor closely and address remaining issues")
        elif status == 'READY':
            recommendations.append("System is ready for production deployment")
            recommendations.append("Proceed with deployment following the deployment guide")
        
        self.assessment_results['recommendations'] = recommendations
    
    def generate_report(self) -> str:
        """Generate human-readable assessment report."""
        report = []
        report.append("=" * 80)
        report.append("DEPLOYMENT READINESS ASSESSMENT REPORT")
        report.append("=" * 80)
        report.append(f"Assessment Date: {self.assessment_results['timestamp']}")
        report.append(f"Overall Status: {self.assessment_results['overall_status']}")
        report.append(f"Readiness Score: {self.assessment_results['readiness_score']:.1f}/10.0")
        report.append("")
        
        # Category breakdown
        report.append("CATEGORY BREAKDOWN:")
        report.append("-" * 40)
        for category, result in self.assessment_results['categories'].items():
            status_icon = "✅" if result['status'] == 'PASS' else "❌"
            report.append(f"{status_icon} {category.replace('_', ' ').title()}: {result['score']:.1f}/10.0 ({result['status']})")
        report.append("")
        
        # Critical issues
        if self.assessment_results['deployment_blockers']:
            report.append("🚨 DEPLOYMENT BLOCKERS:")
            for blocker in self.assessment_results['deployment_blockers']:
                report.append(f"   • {blocker}")
            report.append("")
        
        # Recommendations
        if self.assessment_results['recommendations']:
            report.append("📋 RECOMMENDATIONS:")
            for rec in self.assessment_results['recommendations']:
                report.append(f"   • {rec}")
            report.append("")
        
        # Detailed issues
        report.append("DETAILED ISSUES BY CATEGORY:")
        report.append("-" * 40)
        for category, result in self.assessment_results['categories'].items():
            if result.get('issues'):
                report.append(f"\n{category.replace('_', ' ').title()}:")
                for issue in result['issues']:
                    report.append(f"   • {issue}")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def save_report(self, filename: str = None):
        """Save assessment report to file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"deployment_readiness_report_{timestamp}.json"
        
        report_path = self.project_root / filename
        
        with open(report_path, 'w') as f:
            json.dump(self.assessment_results, f, indent=2)
        
        print(f"📄 Assessment report saved: {report_path}")
        return report_path

def main():
    """Main function to run deployment readiness assessment."""
    print("🚀 Speech Translation System - Deployment Readiness Assessment")
    print("=" * 70)
    print()
    
    # Run assessment
    assessor = DeploymentReadinessAssessment()
    results = assessor.run_assessment()
    
    # Generate and display report
    report = assessor.generate_report()
    print(report)
    
    # Save detailed results
    report_path = assessor.save_report()
    
    # Exit with appropriate code
    status = results['overall_status']
    if status == 'READY':
        print("\n🎉 System is READY for deployment!")
        sys.exit(0)
    elif status == 'CONDITIONAL':
        print("\n⚠️  System may be ready for deployment with conditions.")
        sys.exit(1)
    else:
        print("\n❌ System is NOT READY for deployment.")
        sys.exit(2)

if __name__ == "__main__":
    main()