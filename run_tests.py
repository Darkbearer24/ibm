#!/usr/bin/env python3
"""
Sprint 7: Comprehensive Test Runner

Executes all test suites (unit, integration, edge cases) and generates
a comprehensive testing report for deployment readiness assessment.
"""

import sys
import os
import subprocess
import time
from pathlib import Path
from datetime import datetime
import json
import traceback

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))


class TestRunner:
    """Comprehensive test runner for the speech translation pipeline."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.test_results = {
            'timestamp': datetime.now().isoformat(),
            'test_suites': {},
            'overall_status': 'UNKNOWN',
            'summary': {},
            'recommendations': []
        }
    
    def run_unit_tests(self):
        """Run unit tests."""
        print("\n[*] Running Unit Tests")
        print("=" * 60)
        
        start_time = time.time()
        
        try:
            # Try to run unit tests directly
            from test_unit import run_unit_tests
            success = run_unit_tests()
            
            self.test_results['test_suites']['unit_tests'] = {
                'status': 'PASSED' if success else 'FAILED',
                'execution_time': time.time() - start_time,
                'method': 'direct_import'
            }
            
            return success
            
        except Exception as e:
            print(f"Direct import failed: {e}")
            print("Falling back to subprocess execution...")
            
            # Fallback to subprocess
            try:
                result = subprocess.run([
                    sys.executable, "test_unit.py"
                ], cwd=self.project_root, capture_output=True, text=True, timeout=300)
                
                success = result.returncode == 0
                
                self.test_results['test_suites']['unit_tests'] = {
                    'status': 'PASSED' if success else 'FAILED',
                    'execution_time': time.time() - start_time,
                    'method': 'subprocess',
                    'stdout': result.stdout[-1000:] if result.stdout else '',  # Last 1000 chars
                    'stderr': result.stderr[-1000:] if result.stderr else ''
                }
                
                print(result.stdout)
                if result.stderr:
                    print("STDERR:", result.stderr)
                
                return success
                
            except subprocess.TimeoutExpired:
                print("[ERROR] Unit tests timed out after 5 minutes")
                self.test_results['test_suites']['unit_tests'] = {
                    'status': 'TIMEOUT',
                    'execution_time': time.time() - start_time,
                    'method': 'subprocess'
                }
                return False
                
            except Exception as e:
                print(f"[ERROR] Unit tests failed with error: {e}")
                self.test_results['test_suites']['unit_tests'] = {
                    'status': 'ERROR',
                    'execution_time': time.time() - start_time,
                    'error': str(e),
                    'method': 'subprocess'
                }
                return False
    
    def run_integration_tests(self):
        """Run integration tests."""
        print("\n[*] Running Integration Tests")
        print("=" * 60)
        
        start_time = time.time()
        
        try:
            # Try to run integration tests directly
            from test_integration import run_integration_tests
            success = run_integration_tests()
            
            self.test_results['test_suites']['integration_tests'] = {
                'status': 'PASSED' if success else 'FAILED',
                'execution_time': time.time() - start_time,
                'method': 'direct_import'
            }
            
            return success
            
        except Exception as e:
            print(f"Direct import failed: {e}")
            print("Falling back to subprocess execution...")
            
            # Fallback to subprocess
            try:
                result = subprocess.run([
                    sys.executable, "test_integration.py"
                ], cwd=self.project_root, capture_output=True, text=True, timeout=600)
                
                success = result.returncode == 0
                
                self.test_results['test_suites']['integration_tests'] = {
                    'status': 'PASSED' if success else 'FAILED',
                    'execution_time': time.time() - start_time,
                    'method': 'subprocess',
                    'stdout': result.stdout[-1000:] if result.stdout else '',
                    'stderr': result.stderr[-1000:] if result.stderr else ''
                }
                
                print(result.stdout)
                if result.stderr:
                    print("STDERR:", result.stderr)
                
                return success
                
            except subprocess.TimeoutExpired:
                print("[ERROR] Integration tests timed out after 10 minutes")
                self.test_results['test_suites']['integration_tests'] = {
                    'status': 'TIMEOUT',
                    'execution_time': time.time() - start_time,
                    'method': 'subprocess'
                }
                return False
                
            except Exception as e:
                print(f"[ERROR] Integration tests failed with error: {e}")
                self.test_results['test_suites']['integration_tests'] = {
                    'status': 'ERROR',
                    'execution_time': time.time() - start_time,
                    'error': str(e),
                    'method': 'subprocess'
                }
                return False
    
    def run_edge_case_tests(self):
        """Run edge case tests."""
        print("\n[*] Running Edge Case Tests")
        print("=" * 60)
        
        start_time = time.time()
        
        try:
            # Try to run edge case tests directly
            from test_edge_cases import run_edge_case_tests
            success = run_edge_case_tests()
            
            self.test_results['test_suites']['edge_case_tests'] = {
                'status': 'PASSED' if success else 'FAILED',
                'execution_time': time.time() - start_time,
                'method': 'direct_import'
            }
            
            return success
            
        except Exception as e:
            print(f"Direct import failed: {e}")
            print("Falling back to subprocess execution...")
            
            # Fallback to subprocess
            try:
                result = subprocess.run([
                    sys.executable, "test_edge_cases.py"
                ], cwd=self.project_root, capture_output=True, text=True, timeout=900)
                
                success = result.returncode == 0
                
                self.test_results['test_suites']['edge_case_tests'] = {
                    'status': 'PASSED' if success else 'FAILED',
                    'execution_time': time.time() - start_time,
                    'method': 'subprocess',
                    'stdout': result.stdout[-1000:] if result.stdout else '',
                    'stderr': result.stderr[-1000:] if result.stderr else ''
                }
                
                print(result.stdout)
                if result.stderr:
                    print("STDERR:", result.stderr)
                
                return success
                
            except subprocess.TimeoutExpired:
                print("[ERROR] Edge case tests timed out after 15 minutes")
                self.test_results['test_suites']['edge_case_tests'] = {
                    'status': 'TIMEOUT',
                    'execution_time': time.time() - start_time,
                    'method': 'subprocess'
                }
                return False
                
            except Exception as e:
                print(f"[ERROR] Edge case tests failed with error: {e}")
                self.test_results['test_suites']['edge_case_tests'] = {
                    'status': 'ERROR',
                    'execution_time': time.time() - start_time,
                    'error': str(e),
                    'method': 'subprocess'
                }
                return False
    
    def check_dependencies(self):
        """Check if all required dependencies are available."""
        print("\n[*] Checking Dependencies")
        print("=" * 60)
        
        required_packages = [
            'numpy', 'torch', 'librosa', 'soundfile', 'streamlit',
            'pytest', 'psutil', 'scipy', 'matplotlib'
        ]
        
        missing_packages = []
        available_packages = []
        
        for package in required_packages:
            try:
                __import__(package)
                available_packages.append(package)
                print(f"[OK] {package}")
            except ImportError:
                missing_packages.append(package)
                print(f"[ERROR] {package} - MISSING")
        
        self.test_results['dependencies'] = {
            'available': available_packages,
            'missing': missing_packages,
            'all_available': len(missing_packages) == 0
        }
        
        if missing_packages:
            print(f"\n[WARNING] Missing packages: {', '.join(missing_packages)}")
            print("Install with: pip install " + ' '.join(missing_packages))
            return False
        
        print("\n[OK] All dependencies available")
        return True
    
    def check_project_structure(self):
        """Check if project structure is complete."""
        print("\n[*] Checking Project Structure")
        print("=" * 60)
        
        required_files = [
            'app.py',
            'utils/pipeline_orchestrator.py',
            'utils/logging_config.py',
            'utils/error_handling.py',
            'utils/framing.py',
            'utils/denoise.py',
            'models/encoder_decoder.py',
            'test_unit.py',
            'test_integration.py',
            'test_edge_cases.py'
        ]
        
        missing_files = []
        available_files = []
        
        for file_path in required_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                available_files.append(file_path)
                print(f"[OK] {file_path}")
            else:
                missing_files.append(file_path)
                print(f"[ERROR] {file_path} - MISSING")
        
        self.test_results['project_structure'] = {
            'available_files': available_files,
            'missing_files': missing_files,
            'structure_complete': len(missing_files) == 0
        }
        
        if missing_files:
            print(f"\n[WARNING] Missing files: {', '.join(missing_files)}")
            return False
        
        print("\n[OK] Project structure complete")
        return True
    
    def generate_summary(self):
        """Generate test summary and recommendations."""
        test_suites = self.test_results['test_suites']
        
        # Count results
        passed = sum(1 for suite in test_suites.values() if suite['status'] == 'PASSED')
        failed = sum(1 for suite in test_suites.values() if suite['status'] == 'FAILED')
        errors = sum(1 for suite in test_suites.values() if suite['status'] == 'ERROR')
        timeouts = sum(1 for suite in test_suites.values() if suite['status'] == 'TIMEOUT')
        
        total_tests = len(test_suites)
        
        # Calculate overall status
        if total_tests == 0:
            overall_status = 'NO_TESTS'
        elif passed == total_tests:
            overall_status = 'ALL_PASSED'
        elif failed > 0 or errors > 0 or timeouts > 0:
            overall_status = 'SOME_FAILED'
        else:
            overall_status = 'UNKNOWN'
        
        self.test_results['overall_status'] = overall_status
        self.test_results['summary'] = {
            'total_test_suites': total_tests,
            'passed': passed,
            'failed': failed,
            'errors': errors,
            'timeouts': timeouts,
            'success_rate': (passed / total_tests * 100) if total_tests > 0 else 0
        }
        
        # Generate recommendations
        recommendations = []
        
        if not self.test_results.get('dependencies', {}).get('all_available', True):
            recommendations.append("Install missing dependencies before deployment")
        
        if not self.test_results.get('project_structure', {}).get('structure_complete', True):
            recommendations.append("Complete project structure before deployment")
        
        if failed > 0:
            recommendations.append("Fix failing tests before deployment")
        
        if errors > 0:
            recommendations.append("Resolve test execution errors")
        
        if timeouts > 0:
            recommendations.append("Investigate and fix test timeouts")
        
        if overall_status == 'ALL_PASSED':
            recommendations.append("[OK] System is ready for deployment")
        elif overall_status == 'SOME_FAILED':
            recommendations.append("[WARNING] Address test failures before deployment")
        else:
            recommendations.append("[ERROR] System not ready for deployment")
        
        self.test_results['recommendations'] = recommendations
    
    def save_report(self):
        """Save test report to file."""
        report_file = self.project_root / "test_report.json"
        
        with open(report_file, 'w') as f:
            json.dump(self.test_results, f, indent=2)
        
        print(f"\n[*] Test report saved to: {report_file}")
    
    def print_final_report(self):
        """Print final test report."""
        print("\n" + "=" * 80)
        print("[*] FINAL TEST REPORT")
        print("=" * 80)
        
        summary = self.test_results['summary']
        print(f"\n[*] Test Summary:")
        print(f"   Total Test Suites: {summary['total_test_suites']}")
        print(f"   Passed: {summary['passed']}")
        print(f"   Failed: {summary['failed']}")
        print(f"   Errors: {summary['errors']}")
        print(f"   Timeouts: {summary['timeouts']}")
        print(f"   Success Rate: {summary['success_rate']:.1f}%")
        
        print(f"\n[*] Overall Status: {self.test_results['overall_status']}")
        
        print(f"\n[*] Recommendations:")
        for i, rec in enumerate(self.test_results['recommendations'], 1):
            print(f"   {i}. {rec}")
        
        # Print detailed results
        print(f"\n[*] Detailed Results:")
        for suite_name, suite_result in self.test_results['test_suites'].items():
            status_emoji = {
                'PASSED': '[OK]',
                'FAILED': '[ERROR]',
                'ERROR': '[ERROR]',
                'TIMEOUT': '[TIMEOUT]'
            }.get(suite_result['status'], '[UNKNOWN]')
            
            print(f"   {status_emoji} {suite_name}: {suite_result['status']} ({suite_result['execution_time']:.1f}s)")
    
    def run_all_tests(self):
        """Run all test suites and generate comprehensive report."""
        print("[*] Starting Comprehensive Test Suite")
        print("=" * 80)
        print(f"Timestamp: {self.test_results['timestamp']}")
        print(f"Project Root: {self.project_root}")
        
        # Check prerequisites
        deps_ok = self.check_dependencies()
        structure_ok = self.check_project_structure()
        
        if not deps_ok or not structure_ok:
            print("\n[ERROR] Prerequisites not met. Skipping tests.")
            self.generate_summary()
            self.print_final_report()
            self.save_report()
            return False
        
        # Run test suites
        test_results = []
        
        # Unit tests
        unit_success = self.run_unit_tests()
        test_results.append(unit_success)
        
        # Integration tests
        integration_success = self.run_integration_tests()
        test_results.append(integration_success)
        
        # Edge case tests
        edge_success = self.run_edge_case_tests()
        test_results.append(edge_success)
        
        # Generate final report
        self.generate_summary()
        self.print_final_report()
        self.save_report()
        
        # Return overall success
        return all(test_results)


def main():
    """Main function to run all tests."""
    runner = TestRunner()
    
    try:
        success = runner.run_all_tests()
        
        if success:
            print("\n[OK] All tests passed! System ready for deployment.")
            return 0
        else:
            print("\n[WARNING] Some tests failed. Review report before deployment.")
            return 1
            
    except KeyboardInterrupt:
        print("\n\n[INFO] Test execution interrupted by user")
        return 130
    except Exception as e:
        print(f"\n\n[ERROR] Test runner failed with error: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())