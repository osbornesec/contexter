#!/usr/bin/env python3
"""
RAG System Integration Test Execution Script

Comprehensive test runner that validates the complete RAG system from 
document ingestion to vector search with production readiness assessment.

Usage:
    python run_rag_integration_tests.py [options]
    
    --quick          Run quick validation tests only
    --full           Run comprehensive test suite (default)
    --performance    Run performance-focused tests
    --components     Run component integration tests
    --report-only    Generate report from existing results
    --output-dir     Directory for test results (default: ./test_results)
    --verbose        Enable verbose logging
    --fail-fast      Stop on first failure
"""

import asyncio
import sys
import argparse
import logging
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('rag_integration_tests.log')
    ]
)
logger = logging.getLogger(__name__)


class RAGTestSuite:
    """Main test suite orchestrator for RAG system validation."""
    
    def __init__(self, output_dir: Path, verbose: bool = False):
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)
        self.verbose = verbose
        
        if verbose:
            logging.getLogger().setLevel(logging.DEBUG)
        
        # Test execution tracking
        self.test_results = {}
        self.start_time = None
        self.end_time = None
    
    async def run_quick_validation(self) -> Dict[str, Any]:
        """Run quick validation tests for basic functionality."""
        logger.info("🚀 Running quick validation tests...")
        
        test_command = [
            "python", "-m", "pytest",
            "tests/integration/test_rag_system_end_to_end.py::TestRAGSystemEndToEnd::test_complete_document_ingestion_to_search_workflow",
            "-v", "--tb=short"
        ]
        
        if self.verbose:
            test_command.append("-s")
        
        result = await self._run_pytest_command(test_command, "quick_validation")
        return result
    
    async def run_comprehensive_tests(self) -> Dict[str, Any]:
        """Run the complete comprehensive test suite."""
        logger.info("🔧 Running comprehensive integration tests...")
        
        test_command = [
            "python", "-m", "pytest",
            "tests/integration/test_rag_system_end_to_end.py",
            "-v", "--tb=short",
            "--strict-markers",
            "--durations=10"
        ]
        
        if self.verbose:
            test_command.extend(["-s", "--log-cli-level=INFO"])
        
        result = await self._run_pytest_command(test_command, "comprehensive")
        return result
    
    async def run_performance_tests(self) -> Dict[str, Any]:
        """Run performance-focused tests."""
        logger.info("⚡ Running performance tests...")
        
        test_command = [
            "python", "-m", "pytest",
            "tests/integration/test_rag_system_end_to_end.py::TestRAGSystemPerformance",
            "-v", "--tb=short",
            "-m", "performance"
        ]
        
        if self.verbose:
            test_command.append("-s")
        
        result = await self._run_pytest_command(test_command, "performance")
        return result
    
    async def run_component_integration_tests(self) -> Dict[str, Any]:
        """Run component integration tests."""
        logger.info("🔗 Running component integration tests...")
        
        test_command = [
            "python", "-m", "pytest",
            "tests/integration/test_component_integration.py",
            "-v", "--tb=short"
        ]
        
        if self.verbose:
            test_command.append("-s")
        
        result = await self._run_pytest_command(test_command, "component_integration")
        return result
    
    async def _run_pytest_command(self, command: List[str], test_type: str) -> Dict[str, Any]:
        """Execute pytest command and capture results."""
        import subprocess
        
        logger.info(f"Executing: {' '.join(command)}")
        start_time = time.time()
        
        try:
            # Change to project root directory
            project_root = Path(__file__).parent
            
            result = subprocess.run(
                command,
                cwd=project_root,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            execution_time = time.time() - start_time
            
            # Parse results
            test_result = {
                'test_type': test_type,
                'command': ' '.join(command),
                'execution_time': execution_time,
                'return_code': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'success': result.returncode == 0,
                'timestamp': datetime.now().isoformat()
            }
            
            # Extract test statistics from pytest output
            stats = self._parse_pytest_output(result.stdout)
            test_result.update(stats)
            
            # Log results
            if test_result['success']:
                logger.info(f"✅ {test_type} tests completed successfully in {execution_time:.1f}s")
                if 'tests_passed' in stats:
                    logger.info(f"   Tests: {stats['tests_passed']} passed, {stats.get('tests_failed', 0)} failed")
            else:
                logger.error(f"❌ {test_type} tests failed (exit code: {result.returncode})")
                if result.stderr:
                    logger.error(f"   Error: {result.stderr[:200]}...")
            
            self.test_results[test_type] = test_result
            return test_result
            
        except subprocess.TimeoutExpired:
            logger.error(f"⏰ {test_type} tests timed out after 5 minutes")
            return {
                'test_type': test_type,
                'success': False,
                'error': 'Test execution timed out',
                'execution_time': 300.0
            }
        except Exception as e:
            logger.error(f"💥 {test_type} tests failed with exception: {e}")
            return {
                'test_type': test_type,
                'success': False,
                'error': str(e),
                'execution_time': time.time() - start_time
            }
    
    def _parse_pytest_output(self, output: str) -> Dict[str, Any]:
        """Parse pytest output to extract test statistics."""
        stats = {}
        lines = output.split('\n')
        
        # Look for test results summary
        for line in lines:
            line = line.strip().lower()
            if 'passed' in line or 'failed' in line or 'error' in line:
                # Extract numbers from pytest summary
                import re
                numbers = re.findall(r'(\d+)\s+(passed|failed|error|skipped)', line)
                for count, status in numbers:
                    stats[f'tests_{status}'] = int(count)
        
        # Look for duration information
        for line in lines:
            if 'seconds' in line.lower() and 'test session starts' not in line.lower():
                import re
                duration_match = re.search(r'(\d+\.?\d*)\s*seconds?', line)
                if duration_match:
                    stats['test_duration'] = float(duration_match.group(1))
                    break
        
        return stats
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """Generate comprehensive summary report."""
        logger.info("📊 Generating summary report...")
        
        summary = {
            'execution_metadata': {
                'start_time': self.start_time,
                'end_time': self.end_time,
                'total_duration': (self.end_time - self.start_time) if self.start_time and self.end_time else 0,
                'test_suites_run': len(self.test_results)
            },
            'test_results': self.test_results,
            'overall_assessment': self._assess_overall_results(),
            'recommendations': self._generate_recommendations()
        }
        
        return summary
    
    def _assess_overall_results(self) -> Dict[str, Any]:
        """Assess overall test results and production readiness."""
        total_success = sum(1 for result in self.test_results.values() if result.get('success', False))
        total_tests = len(self.test_results)
        
        success_rate = total_success / total_tests if total_tests > 0 else 0.0
        
        # Calculate production readiness score
        readiness_score = 0.0
        critical_issues = []
        warnings = []
        
        if success_rate >= 1.0:
            readiness_score += 60  # Base score for all tests passing
        elif success_rate >= 0.8:
            readiness_score += 40  # Partial credit for most tests passing
            warnings.append(f"Some tests failed (success rate: {success_rate:.1%})")
        else:
            critical_issues.append(f"Low test success rate: {success_rate:.1%}")
        
        # Check for specific test failures
        for test_type, result in self.test_results.items():
            if not result.get('success', False):
                critical_issues.append(f"{test_type} tests failed")
            
            # Check performance criteria
            if test_type == 'performance':
                if result.get('execution_time', 0) > 120:  # 2 minutes
                    warnings.append("Performance tests took longer than expected")
                else:
                    readiness_score += 20  # Performance bonus
        
        # Component integration bonus
        if 'component_integration' in self.test_results and self.test_results['component_integration'].get('success'):
            readiness_score += 20
        
        # Final production readiness assessment
        production_ready = (
            success_rate >= 0.9 and
            len(critical_issues) == 0 and
            readiness_score >= 80
        )
        
        return {
            'success_rate': success_rate,
            'production_ready': production_ready,
            'readiness_score': min(100, readiness_score),
            'critical_issues': critical_issues,
            'warnings': warnings,
            'total_test_suites': total_tests,
            'successful_test_suites': total_success
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []
        assessment = self._assess_overall_results()
        
        if assessment['production_ready']:
            recommendations.append("✅ System is ready for production deployment!")
            recommendations.append("Consider setting up monitoring for production performance")
        else:
            recommendations.append("❌ System requires fixes before production deployment")
            
            if assessment['success_rate'] < 0.9:
                recommendations.append("Fix failing tests to improve reliability")
            
            for issue in assessment['critical_issues']:
                recommendations.append(f"🔴 CRITICAL: {issue}")
            
            for warning in assessment['warnings']:
                recommendations.append(f"🟡 WARNING: {warning}")
        
        # Performance recommendations
        if 'performance' in self.test_results:
            perf_result = self.test_results['performance']
            if perf_result.get('success'):
                recommendations.append("✅ Performance targets are being met")
            else:
                recommendations.append("🔴 Performance issues need attention")
        
        # Component integration recommendations
        if 'component_integration' in self.test_results:
            comp_result = self.test_results['component_integration']
            if comp_result.get('success'):
                recommendations.append("✅ Component integration is working correctly")
            else:
                recommendations.append("🔴 Component integration issues detected")
        
        return recommendations
    
    async def save_results(self):
        """Save test results to files."""
        summary = self.generate_summary_report()
        
        # Save comprehensive report
        report_file = self.output_dir / f"rag_integration_report_{int(time.time())}.json"
        with open(report_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Save human-readable summary
        summary_file = self.output_dir / f"rag_integration_summary_{int(time.time())}.txt"
        with open(summary_file, 'w') as f:
            f.write(self._format_human_readable_summary(summary))
        
        logger.info(f"📁 Results saved to {self.output_dir}")
        logger.info(f"   Report: {report_file}")
        logger.info(f"   Summary: {summary_file}")
        
        return summary
    
    def _format_human_readable_summary(self, summary: Dict[str, Any]) -> str:
        """Format human-readable summary report."""
        lines = [
            "=" * 80,
            "RAG SYSTEM INTEGRATION TEST SUMMARY",
            "=" * 80,
            "",
            f"Execution Time: {summary['execution_metadata']['start_time']} - {summary['execution_metadata']['end_time']}",
            f"Total Duration: {summary['execution_metadata']['total_duration']:.1f} seconds",
            f"Test Suites Run: {summary['execution_metadata']['test_suites_run']}",
            "",
            "OVERALL ASSESSMENT",
            "-" * 40,
            f"Success Rate: {summary['overall_assessment']['success_rate']:.1%}",
            f"Production Ready: {'YES' if summary['overall_assessment']['production_ready'] else 'NO'}",
            f"Readiness Score: {summary['overall_assessment']['readiness_score']:.0f}/100",
            "",
            "TEST SUITE RESULTS",
            "-" * 40,
        ]
        
        for test_type, result in summary['test_results'].items():
            status = "✅ PASSED" if result.get('success') else "❌ FAILED"
            duration = result.get('execution_time', 0)
            lines.append(f"{test_type:<25} {status} ({duration:.1f}s)")
        
        lines.extend([
            "",
            "RECOMMENDATIONS",
            "-" * 40,
        ])
        
        for rec in summary['recommendations']:
            lines.append(rec)
        
        lines.extend([
            "",
            "=" * 80
        ])
        
        return "\n".join(lines)


async def main():
    """Main entry point for the test runner."""
    parser = argparse.ArgumentParser(
        description="RAG System Integration Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_rag_integration_tests.py --quick
  python run_rag_integration_tests.py --full --verbose
  python run_rag_integration_tests.py --performance --output-dir ./results
        """
    )
    
    parser.add_argument("--quick", action="store_true", help="Run quick validation tests only")
    parser.add_argument("--full", action="store_true", help="Run comprehensive test suite (default)")
    parser.add_argument("--performance", action="store_true", help="Run performance-focused tests")
    parser.add_argument("--components", action="store_true", help="Run component integration tests")
    parser.add_argument("--all", action="store_true", help="Run all test suites")
    parser.add_argument("--output-dir", type=Path, default=Path("./test_results"), help="Directory for test results")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on first failure")
    
    args = parser.parse_args()
    
    # Default to full if no specific test type is specified
    if not any([args.quick, args.full, args.performance, args.components, args.all]):
        args.full = True
    
    # Create test suite
    test_suite = RAGTestSuite(output_dir=args.output_dir, verbose=args.verbose)
    test_suite.start_time = time.time()
    
    logger.info("🎯 Starting RAG System Integration Test Suite")
    logger.info(f"📁 Output directory: {args.output_dir}")
    
    try:
        # Run selected test suites
        if args.quick:
            await test_suite.run_quick_validation()
        
        if args.full or args.all:
            await test_suite.run_comprehensive_tests()
        
        if args.performance or args.all:
            await test_suite.run_performance_tests()
        
        if args.components or args.all:
            await test_suite.run_component_integration_tests()
        
        test_suite.end_time = time.time()
        
        # Generate and save results
        summary = await test_suite.save_results()
        
        # Print final assessment
        assessment = summary['overall_assessment']
        
        print("\n" + "=" * 60)
        print("🏁 RAG SYSTEM INTEGRATION TEST COMPLETE")
        print("=" * 60)
        print(f"Success Rate: {assessment['success_rate']:.1%}")
        print(f"Production Ready: {'YES ✅' if assessment['production_ready'] else 'NO ❌'}")
        print(f"Readiness Score: {assessment['readiness_score']:.0f}/100")
        print(f"Results saved to: {args.output_dir}")
        
        if assessment['critical_issues']:
            print("\n🔴 CRITICAL ISSUES:")
            for issue in assessment['critical_issues']:
                print(f"  - {issue}")
        
        if assessment['warnings']:
            print("\n🟡 WARNINGS:")
            for warning in assessment['warnings']:
                print(f"  - {warning}")
        
        print("\n💡 RECOMMENDATIONS:")
        for rec in summary['recommendations']:
            print(f"  {rec}")
        
        print("=" * 60)
        
        # Exit with appropriate code
        exit_code = 0 if assessment['production_ready'] else 1
        
        if args.fail_fast and not assessment['production_ready']:
            logger.error("Fail-fast mode: exiting due to test failures")
        
        return exit_code
        
    except KeyboardInterrupt:
        logger.warning("Test execution interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)