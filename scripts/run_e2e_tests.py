#!/usr/bin/env python3
"""
E2E Test Runner for C7DocDownloader

This script runs comprehensive end-to-end tests for the C7DocDownloader system.
"""

import subprocess
import sys
import argparse
from pathlib import Path


def run_command(command: list[str], description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(command)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(command, check=True, capture_output=False)
        print(f"✅ {description} - PASSED")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - FAILED (exit code: {e.returncode})")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run C7DocDownloader E2E tests")
    parser.add_argument(
        "--quick", 
        action="store_true", 
        help="Run only quick E2E tests (skip performance tests)"
    )
    parser.add_argument(
        "--performance-only", 
        action="store_true",
        help="Run only performance tests"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Run tests in verbose mode"
    )
    parser.add_argument(
        "--parallel", 
        type=int, 
        default=1,
        help="Number of parallel test processes"
    )
    
    args = parser.parse_args()
    
    # Base pytest command
    pytest_cmd = ["python", "-m", "pytest"]
    
    if args.verbose:
        pytest_cmd.extend(["-v", "-s"])
    
    if args.parallel > 1:
        pytest_cmd.extend(["-n", str(args.parallel)])
    
    # Test suites to run
    test_suites = []
    
    if args.performance_only:
        test_suites.append({
            "name": "Performance Tests",
            "path": "tests/e2e/test_performance_scenarios.py",
            "markers": "-m 'e2e and slow'",
            "description": "Performance and load testing scenarios"
        })
    elif args.quick:
        test_suites.extend([
            {
                "name": "Core E2E Workflow Tests",
                "path": "tests/e2e/test_full_download_workflow.py",
                "markers": "-m 'e2e and not slow'",
                "description": "Core download workflow tests (excluding slow tests)"
            },
            {
                "name": "CLI Integration Tests", 
                "path": "tests/e2e/test_cli_integration.py",
                "markers": "-m 'e2e and not slow'",
                "description": "CLI integration tests"
            }
        ])
    else:
        # Full E2E test suite
        test_suites.extend([
            {
                "name": "Core E2E Workflow Tests",
                "path": "tests/e2e/test_full_download_workflow.py", 
                "markers": "-m 'e2e'",
                "description": "Complete download workflow tests"
            },
            {
                "name": "CLI Integration Tests",
                "path": "tests/e2e/test_cli_integration.py",
                "markers": "-m 'e2e'", 
                "description": "CLI integration tests"
            },
            {
                "name": "Performance Tests",
                "path": "tests/e2e/test_performance_scenarios.py",
                "markers": "-m 'e2e and slow'",
                "description": "Performance and load testing scenarios"
            }
        ])
    
    print("🚀 Starting C7DocDownloader E2E Test Suite")
    print(f"Test mode: {'Performance Only' if args.performance_only else 'Quick' if args.quick else 'Full'}")
    print(f"Parallel processes: {args.parallel}")
    
    # Track results
    passed = 0
    failed = 0
    
    # Run each test suite
    for suite in test_suites:
        command = pytest_cmd + [suite["path"]]
        
        if suite.get("markers"):
            command.extend(suite["markers"].split())
        
        success = run_command(command, f"{suite['name']} - {suite['description']}")
        
        if success:
            passed += 1
        else:
            failed += 1
    
    # Final summary
    print(f"\n{'='*60}")
    print("E2E TEST SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Total: {passed + failed}")
    
    if failed == 0:
        print("\n🎉 All E2E tests passed!")
        sys.exit(0)
    else:
        print(f"\n💥 {failed} test suite(s) failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()