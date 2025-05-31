"""
Test runner script for Vietnamese ID Card OCR.

This script runs all tests and generates coverage reports.
"""

import pytest
import sys
import os
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


def run_tests():
    """Run all tests with coverage."""
    # Test configuration
    pytest_args = [
        "tests/",
        "-v",
        "--tb=short",
        "--cov=src",
        "--cov-report=html:coverage_html",
        "--cov-report=term-missing",
        "--cov-report=xml",
        "--junitxml=test-results.xml"
    ]

    # Run tests
    exit_code = pytest.main(pytest_args)

    if exit_code == 0:
        print("\n✅ All tests passed!")
        print("📊 Coverage report generated in coverage_html/")
    else:
        print("\n❌ Some tests failed!")
        print(f"Exit code: {exit_code}")

    return exit_code


def run_specific_test(test_name):
    """Run a specific test file or test function."""
    pytest_args = [
        f"tests/{test_name}",
        "-v",
        "--tb=short"
    ]

    return pytest.main(pytest_args)


def run_fast_tests():
    """Run only fast tests (excluding integration tests)."""
    pytest_args = [
        "tests/",
        "-v",
        "-m", "not integration",
        "--tb=short"
    ]

    return pytest.main(pytest_args)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_name = sys.argv[1]
        if test_name == "fast":
            exit_code = run_fast_tests()
        else:
            exit_code = run_specific_test(test_name)
    else:
        exit_code = run_tests()

    sys.exit(exit_code)
