#!/usr/bin/env python3
"""
Project structure validation script for Vietnamese ID Card OCR.
This script validates that all required directories and files are in place.
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple


class StructureValidator:
    """Validates the project structure."""

    def __init__(self, root_dir: Path = None):
        self.root_dir = root_dir or Path(__file__).parent
        self.errors = []
        self.warnings = []

    def check_directory(self, path: str, required: bool = True) -> bool:
        """Check if a directory exists."""
        full_path = self.root_dir / path
        exists = full_path.is_dir()

        if not exists and required:
            self.errors.append(f"❌ Missing required directory: {path}")
        elif not exists:
            self.warnings.append(f"⚠️  Optional directory missing: {path}")
        else:
            print(f"✅ Directory exists: {path}")

        return exists

    def check_file(self, path: str, required: bool = True) -> bool:
        """Check if a file exists."""
        full_path = self.root_dir / path
        exists = full_path.is_file()

        if not exists and required:
            self.errors.append(f"❌ Missing required file: {path}")
        elif not exists:
            self.warnings.append(f"⚠️  Optional file missing: {path}")
        else:
            print(f"✅ File exists: {path}")

        return exists

    def validate_structure(self):
        """Validate the complete project structure."""
        print("🔍 Validating Vietnamese ID Card OCR project structure...\n")

        # Core files
        print("📄 Checking core files...")
        self.check_file("README.md")
        self.check_file("requirements_windows.txt")
        self.check_file("setup.py")
        self.check_file("Makefile")
        self.check_file("QUICK_START.md")

        # Entry points
        print("\n🚀 Checking entry points...")
        self.check_file("app.py")
        self.check_file("api_app.py")
        self.check_file("streamlit_app.py")

        # Source code structure
        print("\n📁 Checking source code structure...")
        self.check_directory("src")
        self.check_file("src/__init__.py")
        self.check_directory("src/api")
        self.check_file("src/api/fastapi_app.py")
        self.check_directory("src/core")
        self.check_file("src/core/id_card_processor.py")
        self.check_directory("src/models")
        self.check_file("src/models/model_manager.py")
        self.check_directory("src/utils")
        self.check_file("src/utils/image_processing.py")
        self.check_file("src/utils/text_processing.py")
        self.check_directory("src/ui")
        self.check_file("src/ui/streamlit_app.py")

        # Configuration
        print("\n⚙️ Checking configuration...")
        self.check_directory("config")
        self.check_file("config/settings.py")
        self.check_file("config/.env.example")
        self.check_file(".env", required=False)

        # Data structure
        print("\n💾 Checking data structure...")
        self.check_directory("data")
        self.check_directory("data/models")
        self.check_directory("data/dictionary")
        self.check_directory("data/samples", required=False)
        self.check_directory("data/uploads", required=False)
        self.check_directory("data/outputs", required=False)

        # Deployment structure
        print("\n🚀 Checking deployment structure...")
        self.check_directory("deployment")
        self.check_directory("deployment/docker")
        self.check_file("deployment/docker/Dockerfile")
        self.check_file("deployment/docker/docker-compose.yml")
        self.check_directory("deployment/k8s")
        self.check_directory("deployment/k3d")

        # Scripts
        print("\n🔧 Checking scripts...")
        self.check_directory("scripts")
        self.check_directory("scripts/setup")
        self.check_directory("scripts/dev")

        # Monitoring
        print("\n📊 Checking monitoring structure...")
        self.check_directory("monitoring")
        self.check_directory("monitoring/prometheus", required=False)
        self.check_directory("monitoring/grafana", required=False)

        # Documentation
        print("\n📚 Checking documentation...")
        self.check_directory("docs")
        self.check_file("docs/PROJECT_STRUCTURE.md")

        # Tests
        print("\n🧪 Checking test structure...")
        self.check_directory("tests", required=False)

        # Archive
        print("\n📦 Checking archive...")
        self.check_directory("archive", required=False)

    def create_missing_directories(self):
        """Create missing directories."""
        required_dirs = [
            "logs", "data/uploads", "data/outputs", "data/samples",
            "tests", "tests/unit", "tests/integration", "tests/fixtures"
        ]

        print("\n🏗️ Creating missing directories...")
        for dir_path in required_dirs:
            full_path = self.root_dir / dir_path
            if not full_path.exists():
                full_path.mkdir(parents=True, exist_ok=True)
                print(f"✅ Created directory: {dir_path}")

    def report_results(self):
        """Report validation results."""
        print("\n" + "="*50)
        print("📋 VALIDATION RESULTS")
        print("="*50)

        if not self.errors and not self.warnings:
            print("🎉 Perfect! Project structure is complete and valid.")
        else:
            if self.errors:
                print(f"\n❌ Found {len(self.errors)} errors:")
                for error in self.errors:
                    print(f"  {error}")

            if self.warnings:
                print(f"\n⚠️  Found {len(self.warnings)} warnings:")
                for warning in self.warnings:
                    print(f"  {warning}")

        print(f"\n📊 Summary:")
        print(f"  - Errors: {len(self.errors)}")
        print(f"  - Warnings: {len(self.warnings)}")

        if self.errors:
            print("\n💡 To fix errors, run: make setup-config")
            return False
        return True


def main():
    """Main function."""
    validator = StructureValidator()

    # Validate structure
    validator.validate_structure()

    # Create missing directories
    validator.create_missing_directories()

    # Report results
    success = validator.report_results()

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
