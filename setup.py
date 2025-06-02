"""
Setup script for Vietnamese ID Card OCR package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read requirements


def read_requirements(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]


# Try to read Windows requirements first, fallback to Linux
try:
    requirements = read_requirements('requirements_windows.txt')
except FileNotFoundError:
    requirements = read_requirements('requirements.txt')

setup(
    name="vietnamese-id-card-ocr",
    version="1.0.0",
    author="Vietnamese ID Card OCR Team",
    author_email="",
    description="Vietnamese ID Card OCR system using deep learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/loingtan/Vietnamese_ID_Card_OCR",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "isort>=5.0",
        ],
        "api": [
            "fastapi>=0.68.0",
            "uvicorn>=0.15.0",
            "prometheus-client>=0.11.0",
        ],
        "ui": [
            "streamlit>=1.20.0",
            "pandas>=1.3.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "vnid-streamlit=src.ui.streamlit_app:main",
            "vnid-api=src.api.fastapi_app:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.txt", "*.md", "*.yml", "*.yaml"],
    },
)
