"""
Python version check to enforce Python 3.13 ONLY.

This module ensures that the transformer lab is only run on Python 3.13,
preventing compatibility issues and ensuring modern syntax support.
"""

import sys


def check_python_version() -> None:
    """Verify Python 3.13 is being used, exit with error if not."""
    if sys.version_info[:2] != (3, 13):
        current_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        print(f"❌ Python 3.13 required! Current version: {current_version}")
        print("🔧 Please use Python 3.13:")
        print("   - Use 'python3.13' command")
        print("   - Update your PATH to use Python 3.13")
        print("   - Use a virtual environment with Python 3.13")
        sys.exit(1)


# Run check on import
check_python_version()