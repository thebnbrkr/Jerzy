import os
import sys
from pathlib import Path

# Add the parent directory to sys.path to make the jerzy module importable
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# This file is automatically recognized by pytest and will run before tests
