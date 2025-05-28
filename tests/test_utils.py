import pytest
import os
import sys
from pathlib import Path

# Add the parent directory to sys.path to make the jerzy module importable
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Now we can import from jerzy
from jerzy.utils import *  # Import any utilities that might be defined

class TestUtils:
    """Tests for utility functions in the utils module."""
    
    def test_module_exists(self):
        """Verify that the utils module exists and can be imported."""
        # This is a placeholder test since the utils.py file exists but is currently empty
        # When functions are added to utils.py, specific tests should be added here
        pass
