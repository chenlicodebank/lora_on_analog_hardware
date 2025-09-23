#!/usr/bin/env python3
"""
Wrapper script for lm_eval that ensures custom models are properly registered.

This script imports the custom_analog_model to trigger the @register_model decorator,
then calls lm_eval with the provided arguments.
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import our custom model to register it
try:
    import custom_analog_model
    print("✓ Successfully imported and registered custom_analog model")
except Exception as e:
    print(f"✗ Failed to import custom_analog_model: {e}")
    sys.exit(1)

# Now import and run lm_eval
try:
    from lm_eval.__main__ import cli_evaluate
    
    # Call the main evaluation function with sys.argv
    sys.exit(cli_evaluate())
    
except Exception as e:
    print(f"✗ Failed to run lm_eval: {e}")
    sys.exit(1) 