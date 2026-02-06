#!/usr/bin/env python3
"""
Standalone runner script for LLaVA-Med integration on Lambda Labs.
This script can be run directly without needing to use -m module syntax.
"""

import os
import sys
from pathlib import Path

# Add the parent directory to Python path so we can import llavamed_modules
script_dir = Path(__file__).parent
parent_dir = script_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# Now import and run main
if __name__ == "__main__":
    from llavamed_modules.main import main
    
    print("=" * 70)
    print("LLaVA-Med Integration - Starting Pipeline")
    print("=" * 70)
    print(f"Working directory: {os.getcwd()}")
    print(f"Script directory: {script_dir}")
    print(f"Python path: {sys.path[0]}")
    print("=" * 70)
    print()
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)



