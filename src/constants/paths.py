import os
import sys
from pathlib import Path

# Get the current directory
current_directory = Path(__file__).resolve()

# Get the parent directory
parent_directory = current_directory.parent.parent.parent


RES_PATH = os.path.join(parent_directory, 'res')

DATA_PATH = os.path.join(parent_directory, 'data')
