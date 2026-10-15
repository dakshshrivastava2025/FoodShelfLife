"""Launch the consolidated local Streamlit frontend."""
from pathlib import Path
import subprocess
import sys

if __name__ == '__main__':
    root = Path(__file__).resolve().parent
    raise SystemExit(subprocess.call([sys.executable, '-m', 'streamlit', 'run', str(root / 'demo_app.py'), *sys.argv[1:]], cwd=root))
