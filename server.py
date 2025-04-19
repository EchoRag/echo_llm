import uvicorn
import os
import sys
from pathlib import Path

# Add virtualenv's site-packages to Python path
venv_path = Path.home() / "Envs" / "echoRag"
site_packages = venv_path / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
sys.path.insert(0, str(site_packages))

# Import the FastAPI app instance
from app.main import app

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)