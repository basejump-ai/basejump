"""Load environment variables"""

from pathlib import Path
from dotenv import load_dotenv

dotenv_path = Path(__file__).parents[2] / ".env"
load_dotenv(dotenv_path=dotenv_path)
