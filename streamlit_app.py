"""Streamlit entry point for the second hosted copy of the app (Streamlit Cloud needs a distinct main file
per app in the same repo and branch). Identical to app.py.  Run with:  streamlit run streamlit_app.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(Path(__file__).resolve().parent / ".env")

from frontend.dashboard import main  # noqa: E402

main()
