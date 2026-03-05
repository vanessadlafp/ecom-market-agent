"""Entry point — run with: python main.py  OR  uvicorn main:app"""
import uvicorn
from src.api.routes import app
from config.settings import get_settings

if __name__ == "__main__":
    s = get_settings()
    uvicorn.run("main:app", host=s.api_host, port=s.api_port, reload=s.api_reload, log_level="info")
