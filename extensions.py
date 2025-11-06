import logging
from logging.handlers import RotatingFileHandler
from openai import AsyncOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

# ✅ Initialize immediately so imports always work
aclient = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ✅ Set up default logger
logger = logging.getLogger("extensions")
logger.setLevel(logging.INFO)

def init_extensions(app=None):
    global logger
    
    # Configure logging
    handler = RotatingFileHandler('app.log', maxBytes=10000, backupCount=3)
    handler.setLevel(logging.INFO)
    
    if app and hasattr(app, "logger"):
        app.logger.addHandler(handler)
    else:
        logger.addHandler(handler)
    
    logger.info("✅ Extensions initialized successfully.")
