import logging
from logging.handlers import RotatingFileHandler
from openai import AsyncOpenAI
import os

aclient = None
logger = None

def init_extensions(app):
    global aclient, logger
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Production logging configuration
    handler = RotatingFileHandler('app.log', maxBytes=10000, backupCount=3)
    handler.setLevel(logging.INFO)
    app.logger.addHandler(handler)
    
    # Configure Async OpenAI
    aclient = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))