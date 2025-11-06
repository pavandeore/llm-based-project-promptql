import json
import datetime
import os
import logging

logger = logging.getLogger(__name__)

def save_application_logic(logic_text: str):
    """Save application-level logic text to local JSON file."""
    try:
        logic_path = "./application_logic.json"

        # Save to file
        with open(logic_path, "w") as f:
            json.dump({"logic": logic_text, "updated": datetime.datetime.now().isoformat()}, f, indent=2)

        logger.info("✅ Application logic saved successfully.")
        return {"status": "success"}

    except Exception as e:
        logger.error(f"Error saving application logic: {e}")
        raise e

def get_application_logic():
    """Return saved logic for prefill."""
    try:
        path = "./application_logic.json"
        if os.path.exists(path):
            with open(path, "r") as f:
                data = json.load(f)
            return data
        else:
            return {"logic": ""}
    except Exception as e:
        logger.error(f"Error loading application logic: {e}")
        return {"logic": ""}