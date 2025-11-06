import json
import os
import logging

logger = logging.getLogger(__name__)

class CacheManager:
    def __init__(self, cache_path="./schema_summary_cache.json"):
        self.cache_path = cache_path
        self.cache = self._load_cache()
    
    def _load_cache(self):
        """Load cache from file"""
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load cache: {e}")
                return {}
        else:
            return {}
    
    def save_cache(self):
        """Save cache to file"""
        try:
            with open(self.cache_path, "w") as f:
                json.dump(self.cache, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving cache: {e}")
    
    def get(self, key):
        """Get value from cache"""
        return self.cache.get(key)
    
    def set(self, key, value):
        """Set value in cache and save"""
        self.cache[key] = value
        self.save_cache()
    
    def clear(self):
        """Clear cache"""
        self.cache = {}
        if os.path.exists(self.cache_path):
            os.remove(self.cache_path)