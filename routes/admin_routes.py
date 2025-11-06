from flask import Blueprint, jsonify, session
from services.database.database_manager import DatabaseManager
from utils.db_url import get_db_url
from utils.async_runner import run_async
from sqlalchemy import create_engine
import os
import shutil
import logging

logger = logging.getLogger(__name__)
admin_bp = Blueprint('admin', __name__)

@admin_bp.route('/admin/refresh-schema', methods=['POST', 'GET'])
def refresh_schema():
    """
    Rebuild schema embeddings and GPT summaries asynchronously.
    """
    try:
        if not all(key in session for key in ['db_type', 'db_host', 'db_name', 'db_user']):
            return jsonify({"error": "Database connection details missing in session. Reconnect first."}), 400

        # Load database connection details from session
        db_url = get_db_url(
            session['db_type'],
            session['db_host'],
            session.get('db_port', '5432'),
            session['db_name'],
            session['db_user'],
            session['db_password']
        )

        engine = create_engine(db_url)
        db_manager = DatabaseManager()

        # 🧠 Clear old cache and chroma data
        cache_path = "./schema_summary_cache.json"
        chroma_path = "./chroma_db"

        if os.path.exists(cache_path):
            os.remove(cache_path)
            logger.info("Deleted old schema summary cache.")

        if os.path.exists(chroma_path):
            shutil.rmtree(chroma_path)
            logger.info("Deleted old ChromaDB embeddings.")

        # 🏗️ Rebuild schema and store embeddings ASYNC
        logger.info("🔄 Starting async schema refresh...")
        schema_data = run_async(db_manager.extract_and_store_schema_async(engine, session['db_type']))
        logger.info(f"🎉 Refreshed schema embeddings for {len(schema_data)} tables")

        return jsonify({
            "status": "success",
            "tables_processed": len(schema_data),
            "message": "Schema summaries and embeddings rebuilt successfully."
        }), 200

    except Exception as e:
        logger.error(f"Schema refresh failed: {e}")
        return jsonify({"error": str(e)}), 500