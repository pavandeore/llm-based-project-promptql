from flask import Blueprint, request, jsonify
from services.database.database_manager import DatabaseManager
from services.llm.sql_generation import generate_sql_with_relationships_async
from services.storage.logic_manager import save_application_logic, get_application_logic
from utils.db_url import get_db_url
from utils.async_runner import run_async
from sqlalchemy import create_engine
import re
import logging

logger = logging.getLogger(__name__)
api_bp = Blueprint('api', __name__)

@api_bp.route('/api/generate-sql', methods=['POST'])
def api_generate_sql():
    try:
        data = request.get_json()
        natural_language_query = data.get('query')
        db_config = data.get('db_config')

        if not natural_language_query or not db_config:
            return jsonify({'error': 'Missing query or db_config'}), 400

        # Build db_url from db_config
        db_url = get_db_url(
            db_config.get('db_type'),
            db_config.get('db_host'),
            db_config.get('db_port', '5432'),
            db_config.get('db_name'),
            db_config.get('db_user'),
            db_config.get('db_password')
        )

        db_manager = DatabaseManager()
        schema_info = run_async(db_manager.get_enhanced_schema_for_query_async(
            natural_language_query, n_results=5))

        relevant_tables = re.findall(r'📊 Table: (\w+)', schema_info)

        engine = create_engine(db_url)
        sample_data = db_manager.schema_manager.fetch_sample_rows(engine, relevant_tables)

        sql_query = run_async(generate_sql_with_relationships_async(
            natural_language_query,
            schema_info,
            db_config.get('db_type'),
            sample_data,
            relevant_tables
        ))

        return jsonify({
            'sql_query': sql_query,
            'schema_used': schema_info
        })

    except Exception as e:
        logger.error(f"API error: {e}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/api/save-logic', methods=['POST'])
def save_application_logic_route():
    """Save application-level logic text to local JSON file."""
    try:
        data = request.get_json()
        logic_text = data.get("logic", "").strip()

        if not logic_text:
            return jsonify({"error": "No logic text provided"}), 400

        save_application_logic(logic_text)
        return jsonify({"status": "success"})

    except Exception as e:
        logger.error(f"Error saving application logic: {e}")
        return jsonify({"error": str(e)}), 500

@api_bp.route('/api/get-logic', methods=['GET'])
def get_application_logic_route():
    """Return saved logic for prefill."""
    try:
        logic_data = get_application_logic()
        return jsonify(logic_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500