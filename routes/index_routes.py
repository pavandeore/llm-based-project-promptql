from flask import Blueprint, render_template, request, session, redirect, url_for
from services.database.database_manager import DatabaseManager
from utils.db_url import get_db_url
from utils.async_runner import run_async
from sqlalchemy import create_engine, text
import logging

logger = logging.getLogger(__name__)
index_bp = Blueprint('index', __name__)

@index_bp.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Store connection details in session
        session['db_type'] = request.form.get('db_type')
        session['db_host'] = request.form.get('db_host')
        session['db_port'] = request.form.get('db_port', '5432')
        session['db_name'] = request.form.get('db_name')
        session['db_user'] = request.form.get('db_user')
        session['db_password'] = request.form.get('db_password')
        
        # Test connection and extract schema
        try:
            db_url = get_db_url(
                session['db_type'],
                session['db_host'],
                session['db_port'],
                session['db_name'],
                session['db_user'],
                session['db_password']
            )
            
            engine = create_engine(db_url)
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))  # Test connection
            
            # Initialize database manager and extract schema ASYNC
            db_manager = DatabaseManager()
            logger.info("🔄 Starting async schema extraction and embedding...")
            
            # Run async schema extraction
            schema_data = run_async(db_manager.extract_and_store_schema_async(engine, session['db_type']))
            
            session['schema_loaded'] = True
            
            logger.info(f"🎉 Successfully loaded schema for {len(schema_data)} tables")
            
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            return render_template('index.html', 
                                error=f"Database connection failed: {str(e)}")
        
        return redirect(url_for('query.query'))
    
    return render_template('index.html')