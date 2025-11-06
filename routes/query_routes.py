from flask import Blueprint, render_template, request, session, redirect, url_for
from services.database.database_manager import DatabaseManager
from services.llm.sql_generation import generate_sql_with_relationships_async
from services.llm.query_rewriter import rewrite_user_query_for_quality
from services.database.sql_executor import auto_fix_and_execute_query
from services.visualization.chart_engine import determine_smart_chart_type_multi_step
from services.database.postgres_helper import quote_identifiers_if_postgres
from utils.db_url import get_db_url
from utils.async_runner import run_async
from utils.serialize import universal_serialize
from sqlalchemy import create_engine
import re
import logging

logger = logging.getLogger(__name__)
query_bp = Blueprint('query', __name__)

@query_bp.route('/query', methods=['GET', 'POST'])
def query():
    if not all(key in session for key in ['db_type', 'db_host', 'db_name', 'db_user', 'schema_loaded']):
        return redirect(url_for('index.index'))

    if request.method == 'POST':
        natural_language_query = request.form.get('query')
        
        if not natural_language_query:
            return render_template('results.html', error="Please enter a query")

        try:
            # Create database connection
            db_url = get_db_url(
                session['db_type'],
                session['db_host'],
                session.get('db_port', '5432'),
                session['db_name'],
                session['db_user'],
                session['db_password']
            )
            
            engine = create_engine(db_url)
            
            # Get relevant schema info using embeddings
            db_manager = DatabaseManager()
            schema_info = run_async(db_manager.get_enhanced_schema_for_query_async(natural_language_query))

            # Parse relevant table names from schema_info text
            relevant_tables = re.findall(r'📊 Table: (\w+)', schema_info)

            # Fetch 1 sample row per relevant table
            sample_data = db_manager.schema_manager.fetch_sample_rows(engine, relevant_tables)

            # Step 2: Enhance the question first
            enhanced_query = run_async(rewrite_user_query_for_quality(natural_language_query))
            logger.info(f"Enhanced Query for SQL: {enhanced_query}")

            # Generate SQL
            sql_query = run_async(generate_sql_with_relationships_async(
                enhanced_query,
                schema_info,
                session['db_type'],
                sample_data,
                relevant_tables
            ))

            if not sql_query:
                return render_template('results.html', error="Failed to generate SQL query. Please try again.")

            # Apply PostgreSQL quoting if needed
            sql_query = quote_identifiers_if_postgres(sql_query, session['db_type'])
            logger.info(f"Generated SQL: {sql_query}")

            # Execute the SQL
            from extensions import aclient
            rows, columns, final_sql = run_async(auto_fix_and_execute_query(
                aclient,
                engine,
                sql_query,
                natural_language_query,
                schema_info,
                session['db_type']
            ))

            # Serialize full rows for table display
            serialized_rows = [
                {c: universal_serialize(v) for c, v in zip(columns, r)} for r in rows
            ]

            # Run GPT-based chart analyzer multi-step approach:
            chart_info = run_async(determine_smart_chart_type_multi_step(rows, columns, natural_language_query))

            # Use the new chart_info structure
            recommended_cols = chart_info.get("recommended_columns", columns)
            chart_type = chart_info.get("chart_type", "table")
            chart_description = chart_info.get("description", "")
            series_config = chart_info.get("series_config", [])
            chart_title = chart_info.get("chart_title", "")
            primary_insight = chart_info.get("primary_insight", "")
            all_charts = chart_info.get("all_charts", [])

            # Filtered rows ONLY for visualization
            chart_rows = [
                {col: row[col] for col in recommended_cols if col in row}
                for row in serialized_rows
            ]

            # Pass both: all rows for table, filtered for chart
            chart_data = chart_rows

            logger.info(f"Determined chart type: {chart_type}")
            logger.info(f"Chart data: {chart_data}")

            columns = [str(c) for c in list(columns)]  # ensure JSON-safe strings

            return render_template(
                'results.html',
                query=natural_language_query,
                sql_query=final_sql,
                columns=columns,
                rows=serialized_rows,
                schema_info=schema_info,
                chart_type=chart_type,
                chart_title=chart_title,
                chart_description=chart_description,
                primary_insight=primary_insight,
                all_charts=all_charts,
                series_config=series_config,
                chart_data=chart_data,
                recommended_columns=recommended_cols
            )
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return render_template('results.html', error=f"Error: {str(e)}")

    return render_template('results.html')