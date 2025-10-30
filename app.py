from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from sqlalchemy import create_engine, text, MetaData, Table
from sqlalchemy.exc import SQLAlchemyError
from openai import OpenAI
import chromadb
import os
import re
import json
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
import logging
from logging.handlers import RotatingFileHandler

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'dev-secret-key')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class DatabaseSchemaManager:
    """Manages database schema information and vector embeddings"""
    
    def __init__(self):
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.chroma_client.get_or_create_collection(
            name="database_schema",
            metadata={"description": "Database table and column information"}
        )
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for schema texts"""
        try:
            response = client.embeddings.create(
                input=texts,
                model="text-embedding-3-small"
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    def store_schema_embeddings(self, schema_data: List[Dict[str, Any]]):
        """Store schema information with embeddings"""
        documents = []
        metadatas = []
        ids = []
        
        for item in schema_data:
            doc_text = f"Table: {item['table_name']}. Columns: {', '.join(item['columns'])}. Descriptions: {item.get('descriptions', '')}"
            documents.append(doc_text)
            metadatas.append({
                "table_name": item["table_name"],
                "columns": json.dumps(item["columns"]),
                "column_types": json.dumps(item.get("column_types", [])),
                "database_type": item.get("database_type", "unknown")
            })
            ids.append(f"{item['table_name']}_{hash(doc_text)}")
        
        # Generate embeddings
        embeddings = self.generate_embeddings(documents)
        
        # Store in ChromaDB
        self.collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
    
    def get_relevant_schema(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Get relevant schema information based on user query"""
        try:
            # Generate embedding for the query
            query_embedding = self.generate_embeddings([query])[0]
            
            # Query ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            
            relevant_schema = []
            for i, metadata in enumerate(results['metadatas'][0]):
                relevant_schema.append({
                    "table_name": metadata["table_name"],
                    "columns": json.loads(metadata["columns"]),
                    "column_types": json.loads(metadata.get("column_types", "[]")),
                    "database_type": metadata.get("database_type", "unknown"),
                    "distance": results['distances'][0][i] if results['distances'] else 0
                })
            
            return relevant_schema
        except Exception as e:
            logger.error(f"Error getting relevant schema: {e}")
            return []

class DatabaseManager:
    """Manages database connections and schema extraction"""
    
    def __init__(self):
        self.schema_manager = DatabaseSchemaManager()
    
    def extract_detailed_schema(self, engine, db_type: str) -> List[Dict[str, Any]]:
        """Extract detailed schema information from database"""
        schema_data = []
        metadata = MetaData()
        
        try:
            with engine.connect() as conn:
                # Reflect all tables
                metadata.reflect(bind=engine)
                
                for table_name, table in metadata.tables.items():
                    table_info = {
                        "table_name": table_name,
                        "columns": [],
                        "column_types": [],
                        "descriptions": [],
                        "database_type": db_type
                    }
                    
                    for column in table.columns:
                        table_info["columns"].append(column.name)
                        table_info["column_types"].append(str(column.type))
                        table_info["descriptions"].append(f"{column.name} ({str(column.type)})")
                    
                    schema_data.append(table_info)
            
            # Store schema in vector database
            self.schema_manager.store_schema_embeddings(schema_data)
            
            return schema_data
            
        except Exception as e:
            logger.error(f"Error extracting schema: {e}")
            raise
    
    def get_relevant_schema_for_query(self, query: str, n_results: int = 5) -> str:
        """Get relevant schema information for a natural language query"""
        relevant_tables = self.schema_manager.get_relevant_schema(query, n_results)
        
        schema_info = "Relevant database schema:\n\n"
        for table in relevant_tables:
            schema_info += f"Table: {table['table_name']}\n"
            schema_info += f"Columns: {', '.join(table['columns'])}\n"
            if table.get('column_types'):
                schema_info += f"Types: {', '.join(table['column_types'])}\n"
            schema_info += "\n"
        
        return schema_info

def generate_sql_from_natural_language(natural_language_query: str, schema_info: str, db_type: str) -> Optional[str]:
    """Use OpenAI to generate SQL from natural language with relevant schema"""
    
    prompt = f"""
You are an expert SQL developer. Convert the following natural language query into SQL.

Database schema information:
{schema_info}

Important instructions:
- For is_active column, check with value 'Y' or 'N' only
- Use proper SQL syntax for {db_type}
- Return only the SQL query without any additional explanation
- Do not include markdown formatting or code blocks
- Use appropriate WHERE clauses based on the query

Natural language query: {natural_language_query}

SQL Query:
"""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful SQL assistant that generates accurate SQL queries. Always return only the SQL query without any additional text."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=500
        )
        
        sql_query = response.choices[0].message.content.strip()
        
        # Clean up the response (remove markdown code blocks if present)
        sql_query = re.sub(r'```sql\s*|\s*```', '', sql_query).strip()
        
        return sql_query
        
    except Exception as e:
        logger.error(f"Error generating SQL: {e}")
        return None

def quote_identifiers_if_postgres(sql_query: str, db_type: str) -> str:
    """
    If using PostgreSQL, wrap table and column identifiers in double quotes 
    to preserve case sensitivity.
    """
    if db_type != 'postgresql' or not sql_query:
        return sql_query

    # Simple pattern: match words used after FROM or JOIN or UPDATE etc.
    keywords = ["FROM", "JOIN", "UPDATE", "INTO", "TABLE"]
    for kw in keywords:
        sql_query = re.sub(
            rf"(?i)\b{kw}\s+(\w+)",
            lambda m: f'{m.group(0).split()[0]} "{m.group(1)}"',
            sql_query
        )

    # Quote columns in SELECT, WHERE, GROUP BY, ORDER BY
    sql_query = re.sub(
        r'(?i)(SELECT|WHERE|AND|OR|GROUP BY|ORDER BY|HAVING)\s+(\w+)\s*',
        lambda m: f'{m.group(1)} "{m.group(2)}" ',
        sql_query
    )

    return sql_query

def get_db_url(db_type: str, db_host: str, db_port: str, db_name: str, db_user: str, db_password: str) -> str:
    """Generate the appropriate database URL based on the database type"""
    if db_type == 'postgresql':
        return f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    elif db_type == 'mysql':
        return f"mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    elif db_type == 'sqlite':
        return f"sqlite:///{db_name}"
    else:
        raise ValueError(f"Unsupported database type: {db_type}")

@app.route('/', methods=['GET', 'POST'])
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
            
            # Initialize database manager and extract schema
            db_manager = DatabaseManager()
            schema_data = db_manager.extract_detailed_schema(engine, session['db_type'])
            session['schema_loaded'] = True
            
            logger.info(f"Successfully loaded schema for {len(schema_data)} tables")
            
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            return render_template('index.html', 
                                error=f"Database connection failed: {str(e)}")
        
        return redirect(url_for('query'))
    
    return render_template('index.html')

@app.route('/query', methods=['GET', 'POST'])
def query():
    if not all(key in session for key in ['db_type', 'db_host', 'db_name', 'db_user', 'schema_loaded']):
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        natural_language_query = request.form.get('query')
        
        if not natural_language_query:
            return render_template('query.html', error="Please enter a query")
        
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
            
            # Get relevant schema information using vector embeddings
            db_manager = DatabaseManager()
            schema_info = db_manager.get_relevant_schema_for_query(natural_language_query)
            
            # Generate SQL from natural language
            sql_query = generate_sql_from_natural_language(
                natural_language_query, 
                schema_info, 
                session['db_type']
            )
            
            if not sql_query:
                return render_template('results.html', 
                                     error="Failed to generate SQL query. Please try rephrasing your question.")
            
            # Apply PostgreSQL quoting if needed
            sql_query = quote_identifiers_if_postgres(sql_query, session['db_type'])
            
            logger.info(f"Generated SQL: {sql_query}")
            
            # Execute the SQL query
            with engine.connect() as conn:
                result = conn.execute(text(sql_query))
                rows = result.fetchall()
                columns = result.keys()
                
            return render_template('results.html', 
                                 query=natural_language_query,
                                 sql_query=sql_query,
                                 columns=columns,
                                 rows=rows,
                                 schema_info=schema_info)
            
        except SQLAlchemyError as e:
            logger.error(f"Database error: {e}")
            return render_template('results.html', 
                                 error=f"Database error: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return render_template('results.html', 
                                 error=f"Error: {str(e)}")
    
    return render_template('query.html')

@app.route('/api/generate-sql', methods=['POST'])
def api_generate_sql():
    """API endpoint for generating SQL"""
    try:
        data = request.get_json()
        natural_language_query = data.get('query')
        db_config = data.get('db_config')
        
        if not natural_language_query or not db_config:
            return jsonify({'error': 'Missing query or db_config'}), 400
        
        # Get relevant schema
        db_manager = DatabaseManager()
        schema_info = db_manager.get_relevant_schema_for_query(natural_language_query)
        
        # Generate SQL
        sql_query = generate_sql_from_natural_language(
            natural_language_query,
            schema_info,
            db_config.get('db_type', 'postgresql')
        )
        
        return jsonify({
            'sql_query': sql_query,
            'schema_used': schema_info
        })
        
    except Exception as e:
        logger.error(f"API error: {e}")
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    return render_template('error.html', error="Page not found"), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('error.html', error="Internal server error"), 500

if __name__ == '__main__':
    # Production configuration
    handler = RotatingFileHandler('app.log', maxBytes=10000, backupCount=3)
    handler.setLevel(logging.INFO)
    app.logger.addHandler(handler)
    
    app.run(host='0.0.0.0', port=9000, debug=False)