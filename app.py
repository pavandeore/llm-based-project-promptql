from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from sqlalchemy import create_engine, text, MetaData, Table
from sqlalchemy.exc import SQLAlchemyError
from openai import AsyncOpenAI
import chromadb
import os
import re
import json
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
import logging
from logging.handlers import RotatingFileHandler
import asyncio
import aiohttp
import time

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'dev-secret-key')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Async OpenAI
aclient = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class DatabaseSchemaManager:
    """Manages database schema information and vector embeddings"""
    
    def __init__(self):
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.chroma_client.get_or_create_collection(
            name="database_schema",
            metadata={"description": "Database table and column information"}
        )

        # 🧠 Caching summaries to avoid regenerating for 300+ tables
        self.summary_cache_path = "./schema_summary_cache.json"
        if os.path.exists(self.summary_cache_path):
            try:
                with open(self.summary_cache_path, "r") as f:
                    self.summary_cache = json.load(f)
            except Exception:
                self.summary_cache = {}
        else:
            self.summary_cache = {}
    
    async def generate_embeddings_async(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for schema texts asynchronously"""
        try:
            response = await aclient.embeddings.create(
                input=texts,
                model="text-embedding-3-small"
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise

    async def auto_summarize_tables_batch_async(self, tables: List[Dict[str, Any]], db_type: str) -> Dict[str, Any]:
        """
        Summarize multiple tables in one GPT call asynchronously.
        Returns a dict {table_name: summary_dict}
        """
        # Filter out tables already in cache
        tables_to_summarize = [t for t in tables if t['table_name'] not in self.summary_cache]
        if not tables_to_summarize:
            return {}

        table_descriptions = "\n\n".join([
            f"Table: {t['table_name']}\nColumns: {', '.join(t['columns'])}"
            for t in tables_to_summarize
        ])

        prompt = f"""
        You are an expert database analyst.
        You will receive several {db_type} tables (table name and columns).
        For each table, return a short JSON object describing:
        - "description": one concise sentence describing the table purpose
        - "column_descriptions": a mapping column_name -> short meaning
        - "semantic_tags": 3–5 keywords about the table's purpose

        Example JSON:
        {{
        "users": {{
            "description": "Stores information about registered users.",
            "column_descriptions": {{
            "user_id": "Unique user identifier",
            "email": "User's email address"
            }},
            "semantic_tags": ["user", "account", "email"]
        }},
        "orders": {{
            "description": "Tracks customer orders and their status.",
            "column_descriptions": {{
            "order_id": "Unique order identifier",
            "status": "Current order state"
            }},
            "semantic_tags": ["order", "transaction", "ecommerce"]
        }}
        }}

        Respond ONLY with valid JSON. Do not include explanations, markdown, or prose.

        Tables:
        {table_descriptions}
        """

        try:
            response = await aclient.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            raw = response.choices[0].message.content.strip()
            summaries = json.loads(raw)
        except Exception as e:
            logger.warning(f"⚠️ Batch summarization failed: {e}")
            return {}

        # Update cache and persist
        for name, summary in summaries.items():
            self.summary_cache[name] = summary

        with open(self.summary_cache_path, "w") as f:
            json.dump(self.summary_cache, f, indent=2)

        return summaries
    
    async def process_table_batches_parallel(self, schema_data: List[Dict[str, Any]], db_type: str = "unknown"):
        """Process table batches in parallel with 5 concurrent batches of 10 tables"""
        documents, metadatas, ids = [], [], []
        
        batch_size = 10
        max_concurrent_batches = 5
        
        # Create batches
        batches = [schema_data[i:i + batch_size] for i in range(0, len(schema_data), batch_size)]
        
        logger.info(f"🚀 Starting parallel processing of {len(batches)} batches ({len(schema_data)} total tables)")
        logger.info(f"📊 Configuration: {max_concurrent_batches} concurrent batches, {batch_size} tables per batch")
        
        start_time = time.time()
        
        # Process batches in parallel with semaphore to limit concurrency
        semaphore = asyncio.Semaphore(max_concurrent_batches)
        
        async def process_single_batch(batch, batch_num):
            async with semaphore:
                batch_start = time.time()
                logger.info(f"🧩 Processing batch {batch_num}/{len(batches)} ({len(batch)} tables)")
                
                summaries = await self.auto_summarize_tables_batch_async(batch, db_type)
                
                batch_docs, batch_metas, batch_ids = [], [], []
                
                for item in batch:
                    table_name = item['table_name']
                    columns = item['columns']

                    summary = self.summary_cache.get(table_name, summaries.get(table_name, {}))
                    desc = summary.get("description", "")
                    col_descs = summary.get("column_descriptions", {})
                    tags = summary.get("semantic_tags", [])

                    # Build enhanced document with relationship context
                    doc_parts = [f"Table: {table_name}. {desc}"]
                    
                    # Add columns with descriptions
                    column_text = "Columns: "
                    for c in columns:
                        meaning = col_descs.get(c, "")
                        column_text += f"{c} ({meaning}), "
                    doc_parts.append(column_text.rstrip(', '))
                    
                    # Add primary keys if available
                    if item.get('primary_keys'):
                        doc_parts.append(f"Primary keys: {', '.join(item['primary_keys'])}")
                    
                    # Add relationships if available
                    if item.get('foreign_keys'):
                        rel_text = "Connects to: "
                        relationships = []
                        for fk in item['foreign_keys']:
                            relationships.append(f"{fk['column']}→{fk['references_table']}")
                        doc_parts.append(rel_text + ", ".join(relationships))
                    
                    if tags:
                        doc_parts.append(f"Keywords: {', '.join(tags)}")
                    
                    doc_text = ". ".join(doc_parts)
                    batch_docs.append(doc_text)
                    
                    batch_metas.append({
                        "table_name": table_name,
                        "columns": json.dumps(columns),
                        "column_types": json.dumps(item.get("column_types", [])),
                        "primary_keys": json.dumps(item.get("primary_keys", [])),
                        "foreign_keys": json.dumps(item.get("foreign_keys", [])),
                        "database_type": item.get("database_type", "unknown")
                    })
                    batch_ids.append(f"{table_name}_{batch_num}_{hash(doc_text)}")
                
                batch_duration = time.time() - batch_start
                logger.info(f"✅ Batch {batch_num} completed in {batch_duration:.2f}s")
                
                return batch_docs, batch_metas, batch_ids
        
        # Process all batches concurrently
        batch_tasks = [process_single_batch(batch, i+1) for i, batch in enumerate(batches)]
        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
        
        # Combine results
        for result in batch_results:
            if isinstance(result, Exception):
                logger.error(f"Batch processing error: {result}")
                continue
            batch_docs, batch_metas, batch_ids = result
            documents.extend(batch_docs)
            metadatas.extend(batch_metas)
            ids.extend(batch_ids)
        
        # Generate embeddings for all documents in one go (async)
        logger.info(f"🧠 Generating embeddings for {len(documents)} tables...")
        embeddings_start = time.time()
        embeddings = await self.generate_embeddings_async(documents)
        embedding_duration = time.time() - embeddings_start
        logger.info(f"📈 Embeddings generated in {embedding_duration:.2f}s")
        
        # Store in ChromaDB
        store_start = time.time()
        self.collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        store_duration = time.time() - store_start
        
        total_duration = time.time() - start_time
        logger.info(f"🎉 All embeddings stored successfully. Total time: {total_duration:.2f}s")
        logger.info(f"📊 Performance: {len(schema_data)} tables in {total_duration:.2f}s ({len(schema_data)/total_duration:.2f} tables/sec)")

    async def store_schema_embeddings_async(self, schema_data: List[Dict[str, Any]], db_type: str = "unknown"):
        """Store enriched schema information efficiently using parallel batch processing"""
        await self.process_table_batches_parallel(schema_data, db_type)

    async def get_relevant_schema_async(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Get relevant schema information based on user query asynchronously"""
        try:
            # Generate embedding for the query
            query_embedding = (await self.generate_embeddings_async([query]))[0]
            
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
                    "primary_keys": json.loads(metadata.get("primary_keys", "[]")),
                    "foreign_keys": json.loads(metadata.get("foreign_keys", "[]")),
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
    
    def extract_detailed_schema_with_relationships(self, engine, db_type: str) -> List[Dict[str, Any]]:
        """Extract schema with foreign keys and relationships"""
        schema_data = []
        metadata = MetaData()
        
        try:
            with engine.connect() as conn:
                # Reflect all tables with foreign keys
                metadata.reflect(bind=engine)
                
                for table_name, table in metadata.tables.items():
                    table_info = {
                        "table_name": table_name,
                        "columns": [],
                        "column_types": [],
                        "descriptions": [],
                        "primary_keys": [col.name for col in table.primary_key],
                        "foreign_keys": [],
                        "relationships": [],
                        "database_type": db_type
                    }
                    
                    # Extract columns and foreign keys
                    for column in table.columns:
                        table_info["columns"].append(column.name)
                        table_info["column_types"].append(str(column.type))
                        
                        # Extract foreign key relationships
                        for fk in column.foreign_keys:
                            fk_info = {
                                "column": column.name,
                                "references_table": fk.column.table.name,
                                "references_column": fk.column.name
                            }
                            table_info["foreign_keys"].append(fk_info)
                            table_info["relationships"].append(
                                f"REFERENCES {fk.column.table.name}({fk.column.name})"
                            )
                    
                    schema_data.append(table_info)
            
            return schema_data
            
        except Exception as e:
            logger.error(f"Error extracting schema with relationships: {e}")
            raise

    async def extract_and_store_schema_async(self, engine, db_type: str) -> List[Dict[str, Any]]:
        """Extract schema and store embeddings asynchronously"""
        schema_data = self.extract_detailed_schema_with_relationships(engine, db_type)
        await self.schema_manager.store_schema_embeddings_async(schema_data, db_type)
        return schema_data

    async def get_enhanced_schema_for_query_async(self, query: str, n_results: int = 5) -> str:
        """Get relevant schema with relationship context asynchronously"""
        relevant_tables = await self.schema_manager.get_relevant_schema_async(query, n_results)
        
        schema_info = "Database Schema with Relationships:\n\n"
        
        for table in relevant_tables:
            schema_info += f"📊 Table: {table['table_name']}\n"
            schema_info += f"   Columns: {', '.join(table['columns'])}\n"
            
            # Add primary keys
            if table.get('primary_keys'):
                schema_info += f"   Primary Keys: {', '.join(table['primary_keys'])}\n"
            
            # Add foreign keys and relationships
            if table.get('foreign_keys'):
                schema_info += "   Relationships:\n"
                for fk in table['foreign_keys']:
                    schema_info += f"     - {fk['column']} → {fk['references_table']}.{fk['references_column']}\n"
            
            schema_info += "\n"
        
        return schema_info

async def generate_sql_with_relationships_async(natural_language_query: str, schema_info: str, db_type: str) -> Optional[str]:
    """Generate SQL with understanding of table relationships asynchronously"""
    
    enhanced_prompt = f"""
You are an expert SQL developer. Convert the following natural language request into an accurate SQL query.

Use the provided database schema (with relationships) to determine which tables and joins are needed.

Database Schema with Relationships:
{schema_info}

CRITICAL INSTRUCTIONS:
1. Carefully read the natural language request and determine which tables are relevant.
2. Use JOINs **only if** the query logically requires combining data from multiple related tables.
3. If the query can be satisfied from a single table, use only that table — no unnecessary JOINs.
4. Use foreign key relationships as hints for possible joins (not mandatory unless relevant).
5. Always generate syntactically correct SQL for {db_type}.
6. Return **only the SQL query**, no explanations or commentary.
7. Include all appropriate WHERE, GROUP BY, ORDER BY, or LIMIT clauses based on intent.
8. Use clear table aliases if multiple tables are involved.
9. Handle is_active or status-like columns using typical conventions (e.g., 'Y', 'N', 'active', 'inactive', true, false, 0, 1).
10. Ensure column and table names match the schema exactly.

Natural language query: {natural_language_query}

SQL Query:
"""

    try:
        response = await aclient.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a SQL expert that understands database relationships and generates accurate JOIN queries."},
                {"role": "user", "content": enhanced_prompt}
            ],
            temperature=0.1,
            max_tokens=1000
        )
        
        sql_query = response.choices[0].message.content.strip()
        sql_query = re.sub(r'```sql\s*|\s*```', '', sql_query).strip()
        
        return sql_query
        
    except Exception as e:
        logger.error(f"Error generating SQL with relationships: {e}")
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

def run_async(coro):
    """Run async coroutine in sync context"""
    try:
        # Try to get existing event loop
        loop = asyncio.get_event_loop()
    except RuntimeError:
        # Create new event loop if none exists
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(coro)

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
            
            # Get relevant schema information using vector embeddings ASYNC
            db_manager = DatabaseManager()
            schema_info = run_async(db_manager.get_enhanced_schema_for_query_async(natural_language_query))
            
            # Generate SQL from natural language ASYNC
            sql_query = run_async(generate_sql_with_relationships_async(
                natural_language_query, 
                schema_info, 
                session['db_type']
            ))
            
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
        
        # Get relevant schema ASYNC
        db_manager = DatabaseManager()
        schema_info = run_async(db_manager.get_enhanced_schema_for_query_async(natural_language_query))
        
        # Generate SQL ASYNC
        sql_query = run_async(generate_sql_with_relationships_async(
            natural_language_query,
            schema_info,
            db_config.get('db_type', 'postgresql')
        ))
        
        return jsonify({
            'sql_query': sql_query,
            'schema_used': schema_info
        })
        
    except Exception as e:
        logger.error(f"API error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/admin/refresh-schema', methods=['POST', 'GET'])
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
            import shutil
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