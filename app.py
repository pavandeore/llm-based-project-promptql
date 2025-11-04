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
from datetime import date
import decimal
import datetime, decimal, uuid, numpy as np

import tempfile
import shutil
from pathlib import Path


load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'dev-secret-key')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Async OpenAI
aclient = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def serialize_value(value):
    """Convert complex SQLAlchemy values into JSON-serializable Python values."""
    import datetime, decimal

    if isinstance(value, (datetime.date, datetime.datetime)):
        return value.isoformat()
    elif isinstance(value, decimal.Decimal):
        return float(value)
    elif isinstance(value, bytes):
        return value.decode(errors="ignore")
    elif value is None:
        return None
    else:
        return value

class DatabaseSchemaManager:
    """Manages database schema information and vector embeddings"""
    
    def __init__(self):
        # ✅ Choose default persistent path
        default_path = Path("./chroma_db").resolve()
        os.makedirs(default_path, exist_ok=True)

        # ✅ Check write permission (industry-safe pattern)
        try:
            test_file = default_path / ".write_test"
            with open(test_file, "w") as f:
                f.write("ok")
            test_file.unlink()  # cleanup test file
            chroma_path = str(default_path)
        except (IOError, PermissionError):
            # 🔒 Fallback to a guaranteed writable directory
            fallback_path = Path(tempfile.gettempdir()) / "chroma_db"
            os.makedirs(fallback_path, exist_ok=True)
            chroma_path = str(fallback_path)
            logger.warning(f"⚠️ Primary ChromaDB path not writable. Using fallback: {chroma_path}")

        # ✅ Initialize persistent Chroma safely
        self.chroma_client = chromadb.PersistentClient(path=chroma_path)
        self.collection = self.chroma_client.get_or_create_collection(
            name="database_schema",
            metadata={"description": "Database table and column information"}
        )

        # ✅ Cache path setup
        self.summary_cache_path = "./schema_summary_cache.json"
        if os.path.exists(self.summary_cache_path):
            try:
                with open(self.summary_cache_path, "r") as f:
                    self.summary_cache = json.load(f)
            except Exception:
                self.summary_cache = {}
        else:
            self.summary_cache = {}

    def serialize_value(value):
        if isinstance(value, (datetime, date)):
            return value.isoformat()
        if isinstance(value, bytes):
            return value.decode(errors="ignore")
        return value

    def fetch_sample_rows(self, engine, tables: List[str]) -> Dict[str, Dict[str, Any]]:
        samples = {}
        with engine.connect() as conn:
            for table in tables:
                try:
                    result = conn.execute(
                        quote_identifiers_if_postgres(text(f"SELECT * FROM {table} LIMIT 1"), engine.dialect.name)
                    )
                    row = result.fetchone()
                    if row:
                        samples[table] = {k: serialize_value(v) for k, v in row._mapping.items()}
                except Exception as e:
                    logger.warning(f"⚠️ Could not fetch sample row from table '{table}': {e}")
                    samples[table] = {"error": str(e)}
        return samples
    
    async def generate_embeddings_async(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for schema texts asynchronously"""
        try:
            response = await aclient.embeddings.create(
                input=texts,
                model="text-embedding-3-large"
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise

    async def auto_summarize_tables_batch_async(self, tables: List[Dict[str, Any]], db_type: str = "unknown") -> Dict[str, Any]:
        """
        Automatically summarize multiple tables with richer context and guidance.
        Returns a dict {table_name: summary_dict}.
        """
        summaries_dict = {}

        for table in tables:
            table_name = table["table_name"]
            if table_name in self.summary_cache:
                summaries_dict[table_name] = self.summary_cache[table_name]
                continue

            # Build structured prompt
            columns = ", ".join(table.get("columns", []))
            primary_keys = ", ".join(table.get("primary_keys", [])) if table.get("primary_keys") else "None"
            foreign_keys = []
            if table.get("foreign_keys"):
                foreign_keys = [
                    f"{fk['column']} → {fk['references_table']}.{fk['references_column']}"
                    for fk in table["foreign_keys"]
                ]

            prompt = f"""
            You are an expert data analyst and database architect.
            Summarize the purpose and structure of the {db_type} SQL table named '{table_name}'.

            Columns: {columns}
            Primary Keys: {primary_keys}
            Foreign Keys: {', '.join(foreign_keys) if foreign_keys else 'None'}

            Output a valid JSON object with the following keys:

            - "table_name": name of the table
            - "description": short plain-English description of what data this table stores
            - "column_descriptions": a dictionary mapping each column name to its meaning
            - "primary_keys": list of primary key columns
            - "foreign_keys": list of relationships like "column → referenced_table.referenced_column"
            - "semantic_tags": list of 5–10 keywords that capture the business meaning of this table
            - "when_to_use": one sentence explaining when to use this table for analysis or queries
            - "usage_examples": 1–3 short example natural-language questions this table could help answer

            Example output:
            {{
              "table_name": "user_profile",
              "description": "Stores basic information about users including names and signup dates.",
              "column_descriptions": {{
                "user_id": "Primary key for user",
                "date_created": "Signup date"
              }},
              "primary_keys": ["user_id"],
              "foreign_keys": [],
              "semantic_tags": ["user", "profile", "signup", "account"],
              "when_to_use": "Use this table when you need user information or signup data.",
              "usage_examples": [
                "How many users signed up last month?",
                "List users who updated their profile recently."
              ]
            }}
            """

            try:
                response = await aclient.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You summarize database tables precisely in JSON."},
                        {"role": "user", "content": prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.1
                )

                # after getting response text
                raw_content = response.choices[0].message.content.strip()
                raw_content = raw_content.strip("```json").strip("```").strip()

                summary = json.loads(raw_content)
                summaries_dict[table_name] = summary
                self.summary_cache[table_name] = summary

            except Exception as e:
                logger.error(f"Error summarizing table '{table_name}': {e}")
                summaries_dict[table_name] = {
                    "table_name": table_name,
                    "description": "Error generating summary",
                    "error": str(e)
                }

        # Save updated cache
        with open(self.summary_cache_path, "w") as f:
            json.dump(self.summary_cache, f, indent=2)

        return summaries_dict
    
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

                    # 🧠 Build enhanced semantic document for embedding
                    doc_parts = [f"Table: {table_name}. {desc}"]

                    # Add columns with descriptions
                    if col_descs:
                        col_lines = [f"{col}: {meaning}" for col, meaning in col_descs.items()]
                        doc_parts.append("Columns:\n" + "\n".join(col_lines))
                    else:
                        doc_parts.append(f"Columns: {', '.join(columns)}")

                    # Add primary keys
                    if item.get('primary_keys'):
                        doc_parts.append(f"Primary Keys: {', '.join(item['primary_keys'])}")

                    # Add foreign key relationships
                    if item.get('foreign_keys'):
                        relationships = [
                            f"{fk['column']} → {fk['references_table']}.{fk['references_column']}"
                            for fk in item['foreign_keys']
                        ]
                        doc_parts.append("Relationships:\n" + "\n".join(relationships))

                    # Add semantic tags
                    if tags:
                        doc_parts.append(f"Semantic Keywords: {', '.join(tags)}")

                    # ✅ Add new GPT-generated fields
                    when_to_use = summary.get("when_to_use")
                    if when_to_use:
                        doc_parts.append(f"When to Use: {when_to_use}")

                    usage_examples = summary.get("usage_examples", [])
                    if usage_examples:
                        doc_parts.append("Example Questions:\n" + "\n".join(f"- {q}" for q in usage_examples))

                    # Combine all parts
                    doc_text = "\n\n".join(doc_parts)

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

async def generate_sql_with_relationships_async(
    natural_language_query: str,
    schema_info: str,
    db_type: str,
    sample_data: Optional[Dict[str, Dict[str, Any]]] = None,
    relevant_tables: Optional[List[str]] = None
) -> Optional[str]:
    """Generate SQL with understanding of table relationships, sample data, and cached semantic summaries."""

    # 🗂️ Load only summaries for relevant tables
    summary_context = ""
    schema_summary_path = "./schema_summary_cache.json"

    if relevant_tables and os.path.exists(schema_summary_path):
        try:
            with open(schema_summary_path, "r") as f:
                cache = json.load(f)

            summaries = {t: cache.get(t) for t in relevant_tables if t in cache}

            if summaries:
                summary_context = "\n\nSchema Summaries for Relevant Tables:\n"
                for t, summary in summaries.items():
                    if not summary:
                        continue
                    desc = summary.get("description", "")
                    cols = summary.get("column_descriptions", {})
                    usage = summary.get("usage_examples", [])
                    summary_context += f"\n📘 Table: {t}\nDescription: {desc}\nColumns:\n"
                    for col, meaning in cols.items():
                        summary_context += f"  - {col}: {meaning}\n"
                    if usage:
                        summary_context += "Example Questions:\n"
                        for q in usage:
                            summary_context += f"  - {q}\n"
        except Exception as e:
            logger.warning(f"⚠️ Could not load relevant summaries: {e}")

    # 🧩 Build sample data context
    sample_context = ""
    if sample_data:
        formatted_samples = json.dumps(sample_data, indent=2)
        sample_context = f"\nSample Rows (1 per relevant table):\n{formatted_samples}"

    # 🧩 Load application logic if available
    logic_context = ""
    logic_path = "./application_logic.json"
    if os.path.exists(logic_path):
        try:
            with open(logic_path, "r") as f:
                logic_data = json.load(f)
                logic_text = logic_data.get("logic", "").strip()
                if logic_text:
                    logic_context = f"\nApplication-Level Logic:\n{logic_text}\n"
        except Exception as e:
            logger.warning(f"⚠️ Could not load application logic: {e}")


    # 🧠 Enhanced prompt with semantic + structural context
    enhanced_prompt = f"""
You are an expert SQL developer. Convert the following natural language request into an accurate SQL query.

Use the provided database schema, schema summaries, and sample data to determine which tables, joins, and filters are needed.

Database Schema with Relationships:
{schema_info}

{summary_context}

{sample_context}

Application Level Logic:
{logic_context}

CRITICAL INSTRUCTIONS:
1. Use the semantic meaning of each table/column (from summaries) to select correct tables.
2. Use sample values to infer column value conventions (Y/N, true/false, etc.).
3. Use JOINs only if necessary — single-table queries are preferred when possible.
5. Use case-insensitive matching (e.g., LOWER()).
6. Prefer partial string matches (LIKE) when the exact value is unclear.
7. Use foreign key relationships as hints for joins (not mandatory).
8. Generate syntactically correct SQL for {db_type}.
9. Return **only the SQL query**, no commentary.
10. Never invent tables or columns not in the schema.
11. Include WHERE, GROUP BY, ORDER BY, and LIMIT clauses based on query intent.
12. Use clear table aliases if multiple tables are joined.
13. Respect conventions for status or completion flags ('Y', 'N', true/false, 1/0).

Natural language query: {natural_language_query}

SQL Query:
"""
    
    logger.info("=================================")
    logger.info(f"Generated Prompt Query: {enhanced_prompt}")

    try:
        response = await aclient.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are a SQL expert that understands schema semantics and relationships.",
                },
                {"role": "user", "content": enhanced_prompt},
            ],
            temperature=0.1,
            max_tokens=1000,
        )

        sql_query = response.choices[0].message.content.strip()
        sql_query = re.sub(r"```sql\s*|\s*```", "", sql_query).strip()
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

    try:
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

    except Exception as e:
        logger.error(f"Error quoting identifiers for Postgres: {e}", exc_info=True)
        # On error, return the original query unmodified to avoid breaking execution
        return sql_query
    
async def auto_fix_and_execute_query(
    aclient,
    engine,
    sql_query: str,
    natural_language_query: str,
    schema_info: str,
    db_type: str,
) -> (list, list, str):
    """
    Execute SQL safely — if it fails, automatically re-generate a corrected query using GPT.
    Returns (rows, columns, final_sql_query)
    """
    try:
        with engine.connect() as conn:
            try:
                # ✅ First attempt
                result = conn.execute(text(sql_query))
                rows = result.fetchall()
                columns = result.keys()
                return rows, columns, sql_query

            except SQLAlchemyError as e:
                error_message = str(e.__cause__ or e)
                logger.warning(f"⚠️ Query failed: {error_message}")

                # 🧠 Build a repair prompt for GPT
                repair_prompt = f"""
The following SQL query failed to execute. Correct it based on the schema and error message.

Natural language query: {natural_language_query}

Failed SQL:
{sql_query}

Database error:
{error_message}

Schema information:
{schema_info}

Instructions:
- Output only the corrected SQL query (no explanation).
- Only use existing tables and columns from the schema.
- Simplify joins if necessary.
- Ensure valid syntax for {db_type}.
"""

                logger.info("🔄 Attempting to auto-fix SQL via GPT...")
                logger.info(f"Repair Prompt:\n{repair_prompt}")

                response = await aclient.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are an expert SQL troubleshooter and repair assistant."},
                        {"role": "user", "content": repair_prompt},
                    ],
                    temperature=0.1,
                    max_tokens=800
                )

                fixed_sql = response.choices[0].message.content.strip()
                fixed_sql = re.sub(r"```sql\s*|\s*```", "", fixed_sql).strip()

                logger.info(f"🔧 Regenerated SQL:\n{fixed_sql}")

                # Try executing the corrected SQL
                result = conn.execute(text(fixed_sql))
                rows = result.fetchall()
                columns = result.keys()
                return rows, columns, fixed_sql

    except SQLAlchemyError as e:
        logger.error(f"❌ Query failed even after auto-fix: {e}")
        raise e

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

def universal_serialize(value):
    """Safely convert any SQL/JSON value to a JSON-serializable form."""
    try:
        if value is None:
            return None
        if isinstance(value, (datetime.date, datetime.datetime)):
            return value.isoformat()
        if isinstance(value, decimal.Decimal):
            return float(value)
        if isinstance(value, (bytes, bytearray)):
            return value.decode(errors="ignore")
        if isinstance(value, uuid.UUID):
            return str(value)
        if isinstance(value, (np.int64, np.int32, np.float32, np.float64)):
            return value.item()
        return value
    except Exception:
        return str(value)


def analyze_column_types(rows, columns):
    """
    Simple heuristic-based column type detection from rows.
    Returns: dict {col_name: 'numeric' | 'categorical' | 'datetime' | 'text'}
    """
    import numbers
    types = {}
    if not rows:
        return {c: 'unknown' for c in columns}

    for col_index, col_name in enumerate(columns):
        sample_vals = [r[col_index] for r in rows if r[col_index] is not None]
        if not sample_vals:
            types[col_name] = 'unknown'
            continue

        # Detect datetimes
        if all(isinstance(v, (datetime.date, datetime.datetime)) for v in sample_vals[:10]):
            types[col_name] = 'datetime'
        # Detect numeric
        elif all(isinstance(v, numbers.Number) or str(v).replace('.', '', 1).isdigit() for v in sample_vals[:10]):
            types[col_name] = 'numeric'
        # Detect categorical (few unique values)
        elif len(set(sample_vals[:20])) < len(sample_vals[:20]) / 2:
            types[col_name] = 'categorical'
        else:
            types[col_name] = 'text'
    return types


async def determine_smart_chart_type(rows, columns):
    """
    Decide best chart type using local heuristics first, fallback to OpenAI when ambiguous.
    Returns: chart_type (string)
    """
    if not rows or not columns:
        return "table"

    # Analyze column types
    col_types = analyze_column_types(rows, columns)
    type_counts = {
        "numeric": sum(1 for t in col_types.values() if t == "numeric"),
        "categorical": sum(1 for t in col_types.values() if t == "categorical"),
        "datetime": sum(1 for t in col_types.values() if t == "datetime"),
    }

    # 🧠 Local heuristic rules (fast path)
    if type_counts["datetime"] >= 1 and type_counts["numeric"] >= 1:
        return "line"
    if type_counts["categorical"] >= 1 and type_counts["numeric"] >= 1:
        return "bar"
    if type_counts["numeric"] == 2:
        return "scatter"
    if type_counts["numeric"] > 2:
        return "bubble"
    if type_counts["categorical"] == 1 and len(rows) <= 10:
        return "pie"

    # 🧠 Fallback: ask OpenAI if uncertain
    try:
        sample_data = [
            {col: universal_serialize(val) for col, val in zip(columns, r)}
            for r in rows[:10]
        ]

        prompt = f"""
        You are a data visualization expert.
        Given the following SQL query result (JSON), choose the most appropriate visualization type.

        Data (sample):
        {json.dumps(sample_data, indent=2)}

        Rules:
        - If one categorical column and one numeric column → 'bar'
        - If two numeric columns → 'scatter'
        - If one date/time column and one numeric column → 'line'
        - If one column with few unique values and one numeric → 'pie'
        - If three numeric columns → 'bubble'
        - If time-series with multiple metrics → 'area'
        - Otherwise choose 'bar'

        Return ONLY one word: the chart type.
        """

        response = await aclient.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a visualization expert."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=10,
        )
        chart_type = response.choices[0].message.content.strip().lower()
        if chart_type not in ["bar", "line", "pie", "scatter", "area", "bubble", "donut", "heatmap"]:
            chart_type = "bar"
        return chart_type
    except Exception as e:
        logger.error(f"Error determining chart type: {e}")
        return "bar"

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

            # Generate SQL
            sql_query = run_async(generate_sql_with_relationships_async(
                natural_language_query,
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
            rows, columns, final_sql = run_async(auto_fix_and_execute_query(
                aclient,
                engine,
                sql_query,
                natural_language_query,
                schema_info,
                session['db_type']
            ))

            columns = list(columns)
            chart_type = run_async(determine_smart_chart_type(rows, columns))
            chart_data = [
                {col: universal_serialize(val) for col, val in zip(columns, r)}
                for r in rows
            ]

            logger.info(f"Determined chart type: {chart_type}")
            logger.info(f"Chart data: {chart_data}")

            columns = [str(c) for c in list(columns)]  # ensure JSON-safe strings

            return render_template(
                'results.html',
                query=natural_language_query,
                sql_query=final_sql,
                columns=columns,
                rows=rows,
                schema_info=schema_info,
                chart_type=chart_type,
                chart_data=json.dumps(chart_data)
            )

        except Exception as e:
            logger.error(f"Query failed: {e}")
            return render_template('results.html', error=f"Error: {str(e)}")

    # 🚀 When accessed via GET (first load)
    return render_template('results.html')


@app.route('/api/generate-sql', methods=['POST'])
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
    
@app.route('/api/save-logic', methods=['POST'])
def save_application_logic():
    """Save application-level logic text to local JSON file."""
    try:
        data = request.get_json()
        logic_text = data.get("logic", "").strip()

        if not logic_text:
            return jsonify({"error": "No logic text provided"}), 400

        logic_path = "./application_logic.json"

        # Save to file
        with open(logic_path, "w") as f:
            json.dump({"logic": logic_text, "updated": datetime.now().isoformat()}, f, indent=2)

        logger.info("✅ Application logic saved successfully.")
        return jsonify({"status": "success"})

    except Exception as e:
        logger.error(f"Error saving application logic: {e}")
        return jsonify({"error": str(e)}), 500
    
@app.route('/api/get-logic', methods=['GET'])
def get_application_logic():
    """Return saved logic for prefill."""
    try:
        path = "./application_logic.json"
        if os.path.exists(path):
            with open(path, "r") as f:
                data = json.load(f)
            return jsonify(data)
        else:
            return jsonify({"logic": ""})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

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