import chromadb
import json
import os
import asyncio
import time
from pathlib import Path
import tempfile
from typing import List, Dict, Any
from extensions import aclient
import logging

logger = logging.getLogger(__name__)

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
            metadata={"description": "Database table and column information"},
            embedding_function=None   # ✅ make manual embedding explicit
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

    def fetch_sample_rows(self, engine, tables: List[str]) -> Dict[str, Dict[str, Any]]:
        from utils.serialize import serialize_value
        from services.database.postgres_helper import quote_identifiers_if_postgres
        from sqlalchemy import text
        
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