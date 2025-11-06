import re
import json
import os
from typing import Optional, List, Dict, Any
from extensions import aclient
import logging

logger = logging.getLogger(__name__)

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