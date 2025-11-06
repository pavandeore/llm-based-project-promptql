from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
import re
import logging

logger = logging.getLogger(__name__)

async def auto_fix_and_execute_query(
    aclient,
    engine,
    sql_query: str,
    natural_language_query: str,
    schema_info: str,
    db_type: str,
):
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
                columns = list(result.keys()) 
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
                columns = list(result.keys()) 
                return rows, columns, fixed_sql

    except SQLAlchemyError as e:
        logger.error(f"❌ Query failed even after auto-fix: {e}")
        raise e