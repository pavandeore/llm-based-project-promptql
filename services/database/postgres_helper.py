import re
import logging

logger = logging.getLogger(__name__)

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