import re
from flask import render_template

def handle_error(error, message="An error occurred"):
    """Standard error handler"""
    return render_template('error.html', error=message), 500

def extract_table_names_from_schema(schema_info: str):
    """Extract table names from schema info text"""
    return re.findall(r'📊 Table: (\w+)', schema_info)

def format_sql_query(sql_query: str) -> str:
    """Format SQL query for better readability"""
    # Simple formatting - you can enhance this with a proper SQL formatter
    sql_query = re.sub(r'\b(SELECT|FROM|WHERE|JOIN|LEFT JOIN|RIGHT JOIN|INNER JOIN|OUTER JOIN|GROUP BY|ORDER BY|HAVING|LIMIT)\b', 
                      lambda m: f'\n{m.group(1)}', sql_query)
    return sql_query.strip()