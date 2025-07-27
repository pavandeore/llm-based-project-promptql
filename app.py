from flask import Flask, render_template, request, redirect, url_for, session
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from openai import OpenAI
import os
import re
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'dev-secret-key')

# Configure OpenAI (you'll need to set OPENAI_API_KEY in your .env file)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_sql_from_natural_language(natural_language_query, schema_info=None):
    """Use OpenAI to generate SQL from natural language"""
    
    prompt = f"""
You are an expert SQL developer. Convert the following natural language query into SQL:

Database schema information:
{schema_info or 'No schema information provided'}

- for is_active column value check with this =  'Y' or 'N'  and not using any other value

Natural language query: {natural_language_query}

Return only the SQL query without any additional explanation or formatting.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful SQL assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"❌ Error generating SQL: {e}")
        return None

def quote_identifiers_if_postgres(sql_query, db_type):
    """
    If using PostgreSQL, wrap table and column identifiers in double quotes 
    to preserve case sensitivity. This is a naive implementation and assumes 
    simple queries.
    """
    if db_type != 'postgresql' or not sql_query:
        return sql_query

    # Simple pattern: match words used after FROM or JOIN or UPDATE etc.
    keywords = ["FROM", "JOIN", "UPDATE", "INTO"]
    for kw in keywords:
        sql_query = re.sub(
            rf"(?i)\b{kw}\s+(\w+)",
            lambda m: f'{m.group(0).split()[0]} "{m.group(1)}"',
            sql_query
        )

    # Optionally: quote columns in WHERE clause (can be expanded further)
    sql_query = re.sub(
        r'(?i)(WHERE|AND|OR)\s+(\w+)\s*=',
        lambda m: f'{m.group(1)} "{m.group(2)}" =',
        sql_query
    )

    return sql_query

def get_db_schema(engine, db_type):
    """Get basic schema information from the database"""
    schema_info = ""
    try:
        with engine.connect() as conn:
            if db_type == 'postgresql':
                # PostgreSQL schema detection
                tables = conn.execute(text("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema NOT IN ('pg_catalog', 'information_schema')
                """)).fetchall()
                
                for table in tables:
                    table_name = table[0]
                    schema_info += f"\nTable: {table_name}\nColumns: "
                    
                    columns = conn.execute(text(f"""
                        SELECT column_name, data_type 
                        FROM information_schema.columns 
                        WHERE table_name = '{table_name}'
                    """)).fetchall()
                    
                    schema_info += ", ".join([f"{col[0]} ({col[1]})" for col in columns])
            
            elif db_type == 'mysql':
                # MySQL schema detection
                tables = conn.execute(text("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = DATABASE()
                """)).fetchall()
                
                for table in tables:
                    table_name = table[0]
                    schema_info += f"\nTable: {table_name}\nColumns: "
                    
                    columns = conn.execute(text(f"""
                        SELECT column_name, column_type 
                        FROM information_schema.columns 
                        WHERE table_name = '{table_name}' AND table_schema = DATABASE()
                    """)).fetchall()
                    
                    schema_info += ", ".join([f"{col[0]} ({col[1]})" for col in columns])
            
            else:
                schema_info = "Schema detection not implemented for this database type"
                
    except Exception as e:
        print(f"Error fetching schema: {e}")
        schema_info = f"Error fetching schema: {str(e)}"
    return schema_info

def get_db_url(db_type, db_host, db_port, db_name, db_user, db_password):
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
        session['db_port'] = request.form.get('db_port')
        session['db_name'] = request.form.get('db_name')
        session['db_user'] = request.form.get('db_user')
        session['db_password'] = request.form.get('db_password')
        
        return redirect(url_for('query'))
    
    return render_template('index.html')

@app.route('/query', methods=['GET', 'POST'])
def query():
    if not all(key in session for key in ['db_type', 'db_host', 'db_name', 'db_user']):
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        natural_language_query = request.form.get('query')
        
        # Create database connection
        db_url = get_db_url(
            session['db_type'],
            session['db_host'],
            session.get('db_port', '5432'),
            session['db_name'],
            session['db_user'],
            session['db_password']
        )
        
        try:
            engine = create_engine(db_url)
            
            # Get schema information to help the LLM
            schema_info = get_db_schema(engine, session['db_type'])
            
            # Generate SQL from natural language
            sql_query = generate_sql_from_natural_language(natural_language_query, schema_info)
            sql_query = quote_identifiers_if_postgres(sql_query, session['db_type'])
            
            if not sql_query:
                return render_template('results.html', error="Failed to generate SQL query")
            
            # Execute the SQL query
            with engine.connect() as conn:
                result = conn.execute(text(sql_query))
                rows = result.fetchall()
                columns = result.keys()
                
            return render_template('results.html', 
                                 query=natural_language_query,
                                 sql_query=sql_query,
                                 columns=columns,
                                 rows=rows)
            
        except SQLAlchemyError as e:
            return render_template('results.html', error=f"Database error: {str(e)}")
        except Exception as e:
            return render_template('results.html', error=f"Error: {str(e)}")
    
    return render_template('query.html')

if __name__ == '__main__':
    app.run(debug=True)