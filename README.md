download python for your operating system if its not present then

dowload and open this project in vscode create virtual enviornment using 
- python3 -m venv venv

Activate the env using
- source venv/bin/activate

install the required packages
- pip install -r requirements.txt

Start the project using
- python3 ./app.py


---

Generate Code Flow Diagram with 
- python3 -m pyflowchart app.py > flowchart.flow   

---



---

Explore 90+ end-to-end [GenAI Projects](https://www.projectpro.io/accelerator-program/generative-ai-program?utm_source=pawan&utm_medium=udemy)

---

Vector Embedding version done 

---

Database Connection (PostgreSQL / MySQL / SQLite)

- The user enters connection credentials via the web form.
- The app tests the connection using SQLAlchemy.

Schema Extraction & Enrichment

- It automatically extracts all tables, columns, data types, and relationships (foreign keys).
- Each table is summarized with GPT (description, column meanings, usage examples, semantic tags, etc.).
- These summaries are cached locally (schema_summary_cache.json) for reuse.

Vector Embedding & Storage (ChromaDB)

- The app embeds each table’s semantic summary using OpenAI’s text-embedding-3-large model.
- Embeddings (vector representations) are stored in a local ChromaDB collection for fast semantic search.

Natural Language Query → SQL Conversion
When a user types a question in plain English:

- The query is embedded and matched against ChromaDB to find the most relevant tables.
- Sample rows are fetched from those tables to give GPT example data.
- A GPT model (gpt-4o) is prompted with the schema, summaries, and samples to generate an accurate SQL query.

SQL Execution & Result Display

- The generated SQL is executed safely via SQLAlchemy.
- The results (columns + rows) are rendered in the browser along with the generated SQL query.

Admin Features

- /admin/refresh-schema allows rebuilding all embeddings and summaries from scratch if the database schema changes.
- Errors and performance metrics are logged to app.log.
