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

Current App flow

---

1️⃣ User inputs DB credentials → Flask connects & validates connection.
2️⃣ Schema extraction → SQLAlchemy reflects tables, columns, and relationships.
3️⃣ Table summarization → GPT creates JSON summaries for each table (meaning, usage, tags).
4️⃣ Vector embeddings → Each table summary is embedded into ChromaDB for semantic search.
5️⃣ User enters natural query → e.g. "Show top 5 products by sales last month".
6️⃣ Query rewriting → GPT refines the question for analytical clarity.
7️⃣ Relevant tables retrieval → ChromaDB finds top-matching tables using embeddings.
8️⃣ SQL generation → GPT builds context-aware SQL with schema, summaries, and relationships.
9️⃣ Execution + auto-fix → SQLAlchemy runs it; GPT repairs if errors occur.
🔟 Visualization → GPT suggests best chart type, builds chart config, and Flask renders data + SQL + chart.