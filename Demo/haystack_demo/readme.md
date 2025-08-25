## Haystack RAG PACE demo backed w/Neo4j (v5.28.1) vector db

* Download chat.py, config.py, ingest.py, llm.py and follow the below instructions to run.

First ensure Neo4j is installed (I use the latest community version - v5.28.1) and added to your $PATH. Start the server with:
- neo4j start

Or check it's current status with:
- neo4j status

Now you can ingest PDF files to the vector DB by running:
- python3 ingest.py path/to/pdf

For my backend, I used an open Llama-3 8B model downloaded from Hugging Face. The model can be queried (DB context prepended) as such:
- python3 chat.py "insert prompt"

Or prompted interactively by simply leaving the query argument blank.
