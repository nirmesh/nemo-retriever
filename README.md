# nemo-retriever
 export NVIDIA_API_KEY=nvapi-

sqlite3 milvus.db
sqlite> .tables
bm25_stats       lnt              token_doc_count
collection_meta  meta           
sqlite> .schema lnt 
CREATE TABLE IF NOT EXISTS "lnt" (id INTEGER PRIMARY KEY, milvus_id VARCHAR(1024), data BLOB);
sqlite> SELECT * FROM lnt LIMIT 10;