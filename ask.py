from openai import OpenAI
from nv_ingest_client.util.milvus import nvingest_retrieval
import os

milvus_uri = "milvus.db"
collection_name = "lnt"
sparse=False

queries = ["which one hold more shares Grameen Capital India Private Limited or Graphene Solutions Taiwan Ltd?e"]
#queries = ["what is % of shares held by the listed entity Graphene Solutions Taiwan Ltd."]
#queries = ["what is % of shares held by the listed entity  Grameen Capital India Private Limited."]

retrieved_docs = nvingest_retrieval(
    queries,
    collection_name,
    milvus_uri=milvus_uri,
    hybrid=sparse,
    top_k=5,
)

# simple generation example
#extract = retrieved_docs[0][0]["entity"]["text"]
extract = "\n\n".join(
    [doc["entity"]["text"] for doc in retrieved_docs[0]]
)
client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = os.environ["NVIDIA_API_KEY"]
)

prompt = f"Using the following content: {extract}\n\n Answer the user query: {queries[0]}"
print(f"Prompt: {prompt}")
completion = client.chat.completions.create(
  model="nvidia/llama-3.1-nemotron-70b-instruct",
  messages=[{"role":"user","content": prompt}],
)
response = completion.choices[0].message.content

print(f"Answer: {response}")