import logging, os, time

from nv_ingest.framework.orchestration.ray.util.pipeline.pipeline_runners import run_pipeline
from nv_ingest.framework.orchestration.ray.util.pipeline.pipeline_runners import PipelineCreationSchema
from nv_ingest_api.util.logging.configuration import configure_logging as configure_local_logging
from nv_ingest_client.client import Ingestor, NvIngestClient
from nv_ingest_api.util.message_brokers.simple_message_broker import SimpleClient
from nv_ingest_client.util.process_json_files import ingest_json_results_to_blob

# Start the pipeline subprocess for library mode
config = PipelineCreationSchema()

run_pipeline(config, block=False, disable_dynamic_scaling=True, run_in_subprocess=True)

client = NvIngestClient(
    message_client_allocator=SimpleClient,
    message_client_port=7671,
    message_client_hostname="localhost"
)

# gpu_cagra accelerated indexing is not available in milvus-lite
# Provide a filename for milvus_uri to use milvus-lite
milvus_uri = "milvus.db"
collection_name = "lnt"
sparse = False

# do content extraction from files                                
ingestor = (
    Ingestor(client=client)
    .files("/home/dell/nemo-retriver/test_page.pdf")
    .extract(
        extract_text=True,
        extract_tables=True,
        extract_charts=False,
        extract_images=False,
        paddle_output_format="markdown",
        extract_infographics=False,
        extract_method="nemoretriever_parse", #Slower, but maximally accurate, especially for PDFs with pages that are scanned images
        text_depth="page"
    ).embed()
    .vdb_upload(
        collection_name=collection_name,
        milvus_uri=milvus_uri,
        sparse=sparse,
        # for llama-3.2 embedder, use 1024 for e5-v5
        dense_dim=2048
    )
)

print("Starting ingestion..")
t0 = time.time()

# Return both successes and failures
# Use for large batches where you want successful chunks/pages to be committed, while collecting detailed diagnostics for failures.
results, failures = ingestor.ingest(show_progress=True, return_failures=True)

# Return only successes
# results = ingestor.ingest(show_progress=True)

t1 = time.time()
print(f"Total time: {t1 - t0} seconds")

# results blob is directly inspectable
print(ingest_json_results_to_blob(results[0]))

# (optional) Review any failures that were returned
if failures:
    print(f"There were {len(failures)} failures. Sample: {failures[0]}")