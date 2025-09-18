import logging, os, time
from PyPDF2 import PdfReader, PdfWriter

from nv_ingest.framework.orchestration.ray.util.pipeline.pipeline_runners import run_pipeline
from nv_ingest.framework.orchestration.ray.util.pipeline.pipeline_runners import PipelineCreationSchema
from nv_ingest_api.util.logging.configuration import configure_logging as configure_local_logging
from nv_ingest_client.client import Ingestor, NvIngestClient
from nv_ingest_api.util.message_brokers.simple_message_broker import SimpleClient
from nv_ingest_client.util.process_json_files import ingest_json_results_to_blob


# ============ Utility: Split PDF ============
def split_pdf(input_path, output_dir, chunk_size=100):
    """Split a large PDF into smaller chunks of `chunk_size` pages each."""
    reader = PdfReader(input_path)
    total_pages = len(reader.pages)

    os.makedirs(output_dir, exist_ok=True)
    output_files = []

    for start in range(0, total_pages, chunk_size):
        writer = PdfWriter()
        for page in range(start, min(start + chunk_size, total_pages)):
            writer.add_page(reader.pages[page])

        out_path = os.path.join(output_dir, f"chunk_{start+1}_to_{min(start+chunk_size, total_pages)}.pdf")
        with open(out_path, "wb") as f:
            writer.write(f)

        output_files.append(out_path)

    return output_files


# ============ Start pipeline ============
config = PipelineCreationSchema()
run_pipeline(config, block=False, disable_dynamic_scaling=True, run_in_subprocess=True)

client = NvIngestClient(
    message_client_allocator=SimpleClient,
    message_client_port=7671,
    message_client_hostname="localhost"
)

milvus_uri = "milvus.db"
collection_name = "LnT"
sparse = False

# ============ Split and Ingest ============
input_pdf = "/home/dell/nemo-retriver/annual.pdf"
split_dir = "/home/dell/nemo-retriver/splits"

print("Splitting large PDF...")
chunks = split_pdf(input_pdf, split_dir, chunk_size=100)
print(f"Created {len(chunks)} chunks.")

for i, chunk_file in enumerate(chunks, 1):
    print(f"\n🚀 Ingesting chunk {i}/{len(chunks)}: {chunk_file}")

    ingestor = (
        Ingestor(client=client)
        .files(chunk_file)
        .extract(
            extract_text=True,
            extract_tables=False,    # disable heavy stuff first, re-enable later if needed
            extract_charts=False,
            extract_images=False,
            extract_infographics=False,
            paddle_output_format="markdown",
            extract_method="nemoretriever_parse",  # safer for PDFs with images
            text_depth="page"
        ).embed()
        .vdb_upload(
            collection_name=collection_name,
            milvus_uri=milvus_uri,
            sparse=sparse,
            dense_dim=2048
        )
    )

    t0 = time.time()
    results, failures = ingestor.ingest(show_progress=True, return_failures=True)
    t1 = time.time()

    print(f"Chunk time: {t1 - t0:.2f} seconds")

    if results:
        print("✅ Success: first result blob preview:")
        print(ingest_json_results_to_blob(results[0])[:500], "...")
    else:
        print("⚠️ No results for this chunk.")

    if failures:
        print(f"⚠️ {len(failures)} failures. Example:")
        print(failures[0])
