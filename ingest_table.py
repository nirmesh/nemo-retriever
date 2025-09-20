import os, time
from pypdf import PdfReader, PdfWriter   # <-- use pypdf, not PyPDF2

from nv_ingest.framework.orchestration.ray.util.pipeline.pipeline_runners import run_pipeline
from nv_ingest.framework.orchestration.ray.util.pipeline.pipeline_runners import PipelineCreationSchema
from nv_ingest_client.client import Ingestor, NvIngestClient
from nv_ingest_api.util.message_brokers.simple_message_broker import SimpleClient
from nv_ingest_client.util.process_json_files import ingest_json_results_to_blob


# ============ Utility: Cut one page ============
def save_single_page(input_pdf, page_number, output_pdf):
    """
    Extracts a single page (1-indexed) from input_pdf and saves to output_pdf.
    """
    reader = PdfReader(input_pdf)
    writer = PdfWriter()
    writer.add_page(reader.pages[page_number - 1])  # pypdf is 0-indexed internally

    with open(output_pdf, "wb") as f:
        writer.write(f)

    print(f"✅ Saved page {page_number} as {output_pdf}")
    return output_pdf


def save_page_range(input_pdf, start_page, end_page, output_pdf):
    """
    Extracts a range of pages (1-indexed, inclusive) from input_pdf and saves to output_pdf.
    
    Args:
        input_pdf (str): Path to input PDF file.
        start_page (int): Starting page number (1-indexed).
        end_page (int): Ending page number (1-indexed, inclusive).
        output_pdf (str): Path to output PDF file.
    """
    reader = PdfReader(input_pdf)
    writer = PdfWriter()

    # Validate range
    total_pages = len(reader.pages)
    if start_page < 1 or end_page > total_pages or start_page > end_page:
        raise ValueError(f"❌ Invalid page range {start_page}-{end_page}. PDF has {total_pages} pages.")

    # Add pages
    for i in range(start_page - 1, end_page):  # convert to 0-index
        writer.add_page(reader.pages[i])

    # Write to output
    with open(output_pdf, "wb") as f:
        writer.write(f)

    print(f"✅ Saved pages {start_page}-{end_page} as {output_pdf}")
    return output_pdf

# ============ Start pipeline ============
config = PipelineCreationSchema()
run_pipeline(config, block=False, disable_dynamic_scaling=True, run_in_subprocess=True)

client = NvIngestClient(
    message_client_allocator=SimpleClient,
    message_client_port=7671,
    message_client_hostname="localhost"
)

milvus_uri = "milvus.db"
collection_name = "lnt"
sparse = False

# ============ Extract page and ingest ============
input_pdf = "/home/dell/nemo-retriver/annual.pdf"
test_pdf = "/home/dell/nemo-retriver/test_page.pdf"
output_md = "/home/dell/nemo-retriver/test_page_tables.md"

#save_single_page(input_pdf, page_number=304, output_pdf=test_pdf)

save_page_range(input_pdf, 295, 345, test_pdf)  

# Ingest one-page PDF
ingestor = (
    Ingestor(client=client)
    .files(test_pdf)
    .extract(
        extract_text=True,     # only tables
        extract_tables=True,
        extract_charts=False,
        extract_images=False,
        extract_infographics=False,
        paddle_output_format="markdown",
        extract_method="nemoretriever_parse",  # safer for PDFs with mixed content
        text_depth="page"
    )
    .embed()
    .vdb_upload(
        collection_name=collection_name,
        milvus_uri=milvus_uri,
        sparse=sparse,
        dense_dim=2048
    )
)

print("🚀 Starting ingestion..")
t0 = time.time()
results, failures = ingestor.ingest(show_progress=True, return_failures=True)
t1 = time.time()
print(f"⏱️ Total time: {t1 - t0:.2f} seconds")

# Handle results
if results:
    blob = ingest_json_results_to_blob(results[0])
    with open(output_md, "w", encoding="utf-8") as f:
        f.write(blob)
    print(f"✅ Extracted table(s) written to {output_md}")
else:
    print("⚠️ No successful results.")
    if failures:
        print("Sample failure:", failures[0])
