import tempfile

from dotenv import load_dotenv
from llama_cloud_services import LlamaParse

import cocoindex

class ToMarkdown(cocoindex.op.FunctionSpec):
    """Convert a PDF to markdown."""

@cocoindex.op.executor_class(gpu=True, cache=True, behavior_version=1)
class LlamaParseExecutor:
    """Executor for LlamaParse to parse files.
       Supported file types: https://docs.llamaindex.ai/en/stable/llama_cloud/llama_parse/
    """

    spec: ToMarkdown
    _parser: LlamaParse

    def prepare(self):
        # Initialize LlamaParse
        self._parser = LlamaParse(
            # API key can be set in environment as LLAMA_CLOUD_API_KEY
            num_workers=1,
            verbose=True,
            language="en",
        )

    async def __call__(self, content: bytes) -> str:
        with tempfile.NamedTemporaryFile(delete=True, suffix=".pdf") as temp_file:
            temp_file.write(content)
            temp_file.flush()
            # Parse the PDF using LlamaParse
            result = await self._parser.aparse(temp_file.name)
            # Get the markdown content
            markdown_documents = result.get_markdown_documents(split_by_page=False)
            # Combine all markdown content if there are multiple documents
            return "\n\n".join([doc.text for doc in markdown_documents])


def text_to_embedding(text: cocoindex.DataSlice) -> cocoindex.DataSlice:
    """
    Embed the text using a SentenceTransformer model.
    """
    return text.transform(
        cocoindex.functions.SentenceTransformerEmbed(
            model="sentence-transformers/all-MiniLM-L6-v2"))

@cocoindex.flow_def(name="PdfEmbedding")
def pdf_embedding_flow(flow_builder: cocoindex.FlowBuilder, data_scope: cocoindex.DataScope):
    """
    Define an example flow that embeds files into a vector database.
    """
    data_scope["documents"] = flow_builder.add_source(cocoindex.sources.LocalFile(path="pdf_files", binary=True))

    doc_embeddings = data_scope.add_collector()

    with data_scope["documents"].row() as doc:
        doc["markdown"] = doc["content"].transform(ToMarkdown())
        doc["chunks"] = doc["markdown"].transform(
            cocoindex.functions.SplitRecursively(),
            language="markdown", chunk_size=2000, chunk_overlap=500)

        with doc["chunks"].row() as chunk:
            chunk["embedding"] = chunk["text"].call(text_to_embedding)
            doc_embeddings.collect(id=cocoindex.GeneratedField.UUID,
                                   filename=doc["filename"], location=chunk["location"],
                                   text=chunk["text"], embedding=chunk["embedding"])

    doc_embeddings.export(
        "doc_embeddings",
        cocoindex.storages.Postgres(),
        primary_key_fields=["id"],
        vector_indexes=[
            cocoindex.VectorIndexDef(
                field_name="embedding",
                metric=cocoindex.VectorSimilarityMetric.COSINE_SIMILARITY)])

query_handler = cocoindex.query.SimpleSemanticsQueryHandler(
    name="SemanticsSearch",
    flow=pdf_embedding_flow,
    target_name="doc_embeddings",
    query_transform_flow=text_to_embedding,
    default_similarity_metric=cocoindex.VectorSimilarityMetric.COSINE_SIMILARITY)

@cocoindex.main_fn()
def _run():
    # Run queries in a loop to demonstrate the query capabilities.
    while True:
        try:
            query = input("Enter search query (or Enter to quit): ")
            if query == '':
                break
            results, _ = query_handler.search(query, 10)
            print("\nSearch results:")
            for result in results:
                print(f"[{result.score:.3f}] {result.data['filename']}")
                print(f"    {result.data['text']}")
                print("---")
            print()
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    load_dotenv(override=True)
    _run()
