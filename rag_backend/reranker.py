from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_classic.retrievers import ContextualCompressionRetriever
from rag_backend.retriever import retriever
from langchain_core.runnables import RunnableLambda

model_cross_encoder = HuggingFaceCrossEncoder(
    model_name = 'BAAI/bge-reranker-v2-m3'
)

reranker = CrossEncoderReranker(
    model = model_cross_encoder,
    top_n = 4
)

compress_retriever = ContextualCompressionRetriever(
    base_retriever = RunnableLambda(retriever),
    base_compressor = reranker
)

if __name__ == "__main__":
    result = compress_retriever.invoke("When did the Digital Personal Data Protection Act, 2023 receive presidential assent, and what is its stated purpose?")
    print(result)