from rag_backend.reranker import compress_retriever
from rag_backend.generator import generator

def rag_pipeline(query : str):

    retrieved_docs = compress_retriever.invoke(query)
    context = [doc.page_content for doc in retrieved_docs]
    generated_answer = generator(query, context)

    return {
        "query" : query,
        "retrieved_context" : context,
        "generated_answer" : generated_answer
    }

if __name__ == "__main__":
    query = "When did the Digital Personal Data Protection Act, 2023 receive presidential assent, and what is its stated purpose ?"
    result = rag_pipeline(query)
    print(result)