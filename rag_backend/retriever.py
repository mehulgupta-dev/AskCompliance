from langchain_community.vectorstores import Chroma
from embedding import dense_embeddings
from rag_backend.vector_store import vector_store_path
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal
from prompt import filter_query_prompt
from llm_model import model

class QueryClassification(BaseModel):
    source_type: Literal["act", "rules", "both"] = Field(..., description="The type of source the query is related to. It can be 'act', 'rules', or 'both'.")
    reasoning: str = Field(..., description="The reasoning behind the classification of the query.")

parser = PydanticOutputParser(pydantic_object = QueryClassification)

format_instructions = parser.get_format_instructions()

filter_query = filter_query_prompt | model | parser

vector_store = Chroma(
    persist_directory = vector_store_path,
    embedding_function = dense_embeddings
)

def retriever(query : str):

    classification = filter_query.invoke({"user_query" : query, "format_instructions": format_instructions})

    source_type = classification.source_type

    if source_type == "both":
        filter_dict = None
    else:
        filter_dict = {"source_type" : source_type}

    retrieve = vector_store.as_retriever(
        search_type = 'similarity',
        search_kwargs = {"k" : 10, "filter" : filter_dict}
    )

    return retrieve.invoke(query)

if __name__ == "__main__":
    result = retriever("When did the Digital Personal Data Protection Act, 2023 receive presidential assent, and what is its stated purpose ?")

    print(result)