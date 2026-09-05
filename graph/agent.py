from langchain.agents import create_agent
from rag_backend.rag_pipeline import rag_pipeline
from langchain_core.tools import tool
from llm_model import model

@tool
def query_dpdp_knowledge_base(question: str) -> str:
    """Retrieves and answers questions about the DPDP Act and Rules from the compliance knowledge base."""
    result = rag_pipeline(question)
    return result["generated_answer"]

agent = create_agent(
    model = model,
    tools = [query_dpdp_knowledge_base],
    system_prompt="You are AskCompliance..."
)