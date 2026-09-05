from graph.agent import agent

def agentic_rag(user_query : str, thread_id: str = "default") -> str:
    """
    Run the agent for a single user query and return the final text answer.
 
    Args:
        user_query: The user's question, e.g. "Should I look deeper into Reliance?"
 
    Returns:
        The agent's final synthesized text answer
    """
    config = {"configurable": {"thread_id": thread_id}}
    result = agent.invoke(
        {"messages": [{"role": "user", "content": user_query}]},
        config=config,
    )
 
    final_message = result["messages"][-1]
    return final_message.content