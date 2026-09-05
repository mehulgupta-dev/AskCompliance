import streamlit as st
from langchain_core.messages import HumanMessage
from graph.agentic_rag import agentic_rag
import uuid

st.set_page_config(
    page_title = "DPDP Act & Rules Chatbot",
    page_icon = "🤖"
)

st.title("🤖 DPDP Act & Rules Chatbot")
st.caption("Ask me anything about DPDP")

# --- Session state setup ---

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "display_messages" not in st.session_state:
    st.session_state.display_messages = []

# -- Render existing chat history --

for msg in st.session_state.display_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Chat input __

user_input = st.chat_input("Ask me anything about DPDP")

if user_input:
    # Show user message immediately
    st.session_state.display_messages.append({"role" : "user", "content" : user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    # call our agent

    thread_id = st.session_state.thread_id

    with st.chat_message("assistant"), st.spinner("Thinking..."):
            result = agentic_rag(user_query = user_input, thread_id = thread_id)

            st.markdown(result)

    st.session_state.display_messages.append({"role": "assistant", "content": result})