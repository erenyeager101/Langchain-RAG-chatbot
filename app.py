import streamlit as st
import os
import shutil
from src.ingest import IngestionPipeline
from src.agent import ChatAgent
from langchain_core.messages import HumanMessage, AIMessage

# Page Config
st.set_page_config(page_title="Advanced RAG Chatbot", layout="wide")

# Initialize Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "agent" not in st.session_state:
    # Check API Key
    if "OPENAI_API_KEY" not in os.environ and "OPENAI_API_KEY" not in st.session_state:
         pass # Wait for user input
    else:
        # Initialize Agent
        try:
             st.session_state.agent = ChatAgent()
        except Exception as e:
             st.error(f"Failed to initialize agent: {e}")

# Sidebar
with st.sidebar:
    st.title("Settings")

    # API Key Input if not in env
    if "OPENAI_API_KEY" not in os.environ:
        api_key = st.text_input("OpenAI API Key", type="password")
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
            st.session_state["OPENAI_API_KEY"] = api_key
            if "agent" not in st.session_state:
                 st.session_state.agent = ChatAgent()
            st.rerun()

    st.divider()

    st.header("Data Source")
    uploaded_files = st.file_uploader("Upload Documents (PDF, MD, TXT)", accept_multiple_files=True)

    if st.button("Process Documents"):
        if uploaded_files:
            with st.spinner("Processing documents..."):
                # Save uploaded files temporarily
                temp_paths = []
                os.makedirs("temp_uploads", exist_ok=True)
                for uploaded_file in uploaded_files:
                    file_path = os.path.join("temp_uploads", uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    temp_paths.append(file_path)

                # Ingest
                try:
                    pipeline = IngestionPipeline()
                    # ingest expects a list of file paths
                    # Ingest returns nothing but updates disk
                    pipeline.ingest(temp_paths)
                    st.success("Documents processed successfully!")

                    # Reload agent to pick up new retriever
                    # Preserve memory if possible, but context changed so maybe start fresh or just keep history?
                    # We'll just re-init. The chat history in session_state remains.
                    # Ideally we should hydrate the new agent's memory with chat_history.
                    new_agent = ChatAgent()
                    # Hydrate memory from session state history
                    for msg in st.session_state.chat_history:
                        new_agent.memory.chat_memory.add_message(msg)
                    st.session_state.agent = new_agent

                except Exception as e:
                    st.error(f"Error processing documents: {e}")
                finally:
                    if os.path.exists("temp_uploads"):
                        shutil.rmtree("temp_uploads")
        else:
            st.warning("Please upload files first.")

    st.divider()
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        if "agent" in st.session_state:
            st.session_state.agent.memory.clear()
        st.rerun()

# Main Chat Interface
st.title("🤖 User-Centric RAG Chatbot")
st.markdown("Chat with your documents using advanced retrieval techniques.")

# Display Chat History
for message in st.session_state.chat_history:
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(message.content)

# User Input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message to history
    st.session_state.chat_history.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Response
    if "agent" in st.session_state:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.agent.chat(prompt)

                output_text = response.get("output", "No response.")
                st.markdown(output_text)

                # Add assistant message to history
                st.session_state.chat_history.append(AIMessage(content=output_text))

                # Show Sources/Intermediate Steps
                steps = response.get("intermediate_steps", [])
                if steps:
                    with st.expander("View Retrieval Steps & Sources"):
                        for action, observation in steps:
                            st.markdown(f"**Tool:** `{action.tool}`")
                            st.markdown(f"**Input:** `{action.tool_input}`")
                            st.markdown(f"**Output:** {observation}")
                            st.divider()
    else:
        st.error("Agent not initialized. Please provide API Key.")
