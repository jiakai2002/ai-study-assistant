import os
import streamlit as st
import tempfile
from dotenv import load_dotenv

from langchain.document_loaders import PyPDFLoader
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import MessagesState, StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

# --- Setup ---
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
llm = init_chat_model("gpt-4o-mini", model_provider="openai")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vector_store = InMemoryVectorStore(embeddings)

st.set_page_config(page_title="Study Assistant", layout="centered")
st.title("📚 Study Assistant 🤖")

# --- Session State ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "processed_filenames" not in st.session_state:
    st.session_state.processed_filenames = set()

# --- File Upload ---
uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        if uploaded_file.name not in st.session_state.processed_filenames:
            with st.spinner(f"📄 Processing {uploaded_file.name}..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    book_path = tmp_file.name

                loader = PyPDFLoader(book_path)
                docs = loader.load()

                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000, chunk_overlap=200, add_start_index=True
                )
                all_splits = text_splitter.split_documents(docs)

                _ = vector_store.add_documents(documents=all_splits)

                st.session_state.processed_filenames.add(uploaded_file.name)
                os.remove(book_path)
                st.success(f"✅ {uploaded_file.name} - {len(all_splits)} chunks stored.")

# --- Tool: Document Retriever ---
@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve relevant documents from the book given a user query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        f"SOURCE {i+1} [Page {doc.metadata.get('page', 'unknown')}, "
        f"Start index: {doc.metadata.get('start_index', 'unknown')}]:\n{doc.page_content}"
        for i, doc in enumerate(retrieved_docs)
    )
    return serialized, retrieved_docs

# --- LangGraph Definition ---
@st.cache_resource
def create_langgraph():
    graph_builder = StateGraph(MessagesState)

    def query_or_respond(state: MessagesState):
        llm_with_tools = llm.bind_tools([retrieve])
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    def generate(state: MessagesState):
        recent_tool_messages = [
            msg for msg in reversed(state["messages"]) if msg.type == "tool"
        ]
        tool_messages = list(reversed(recent_tool_messages))
        docs_content = "\n\n".join(doc.content for doc in tool_messages)

        system_message_content = (
            "You are an expert academic study assistant analyzing documents. "
            "Follow these guidelines precisely when responding to questions:\n\n"
            "1. ONLY use information from the retrieved context below. Do not use prior knowledge.\n"
            "2. If the context contains conflicting information, acknowledge the conflict.\n"
            "3. If the question can't be answered using the context, respond: 'The provided documents don't contain information about [topic].'\n"
            "4. Cite specific page numbers when answering (e.g., 'According to page 5...').\n"
            "5. Format your answers with key concepts in **bold**.\n"
            "6. If numerical data exists, present it precisely as stated in the text.\n\n"
            "Retrieved context:\n\n"
            f"{docs_content}"
        )

        conversation_messages = [
            m for m in state["messages"]
            if m.type in ("human", "system") or (m.type == "ai" and not m.tool_calls)
        ]

        prompt = [SystemMessage(system_message_content)] + conversation_messages
        response = llm.invoke(prompt)
        return {"messages": [response]}

    tools = ToolNode([retrieve])

    graph_builder.add_node(query_or_respond)
    graph_builder.add_node(tools)
    graph_builder.add_node(generate)
    graph_builder.set_entry_point("query_or_respond")
    graph_builder.add_conditional_edges(
        "query_or_respond",
        tools_condition,
        {END: END, "tools": "tools"},
    )
    graph_builder.add_edge("tools", "generate")
    graph_builder.add_edge("generate", END)
    graph_builder.add_node(verify_answer)
    graph_builder.add_edge("generate", "verify_answer")
    graph_builder.add_edge("verify_answer", END)

    memory = MemorySaver()
    graph = graph_builder.compile(checkpointer=memory)
    return graph, memory

def verify_answer(state: MessagesState):
    """Verify the generated answer against retrieved context"""
    # Get the last AI message
    last_ai_message = [m for m in reversed(state["messages"]) if isinstance(m, AIMessage)][0]
    
    # Get retrieved context
    tool_messages = [m for m in state["messages"] if m.type == "tool"]
    context = "\n".join(m.content for m in tool_messages)
    
    verification_prompt = [
        SystemMessage(content=(
            "You are a fact-checker. Verify if the following answer is fully supported by the context.\n"
            "If there are any statements not supported by the context, revise the answer to remove them.\n"
            f"Context: {context}\n\n"
            f"Answer to verify: {last_ai_message.content}\n\n"
            "Respond with ONLY the verified/corrected answer and nothing else."
        ))
    ]
    
    verified_response = llm.invoke(verification_prompt)
    return {"messages": [verified_response]}



graph, memory = create_langgraph()

# --- Display Chat History ---
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

# --- User Interaction ---
user_input = st.chat_input("Ask me anything...")

if user_input and st.session_state.processed_filenames:

    human_message = HumanMessage(content=user_input)
    st.session_state.chat_history.append(human_message)

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        response_area = st.empty()

        input_state = {"messages": [human_message]}
        config = {"configurable": {"thread_id": "thread-001"}}
        full_response = ""

        for step in graph.stream(input_state, stream_mode="values", config=config):
            if step["messages"] and isinstance(step["messages"][-1], AIMessage):
                chunk = step["messages"][-1].content
                full_response += chunk
                response_area.markdown(full_response)

        st.session_state.chat_history.append(AIMessage(content=full_response))

elif user_input and not st.session_state.processed_filenames:
    st.warning("Please upload at least one PDF file first.")
