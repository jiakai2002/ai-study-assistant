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
            "You are an academic assistant that helps students by processing the following context. "
            "Based on the retrieved information, provide a concise and clear response to the question. "
            "If the context is not sufficient to answer or generate relevant content, acknowledge the lack of information. "
            "Your response should be brief, to the point, and relevant to the user's query.\n\n"
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

    memory = MemorySaver()
    graph = graph_builder.compile(checkpointer=memory)
    return graph, memory


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
