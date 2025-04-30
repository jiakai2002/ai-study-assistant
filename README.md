# 📚 AI Study Assistant 🤖

An interactive AI-powered chatbot built with **Streamlit** and **LangChain** that allows users to upload PDF documents and ask questions based on their content. This tool uses **OpenAI's GPT-4o** and **vector embeddings** for context-aware answers, making studying more efficient and personalized.

<img width="1069" alt="Screenshot 2025-04-30 at 2 46 09 PM" src="https://github.com/user-attachments/assets/1d78b46c-a2b2-4a68-bb18-e4a720034a1c" />

## 🚀 Features

- 📄 Upload multiple PDF files for contextual Q&A
- 🔍 Semantic document retrieval using OpenAI embeddings
- 🧠 Memory-aware conversation powered by LangGraph
- 🛠️ Tool-augmented reasoning using LangChain tools
- 💬 Chat interface with real-time response streaming via Streamlit
- 📚 Supports large documents via intelligent text chunking

## 🧰 Tech Stack

- [Streamlit](https://streamlit.io/)
- [LangChain](https://www.langchain.com/)
- [LangGraph](https://www.langgraph.dev/)
- [OpenAI GPT-4o](https://platform.openai.com/docs/models/gpt-4)
- [text-embedding-3-large](https://platform.openai.com/docs/guides/embeddings)
- In-Memory Vector Store (for simplicity and prototyping)

## 📝 How It Works

1. **Upload PDFs** – The app extracts and chunks the document text.  
2. **Embed Content** – Text chunks are embedded into a vector store using OpenAI's embedding model.  
3. **Ask Questions** – The user types a question; relevant chunks are retrieved via similarity search.  
4. **LLM Reasoning** – LangGraph and the GPT model use the retrieved context to answer the user's query.  
5. **Conversational Memory** – Your chat history is preserved and updated for contextual replies.

## 🧑‍💻 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/ai-study-assistant.git
cd ai-study-assistant
```
### 2. Create Virtual Environment and Install Dependencies

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
### 3. Setup Environment Variables

Create a .env file in the root directory and add your OpenAI API key:

```ini
OPENAI_API_KEY=your-openai-api-key
```
### 4. Run the App

```bash
streamlit run app.py
```
