from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.config import get_settings

settings = get_settings()
embeddings = OllamaEmbeddings(
    model=settings.llm_embed_model,
    base_url=settings.ollama_base_url,
)
llm = ChatOpenAI(
    model=settings.groq_model,
    api_key=settings.groq_api_key,
    base_url=settings.groq_base_url,
    temperature=settings.llm_temperature_insight,
    timeout=settings.llm_timeout_sec,
)

vectorstore = Chroma(
    collection_name="quantai_docs",
    embedding_function=embeddings,
    persist_directory="data/chroma_db"
)

# Simple in-memory history
chat_history = []

def add_document_to_store(text: str, metadata: dict | None = None):
    if not text.strip():
        raise ValueError("No readable text was found in this document")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([text], metadatas=[metadata or {}])
    vectorstore.add_documents(chunks)
    return len(chunks)

class CustomChatbotChain:
    def invoke(self, inputs: dict):
        question = inputs.get("question", "")
        
        # Retrieve context
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        docs = retriever.invoke(question)
        context = "\n\n".join([d.page_content for d in docs])
        
        # Build prompt
        history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history[-5:]])
        
        prompt = f"""You are a helpful AI data analyst. Answer the question based ONLY on the provided context.
        
Context:
{context}

Chat History:
{history_str}

Question: {question}

Answer:"""
        
        response = llm.invoke(prompt)
        answer = str(response.content)
        
        # Update history
        chat_history.append({"role": "User", "content": question})
        chat_history.append({"role": "Assistant", "content": answer})
        
        return {"answer": answer}

def get_chatbot_chain():
    return CustomChatbotChain()
