import os
from langchain.chains import RetrievalQA
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# 1. Set up your Google Gemini API Key
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")  # Replace

# 2. Define source document library (Simulated dataset)
raw_text = """
Retrieval-Augmented Generation (RAG) is an architectural pattern that improves the accuracy 
and reliability of generative AI models by fetching facts from an external knowledge base. 
LangChain is a robust framework designed to simplify the creation of applications using large 
language models. By combining LangChain with vector databases like FAISS, developers can perform 
lightning-fast similarity searches over massive datasets. FAISS (Facebook AI Similarity Search) 
is a library developed by Meta for efficient similarity search and clustering of dense vectors. 
Google Gemini models provide state-of-the-art natural language understanding and generation capabilities, 
making them ideal backend engines for context-aware Q&A systems.
"""

document = Document(page_content=raw_text, metadata={"source": "tech_manual"})

# 3. Split the text into manageable chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300, 
    chunk_overlap=50
    )
docs_chunks = text_splitter.split_documents([document])
print(f"Total Chunks Created: {len(docs_chunks)}")

# 4. Initialize Google Generative AI Embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# 5. Build FAISS Vector Store Index from chunks
vector_store = FAISS.from_documents(docs_chunks, embeddings)
print("FAISS Vector Index successfully built and stored locally!")

# 6. Configure the Retriever and Gemini LLM Engine
retriever = vector_store.as_retriever(
    search_type="similarity", 
    search_kwargs={"k": 2}
    )

llm = ChatGoogleGenerativeAI(
model="gemini-3.1-flash-lite", 
    temperature=0.2
    )

# 7. Construct the RetrievalQA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
    )

# 8. Submit a Natural Query and Evaluate Outpu
query = "What is FAISS and who developed it?"
response = qa_chain.invoke({"query": query})

print("\n--- QUERY RESULT ---")
print(f"User Query: {query}")
print(f"Generated Answer:\n{response['result']}")

print("\n--- RETRIEVED SOURCE DOCUMENTS (VECTOR QUERY ANALYTICS) ---")
for i, doc in enumerate(response["source_documents"]):
    print(f"Chunk {i+1}: {doc.page_content}")
    print(f"Metadata: {doc.metadata}")
    print("-" * 40)
                                                    