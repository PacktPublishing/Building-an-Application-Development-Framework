# --- 60‑second RAG proof of concept -----------------
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

docs = [
    "John bought a new MacBook in Berlin.",
    "Apple’s M3 chip was released in 2023.",
    "Berlin’s Apple Store is near Ku’damm.",
]
vectordb = FAISS.from_texts(docs, OpenAIEmbeddings())

rag = RetrievalQA.from_chain_type(
    llm=OpenAI(max_tokens=64), chain_type="stuff", retriever=vectordb.as_retriever()
)

print(rag.invoke("Where did John get his laptop?"))
# ----------------------------------------------------
