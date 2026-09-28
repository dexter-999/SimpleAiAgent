import os
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
import pandas as pd
from langchain.tools import tool


embeddings = OllamaEmbeddings(model="mxbai-embed-large")

df = pd.read_csv("realistic_restaurant_reviews.csv")



dbLocation = "./vectorDatabase"
add_document = not os.path.exists(dbLocation)
documents = []

if add_document:
    for i,row in df.iterrows():
        document =Document(
            page_content=row["Title"] + " " + row["Review"],
            metadata={"rating": row["Rating"], "date": row["Date"]},
            id=str(i))
        documents.append(document)

chroma = Chroma(
	embedding_function = embeddings,
	collection_name="knowledge_vector",
    persist_directory=dbLocation,

)


if add_document:
    chroma.add_documents(documents=documents)

retrieve = chroma.as_retriever(search_kwargs={"k":2})

@tool
def search_company_docs(query: str) -> str:
    """Searches a database of restaurant reviews (customer reviews, ratings, pizza, service, prices, etc.).
    Use this tool first for any questions related to restaurants, customer reviews, food quality, or service,
    before resorting to any external internet searches."""
    results = retrieve.invoke(query)
    if not results:
        return "لم أجد معلومات متعلقة بهذا السؤال في المستندات."
    return "\n".join(doc.page_content for doc in results)

