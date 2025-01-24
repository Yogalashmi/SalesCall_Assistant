from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_core.documents import Document  # Import the Document class
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient, models
from langchain_ollama import OllamaEmbeddings
from uuid import uuid4
import os


loader = CSVLoader(file_path='Objections.csv')  
data = loader.load()
documents = []

for item in data:

    lines = item.page_content.split('\n')
    metadata = {}
    for line in lines:
        if ':' in line:
            key, value = line.split(':', 1)  
            metadata[key.strip()] = value.strip()  
    Category = metadata.get('ï»¿Category', 'No Category')
    Keywords = metadata.get('Keywords', 'No Keywords')
    Suggestions = metadata.get('ResponseStrategies', 'Not specified')  
    page_content = (
        f"Category: {Category}\n"
        f"Keywords: {Keywords}\n"
        f"Suggestions: {Suggestions}\n"

    )

    document = Document(
        metadata={
            'Suggestions': Suggestions,
            'Keywords': Keywords,
            'Category': Category,
        },
        page_content=page_content  
    )
    
    documents.append(document)

print(type(documents[0]))
print(documents[0].metadata)
print(documents[0].page_content)
qdrant_api_key = os.getenv('QDRANT_API_KEY')
url = os.getenv('Qdrant_URL')
groq_api_key = os.getenv('GROQ_API_KEY')

client = QdrantClient(url,api_key=qdrant_api_key)  
client.create_collection(
    collection_name="objection_dataset",
    vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE),
)

embedding_generator = OllamaEmbeddings(model="nomic-embed-text")


vector_store = Qdrant(
    client=client,
    collection_name="objection_dataset",
    embeddings=embedding_generator
)


uuids = [str(uuid4()) for _ in range(len(documents))]

vector_store.add_documents(documents=documents, ids=uuids)
