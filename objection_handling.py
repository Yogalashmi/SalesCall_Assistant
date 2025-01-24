import os
import streamlit as st
import cohere
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from langchain_ollama import OllamaEmbeddings

cohere_api_key = os.getenv("COHERE_API_KEY") 
co = cohere.Client(cohere_api_key)
qdrant_api_key = os.getenv('QDRANT_API_KEY')
url = os.getenv('Qdrant_URL')
client = QdrantClient(url,api_key=qdrant_api_key)

categories = []
keywords = []
suggestions = []

def extract_details(retrieved_items):
    objections = []
    for item in retrieved_items:
        metadata = item.payload.get('metadata', {})
        category = metadata.get('Category', 'No Category')
        suggestion = metadata.get('Suggestions', 'No Suggestions')
        keyword = metadata.get('Keywords', 'No Keywords')

        categories.append(category)
        suggestions.append(suggestion)

        objections.append({"keyword": keyword, "suggestion": suggestion})
    return objections, categories, suggestions

def retrieve_documents(query):
    try:
        results = client.search(
            collection_name="objection_dataset",
            query_vector=query,
            limit=3,
            score_threshold=0.6,  
        )
        if not results:
            return [], [], []
        objections, categories, suggestions = extract_details(results) 
        return objections, categories, suggestions
    except Exception as e:
        return [], [], []


def generate_response(query, context):
    prompt = f"""
    You are an expert conversational AI that helps users refine their responses to objections in a professional and persuasive manner.
    Below is a user's sentence, along with objections (keywords) and response strategies.
    Your task is to create a natural and professional dialogue where the user's response incorporates the provided strategies in a smooth and engaging way.

    User's Sentence:
    {query}

    Objections:
    {context}

    Instructions:
    1. Analyze the user's sentence and objections.
    2. Incorporate the response strategies into a well-structured and natural dialogue.
    3. Ensure the tone is professional, empathetic, and persuasive.

    Output:
    Provide only the dialogue in the response, that too a short and crisp 1 sentence dialogue where the user's response addresses each objection effectively.
    """
    response = co.generate(
        model='command-r-plus',
        prompt=prompt,
        max_tokens=100,
        temperature=0.5
    )
    return response.generations[0].text.strip()
def append_category(category):
    for cat in category:
        st.session_state.found_categories.append(cat)

def objection_rag(query):

    embedding_generator = OllamaEmbeddings(model="nomic-embed-text")
    query_vector = embedding_generator.embed_query(query)

    objections, categories, suggestions = retrieve_documents(query_vector)
    
    if not objections:
        st.write("No relevant objections found in the database for the given query.")
        return
    categories = list(set(categories))   
    append_category(categories)
    context = "\n".join([f"{obj['keyword']}: {obj['suggestion']}" for obj in objections])

    answer = generate_response(query, context)
    st.write(f"Response: {answer}")
