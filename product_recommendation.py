from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from langchain_ollama import OllamaEmbeddings
from langchain_groq import ChatGroq
import os
import streamlit as st

qdrant_api_key = os.getenv('QDRANT_API_KEY')
url = os.getenv('Qdrant_URL')
groq_api_key = os.getenv('GROQ_API_KEY')

client = QdrantClient(url,api_key=qdrant_api_key)  
ollama_embeddings = OllamaEmbeddings(model="nomic-embed-text")
llm=ChatGroq(model="llama3-8b-8192", api_key=groq_api_key)

def generate_preference_extraction_prompt(user_inputs):
    return (
        "You are a personal shopping assistant. Your task is to analyze the following conversation and extract "
        "the user's preferences, likes, or what they are looking for in clothing. Please focus on the following attributes:\n"
        "1. Color preferences\n"
        "2. Clothing types (e.g., shirts, dresses, ethnic wear)\n"
        "3. Fabric preferences (e.g., cotton, silk)\n"
        "4. Size preferences\n"
        "5. Gender (male/female)\n"
        "Please summarize the user's preferences in a single line, "
        "formatted as follows: 'Color Fabric ClothingType Gender and any other attributes value that you extracted"
        "For example, if the input is: 'I really like blue color and I want a cotton dress in that color', "
        "your response should be: 'Blue cotton dress women.'\n"
        "another example, if the input is:'is there any pants available under selling price 400.',"
        "your response should be: 'pants ,men/women and selling price 1000.'"
        "Here is the conversation:\n"
        f"{user_inputs}\n"
        "Please provide the response in the specified format only and you must extract the prefrences if it is there in the provided context "
    )

def extract_titles(retrieved_items):
    titles = []
    for item in retrieved_items:
        title = item.payload['page_content'].split('\n')[0]  
        titles.append(title) 
    return titles

def process_user_input(user_inputs, llm):
    """
    Process user inputs, check for context, summarize conversation,
    retrieve recommendations, and generate a response.
    """
    context_check_prompt = f"Based on the following conversation, is there any context/ reference to a product for which we can recommend products where in our products are clothings and footwear?\n{user_inputs}\nAnswer strictly with 'yes' or 'no':"
    context_check_response = llm.invoke(context_check_prompt)
    llm_output = (context_check_response.content).lower()
    if llm_output == "yes":
        summary_vector = ollama_embeddings.embed_query(user_inputs)
        results = client.search(
            collection_name="Clothing_Dataset",
            query_vector=summary_vector,
            limit=3,
            score_threshold=0.6,  
        )
        product_list = extract_titles(results)
        return product_list
    else:
        list =[]
        st.write("No recommendation for now")
        return list