from dotenv import load_dotenv
import streamlit as st 
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import OllamaEmbeddings
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate  # Import RunnableMap
import qdrant_client
import getpass
import os

'''def get_vector_store():
    # Connect to Qdrant
    client = qdrant_client.QdrantClient("https://22f58e0e-66ae-4eee-a24a-0867ac2c19c8.europe-west3-0.gcp.cloud.qdrant.io",
                      api_key="KpJ6RTfOcF6E5GO1Tq3O1ilH1fx35fYQbPlfcU06qkMPX84VWbLNtw")
    
    # Use Ollama embeddings
    embeddings = OllamaEmbeddings(model="nomic-embed-text")  # Replace with your Ollama embedding model name

    # Initialize Qdrant vector store
    vector_store = Qdrant(
        client=client, 
        collection_name="ProductDetails", 
        embeddings=embeddings,
    )
    
    return vector_store

def get_groq_llama_model():
    if not os.environ.get("GROQ_API_KEY"):
        os.environ["GROQ_API_KEY"] = getpass.getpass("Enter API key for Groq: ")
    
    # Retrieve the API key from the environment variable
    groq_api_key = os.environ["GROQ_API_KEY"]
    
    # Use ChatGroq to load the Llama model
    return ChatGroq(model="llama3-8b-8192", api_key=groq_api_key)  # Use the retrieved API key

def expand_query(user_question):
    # Define categories and their related terms
    categories = {
        "bottomwear": ["jeans", "pants", "trousers", "shorts","trackpants"],
        "topwear": ["kurti", "saree", "t-shirt", "shirt", "blouse","frock"],
        "silk": ["silk", "satin"],
        "men" : ["trackpants","jean","kurta","track suit"],
        "women":["saree","lehenga","frock","ethinic kurtis","blouse"],
    }
    
    # Initialize an expanded query
    expanded_query = user_question
    
    # Expand the query based on the user's input
    for category, terms in categories.items():
        for term in terms:
            if term in user_question.lower():
                expanded_query += f" {category}"
                break  # Only add the category once

    return expanded_query


def main():
    load_dotenv()
    
    st.set_page_config(page_title="Dynamic Question prompt handler")
    st.header("Ask your question 💬")
    
    # Create vector store
    vector_store = get_vector_store()
    
    # Create Groq LLM
    llm = get_groq_llama_model()

    # Create retriever with similarity search type
    retriever = vector_store.as_retriever(
        search_type="similarity", 
        search_kwargs={"k": 1}
    )

    # Create the prompt template
    template = """ template = You are going to be a sales agent's assistant. You will help them answer the question asked by the customer. If the given context  or product decription doesn't match the query, strictly say the following : "Currently, we don't have that specific item. However, we have a variety of alternatives that might interest you, including other styles and types of ethnic wear."

If the context matches the query, provide a brief response including titles, prices, and brands. If brand names and prices are not mentioned, provide a name and price in rupees based on average pricing from internet searches. Finally, let them know if any discounts are available.

{context}

Question: {question}
Your response::\n\n"
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    {context}

    Question: {question}
    Your response:"""

    PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])

    chain_type_kwargs = {"prompt": PROMPT}
    # Create a RunnableMap that combines the prompt and the LLM
    #llm_sequence = RunnableMap({"prompt": prompt_template, "llm": llm})
    chain_type_kwargs = {"prompt": PROMPT}

    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type="stuff",
                                       retriever=retriever,
                                       return_source_documents=True,
                                       chain_type_kwargs=chain_type_kwargs)

    # User input for the question
    user_question = st.text_input("Ask a question about products:")

    if st.button("Submit"):
        if user_question:
            expanded_query = expand_query(user_question)
            response = qa_chain({"query": expanded_query})
            llm_output = response['result']  # Assuming 'result' contains the LLM's output
            # Display only the LLM's output
            st.write(llm_output)

        else:
            st.warning("Please enter a question.") 

if __name__ == "__main__":
    main()

from dotenv import load_dotenv
import streamlit as st
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import OllamaEmbeddings
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
import qdrant_client
import getpass
import os'''

def get_vector_store():
    client = qdrant_client.QdrantClient(
        "https://22f58e0e-66ae-4eee-a24a-0867ac2c19c8.europe-west3-0.gcp.cloud.qdrant.io",
        api_key="KpJ6RTfOcF6E5GO1Tq3O1ilH1fx35fYQbPlfcU06qkMPX84VWbLNtw"
    )
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vector_store = Qdrant(
        client=client,
        collection_name="Clothing_Dataset",
        embeddings=embeddings,
    )
    return vector_store

def get_groq_llama_model():
    if not os.environ.get("GROQ_API_KEY"):
        os.environ["GROQ_API_KEY"] = getpass.getpass("Enter API key for Groq: ")
    groq_api_key = os.environ["GROQ_API_KEY"]
    return ChatGroq(model="llama3-8b-8192", api_key=groq_api_key)

def create_qa_chain():
    vector_store = get_vector_store()
    llm = get_groq_llama_model()

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 1}
    )
    template = """ You are going to be a sales agent's assistant. You will help them answer the question asked by the customer. If the given context  or product decription doesn't match the query, strictly say the following : "Currently, we don't have that specific item. However, we have a variety of alternatives that might interest you, including other styles and types of ethnic wear."

If the context matches the query, provide a brief response including titles, prices, and brands. If brand names and prices are not mentioned, provide a name and price in rupees based on average pricing from internet searches. Finally, let them know if any discounts are available.

{context}

Question: {question}
Your response::\n\n"
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    {context}

    Question: {question}
    Your response:"""

    PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])
    chain_type_kwargs = {"prompt": PROMPT}
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type="stuff",
                                       retriever=retriever,
                                       return_source_documents=True,
                                       chain_type_kwargs=chain_type_kwargs)
    return qa_chain

def handle_query(user_question):
    load_dotenv()
    qa_chain = create_qa_chain()
    response = qa_chain({"query": user_question})
    return response.get("result", "No result found.")
