from qdrant_client.http.exceptions import ResponseHandlingException
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import os
import qdrant_client
import getpass
from collections import defaultdict
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.vectorstores import Qdrant
from langchain.embeddings import OllamaEmbeddings
from langchain.chat_models import ChatGroq


@retry(
    stop=stop_after_attempt(5),  # Retry up to 5 times
    wait=wait_exponential(multiplier=1, min=1, max=10),  # Exponential backoff
    retry=retry_if_exception_type(ResponseHandlingException),  # Retry only on Qdrant-specific exceptions
)
def get_vector_store():
    client = qdrant_client.QdrantClient(
        "https://22f58e0e-66ae-4eee-a24a-0867ac2c19c8.europe-west3-0.gcp.cloud.qdrant.io",
        api_key="KpJ6RTfOcF6E5GO1Tq3O1ilH1fx35fYQbPlfcU06qkMPX84VWbLNtw"
    )
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vector_store = Qdrant(
        client=client,
        collection_name="ProductDetails",
        embeddings=embeddings,
    )
    return vector_store


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(Exception),  # Retry on any exception while fetching the LLM
)
def get_groq_llama_model():
    if not os.environ.get("GROQ_API_KEY"):
        os.environ["GROQ_API_KEY"] = getpass.getpass("Enter API key for Groq: ")
    groq_api_key = os.environ["GROQ_API_KEY"]
    return ChatGroq(model="llama3-8b-8192", api_key=groq_api_key)


def expand_query(user_question):
    categories = {
        "bottomwear": ["jeans", "pants", "trousers", "shorts", "trackpants"],
        "topwear": ["kurti", "saree", "t-shirt", "shirt", "blouse", "frock"],
        "silk": ["silk", "satin"],
        "men": ["trackpants", "jean", "kurta", "track suit"],
        "women": ["saree", "lehenga", "frock", "ethnic kurtis", "blouse"],
    }
    expanded_query = user_question
    for category, terms in categories.items():
        for term in terms:
            if term in user_question.lower():
                expanded_query += f" {category}"
                break
    return expanded_query


def create_qa_chain():
    vector_store = get_vector_store()
    llm = get_groq_llama_model()

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 1}
    )
    template = """ You are going to be a sales agent's assistant. You will help them answer the question asked by the customer. If the given context or product description doesn't match the query, strictly say the following: "Currently, we don't have that specific item. However, we have a variety of alternatives that might interest you, including other styles and types of ethnic wear."

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

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs=chain_type_kwargs
    )
    return qa_chain


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(Exception),  # Retry on general exceptions during query handling
)
def handle_query(user_question):
    load_dotenv()
    qa_chain = create_qa_chain()
    expanded_query = expand_query(user_question)
    response = qa_chain({"query": expanded_query})
    return response.get("result", "No result found.")
