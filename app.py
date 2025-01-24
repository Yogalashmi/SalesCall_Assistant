import os
import asyncio
from collections import Counter
import logging
import vosk
from datetime import datetime
import pandas as pd
import pyaudio
from pdf_geneartion import generate_pdf
from transformers import pipeline
from dotenv import load_dotenv
from transcript import TranscriptCollector
from sentiment_analysis import SentimentTracker
import streamlit as st
from product_recommendation import process_user_input
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from langchain_community.embeddings import OllamaEmbeddings
from langchain_groq import ChatGroq
from sales_call_analyzer import SalesCallAnalyzer
from transformers import pipeline
from analatical_graph import create_line_chart,create_pie_chart,create_bar_chart
from question_handler import handle_query 
import re
from customer_metrics import APIClient
from objection_handling import objection_rag
import logging
import os

load_dotenv()

#st.set_page_config(page_title="Dyr")

st.sidebar.title("Navigation")
options = st.sidebar.radio("Select a page:", ["Start the call", "Summary", "Dynamic Search Query"])

sent_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
transcript_collector = TranscriptCollector()
sentiment_tracker = SentimentTracker()


vosk_model = vosk.Model('vosk-model-small-en-in-0.4')
ollama_embeddings = OllamaEmbeddings(model="nomic-embed-text")

groq_api_key = os.getenv('GROQ_API_KEY')
cohere_api_key = os.getenv('COHERE_API_KEY')
qdrant_api_key = os.getenv('QDRANT_API_KEY')
url = os.getenv('Qdrant_URL')
llm=ChatGroq(model="llama3-8b-8192", api_key=groq_api_key)

client = QdrantClient(url,api_key=qdrant_api_key)
api_client = APIClient(groq_api_key)
analyzer = SalesCallAnalyzer(cohere_api_key)

st.title("Real-Time Speech to Text Transcription")
st.write("Press the button below to start or stop recording.")

if 'recording' not in st.session_state:
    st.session_state.recording = False
if 'full_transcript' not in st.session_state:
    st.session_state.full_transcript = ""
if 'sentiment' not in st.session_state:
    st.session_state.sentiment = None
if 'extracted_data' not in st.session_state:
    st.session_state.extracted_data = []
# Initialize session state for recommended products
if 'recommended_products' not in st.session_state:
    st.session_state.recommended_products = []
if "found_categories" not in st.session_state:
    st.session_state.found_categories = []

stop_recording_event = None
if options == "Start the call":
    async def get_transcript():
        global stop_recording_event
        stop_recording_event = asyncio.Event()  
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=16000,
                        input=True,
                        frames_per_buffer=8000)
        stream.start_stream()

        recognizer = vosk.KaldiRecognizer(vosk_model, 16000)
        st.write("Start speaking...")
        try:
            while not stop_recording_event.is_set():
                data = stream.read(8000, exception_on_overflow=False)
                if recognizer.AcceptWaveform(data):
                    result = recognizer.Result()
                    latest_transcript = result[14:-3]

                    if latest_transcript:
                        transcript_collector.add_part(latest_transcript)
                        full_transcript = transcript_collector.get_full_transcript()
                        st.session_state.full_transcript = full_transcript
                        st.write(f"Latest Transcript: {latest_transcript}")
                        st.write(f"Transcript So Far: {st.session_state.full_transcript.strip()}")
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        overall_sentiment = sent_analyzer(full_transcript)[0]
                        st.write(f"Sentiment: {overall_sentiment['label']}, Score: {overall_sentiment['score']}")
                        products = process_user_input(latest_transcript, llm)
                        if(products!=[]): 
                            st.write("Top Recommendationds")
                            for i in products:
                                st.write(i)
                                st.session_state.recommended_products.append({i})
                        st.write("Recommended Products")
                        objection_rag(latest_transcript)
    
                        st.session_state.extracted_data.append({
                            "Timestamp": timestamp,
                            "Sentence": latest_transcript,
                            "Sentiment": overall_sentiment['label'],
                            "Score": overall_sentiment['score']
                        })

                        st.write()
           
            stream.stop_stream()
            stream.close()
            p.terminate()

            full_transcript = transcript_collector.get_full_transcript()
            overall_sentiment = sent_analyzer(full_transcript)[0]
            st.session_state.full_transcript = full_transcript
            st.session_state.sentiment = overall_sentiment
            st.write("\n=== Full Transcript ===")
            st.write(full_transcript)
            st.write("\n=== Overall Sentiment ===")
            st.write(f"Sentiment: {overall_sentiment['label']}, Score: {overall_sentiment['score']}")

        except Exception as e:
            logging.error(f"Error during transcription: {e}")
            stream.stop_stream()
            stream.close()
            p.terminate()
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Start Recording"):
            st.session_state.recording = True
            asyncio.run(get_transcript())

    with col2:
        if st.button("Stop Recording"):
            if stop_recording_event is not None:  
                stop_recording_event.set()  
            st.write("Recording stopped.")
            st.session_state.recording = False
    if not st.session_state.recording:
        st.write("Transcript So Far:")
        st.write(st.session_state.full_transcript)

        if st.session_state.sentiment:
            st.write("Overall Sentiment Analysis Results:")
            st.write(f"Sentiment: {st.session_state.sentiment['label']}, Score: {st.session_state.sentiment[' score']}")
elif options == "Summary":
    tab1, tab2, tab3 = st.tabs(["Insights about customer", "Sales Metrics", "Analatical Dashboard"])
    with tab1:
        st.markdown("## Analysis based on Sales Conversation")
        if st.session_state.full_transcript:
            response_1 = api_client.customer_insights(st.session_state.full_transcript)
            clothing_questions, rest_of_response = api_client.extract_clothing_questions(st.session_state.full_transcript)
            if response_1 != 'Nothing':
                sections = response_1.split("\n\n")  
                customer_concerns = sections[0].split("Customer Concerns:")[-1].strip() if len(sections) > 0 else "No concerns found."
                issues_faced = sections[1].split("Issues Faced:")[-1].strip() if len(sections) > 1 else "No issues found."
                customer_feedback = sections[2].split("Customer Feedback:")[-1].strip() if len(sections) > 2 else "No feedback found."
                initial_needs = sections[3].split("Initial Needs/Interests:")[-1].strip() if len(sections) > 3 else "No initial needs found."
                col1, col2 = st.columns(2)
                with col1:
                    st.write(issues_faced)
                    st.write(initial_needs)
                with col2:
                    st.write(customer_concerns)
                    st.write(customer_feedback)
            st.markdown("-----")
            if clothing_questions:
                st.subheader("Questions asked by the user")
                for i in clothing_questions:
                    st.write(i)
            st.markdown("-----")
            st.subheader("Products Recommedated to the user")
            if "recommendation" in st.session_state:
                product_recommendations = st.session_state.recommendation
                if product_recommendations: 
                    product_counts = Counter(product_recommendations)
                    top_5_products = [product for product, _ in product_counts.most_common(5)]
                    st.subheader("Top Product Recommendations for this user")
                    for product in top_5_products:
                        st.write(f"**{product}**")
                else:
                    st.warning("No recommendations for this user.")
            else:
                st.error("No product recommendations found in the session state.")

            if st.button("Generate PDF for Customer Insights"):
                pdf_file_name = "customer_insights_report.pdf"
                content = f"""
                <h1>Customer Insights</h1>
                <h2>Customer Concerns</h2>
                <p>{customer_concerns}</p>
                <h2>Issues Faced</h2>
                <p>{issues_faced}</p>
                <h2>Customer Feedback</h2>
                <p>{customer_feedback}</p>
                <h2>Initial Needs</h2>
                <p>{initial_needs}</p>
                <h2>Questions Asked by the User</h2>
                <ul>
                    {"".join(f"<li>{q}</li>" for q in clothing_questions)}
                </ul>
                <h2>Product Recommedated to the user</h2>
                """
                pdf_data=generate_pdf(content, pdf_file_name)
                if pdf_data:
                    st.download_button(
                label="Download PDF",
                data=pdf_data,
                file_name=pdf_file_name,
                mime="application/pdf"
            )
        else:
            st.error("Please first start the call to generate the transcript.")
    with tab2:
        st.markdown("## Sales Metrics Report")
        response = analyzer.analyze_transcript(st.session_state.full_transcript)
        print(response)
        likelihood_match = re.search(r"Likelihood of closing the sale: (.*)%", response)
        stage_match = re.search(r"Stage of the call: (.*)", response)
        reason_match = re.search(r"Reason: (.*)", response)
        suggestion_match = re.search(r"Suggestions:\s*(.*)", response, re.DOTALL)
        likelihood = likelihood_match.group(1) if likelihood_match else "N/A"
        stage = stage_match.group(1) if stage_match else "N/A"
        reason = reason_match.group(1) if reason_match else "N/A"
        suggestions = []
        if suggestion_match:
            suggestions_text = suggestion_match.group(1).strip()
            suggestions = [s.strip("- ").strip() for s in suggestions_text.split("\n") if s.strip()]
        st.success("Analysis Complete!")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Likelihood of Closing the Sale")
            st.markdown(f":chart_with_upwards_trend: **{likelihood}%**")
            
            st.subheader("Stage of the Call")
            st.markdown(f":speech_balloon: **{stage}**")
        with col2:
            st.subheader("Reason")
            st.markdown(f":bulb: {reason}")
        st.markdown("-----")
        st.subheader("Actionable Insights for Deal Success")
        for suggestion in suggestions:
            st.markdown(f":dart: {suggestion}")
        st.markdown("-----")
    with tab3:
        st.markdown("## Sentiment Analysis and Analytical Dashboard")
        if st.session_state.extracted_data:
            st.write("### Extracted Data")
            extracted_df = pd.DataFrame(st.session_state.extracted_data)  # Convert to DataFrame
            st.write(extracted_df)
        st.markdown("<h2 style='text-align: center;'>Sentiment Report</h2>", unsafe_allow_html=True) 
        st.write(rest_of_response)
        st.markdown("-----") 
        fig_line = create_line_chart(extracted_df)
        st.plotly_chart(fig_line)
        fig_pie = create_pie_chart(extracted_df)
        st.plotly_chart(fig_pie)
        st.markdown("-----")
        st.subheader("Ojections raised by customer")
        create_bar_chart(st.session_state.found_categories)

elif options == "Dynamic Search Query":
    st.header("Dynamic question handling 💬")
    user_question = st.text_input("Ask anything related to or about products:")
    if st.button("Submit"):
        if user_question:
            result = handle_query(user_question)
            st.write(result)
        else:
            st.warning("Please enter a question.")