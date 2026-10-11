from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_huggingface import ChatHuggingFace
import streamlit as st
from dotenv import load_dotenv

from youtube_transcript_api import YouTubeTranscriptApi

# Fetching the transcript of a YouTube video using its video ID
# video_id = "lb4PSWzgvrY"  # paste your video id here

load_dotenv()

# Creating embeddings for the chunks using HuggingFace embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def process_video(video_id):
    try:
        transcript_list = YouTubeTranscriptApi().fetch(video_id)
        transcript = " ".join(chunk.text for chunk in transcript_list)
    except Exception as e:
        return None, f"Error fetching transcript: {e}"
    
    # Splitting the transcript into smaller chunks for processing

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(transcript)   

    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore, None 


# Augmenting the query with a prompt template for better context

from langchain_google_genai import ChatGoogleGenerativeAI
import os 

model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    google_api_key= os.getenv("GEMINI_API_KEY")
)

def generate_suggested_questions(vectorstore):
    # Pull a sample of content from the vectorstore to summarize
    docs = vectorstore.similarity_search("summary of the video", k=4)
    context_text = "\n\n".join(doc.page_content for doc in docs)

    prompt = f"""
Based on the following video transcript excerpt, suggest exactly 3 important questions 
a viewer might want to ask to understand the video better.
Return ONLY the 3 questions, one per line, no numbering, no extra text.

Transcript:
{context_text}
"""
    response = model.invoke(prompt)
    questions = [q.strip() for q in response.content.split("\n") if q.strip()]
    return questions[:3]

#---------------- ST UI ----------------

st.set_page_config(page_title="YouTube RAG Chatbot", layout="wide")

#------------------ FUNCTIONS ----------------

import re

def extract_video_id(url):
    pattern = r"(?:v=|youtu\.be/|embed/)([a-zA-Z0-9_-]{11})"
    match = re.search(pattern, url)
    if match:
        return match.group(1)
    return None

# ---------------- SIDEBAR ----------------

with st.sidebar:
    st.header("📺 Video Input")
    video_url = st.text_input("Paste YouTube Video URL")
    process_btn = st.button("Process Video")

if process_btn:
    video_id = extract_video_id(video_url)
    if not video_id:
        st.error("Invalid YouTube URL. Please check and try again.")
    else:
        with st.spinner("Processing video..."):
            vectorstore, error = process_video(video_id)
        if error:
            st.error(error)
        else:
            st.session_state.vectorstore = vectorstore
            with st.spinner("Generating suggested questions..."):
                st.session_state.suggested_questions = generate_suggested_questions(vectorstore)
            st.success("Video processed! You can now ask questions.")

# ---------------- MAIN AREA ----------------

# st.title("YouTube Video Q&A")

# if process_btn:
#     st.write("You pasted:", video_url)
#     st.info("Processing logic will go here in the next step.")

# st.subheader("Chat")

st.subheader("💬 Chat")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def answer_question(question):
    retriever = st.session_state.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    retrieved_docs = retriever.invoke(question)
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

    prompt_template = """
You are a helpful assistant that answers questions based on the provided context.
Context: {context}
Question: {question}
Answer:
"""
    prompt = prompt_template.format(context=context_text, question=question)
    response = model.invoke(prompt)
    return response.content

# Show suggested questions as buttons (only if video is processed)
if "vectorstore" in st.session_state:
    if "suggested_questions" in st.session_state:
        st.write("**Suggested questions:**")
        cols = st.columns(len(st.session_state.suggested_questions))
        for i, q in enumerate(st.session_state.suggested_questions):
            if cols[i].button(q):
                with st.spinner("Thinking..."):
                    answer = answer_question(q)
                st.session_state.chat_history.append((q, answer))

    # Manual question input
    user_question = st.chat_input("Ask a question about the video")
    if user_question:
        with st.spinner("Thinking..."):
            answer = answer_question(user_question)
        st.session_state.chat_history.append((user_question, answer))

    # Display chat history
    for q, a in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(q)
        with st.chat_message("assistant"):
            st.write(a)
else:
    st.info("Please process a video first using the sidebar.")