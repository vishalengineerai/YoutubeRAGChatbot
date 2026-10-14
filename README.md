# YouTube RAG Chatbot

An AI-powered application that uses Retrieval-Augmented Generation (RAG) to help users interact with YouTube video content and get answers based on the information available in the video. It use streamlit for the UI

## Features

- Ask questions about YouTube video content.
- Retrieve relevant information to answer user queries.
- Generate context-aware responses using an LLM.
- Use RAG to ground answers in retrieved video content.

## Tech Stack

- **Language:** Python, Langchain, Streamlit, 
- **AI Architecture:** Retrieval-Augmented Generation (RAG)
- **Dependencies:** Listed in `requirements.txt`

## Project Structure

```text
YoutubeRAGChatbot/
├── YoutubeRAGChatbot.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation

### 1. Clone the repository

### 2. Create a virtual environment

### 3. Install dependencies

### 4. Configure environment variables

Add the API keys required by your application to a `.env` file, if applicable.

### 5. Run the application

```bash
python3 YoutubeRAGChatbot.py
```

## How It Works

1. Processes YouTube video content.
2. Retrieves relevant information for a user's question.
3. Uses an AI language model to generate a response based on the retrieved context.

## Future Improvements

- Support multiple YouTube videos.
- Improve retrieval accuracy.
- Add conversation history.
- Build a user-friendly web interface.

## Author

Vishal
