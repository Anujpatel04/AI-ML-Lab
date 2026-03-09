# Chat with Gmail Inbox

This is a simple LLM application that uses RAG (Retrieval Augmented Generation) to let you chat with your Gmail inbox. You can ask questions about your emails and get answers based on the actual content in your inbox.

## Features

- Connect to your Gmail inbox using the Gmail API
- Ask questions about your emails in natural language
- Get accurate answers using RAG with DeepSeek API
- Simple Streamlit interface for easy interaction

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Gmail API

You'll need to set up Gmail API access:

1. Go to the [Google Cloud Console](https://console.cloud.google.com/) and create a new project
2. Navigate to "APIs & Services > OAuth consent screen" and configure it
3. Enable the Gmail API for your project
4. Create OAuth client ID credentials (Desktop app type)
5. Download the credentials as JSON and save it as `credentials.json` in this directory

### 3. Set Up API Key

The app uses DeepSeek API for the LLM. Add your API key to the `.env` file in the root directory:

```
DEEPSEEK_API_KEY=your-api-key-here
```

You can get your DeepSeek API key from [DeepSeek Platform](https://platform.deepseek.com/).

## Running the App

```bash
streamlit run chat_gmail.py
```

The app will:
1. Load your Gmail inbox emails
2. Store them in a vector database
3. Allow you to ask questions about your emails

## Repository

This project is part of the [Anuj-AI-ML-Lab](https://github.com/Anujpatel04/Anuj-AI-ML-Lab/tree/main/All_LLMs/chat_with_gmail) repository.


