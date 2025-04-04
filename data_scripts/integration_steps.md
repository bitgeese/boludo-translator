# Integration Guide: Adding Ventureout Spanish Content to Chainlit App

This document outlines the steps needed to integrate the scraped Argentinian Spanish content from ventureoutspanish.com into a Chainlit RAG (Retrieval Augmented Generation) application.

## 1. Data Processing Pipeline

The data processing pipeline consists of two main scripts:

1. `scrape_ventureout.py`: Scrapes content from ventureoutspanish.com
2. `embed_data.py`: Processes, cleans, chunks, and embeds the content

### Running the Pipeline

```bash
# Step 1: Install dependencies
pip install -r requirements.txt

# Step 2: Run the scraping script
python scrape_ventureout.py
# This creates ventureout_data.jsonl

# Step 3: Set your OpenAI API key for embeddings
export OPENAI_API_KEY=your_api_key_here

# Step 4: Run the embedding script
python embed_data.py
# This creates a Chroma vector store in ../chroma_db
```

## 2. Chainlit Application Integration

### Setup

1. Create or update your Chainlit app's requirements:

```
# Add to your main app requirements.txt
langchain>=0.0.284
langchain-community>=0.0.1
langchain-openai>=0.0.1
openai>=1.0.0
chromadb>=0.4.18
chainlit>=0.7.0
```

### Code Integration

Here's how to integrate the vector store into a Chainlit application:

```python
import os
import chainlit as cl
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Path to the Chroma DB directory
CHROMA_PATH = "./chroma_db"  # Adjust path as needed

# Initialize the embedding function
embeddings = OpenAIEmbeddings()

# Template for a custom prompt
template = """You are an assistant specialized in Argentinian Spanish (Rioplatense Spanish).
Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context:
{context}

Question: {question}
Helpful Answer:"""

@cl.on_chat_start
async def start():
    # Send initial message
    await cl.Message(
        content="Hello! I'm your Argentinian Spanish assistant. Ask me anything about Rioplatense Spanish!"
    ).send()
    
    # Load the vector store
    try:
        vectorstore = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=embeddings
        )
        
        # Create a retriever from the vector store
        retriever = vectorstore.as_retriever(
            search_kwargs={"k": 4}  # Retrieve top 4 most relevant chunks
        )
        
        # Create a prompt
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        # Create a chain
        chain = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(model="gpt-4-turbo", temperature=0),
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )
        
        # Store the chain in the user session
        cl.user_session.set("chain", chain)
        
    except Exception as e:
        await cl.Message(
            content=f"Error initializing the assistant: {str(e)}"
        ).send()

@cl.on_message
async def on_message(message: cl.Message):
    # Get the chain from the user session
    chain = cl.user_session.get("chain")
    
    # Get response from chain
    response = await cl.make_async(chain)(message.content)
    
    # Extract the answer and sources
    answer = response["result"]
    source_documents = response["source_documents"]
    
    # Prepare sources text
    sources_text = ""
    for i, doc in enumerate(source_documents):
        source_url = doc.metadata.get("url", "Unknown source")
        source_title = doc.metadata.get("title", "Untitled")
        sources_text += f"{i+1}. [{source_title}]({source_url})\n"
    
    # Send the response with sources
    await cl.Message(
        content=answer,
        elements=[
            cl.Text(name="Sources", content=sources_text, display="inline")
        ] if sources_text else []
    ).send()
```

## 3. Notes on Deployment

1. **Environment Variables**: Ensure your OpenAI API key is set on the deployment environment.
2. **Vector Store Access**: Both your data processing scripts and Chainlit app need access to the same Chroma DB directory. Ensure paths are properly configured.
3. **Periodic Updates**: To keep content fresh, consider running the scraper/embedding pipeline periodically and updating the vector store.

## 4. Future Improvements

1. **Additional Sources**: You could expand the data collection to include other sources about Argentinian Spanish.
2. **Custom Embeddings**: Consider using a Spanish-specific embedding model like E5, BAAI or other multilingual models.
3. **Better Chunking**: Text splitting could be improved to preserve more context around topics.
4. **Use Metadata for Filtering**: Add capabilities to filter by content type, topic, etc.
5. **Hybrid Search**: Implement keyword+semantic search for better retrieval in edge cases. 