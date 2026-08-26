# Research RAG Chatbot

An AI-powered chatbot that allows users to ask questions about a research paper using Retrieval-Augmented Generation (RAG).

## Overview

This project takes a research paper in PDF format, processes its content, and allows users to ask questions about the paper.

Instead of sending the entire document to the LLM, the system retrieves the most relevant sections of the paper and provides them as context to the language model.

## Architecture

```text
Research Paper (PDF)
        ↓
   PDF Loader
        ↓
   Text Chunking
        ↓
    Embeddings
        ↓
      FAISS
        ↓
    Retriever
        ↓
 Relevant Chunks
        ↓
       LLM
        ↓
   Final Answer