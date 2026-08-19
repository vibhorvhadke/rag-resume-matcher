# RAG-Based Resume-Job Matching System

A Retrieval-Augmented Generation (RAG) system that matches candidate resumes against job descriptions using semantic search, hybrid scoring, and metadata filtering.

## Features
- Section-aware document chunking (Education, Experience, Skills)
- Embeddings via Sentence-Transformers (all-MiniLM-L6-v2)
- Vector storage and retrieval using ChromaDB
- Hybrid search (semantic similarity + keyword matching on critical skills)
- Must-have requirement filtering (e.g., "5+ years Python")
- Match scoring (0-100) with reasoning

## Project Structure
├── resume_rag.py # Document processing, chunking, embedding, storage
├── job_matcher.py # Semantic search, hybrid scoring, ranking
├── RAG_Based_Profile_Matching.ipynb # Full notebook with experimentation & analysis
├── resumes/ # Sample resume dataset (30 resumes)
├── job_descriptions/ # Sample job descriptions (6 JDs)
├── outputs/ # Match results and performance metrics
└── vector_db/ # ChromaDB persistent storage

## Performance
- Average Precision@5: 0.73
- Average Recall: 0.89
- Average Latency: ~0.11 seconds per query

## Tech Stack
Python, Sentence-Transformers, ChromaDB, Regex-based document parsing
