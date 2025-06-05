# Local LLM-Based Candidate Recommendation System

This project showcases the implementation of a local, privacy-preserving candidate recommendation system using open-source language models. The goal is to simulate how a hiring assistant can interactively shortlist candidates based on a conversation with the user — all without relying on any cloud-based APIs.

## Project Description

Traditional hiring systems often depend on centralized services or third-party platforms for processing candidate data. In contrast, this project demonstrates how to run a complete recommendation system **locally**, powered by transformer models and vector search.

We use a pre-trained language model (such as GPT-2) and reduce its complexity using a **layer-skipping technique**, inspired by the LayerSkip research paper. The system generates fictional candidate CVs, encodes them using **Sentence Transformers**, and stores them in a **FAISS vector index** for efficient similarity search.

During interaction, the system collects user preferences (e.g., experience, skills, salary range), refines the candidate pool using semantic search, and returns the most relevant candidates through multi-turn filtering.

This makes it a useful prototype for building intelligent recruiting tools, especially in environments where privacy, cost-efficiency, or offline availability are critical.

## Key Features

- Generates fictional CVs with realistic attributes
- Encodes and indexes CVs using Sentence Transformers and FAISS
- Allows interactive, multi-turn candidate filtering through dialogue
- Demonstrates a reduced-layer GPT-2 model for faster local inference
- Runs entirely offline (no external API calls)

## Project Files

- `New_Assignment.ipynb`: Jupyter notebook with complete implementation and explanations
- `requirements.txt`: Required libraries for setup
- `README.md`: Project overview and usage instructions

