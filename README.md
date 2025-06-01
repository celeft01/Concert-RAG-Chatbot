# Concert RAG Chatbot

A domain-specific Retrieval-Augmented Generation (RAG) system for concert and tour information. This application allows users to:
- Upload concert-related documents for knowledge base ingestion
- Ask questions about uploaded documents
- Search for upcoming concerts for specific artists

## Architecture Overview

The system consists of two main components:

1. **RAG Service Backend** (`rag_service.py`) - FastAPI-based service that:
   - Processes and indexes documents
   - Responds to questions using a retrieval-augmented approach
   - Searches for concert information

2. **UI Application** (`ui_app.py`) - Streamlit-based web interface that:
   - Provides document upload functionality
   - Shows chat history
   - Displays concert search results

## Technical Features

- **Document Processing**: Chunks documents with overlap for better context preservation
- **Semantic Search**: Uses FAISS for efficient similarity search
- **Embeddings**: Utilizes Sentence Transformers for generating embeddings
- **Text Generation**: Implements FLAN-T5-XL for generating coherent answers
- **Domain Filtering**: Validates that documents are concert-related
- **Concert Search Integration**: Uses external API to find upcoming concerts

## Model Selection Rationale

### Embedding Model: all-mpnet-base-v2
- **Reasoning**: Selected for its strong performance on semantic similarity tasks with 768-dimensional embeddings
- **Benefits**:
  - Superior performance on semantic textual similarity benchmarks
  - Effective at capturing nuanced meaning in short text chunks
  - Good balance between quality and computational efficiency
  - Trained on diverse datasets, making it robust for varied concert documentation

### Generation Model: FLAN-T5-LARGE
- **Reasoning**: Chosen for its strong instruction-following ability and solid performance on generation tasks
- **Benefits**:
  - Fine-tuned on a wide range of instructional data, making it well-suited for question-answering and summarization
  - Delivers coherent and context-aware responses with lower computational overhead
  - Faster inference and lower memory usage compared to models like FLAN-T5-XL, making it a practical choice for limited-resource environments
  - High quality in generating relevant answers from retrieved document chunks

### Vector Database: FAISS (Facebook AI Similarity Search)
- **Reasoning**: Selected for efficient similarity search operations
- **Benefits**:
  - Optimized for rapid nearest neighbor search
  - Scales well with increasing document volume
  - Supports L2 distance metric suitable for our embedding space
  - Low memory footprint for in-memory operation

## Setup Instructions

### Prerequisites

- Python 3.8+
- Virtual environment tool (recommended)

### Installation

1. Clone this repository:
   ```
   git clone https://github.com/celeft01/ProvectusInternship_ChristosEleftheriou.git
   cd ProvectusInternship_ChristosEleftheriou
   ```

2. Create and activate a virtual environment (Optional):

   - For Windows (Powershell)
   ```
   python -m venv venv
   Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process
   .\venv\Scripts\Activate.ps1
   ```
   - For Windows (Command Prompt)
   ```
   python -m venv venv
   venv\Scripts\activate.bat
   ```
   - For macOS/Linux
   ```
   python -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

### Running the Application

1. Start the RAG service backend:
   ```
   python rag_service.py
   ```

2. In a separate terminal (with the virtual environment activated if activated before), start the UI:
   ```
   streamlit run ui_app.py
   ```

3. Open your browser and navigate to:
   ```
   http://localhost:8501
   ```

## Usage Guide

### Uploading Documents

- Use the document upload section to add concert-related information
- Either upload a PDF/TXT file or paste text directly
- The system will validate if the content is concert-related
- You'll receive a summary of the ingested document

### Asking Questions

- Type your question in the question input box
- The system will retrieve relevant information from uploaded documents
- Generated answers are based on the context of your documents

### Searching for Concerts

- Enter an artist or band name in the search field
- The system will search for upcoming concerts
- Results include event details and links

## Design Choices

1. **Domain-Specific Approach**:
   - The system is intentionally focused on concert and tour information
   - Domain filtering ensures high-quality, relevant responses

2. **Chunking Strategy**:
   - Documents are split into overlapping chunks to maintain context
   - Each chunk is embedded and indexed separately
   - The 200-character chunk size with 100-character overlap was chosen to balance context preservation with retrieval precision

3. **Context Augmentation**:
   - When answering questions, the system combines relevant chunks
   - Document summaries are included for additional context
   - Fallback mechanisms ensure responses even with limited relevant chunks

4. **User Interface**:
   - Streamlit provides an intuitive, easy-to-use interface
   - Chat history is maintained for reference
   - Support for both text input and document uploads increases flexibility

## Future Improvements

- Add user authentication
- Implement document persistence using a database
- Add support for more document formats
- Improve concert search with more detailed information
- Enhance similarity search with hybrid retrieval methods
- Explore fine-tuning the generation model on concert-specific data
- Implement vector database persistence for system restarts
