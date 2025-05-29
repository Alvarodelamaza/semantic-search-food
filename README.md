# 🔍🛒 Semantic Search Project

This repository contains a semantic search system for food items using embeddings and FAISS for fast vector similarity search. The system includes a Streamlit web interface with multiple search methods including basic semantic search, LLM re-ranking, and filtered search.

The explanation and Analysis of experiments can be found in:
https://www.overleaf.com/read/dbdnkdksnrvx#2a3038
Done by Alvaro de la Maza


## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Setup

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/yourusername/semantic-search-system.git
   cd semantic-search-system

2. **Create and activate Virtual Environment**

   
   ```bash
   # For macOS/Linux
   python -m venv venv
   source venv/bin/activate

  
   ```bash
   # For Windows
   python -m venv venv
   venv\Scripts\activate

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt

4. **Download the Dataset**
   Create a data directory and download the required data files into it**

5. **Start the Streamlit App**
   Navigate to the src directory and run:
   ```bash
   cd src
   streamlit run search_app.py
