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

  
   
   # For Windows
   python -m venv venv
   venv\Scripts\activate

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt

4. **Download the Dataset**
   Create a data directory and download the required data files into it**

5. **Start the UI**
    There are two option to visualize the results
         
      ### Streamlit App
      Navigate to the `src` directory and run:
      
      cd src
      streamlit run search_app.py
        
      ### Jupyter Notebook UI
      1. Go to the notebook `data_exploration.ipynb`
         Run all the cells to prepare the dataset

      2. Go to the notebook `evaluation.ipynb`
         Run all the cells of the section Data loading and Run the first cell of Model Evaluations

      3. Go to Display subsection and run the UI cell

      4. Ready to start your search!


## Notebooks

   #### data_exploration.ipynb 
   Contains all the initial EDA and the feature engineering necessry for next parts
   #### evaluation.ipynb 
   Contains all the evaluations of diffferent approaches, as well as metric calculations
   #### generate_test_set.ipynb 
   Contains thegeneration of the 200 queries test set

## Files
   #### evaluation.py 
   Contains functions to calculate metrics
   #### semantic_search.py 
   Contains all the funtions to generate search with the different approaches
   #### preprocess.py 
   Contains functions to generate new features
   #### utils.py 
   Contains auxiliary functions 
