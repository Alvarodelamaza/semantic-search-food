import streamlit as st
import numpy as np
import pandas as pd
import json
import requests
import os
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import faiss
import sys

# Add better error handling for imports
try:
    from semantic_search import generate_embeddings, enhanced_search, build_faiss_index, faiss_search, enhanced_search_with_faiss
    from utils import convert_embedding_string_to_array
except ImportError as e:
    st.error(f"Failed to import required modules: {e}")
    st.info("Make sure you're running the app from the correct directory and all dependencies are installed.")
    st.stop()

def create_search_ui():
    st.title("🛒 Semantic Item Search System")
    
    # Check if data is loaded
    if 'data_loaded' not in st.session_state or not st.session_state.data_loaded:
        st.warning("Data is still loading or failed to load. Please check the logs.")
        return
    
    # Sidebar for search options
    st.sidebar.header("Search Options")
    search_type = st.sidebar.radio(
        "Select Search Method:",
        ["Basic Semantic Search", "Enhanced Search (LLM Re-ranking)", "Filtered Search", "FAISS Search", "FAISS + LLM Re-ranking"]
    )
    
    top_k = st.sidebar.slider("Number of results to display", min_value=1, max_value=20, value=5)
    
    # Main search interface
    query = st.text_input("Enter your search query:", "")
    
    # Additional options for filtered search
    filter_criteria = None
    if search_type in ["Filtered Search", "FAISS + LLM Re-ranking"]:
        filter_criteria = st.text_input(
            "Enter filter criteria (e.g., 'Only vegetarian items', 'Price under R$30'):",
            ""
        )
    
    # Search button
    search_button = st.button("🔍 Search")
    
    # Initialize session state for storing results
    if 'search_results' not in st.session_state:
        st.session_state.search_results = None
    
    # Perform search when button is clicked
    if search_button and query:
        try:
            with st.spinner("Searching..."):
                # Generate query embedding
                query_embedding = generate_embeddings([query])[0]
                
                # Perform search based on selected method
                if search_type == "Basic Semantic Search":
                    from semantic_search import semantic_search
                    top_indices, top_scores = semantic_search(query_embedding, st.session_state.item_embeddings, top_k=top_k)
                    st.session_state.search_results = (top_indices, top_scores)
                
                elif search_type == "Enhanced Search (LLM Re-ranking)":
                    top_indices = enhanced_search(
                        query=query,
                        item_embeddings=st.session_state.item_embeddings,
                        items_df=st.session_state.items_df,
                        top_k=top_k,
                        apply_filter=False
                    )
                    # Calculate scores for display purposes
                    similarities = cosine_similarity([query_embedding], st.session_state.item_embeddings)[0]
                    top_scores = similarities[top_indices]
                    st.session_state.search_results = (top_indices, top_scores)
                
                elif search_type == "Filtered Search":
                    top_indices = enhanced_search(
                        query=query,
                        item_embeddings=st.session_state.item_embeddings,
                        items_df=st.session_state.items_df,
                        top_k=top_k,
                        apply_filter=True,
                        filter_criteria=filter_criteria if filter_criteria else None
                    )
                    # Calculate scores for display purposes
                    similarities = cosine_similarity([query_embedding], st.session_state.item_embeddings)[0]
                    top_scores = similarities[top_indices]
                    st.session_state.search_results = (top_indices, top_scores)
                    
                elif search_type == "FAISS Search":
                    # Use FAISS for fast retrieval
                    top_indices, top_scores = faiss_search(query_embedding, st.session_state.faiss_index, top_k=top_k)
                    st.session_state.search_results = (top_indices, top_scores)
                    
                elif search_type == "FAISS + LLM Re-ranking":
                    # Use FAISS with LLM re-ranking
                    top_indices = enhanced_search_with_faiss(
                        query=query,
                        faiss_index=st.session_state.faiss_index,
                        items_df=st.session_state.items_df,
                        top_k=top_k,
                        apply_filter=True if filter_criteria else False,
                        filter_criteria=filter_criteria if filter_criteria else None
                    )
                    # Calculate scores for display purposes
                    similarities = cosine_similarity([query_embedding], st.session_state.item_embeddings)[0]
                    top_scores = similarities[top_indices]
                    st.session_state.search_results = (top_indices, top_scores)
        except Exception as e:
            st.error(f"Error during search: {e}")
            return
    
    # Display results if available
    if st.session_state.search_results:
        display_results(query, st.session_state.search_results[0], st.session_state.search_results[1], st.session_state.items_df)

def display_results(query, top_indices, top_scores, items_df):
    """Display search results in a user-friendly format"""
    st.subheader(f"Search Results for: '{query}'")
    
    # Create columns for results
    cols_per_row = 3
    
    for i in range(0, len(top_indices), cols_per_row):
        # Create a row of columns
        cols = st.columns(cols_per_row)
        
        # Fill each column with a result
        for j in range(cols_per_row):
            if i + j < len(top_indices):
                idx = top_indices[i + j]
                score = top_scores[i + j]
                
                with cols[j]:
                    display_item(idx, score, j+i+1, items_df)
                    
def display_item(idx, score, rank, items_df):
    """Display a single item in the results"""
    try:
        row = items_df.iloc[idx]
        
        # Parse itemMetadata
        item_meta = json.loads(row['itemMetadata'])
        
        # Extract item details
        item_name = item_meta.get('name', 'Unknown Item')
        item_description = item_meta.get('description', 'No description')
        item_price = item_meta.get('price', 'Price not available')
        item_category = item_meta.get('category_name', 'No category')
        
        # Create a card-like display
        st.markdown(f"### {rank}. {item_name}")
        
        # Get image URL
        images = item_meta.get('images', [])
        if images:
            image_str = images[0]
            image_url = f"https://static.ifood-static.com.br/image/upload/t_low/pratos/{image_str}"
            
            # Try to display the image
            try:
                response = requests.get(image_url)
                img = Image.open(BytesIO(response.content))
                st.image(img, use_container_width=True)
            except Exception as e:
                st.info(f"Image not available")
        else:
            st.info("No image available")
        
        # Display item details
        st.markdown(f"**Description:** {item_description}")
        st.markdown(f"**Price:** R${item_price:.2f}" if isinstance(item_price, (int, float)) else f"**Price:** {item_price}")
        st.markdown(f"**Category:** {item_category}")
        st.markdown(f"**Similarity Score:** {score:.4f}")
        
        # Add a separator
        st.markdown("---")
    except Exception as e:
        st.error(f"Error displaying item {idx}: {e}")

def load_data():
    """Load all required data with proper error handling"""
    st.session_state.data_loaded = False
    
    try:
        # Display current working directory for debugging
        st.sidebar.info(f"Current working directory: {os.getcwd()}")
        
        # Try to load data files
        data_path = "../data"
        
        # Check if data directory exists
        if not os.path.exists(data_path):
            st.sidebar.error(f"Data directory not found: {data_path}")
            st.sidebar.info("Please make sure you're running the app from the correct directory")
            return False
        
        # Load queries
        queries_path = os.path.join(data_path, "queries.csv")
        if os.path.exists(queries_path):
            st.session_state.queries = pd.read_csv(queries_path)
            st.sidebar.success("Queries loaded successfully")
        else:
            st.sidebar.error(f"File not found: {queries_path}")
            return False
        
        # Load test queries
        test_queries_path = os.path.join(data_path, "test_queries.csv")
        if os.path.exists(test_queries_path):
            test_queries_df = pd.read_csv(test_queries_path)
            st.session_state.test_queries = list(test_queries_df['query'])
            st.sidebar.success("Test queries loaded successfully")
        else:
            st.sidebar.error(f"File not found: {test_queries_path}")
            return False
        
        # Load items
        items_path = os.path.join(data_path, "5k_items_processed.csv")
        if os.path.exists(items_path):
            st.session_state.items_df = pd.read_csv(items_path)
            st.sidebar.success("Items loaded successfully")
        else:
            st.sidebar.error(f"File not found: {items_path}")
            return False
        
        # Convert embeddings
        try:
            st.session_state.item_embeddings = np.array([
                convert_embedding_string_to_array(emb) 
                for emb in st.session_state.items_df['embeddings_jointText']
            ])
            st.session_state.item_embeddings_natural = np.array([
                convert_embedding_string_to_array(emb) 
                for emb in st.session_state.items_df['embeddings_jointTextNatural']
            ])
            st.sidebar.success("Embeddings converted successfully")
        except Exception as e:
            st.sidebar.error(f"Error converting embeddings: {e}")
            return False
        
        # Build FAISS index
        try:
            with st.spinner("Building FAISS index..."):
                st.session_state.faiss_index = build_faiss_index(st.session_state.item_embeddings)
                st.sidebar.success("FAISS index built successfully!")
        except Exception as e:
            st.sidebar.error(f"Error building FAISS index: {e}")
            return False
        
        st.session_state.data_loaded = True
        return True
    
    except Exception as e:
        st.sidebar.error(f"Error loading data: {e}")
        return False

# Add a function to run the app
if __name__ == "__main__":
    st.set_page_config(
        page_title="Semantic Item Search",
        page_icon="🛒",
        layout="wide"
    )
    
    # Load data first
    if 'data_loaded' not in st.session_state:
        with st.spinner("Loading data and initializing models..."):
            success = load_data()
            if not success:
                st.error("Failed to load data. Please check the logs in the sidebar.")
    
    # Create the UI
    create_search_ui()