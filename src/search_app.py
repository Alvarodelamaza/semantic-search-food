import streamlit as st
import numpy as np
import pandas as pd
import json
import requests
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import faiss

from semantic_search import generate_embeddings, enhanced_search, build_faiss_index, faiss_search, enhanced_search_with_faiss
from utils import convert_embedding_string_to_array

def create_search_ui():
    st.title("🛒 Semantic Item Search System")
    
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
        with st.spinner("Searching..."):
            # Generate query embedding
            query_embedding = generate_embeddings([query])[0]
            
            # Perform search based on selected method
            if search_type == "Basic Semantic Search":
                from semantic_search import semantic_search
                top_indices, top_scores = semantic_search(query_embedding, item_embeddings, top_k=top_k)
                st.session_state.search_results = (top_indices, top_scores)
            
            elif search_type == "Enhanced Search (LLM Re-ranking)":
                top_indices = enhanced_search(
                    query=query,
                    item_embeddings=item_embeddings,
                    items_df=items_df,
                    top_k=top_k,
                    apply_filter=False
                )
                # Calculate scores for display purposes
                similarities = cosine_similarity([query_embedding], item_embeddings)[0]
                top_scores = similarities[top_indices]
                st.session_state.search_results = (top_indices, top_scores)
            
            elif search_type == "Filtered Search":
                top_indices = enhanced_search(
                    query=query,
                    item_embeddings=item_embeddings,
                    items_df=items_df,
                    top_k=top_k,
                    apply_filter=True,
                    filter_criteria=filter_criteria if filter_criteria else None
                )
                # Calculate scores for display purposes
                similarities = cosine_similarity([query_embedding], item_embeddings)[0]
                top_scores = similarities[top_indices]
                st.session_state.search_results = (top_indices, top_scores)
                
            elif search_type == "FAISS Search":
                # Use FAISS for fast retrieval
                top_indices, top_scores = faiss_search(query_embedding, faiss_index, top_k=top_k)
                st.session_state.search_results = (top_indices, top_scores)
                
            elif search_type == "FAISS + LLM Re-ranking":
                # Use FAISS with LLM re-ranking
                top_indices = enhanced_search_with_faiss(
                    query=query,
                    faiss_index=faiss_index,
                    items_df=items_df,
                    top_k=top_k,
                    apply_filter=True if filter_criteria else False,
                    filter_criteria=filter_criteria if filter_criteria else None
                )
                # Calculate scores for display purposes
                similarities = cosine_similarity([query_embedding], item_embeddings)[0]
                top_scores = similarities[top_indices]
                st.session_state.search_results = (top_indices, top_scores)
    
    # Display results if available
    if st.session_state.search_results:
        display_results(query, st.session_state.search_results[0], st.session_state.search_results[1], items_df)

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
            st.error(f"Error loading image: {e}")
    else:
        st.info("No image available")
    
    # Display item details
    st.markdown(f"**Description:** {item_description}")
    st.markdown(f"**Price:** R${item_price:.2f}" if isinstance(item_price, (int, float)) else f"**Price:** {item_price}")
    st.markdown(f"**Category:** {item_category}")
    st.markdown(f"**Similarity Score:** {score:.4f}")
    
    # Add a separator
    st.markdown("---")

# Add a function to run the app
if __name__ == "__main__":
    # Load data and models
    queries = pd.read_csv("../data/queries.csv")
    test_queries = pd.read_csv("../data/test_queries.csv")
    test_queries = list(test_queries['query'])
    items_df = pd.read_csv("../data/5k_items_processed.csv")

    # Convert embeddings
    item_embeddings = np.array([convert_embedding_string_to_array(emb) for emb in items_df['embeddings_jointText']])
    item_embeddings_natural = np.array([convert_embedding_string_to_array(emb) for emb in items_df['embeddings_jointTextNatural']])
    
    # Build FAISS index (do this once at startup)
    with st.spinner("Building FAISS index..."):
        faiss_index = build_faiss_index(item_embeddings)
        st.success("FAISS index built successfully!")
    
    create_search_ui()