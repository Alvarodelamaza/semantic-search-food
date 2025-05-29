
import json
import numpy as np
import ast
import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO
import ipywidgets as widgets
from semantic_search import semantic_search ,enhanced_search, generate_embeddings, cosine_similarity,  build_faiss_index, enhanced_search_with_faiss
from IPython.display import display, clear_output


def display_search_results(query, item_embeddings, items_df, top_k=5, search_method="Basic Semantic Search", filter_criteria=None):
    """
    Display search results including item details and images, with selectable search methodology.
    
    Args:
        query: The original search query text
        item_embeddings: A 2D array of item embedding vectors
        items_df: DataFrame containing item data
        top_k: Number of top results to display
        search_method: The search methodology to use ("Basic Semantic Search", "Enhanced Search", "Filtered Search")
        filter_criteria: Criteria for filtering items (used only for "Filtered Search")
    """
    # Generate query embedding
    query_embedding = generate_embeddings([query])[0]
    
    # Perform search based on selected methodology
    if search_method == "Basic Semantic Search":
        top_indices, top_scores = semantic_search(query_embedding, item_embeddings, top_k=top_k)
    elif search_method == "Enhanced Search":
        top_indices = enhanced_search(query, item_embeddings, items_df, top_k=top_k, apply_filter=False)
        top_scores = cosine_similarity([query_embedding], item_embeddings)[0][top_indices]
    elif search_method == "Filtered Search":
        top_indices = enhanced_search(query, item_embeddings, items_df, top_k=top_k, apply_filter=True, filter_criteria=filter_criteria)
        top_scores = cosine_similarity([query_embedding], item_embeddings)[0][top_indices]
    else:
        print("Invalid search method selected.")
        return
    
    # Display results
    print(f"Search Query: '{query}'")
    print(f"Top {top_k} Results:")
    print("-" * 80)
    
    # Create a figure for displaying images
    fig, axes = plt.subplots(top_k, 1, figsize=(10, 30))
    if top_k == 1:
        axes = [axes]  # Make axes iterable when top_k=1
    
    for i, (idx, score) in enumerate(zip(top_indices, top_scores)):
        row = items_df.iloc[idx]
        
        # Parse itemMetadata
        item_meta = json.loads(row['itemMetadata'])
        
        # Extract item details
        item_name = item_meta.get('name', 'Unknown Item')
        item_description = item_meta.get('description', 'No description')
        item_price = item_meta.get('price', 'Price not available')
        item_category = item_meta.get('category_name', 'No category')
        
        # Print item details
        print(f"Rank {i+1}: {item_name}")
        print(f"  Description: {item_description}")
        print(f"  Price: R${item_price:.2f}" if isinstance(item_price, (int, float)) else f"  Price: {item_price}")
        print(f"  Category: {item_category}")
        print(f"  Similarity Score: {score:.4f}")
        
        # Get image URL
        images = item_meta.get('images', [])
        if images:
            image_str = images[0]
            image_url = f"https://static.ifood-static.com.br/image/upload/t_low/pratos/{image_str}"
            print(f"  Image URL: {image_url}")
            
            # Try to display the image
            try:
                response = requests.get(image_url)
                img = Image.open(BytesIO(response.content))
                axes[i].imshow(img)
                axes[i].set_title(f"{i+1}. {item_name[:20]}..." if len(item_name) > 20 else f"{i+1}. {item_name}")
                axes[i].axis('off')
            except Exception as e:
                print(f"  Error loading image: {e}")
                axes[i].text(0.5, 0.5, "Image not available", ha='center', va='center')
                axes[i].axis('off')
        else:
            print("  No image available")
            axes[i].set_title(f"{i+1}. {item_name[:20]}..." if len(item_name) > 20 else f"{i+1}. {item_name}")
            axes[i].text(0.5, 0.5, "No image", ha='center', va='center')
            axes[i].axis('off')
        
        print("-" * 80)
    
    plt.tight_layout()
    plt.show()
    
    return top_indices, top_scores

# Interactive UI for Jupyter Notebook
def search_ui(item_embeddings, items_df):
    """
    Create an interactive UI for searching in Jupyter Notebook, including filtering criteria.
    """
    query_input = widgets.Text(description="Query:")
    search_method_dropdown = widgets.Dropdown(
        options=["Basic Semantic Search", "Enhanced Search", "Filtered Search"],
        description="Method:"
    )
    top_k_slider = widgets.IntSlider(value=5, min=1, max=20, step=1, description="Top K:")
    filter_criteria_input = widgets.Text(description="Filter Criteria:")  # New widget for filtering criteria
    search_button = widgets.Button(description="Search")
    output = widgets.Output()

    def on_search_button_clicked(b):
        with output:
            clear_output()
            query = query_input.value
            search_method = search_method_dropdown.value
            top_k = top_k_slider.value
            filter_criteria = filter_criteria_input.value  # Get the filter criteria input
            if query:
                if search_method == "Filtered Search" and not filter_criteria:
                    print("Please enter filter criteria for the filtered search.")
                else:
                    display_search_results(
                        query,
                        item_embeddings,
                        items_df,
                        top_k=top_k,
                        search_method=search_method,
                        filter_criteria=filter_criteria if search_method == "Filtered Search" else None
                    )
            else:
                print("Please enter a query.")

    search_button.on_click(on_search_button_clicked)

    display(widgets.VBox([query_input, search_method_dropdown, top_k_slider, filter_criteria_input, search_button, output]))

def convert_embedding_string_to_array(embedding_str):
    # Remove any outer quotes if present
    if embedding_str.startswith("'") and embedding_str.endswith("'"):
        embedding_str = embedding_str[1:-1]
    
    try:
        # Try using ast.literal_eval which is safer than eval
        return np.array(ast.literal_eval(embedding_str))
    except:
        try:
            # Try using json.loads as a fallback
            return np.array(json.loads(embedding_str))
        except:
            # If all else fails, try a manual string parsing approach
            values = embedding_str.strip('[]').split(',')
            return np.array([float(v.strip()) for v in values])
