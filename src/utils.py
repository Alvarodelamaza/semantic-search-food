from semantic_search import semantic_search
import json
import numpy as np
import ast
import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO


def display_search_results(query, query_embedding, item_embeddings, items_df, top_k=5):
    """
    Display search results including item details and images.
    
    Args:
        query: The original search query text
        query_embedding: The embedding vector of the query
        item_embeddings: A 2D array of item embedding vectors
        items_df: DataFrame containing item data
        top_k: Number of top results to display
    """
    # Perform semantic search
    top_indices, top_scores = semantic_search(query_embedding, item_embeddings, top_k=top_k)
    
    print(f"Search Query: '{query}'")
    print(f"Top {top_k} Results:")
    print("-" * 80)
    
    # Create a figure for displaying images
    fig, axes = plt.subplots( top_k,1, figsize=(10, 30))
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
