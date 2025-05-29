from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
import matplotlib.pyplot as plt

import openai
from dotenv import load_dotenv
import os 
import faiss
import numpy as np

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")

client = openai.OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_API_BASE
)

def generate_embeddings(texts, model="text-embedding-3-large", batch_size=10):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]  
        response = client.embeddings.create(input=batch, model=model)  
        embeddings.extend([item.embedding for item in response.data])  
    return embeddings


def semantic_search(query_embedding, item_embeddings, top_k=30):
    # Compute cosine similarity
    similarities = cosine_similarity([query_embedding], item_embeddings)[0]
    
    # Get top-k most similar items
    top_k_indices = np.argsort(similarities)[-top_k:][::-1]
    return top_k_indices, similarities[top_k_indices]



def rerank_with_llm(query, candidate_items, items_df, top_k=10):
    """
    Re-rank candidate items using LLM with function calling.
    """
    tools = [
        {
            "type": "function",
            "function": {
                "name": "rank_items",
                "description": "Rank items based on relevance to query and other factors",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ranked_items": {
                            "type": "array",
                            "description": "List of item IDs ranked by relevance, with most relevant first",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "item_id": {
                                        "type": "integer",
                                        "description": "ID of the item"
                                    },
                                    "relevance_score": {
                                        "type": "number",
                                        "description": "Score from 0-10 indicating relevance to the query"
                                    },
                                    "reasoning": {
                                        "type": "string",
                                        "description": "Brief explanation of why this item is relevant"
                                    }
                                },
                                "required": ["item_id", "relevance_score"]
                            }
                        }
                    },
                    "required": ["ranked_items"]
                }
            }
        }
    ]
    
    # Create a prompt with item metadata to help the LLM reason
    items_context = []
    for idx in candidate_items:
        item_context = f"Item ID: {idx}\n Information: {items_df['jointText'][idx]} \n Total Orders: {items_df['total_orders'][idx]}"
        
        items_context.append(item_context)
    
    items_text = "\n\n".join(items_context)
    
    # Create the message for the LLM
    messages = [
        {"role": "system", "content": "You are a search ranking expert. Your task is to re-rank search results based on relevance to the query, considering factors like semantic match, popularity, and recency."},
        {"role": "user", "content": f"Query: {query}\n\nCandidate items:\n{items_text}\n\nPlease rank these items by relevance to the query. Consider both semantic relevance and other factors like popularity or recency when appropriate. Call the rank_items function with your ranking."}
    ]
    
    # Call the LLM with function calling
    response = client.chat.completions.create(
        model="gpt-4.1", 
        messages=messages,
        tools=tools,
        tool_choice={"type": "function", "function": {"name": "rank_items"}}
    )
    
    # Extract the function call arguments
    function_call = response.choices[0].message.tool_calls[0].function
    arguments = json.loads(function_call.arguments)
    
    # Add error handling for the response format
    try:
        # Extract the ranked item IDs
        ranked_item_ids = [item["item_id"] for item in arguments["ranked_items"]]
    except KeyError:
        # If the expected format is not found, try to handle alternative formats
        if "ranked_items" in arguments:
            # Check what keys are actually in the items
            if arguments["ranked_items"] and isinstance(arguments["ranked_items"][0], dict):
                # Get the first item to check its keys
                first_item = arguments["ranked_items"][0]
                if "id" in first_item:
                    ranked_item_ids = [item["id"] for item in arguments["ranked_items"]]
                elif "item" in first_item:
                    ranked_item_ids = [item["item"] for item in arguments["ranked_items"]]
                elif "index" in first_item:
                    ranked_item_ids = [item["index"] for item in arguments["ranked_items"]]
                else:
                    # If we can't find a suitable key, just use the original order
                    print("Warning: Could not parse ranked items properly. Using original order.")
                    ranked_item_ids = candidate_items[:top_k]
            elif arguments["ranked_items"] and isinstance(arguments["ranked_items"][0], (int, str)):
                # If the items are directly IDs without being in objects
                ranked_item_ids = [int(item) if isinstance(item, str) and item.isdigit() else item 
                                  for item in arguments["ranked_items"]]
            else:
                # Fallback to original order
                print("Warning: Unexpected format in ranked_items. Using original order.")
                ranked_item_ids = candidate_items[:top_k]
        else:
            # If ranked_items key is missing entirely
            print("Warning: No ranked_items found in response. Using original order.")
            ranked_item_ids = candidate_items[:top_k]
    
    # Return the top_k items
    return ranked_item_ids[:top_k]

def filter_items_with_criteria(query, candidate_items, items_df, criteria=None):
    """
    Filter candidate items based on specific criteria.
    """
    tools = [
        {
            "type": "function",
            "function": {
                "name": "filter_items",
                "description": "Filter items based on specific criteria",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filtered_items": {
                            "type": "array",
                            "description": "List of item IDs that meet the criteria",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "item_id": {
                                        "type": "integer",
                                        "description": "ID of the item"
                                    },
                                    "meets_criteria": {
                                        "type": "boolean",
                                        "description": "Whether the item meets the criteria"
                                    },
                                    "explanation": {
                                        "type": "string",
                                        "description": "Explanation of why the item meets or doesn't meet the criteria"
                                    }
                                },
                                "required": ["item_id", "meets_criteria"]
                            }
                        }
                    },
                    "required": ["filtered_items"]
                }
            }
        }
    ]
    
    # Process items in batches
    batch_size = 20
    all_filtered_items = []
    
    for i in range(0, len(candidate_items), batch_size):
        batch_items = candidate_items[i:i + batch_size]
        
        # Create context for current batch
        items_context = []
        for idx in batch_items:
            item_context = f"Item ID: {idx}\n Information: {items_df['jointText'][idx]} \n Total Orders: {items_df['total_orders'][idx]}"
            items_context.append(item_context)
        
        items_text = "\n\n".join(items_context)
        
        # Create criteria text
        criteria_text = ""
        if criteria:
            criteria_text = f"Apply these specific criteria: {criteria}"
        else:
            criteria_text = "Filter items that are most relevant to the query and meet user expectations."
        
        messages = [
            {"role": "system", "content": "You are a search filtering expert. Your task is to filter search results based on specific criteria."},
            {"role": "user", "content": f"Query: {query}\n\nCandidate items:\n{items_text}\n\n{criteria_text}\n\nPlease filter these items and call the filter_items function with your results."}
        ]
        
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=messages,
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "filter_items"}}
        )
        
        function_call = response.choices[0].message.tool_calls[0].function
        arguments = json.loads(function_call.arguments)
        
        # Add batch results to overall results
        all_filtered_items.extend(arguments["filtered_items"])
    
    # Extract the filtered item IDs (only those that meet criteria)
    filtered_item_ids = [item["item_id"] for item in all_filtered_items if item["meets_criteria"]]
    
    return filtered_item_ids

def enhanced_search(query, item_embeddings, items_df, top_k=10, apply_filter=False, filter_criteria=None):
    """
    Complete search pipeline with embedding search, optional filtering, and re-ranking.
    """
    # Generate query embedding
    query_embedding = generate_embeddings([query])[0]
    
    # First stage: Semantic search to get candidate items
    candidate_indices, _ = semantic_search(query_embedding, item_embeddings, top_k=50)
    
    # Optional filtering stage
    if apply_filter:
        filtered_indices = filter_items_with_criteria(query, candidate_indices, items_df, filter_criteria)
        
        # If filtering returns too few results, use original candidates
        if len(filtered_indices) < 5:
            candidate_indices = candidate_indices
        else:
            candidate_indices = filtered_indices
    
    # If we have too few items, return what we have
    if len(candidate_indices) <= top_k:
        return candidate_indices
    
    # Second stage: Re-ranking
    reranked_indices = rerank_with_llm(query, candidate_indices, items_df, top_k)
    
    return reranked_indices


def tiered_search(query, item_embeddings, items_df, top_k=10):
    """Two-tier search that only uses LLM for ambiguous queries"""
    query_embedding = generate_embeddings([query])[0]
    
    # First tier: Basic semantic search
    candidate_indices, scores = semantic_search(query_embedding, item_embeddings, top_k=top_k)
    
    # Check confidence of top results
    top_score = scores[0] if len(scores) > 0 else 0
    second_score = scores[1] if len(scores) > 1 else 0
    
    # If top result is significantly better than second, return without LLM
    if top_score > 0.85 or (top_score - second_score) > 0.15:
        return candidate_indices
    
    # Second tier: Only use LLM reranking for ambiguous results
    return rerank_with_llm(query, candidate_indices, items_df, top_k)



def build_faiss_index(embeddings):
    """
    Build a FAISS index for fast vector similarity search.
    
    Args:
        embeddings: List or array of embedding vectors
    
    Returns:
        FAISS index
    """
    # Convert embeddings to a proper numpy array
    if isinstance(embeddings, np.ndarray) and embeddings.dtype == object:
        # If embeddings is a numpy array of Python lists
        embeddings_list = [np.array(emb, dtype='float32') for emb in embeddings]
        embeddings = np.vstack(embeddings_list)
    elif isinstance(embeddings, list):
        # If embeddings is a list of lists or numpy arrays
        if all(isinstance(emb, list) for emb in embeddings):
            embeddings = np.array(embeddings, dtype='float32')
        elif all(isinstance(emb, np.ndarray) for emb in embeddings):
            embeddings = np.vstack(embeddings).astype('float32')
    
   
    embeddings = np.ascontiguousarray(embeddings, dtype='float32')
    dimension = embeddings.shape[1]
    
    # Create index - using IndexFlatIP for inner product 
    index = faiss.IndexFlatIP(dimension)
    
    #  normalization
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized_embeddings = embeddings / norms
    normalized_embeddings = np.ascontiguousarray(normalized_embeddings, dtype='float32')
    
    index.add(normalized_embeddings)
    
    return index


def faiss_search(query_embedding, faiss_index, top_k=30):
    """
    Search for similar items using FAISS index.
    
    Args:
        query_embedding: Embedding vector of the query
        faiss_index: FAISS index containing item embeddings
        top_k: Number of top results to return
    
    Returns:
        Tuple of (indices, distances)
    """
    # Convert to numpy 
    if not isinstance(query_embedding, np.ndarray):
        query_embedding = np.array(query_embedding).astype('float32')
    
    # Reshape 
    if len(query_embedding.shape) == 1:
        query_embedding = query_embedding.reshape(1, -1)
    
    # Normalize 
    faiss.normalize_L2(query_embedding)
    
    # Search the index
    distances, indices = faiss_index.search(query_embedding, top_k)
    
    return indices[0], distances[0]

def enhanced_search_with_faiss(query, faiss_index, items_df, top_k=10, apply_filter=False, filter_criteria=None):
    """
    Complete search pipeline with FAISS for fast retrieval, optional filtering, and re-ranking.
    """
    # Generate query embedding
    query_embedding = generate_embeddings([query])[0]
    
    # 1. Fast vector search with FAISS to get candidate items
    candidate_indices, _ = faiss_search(query_embedding, faiss_index, top_k=50)
    
 
    
    # If we have too few items
    if len(candidate_indices) <= top_k:
        return candidate_indices
    
    # 2. Re-ranking
    reranked_indices = rerank_with_llm(query, candidate_indices, items_df, top_k)
    
    return reranked_indices
