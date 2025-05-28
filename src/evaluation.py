import numpy as np
from typing import List, Dict

def evaluate_search_system(
    queries: List[str],
    search_results: List[List[int]],
    k_values: List[int] = [1, 3, 5, 10]
) -> Dict:
    """
    Evaluate search system performance when there's only one true positive per query,
    and the true product IDs are integers starting from 0 that match the query order.
    
    Args:
        queries: List of query strings
        search_results: List of lists, where each inner list contains ranked product IDs for a query
        k_values: List of k values for Precision@k, Recall@k, etc.
        
    Returns:
        Dictionary with evaluation metrics
    """
    metrics = {}
    
    # The true product ID for query at index i is simply i
    true_product_ids = list(range(len(queries)))
    
    # Calculate Mean Reciprocal Rank (MRR)
    reciprocal_ranks = []
    average_ranks = []
    
    for i, results in enumerate(search_results):
        true_id = i  # The true product ID for query i is i
        if true_id in results:
            rank = results.index(true_id) + 1
            reciprocal_ranks.append(1.0 / rank)
            average_ranks.append(rank)
        else:
            reciprocal_ranks.append(0.0)
            average_ranks.append(len(results) + 1)  # Assign a rank worse than the last result
    
    metrics['mrr'] = np.mean(reciprocal_ranks)
    metrics['average_rank'] = np.mean(average_ranks)
    
    # Calculate Hit Rate and other metrics at different k values
    for k in k_values:
        # Hit Rate@k (aka Success@k)
        hit_at_k = []
        for i, results in enumerate(search_results):
            true_id = i  # The true product ID for query i is i
            hit_at_k.append(1.0 if true_id in results[:k] else 0.0)
        
        metrics[f'hit_rate@{k}'] = np.mean(hit_at_k)
        
        # Precision@k and Recall@k for each query
        precision_at_k = []
        recall_at_k = []
        
        for i, results in enumerate(search_results):
            true_id = i  # The true product ID for query i is i
            # For a single relevant item, precision@k = 1/k if item is in top k, 0 otherwise
            if true_id in results[:k]:
                precision_at_k.append(1.0 / k)
                recall_at_k.append(1.0)  # Recall is 1 if the item is found, 0 otherwise
            else:
                precision_at_k.append(0.0)
                recall_at_k.append(0.0)
        
        metrics[f'precision@{k}'] = np.mean(precision_at_k)
        metrics[f'recall@{k}'] = np.mean(recall_at_k)
    
    return metrics


# Full evaluation 
def evaluate_full_system(queries, search_results_function):
    """
    Evaluate the full system with all queries
    
    Args:
        queries: List of all query strings
        search_results_function: A function that takes a query and returns ranked results
    """
    all_search_results = []
    
    # Get search results for each query
    for query in queries:
        results = search_results_function(query)
        all_search_results.append(results)
    
    # Evaluate
    metrics = evaluate_search_system(queries, all_search_results)
    
    # Print results
    print("Evaluation Results:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    return metrics

