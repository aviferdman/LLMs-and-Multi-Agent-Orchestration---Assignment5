"""
Metrics Module

Provides comprehensive evaluation metrics for LLM experiments.

Classes:
    MetricsCalculator: Main metrics calculation class

Functions:
    accuracy: Calculate accuracy score
    semantic_similarity: Calculate semantic similarity
    precision_at_k: Calculate Precision@k
    recall_at_k: Calculate Recall@k
    mrr: Mean Reciprocal Rank
    ndcg: Normalized Discounted Cumulative Gain
"""

from typing import List, Any
import numpy as np
from loguru import logger


class MetricsCalculator:
    """
    Calculate various evaluation metrics for experiments.
    
    Supports:
    - Accuracy metrics
    - Retrieval metrics (P@k, R@k, MRR, nDCG)
    - Semantic similarity
    - Latency statistics
    """
    
    def __init__(self):
        """Initialize metrics calculator."""
        logger.info("MetricsCalculator initialized")
    
    def accuracy(
        self, 
        predictions: List[str], 
        ground_truth: List[str]
    ) -> float:
        """
        Calculate accuracy between predictions and ground truth.
        
        Args:
            predictions: List of predicted answers
            ground_truth: List of correct answers
            
        Returns:
            Accuracy score between 0 and 1
            
        Raises:
            ValueError: If lists have different lengths
        """
        if len(predictions) != len(ground_truth):
            raise ValueError("Lists must have same length")
        
        if len(predictions) == 0:
            return 0.0
        
        correct = sum(p == g for p, g in zip(predictions, ground_truth))
        return correct / len(predictions)
    
    def semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0 and 1
        """
        # TODO: Implement semantic similarity using embeddings
        # Placeholder: simple word overlap
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if len(words1) == 0 or len(words2) == 0:
            return 0.0
        
        overlap = len(words1 & words2)
        union = len(words1 | words2)
        
        return overlap / union if union > 0 else 0.0
    
    def precision_at_k(
        self, 
        retrieved: List[Any], 
        relevant: List[Any], 
        k: int
    ) -> float:
        """
        Calculate Precision@k.
        
        Args:
            retrieved: List of retrieved items
            relevant: List of relevant items
            k: Number of top items to consider
            
        Returns:
            Precision@k score
        """
        if k <= 0 or len(retrieved) == 0:
            return 0.0
        
        retrieved_k = retrieved[:k]
        relevant_set = set(relevant)
        
        correct = sum(1 for item in retrieved_k if item in relevant_set)
        return correct / k
    
    def recall_at_k(
        self, 
        retrieved: List[Any], 
        relevant: List[Any], 
        k: int
    ) -> float:
        """
        Calculate Recall@k.
        
        Args:
            retrieved: List of retrieved items
            relevant: List of relevant items
            k: Number of top items to consider
            
        Returns:
            Recall@k score
        """
        if len(relevant) == 0 or k <= 0:
            return 0.0
        
        retrieved_k = retrieved[:k]
        relevant_set = set(relevant)
        
        correct = sum(1 for item in retrieved_k if item in relevant_set)
        return correct / len(relevant)
    
    def mrr(
        self, 
        retrieved_lists: List[List[Any]], 
        relevant_lists: List[List[Any]]
    ) -> float:
        """
        Calculate Mean Reciprocal Rank.
        
        Args:
            retrieved_lists: List of retrieved item lists (one per query)
            relevant_lists: List of relevant item lists (one per query)
            
        Returns:
            MRR score
        """
        if len(retrieved_lists) != len(relevant_lists):
            raise ValueError("Lists must have same length")
        
        if len(retrieved_lists) == 0:
            return 0.0
        
        reciprocal_ranks = []
        for retrieved, relevant in zip(retrieved_lists, relevant_lists):
            relevant_set = set(relevant)
            for i, item in enumerate(retrieved, 1):
                if item in relevant_set:
                    reciprocal_ranks.append(1.0 / i)
                    break
            else:
                reciprocal_ranks.append(0.0)
        
        return np.mean(reciprocal_ranks)
    
    def ndcg_at_k(
        self, 
        retrieved: List[Any], 
        relevance_scores: List[float], 
        k: int
    ) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain at k.
        
        Args:
            retrieved: List of retrieved items
            relevance_scores: Relevance scores for each item
            k: Number of top items to consider
            
        Returns:
            nDCG@k score
        """
        if k <= 0 or len(retrieved) == 0:
            return 0.0
        
        # Calculate DCG@k
        dcg = sum(
            (2 ** rel - 1) / np.log2(i + 2)
            for i, rel in enumerate(relevance_scores[:k])
        )
        
        # Calculate IDCG@k (ideal DCG)
        ideal_scores = sorted(relevance_scores, reverse=True)[:k]
        idcg = sum(
            (2 ** rel - 1) / np.log2(i + 2)
            for i, rel in enumerate(ideal_scores)
        )
        
        return dcg / idcg if idcg > 0 else 0.0
