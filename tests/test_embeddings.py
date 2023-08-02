import numpy as np
from guardrail.dataset.embeddings import EmbeddingModel

# Test case 1: Testing the `generate_freshness_score` method
embedding_data = {
    'result1': np.array([0.8, 0.2, 0.5]),
    'result2': np.array([0.9, 0.3, 0.6]),
    'result3': np.array([0.7, 0.1, 0.4]),
    # Add more text data and corresponding embeddings here
}

embedding_model = EmbeddingModel(embedding_data, non_linear_function='sigmoid')
query_embedding = np.array([0.8, 0.2, 0.5])
freshness_score = embedding_model.generate_freshness_score(query_embedding, weight=1.5, bias=-0.5)
print("Freshness Score:", freshness_score)

# Test case 2: Testing the `improve_embeddings_with_diversity` method
query_embedding = np.array([0.8, 0.2, 0.5])
diverse_results = embedding_model.improve_embeddings_with_diversity(query_embedding, diversity_weight=0.7, top_k=3)
print("Diverse Results:", diverse_results)

# Test case 3: Testing the `reduce_embedding_dimension_pca` method
embeddings_data = np.random.rand(100, 50)  # Example embeddings matrix with 100 samples and 50 dimensions
target_dim_pca = 10  # The desired dimensionality of the reduced embeddings (PCA)
reduced_embeddings_pca = EmbeddingModel.reduce_embedding_dimension_pca(embeddings_data, target_dim_pca)
print("Reduced Embeddings Shape (PCA):", reduced_embeddings_pca.shape)

# Test case 4: Testing the `reduce_embedding_dimension_tsne` method
target_dim_tsne = 2  # The desired dimensionality of the reduced embeddings (t-SNE)
reduced_embeddings_tsne = EmbeddingModel.reduce_embedding_dimension_tsne(embeddings_data, target_dim_tsne)
print("Reduced Embeddings Shape (t-SNE):", reduced_embeddings_tsne.shape)

# Test case 5: Testing the `reduce_embedding_dimension_umap` method
target_dim_umap = 3  # The desired dimensionality of the reduced embeddings (UMAP)
reduced_embeddings_umap = EmbeddingModel.reduce_embedding_dimension_umap(embeddings_data, target_dim_umap)
print("Reduced Embeddings Shape (UMAP):", reduced_embeddings_umap.shape)

import numpy as np

# Test case 1: Diversity
embedding_data_diversity = {
    'get python': np.array([0.8, 0.2, 0.5]),
    'HTTP method': np.array([0.9, 0.3, 0.6]),
    'places to purchase a live python': np.array([0.7, 0.1, 0.4]),
    # Add more text data and corresponding embeddings here
}

embedding_model_diversity = EmbeddingModel(embedding_data_diversity, non_linear_function='sigmoid')
query_embedding_diversity = np.array([0.8, 0.2, 0.5])

# Set diversity_weight to 0.7 to prioritize diversity over relevance
diverse_results = embedding_model_diversity.improve_embeddings_with_diversity(query_embedding_diversity, diversity_weight=0.7, top_k=3)
print("Diverse Results:", diverse_results)
# Expected output: The diverse_results should contain a mix of relevant and diverse results.

# Test case 2: Non-linear scoring (Freshness)
embedding_data_freshness = {
    'election results today': np.array([0.9, 0.2, 0.6]),
    'election results yesterday': np.array([0.8, 0.1, 0.4]),
    'election results seven days ago': np.array([0.7, 0.05, 0.3]),
    # Add more text data and corresponding embeddings here
}

embedding_model_freshness = EmbeddingModel(embedding_data_freshness, non_linear_function='sigmoid')

# Test for today's data
query_embedding_freshness_today = np.array([0.9, 0.2, 0.6])
freshness_score_today = embedding_model_freshness.generate_freshness_score(query_embedding_freshness_today, weight=1.5, bias=-0.5)
print("Freshness Score (Today's data):", freshness_score_today)
# Expected output: The freshness_score_today should be high due to the non-linear scoring for today's data.

# Test for data from yesterday
query_embedding_freshness_yesterday = np.array([0.8, 0.1, 0.4])
freshness_score_yesterday = embedding_model_freshness.generate_freshness_score(query_embedding_freshness_yesterday, weight=1.5, bias=-0.5)
print("Freshness Score (Yesterday's data):", freshness_score_yesterday)
# Expected output: The freshness_score_yesterday should be lower than today's score due to non-linear scoring.

# Test for data from seven days ago
query_embedding_freshness_seven_days_ago = np.array([0.7, 0.05, 0.3])
freshness_score_seven_days_ago = embedding_model_freshness.generate_freshness_score(query_embedding_freshness_seven_days_ago, weight=1.5, bias=-0.5)
print("Freshness Score (Seven days ago data):", freshness_score_seven_days_ago)
# Expected output: The freshness_score_seven_days_ago should be similar to yesterday's score, indicating similar value in the non-linear scoring.

# Test case 3: Invalid non-linear function
try:
    embedding_model_invalid_function = EmbeddingModel(embedding_data_freshness, non_linear_function='invalid_function')
except ValueError as e:
    print("Error:", str(e))
# Expected output: An error message indicating that the non-linear function provided is invalid.

