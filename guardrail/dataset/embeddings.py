import numpy as np
import umap
from sklearn.manifold import TSNE

class EmbeddingModel:
    def __init__(self, embedding_data, non_linear_function='sigmoid'):
        # 'embedding_data' is a dictionary mapping text data to its corresponding embeddings
        # For example: {'get python': np.array([0.8, 0.2, 0.5]), 'HTTP method': np.array([0.9, 0.3, 0.6]), ...}
        self.embedding_data = embedding_data

        # Define the non-linear scoring function based on the user's choice
        if non_linear_function == 'sigmoid':
            self.non_linear_function = self.sigmoid
        elif non_linear_function == 'relu':
            self.non_linear_function = self.relu
        elif non_linear_function == 'tanh':
            self.non_linear_function = self.tanh
        else:
            raise ValueError(f"Invalid non-linear function: {non_linear_function}. Supported functions are 'sigmoid', 'relu', and 'tanh'.")

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def relu(self, x):
        return np.maximum(0, x)

    def tanh(self, x):
        return np.tanh(x)

    def calculate_similarity(self, embedding1, embedding2):
        # Assuming 'embedding1' and 'embedding2' are NumPy arrays representing the embeddings of two results
        # Calculate the similarity between two embeddings (e.g., using cosine similarity)
        return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

    def generate_freshness_score(self, embedding, weight=1.5, bias=-0.5):
        # Apply the selected non-linear transformation on the embedding

        transformed_score = weight * embedding + bias

        # Apply the selected non-linear function to map the transformed score
        freshness_score = self.non_linear_function(transformed_score)

        return freshness_score
    
    def calculate_similarity(self, embedding1, embedding2):
        # Assuming 'embedding1' and 'embedding2' are NumPy arrays representing the embeddings of two results
        # Calculate the similarity between two embeddings (e.g., using cosine similarity)
        return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
    
    def improve_embeddings_with_diversity(self, query_embedding, diversity_weight=0.5, top_k=5):
        # Assuming 'query_embedding' is a NumPy array representing the embedding of the search query
        # 'diversity_weight' is a trade-off parameter between relevance and diversity (0.0 for pure relevance, 1.0 for pure diversity)
        # 'top_k' is the number of diverse results to select and return
        # Goal is to encourage the retrieval of not only the nearest neighbors but also diverse results that cover various aspects of the query's intent.
        # Maximal Marginal Relevance" (MMR), which balances relevance and diversity.
        # MMR selects results that are both relevant to the query and different from the ones already 
        # selected. MMR uses a trade-off parameter called the diversity weight to control the balance 
        # between relevance and diversity.

        # List to keep track of selected results
        selected_results = []

        # Select the most relevant result (nearest neighbor) as the first selected result
        result_texts = list(self.embedding_data.keys())
        similarities = [self.calculate_similarity(query_embedding, self.embedding_data[text]) for text in result_texts]
        most_relevant_idx = np.argmax(similarities)
        selected_results.append(self.embedding_data[result_texts[most_relevant_idx]])

        # Remove the most relevant result from the list of candidate results to avoid duplication
        result_texts.pop(most_relevant_idx)

        while len(selected_results) < top_k and len(result_texts) > 0:
            # Calculate the diversity score between each result and the selected results
            diversity_scores = [np.mean([self.calculate_similarity(emb_i, emb_j) for emb_i in selected_results]) for emb_j in [self.embedding_data[text] for text in result_texts]]

            # Calculate the diversified score (relevance - diversity * diversity_weight) for each result
            diversified_scores = [sim - diversity_weight * div_score for sim, div_score in zip(similarities, diversity_scores)]

            # Select the result with the highest diversified score
            next_idx = np.argmax(diversified_scores)

            # Add the selected result to the list of selected results
            selected_results.append(self.embedding_data[result_texts[next_idx]])

            # Remove the selected result from the list of candidate results to avoid duplication
            result_texts.pop(next_idx)

        return selected_results
    
    def reduce_embedding_dimension_pca(embeddings, target_dim):
        """
        Reduce the dimensionality of embeddings using Principal Component Analysis (PCA).

        Parameters:
            embeddings (numpy.ndarray): The embeddings matrix with shape (num_samples, original_dim).
            target_dim (int): The desired dimensionality of the reduced embeddings.

        Returns:
            numpy.ndarray: The reduced embeddings matrix with shape (num_samples, target_dim).
        """
        # Center the embeddings by subtracting the mean
        centered_embeddings = embeddings - np.mean(embeddings, axis=0)

        # Calculate the covariance matrix of the centered embeddings
        covariance_matrix = np.cov(centered_embeddings, rowvar=False)

        # Perform PCA to obtain the principal components and eigenvalues
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

        # Sort the eigenvalues and eigenvectors in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1]
        sorted_eigenvalues = eigenvalues[sorted_indices]
        sorted_eigenvectors = eigenvectors[:, sorted_indices]

        # Select the top 'target_dim' eigenvectors (principal components) to reduce the dimensionality
        reduced_embeddings = np.dot(centered_embeddings, sorted_eigenvectors[:, :target_dim])

        # Add back the mean to obtain the final reduced embeddings
        final_reduced_embeddings = reduced_embeddings + np.mean(embeddings, axis=0)

        return final_reduced_embeddings
    
    def reduce_embedding_dimension_tsne(embeddings, target_dim):
        """
        Reduce the dimensionality of embeddings using t-SNE.

        Parameters:
            embeddings (numpy.ndarray): The embeddings matrix with shape (num_samples, original_dim).
            target_dim (int): The desired dimensionality of the reduced embeddings.

        Returns:
            numpy.ndarray: The reduced embeddings matrix with shape (num_samples, target_dim).
        """
        # Create a t-SNE model with the desired target dimension
        tsne_model = TSNE(n_components=target_dim)

        # Perform t-SNE to obtain the reduced embeddings
        reduced_embeddings = tsne_model.fit_transform(embeddings)

        return reduced_embeddings
    
    def reduce_embedding_dimension_umap(embeddings, target_dim):
        """
        Reduce the dimensionality of embeddings using UMAP.

        Parameters:
            embeddings (numpy.ndarray): The embeddings matrix with shape (num_samples, original_dim).
            target_dim (int): The desired dimensionality of the reduced embeddings.

        Returns:
            numpy.ndarray: The reduced embeddings matrix with shape (num_samples, target_dim).
        """
        # Create a UMAP model with the desired target dimension
        umap_model = umap.UMAP(n_components=target_dim)

        # Perform UMAP to obtain the reduced embeddings
        reduced_embeddings = umap_model.fit_transform(embeddings)

        return reduced_embeddings