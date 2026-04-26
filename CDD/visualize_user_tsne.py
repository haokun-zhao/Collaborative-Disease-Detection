'''
t-SNE Visualization for User (Patient) Embeddings
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')

def load_disease_burden(diag_adj_csv_path):
    """
    Load disease burden (number of diseases per patient) from diag_adj_with_ID.csv
    
    Args:
        diag_adj_csv_path: path to diag_adj_with_ID.csv file
    
    Returns:
        disease_counts: numpy array of shape (n_patients,) containing number of diseases per patient
        subject_ids: numpy array of subject IDs (for reference)
    """
    print(f"Loading disease burden from {diag_adj_csv_path}...")
    df = pd.read_csv(diag_adj_csv_path)
    
    # Get subject IDs (first column)
    subject_ids = df.iloc[:, 0].values
    
    # Count number of diseases per patient (sum of 1s in each row, excluding first column)
    disease_counts = df.iloc[:, 1:].sum(axis=1).values
    
    print(f"Loaded disease burden for {len(disease_counts)} patients")
    print(f"Disease count statistics:")
    print(f"  Min: {disease_counts.min()}")
    print(f"  Max: {disease_counts.max()}")
    print(f"  Mean: {disease_counts.mean():.2f}")
    print(f"  Median: {np.median(disease_counts):.2f}")
    
    return disease_counts, subject_ids

def visualize_user_tsne(user_embeddings, perplexity=30, learning_rate=200, 
                        random_state=42, save_path='user_tsne_visualization.png',
                        sample_size=None, disease_burden=None):
    """
    Visualize user embeddings using t-SNE
    
    Args:
        user_embeddings: numpy array of shape (n_users, embedding_dim)
        perplexity: t-SNE perplexity parameter
        learning_rate: t-SNE learning rate
        random_state: random seed for reproducibility
        save_path: path to save the visualization
        sample_size: if specified, randomly sample this many users for visualization
    """
    print(f"Input embeddings shape: {user_embeddings.shape}")
    
    # Sample users if specified (for faster computation with large datasets)
    disease_burden_to_use = None
    if sample_size is not None and sample_size < user_embeddings.shape[0]:
        print(f"Sampling {sample_size} users from {user_embeddings.shape[0]} total users...")
        np.random.seed(random_state)
        sample_indices = np.random.choice(user_embeddings.shape[0], sample_size, replace=False)
        embeddings_to_visualize = user_embeddings[sample_indices]
        if disease_burden is not None:
            disease_burden_to_use = disease_burden[sample_indices]
        print(f"Sampled embeddings shape: {embeddings_to_visualize.shape}")
    else:
        embeddings_to_visualize = user_embeddings
        disease_burden_to_use = disease_burden
        sample_indices = None
    
    # Compute t-SNE
    print(f"\nComputing t-SNE with perplexity={perplexity}, learning_rate={learning_rate}...")
    tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate,
                random_state=random_state, verbose=1, n_iter=1000)
    tsne_results = tsne.fit_transform(embeddings_to_visualize)
    print(f"t-SNE results shape: {tsne_results.shape}")
    
    # Create visualization
    plt.figure(figsize=(12, 10))
    
    # Plot all users with disease burden coloring if available
    if disease_burden_to_use is not None:
        # Use log scale for better visual distinction
        log_disease_burden = np.log1p(disease_burden_to_use)
        scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], 
                             c=log_disease_burden, cmap='viridis', 
                             alpha=0.7, s=30, edgecolors='black', linewidth=0.2)
        cbar = plt.colorbar(scatter, label='Log(1 + Number of Diseases)')
        cbar.ax.tick_params(labelsize=10)
        # Format colorbar ticks to show log scale values
        tick_values = np.linspace(log_disease_burden.min(), log_disease_burden.max(), 6)
        cbar.set_ticks(tick_values)
        cbar.set_ticklabels([f'{val:.2f}' for val in tick_values])
        title_suffix = f'\nColored by Disease Burden (Log Scale)'
    else:
        scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], 
                             alpha=0.6, s=20, c='steelblue', edgecolors='black', linewidth=0.1)
        title_suffix = ''
    
    # plt.title(f'Patient Embeddings t-SNE Visualization{title_suffix}\n'
    #           f'(perplexity={perplexity}, learning_rate={learning_rate}, '
    #           f'n_users={embeddings_to_visualize.shape[0]})', 
    #           fontsize=14, fontweight='bold')
    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add text annotation
    if sample_size is not None:
        plt.text(0.02, 0.98, f'Sampled {sample_size} users', 
                transform=plt.gca().transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to {save_path}")
    plt.close()

def visualize_user_tsne_with_density(user_embeddings, perplexity=30, learning_rate=200, 
                                     random_state=42, save_path='user_tsne_visualization/user_tsne_visualization_density.png',
                                     sample_size=None, disease_burden=None):
    """
    Visualize user embeddings using t-SNE with density coloring
    
    Args:
        user_embeddings: numpy array of shape (n_users, embedding_dim)
        perplexity: t-SNE perplexity parameter
        learning_rate: t-SNE learning rate
        random_state: random seed for reproducibility
        save_path: path to save the visualization
        sample_size: if specified, randomly sample this many users for visualization
    """
    print(f"Input embeddings shape: {user_embeddings.shape}")
    
    # Sample users if specified
    disease_burden_to_use = None
    if sample_size is not None and sample_size < user_embeddings.shape[0]:
        print(f"Sampling {sample_size} users from {user_embeddings.shape[0]} total users...")
        np.random.seed(random_state)
        sample_indices = np.random.choice(user_embeddings.shape[0], sample_size, replace=False)
        embeddings_to_visualize = user_embeddings[sample_indices]
        if disease_burden is not None:
            disease_burden_to_use = disease_burden[sample_indices]
            print(f"Disease burden sampled: shape={disease_burden_to_use.shape}, range=[{disease_burden_to_use.min()}, {disease_burden_to_use.max()}]")
        else:
            print("Warning: disease_burden is None, cannot sample disease burden data.")
        print(f"Sampled embeddings shape: {embeddings_to_visualize.shape}")
    else:
        embeddings_to_visualize = user_embeddings
        disease_burden_to_use = disease_burden
        if disease_burden_to_use is not None:
            print(f"Disease burden to use: shape={disease_burden_to_use.shape}, range=[{disease_burden_to_use.min()}, {disease_burden_to_use.max()}]")
        else:
            print("Warning: disease_burden_to_use is None.")
        sample_indices = None
    
    # Compute t-SNE
    print(f"\nComputing t-SNE with perplexity={perplexity}, learning_rate={learning_rate}...")
    tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate,
                random_state=random_state, verbose=1, n_iter=1000)
    tsne_results = tsne.fit_transform(embeddings_to_visualize)
    print(f"t-SNE results shape: {tsne_results.shape}")
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Subplot 1: Scatter plot colored by disease burden
    if disease_burden_to_use is not None:
        # Use log scale for better visual distinction
        log_disease_burden = np.log1p(disease_burden_to_use)
        scatter1 = ax1.scatter(tsne_results[:, 0], tsne_results[:, 1], 
                              c=log_disease_burden, cmap='viridis',
                              alpha=0.7, s=30, edgecolors='black', linewidth=0.2)
        cbar1 = plt.colorbar(scatter1, ax=ax1, label='Log(1 + Number of Diseases)')
        cbar1.ax.tick_params(labelsize=10)
        # Format colorbar ticks to show log scale values
        tick_values = np.linspace(log_disease_burden.min(), log_disease_burden.max(), 6)
        cbar1.set_ticks(tick_values)
        cbar1.set_ticklabels([f'{val:.2f}' for val in tick_values])
        # ax1.set_title('Patient Embeddings t-SNE\n(Colored by Disease Burden, Log Scale)', fontsize=14, fontweight='bold')
    else:
        scatter1 = ax1.scatter(tsne_results[:, 0], tsne_results[:, 1], 
                              alpha=0.6, s=20, c='steelblue', edgecolors='black', linewidth=0.1)
        ax1.set_title('Patient Embeddings t-SNE', fontsize=14, fontweight='bold')
    ax1.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax1.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: Density plot using hexbin
    if disease_burden_to_use is not None:
        log_disease_burden = np.log1p(disease_burden_to_use)
        hb = ax2.hexbin(tsne_results[:, 0], tsne_results[:, 1], 
                        C=log_disease_burden, reduce_C_function=np.mean,
                        gridsize=60, cmap='viridis', mincnt=1)
        cb = plt.colorbar(hb, ax=ax2, label='Average Log(1 + Number of Diseases)')
        cb.ax.tick_params(labelsize=10)
        # Format colorbar ticks to show log scale values
        # Get the actual range from the hexbin result
        vmin, vmax = hb.get_clim()
        tick_values = np.linspace(vmin, vmax, 6)
        cb.set_ticks(tick_values)
        cb.set_ticklabels([f'{val:.2f}' for val in tick_values])
    else:
        hb = ax2.hexbin(tsne_results[:, 0], tsne_results[:, 1], 
                        gridsize=60, cmap='YlOrRd', mincnt=1)
        cb = plt.colorbar(hb, ax=ax2, label='Number of patients')
    # ax2.set_title('Patient Density Distribution', fontsize=14, fontweight='bold')
    ax2.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax2.set_ylabel('t-SNE Dimension 2', fontsize=12)
    
    # Add text annotation
    if sample_size is not None:
        fig.text(0.02, 0.98, f'Sampled {sample_size} users from {user_embeddings.shape[0]} total', 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # plt.suptitle(f'Patient Embeddings t-SNE Visualization\n'
    #              f'(perplexity={perplexity}, learning_rate={learning_rate})', 
    #              fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to {save_path}")
    plt.close()

def main():
    # Load user embeddings
    user_emb_final_path = 'user_emb_final.npy'
    
    if not os.path.exists(user_emb_final_path):
        print(f"Error: User embeddings file not found at {user_emb_final_path}")
        print("Please run extract_user_embeddings.py first to generate the embeddings file.")
        return
    
    print(f"Loading user embeddings from {user_emb_final_path}...")
    user_embeddings = np.load(user_emb_final_path)
    print(f"User embeddings shape: {user_embeddings.shape}")
    
    # Load disease burden from diag_adj_with_ID.csv
    diag_adj_csv_path = 'diag_adj_with_ID.csv'
    disease_burden = None
    if os.path.exists(diag_adj_csv_path):
        try:
            disease_counts, subject_ids = load_disease_burden(diag_adj_csv_path)
            print(f"\nDisease burden loading check:")
            print(f"  - Number of patients in CSV: {len(disease_counts)}")
            print(f"  - Number of user embeddings: {user_embeddings.shape[0]}")
            # Ensure disease_counts matches the number of user embeddings
            if len(disease_counts[:user_embeddings.shape[0]]) == user_embeddings.shape[0]:
                disease_burden = disease_counts[:user_embeddings.shape[0]]
                print(f"  ✓ Disease burden loaded successfully and matches user embeddings shape.")
                print(f"  - Disease burden range: [{disease_burden.min()}, {disease_burden.max()}]")
            else:
                print(f"  ✗ Warning: Disease burden count ({len(disease_counts)}) doesn't match user embeddings ({user_embeddings.shape[0]})")
                print("  Visualization will proceed without disease burden coloring.")
        except Exception as e:
            print(f"Error loading disease burden: {e}")
            import traceback
            traceback.print_exc()
            print("Visualization will proceed without disease burden coloring.")
    else:
        print(f"Warning: {diag_adj_csv_path} not found. Visualization will proceed without disease burden coloring.")
    
    # Determine if we should sample (for very large datasets)
    n_users = user_embeddings.shape[0]
    sample_size = None
    if n_users > 10000:
        print(f"\nLarge dataset detected ({n_users} users).")
        print("Consider using sample_size parameter for faster t-SNE computation.")
        # Uncomment the line below to enable sampling
        # sample_size = 10000
    
    # Create output directory if it doesn't exist
    output_dir = 'user_tsne_visualization'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Generate visualizations with different parameters
    perplexity_values = [50]
    learning_rate_values = [200]
    
    for perplexity in perplexity_values:
        for learning_rate in learning_rate_values:
            save_path = f'{output_dir}/user_tsne_p{perplexity}_lr{learning_rate}_sampled.png'
            print(f"\n{'='*60}")
            print(f"Generating visualization: {save_path}")
            print(f"{'='*60}")
            visualize_user_tsne(user_embeddings, perplexity=perplexity, 
                               learning_rate=learning_rate, random_state=42,
                               save_path=save_path, sample_size=sample_size,
                               disease_burden=disease_burden)
    
    # Generate density visualization
    print(f"\n{'='*60}")
    print("Generating density visualization...")
    print(f"{'='*60}")
    visualize_user_tsne_with_density(user_embeddings, perplexity=50, 
                                     learning_rate=200, random_state=42,
                                     save_path=f'{output_dir}/user_tsne_density_sampled.png', 
                                     sample_size=sample_size,
                                     disease_burden=disease_burden)
    
    print("\nAll visualizations completed!")

if __name__ == '__main__':
    main()

