# Cell: Load and Prepare Data
import pandas as pd
import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import random
import sys

# Set seed for all random operations
random.seed(42)
np.random.seed(42)

class TeeOutput:
    """
    Class to duplicate output to both console and file
    """
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.logfile = open(filename, "w", encoding="utf-8")
        self.timestamp_start = datetime.now()
    
    def write(self, message):
        self.terminal.write(message)
        self.logfile.write(message)
    
    def flush(self):
        self.terminal.flush()
        self.logfile.flush()
    
    def close(self):
        elapsed = datetime.now() - self.timestamp_start
        self.write(f"\n\nExecution completed in {elapsed}\n")
        self.logfile.close()

def log_section(title):
    """Helper function to log a section header"""
    timestamp = datetime.now().strftime('%H:%M:%S')
    separator = "=" * 80
    section_header = f"\n{separator}\n{timestamp} - {title}\n{separator}\n"
    print(section_header)

def load_and_prepare_data(copurchase_path='copurchase.csv', products_path='products.csv', verbose=True):
    """
    Load and prepare data with comprehensive data cleaning and preprocessing
    
    Parameters:
    -----------
    copurchase_path : str
        Path to the copurchase CSV file
    products_path : str
        Path to the products CSV file
    verbose : bool
        Whether to print detailed information during processing
    
    Returns:
    --------
    tuple
        Clean copurchase_df and products_df
    """
    # 1. Validate file paths
    for path in [copurchase_path, products_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
    
    # 2. Load data with appropriate data types
    if verbose:
        print(f"Loading data from {copurchase_path} and {products_path}...")
    
    # Load copurchase data with appropriate data types
    try:
        copurchase_df = pd.read_csv(copurchase_path)
        products_df = pd.read_csv(products_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        raise
        
    # 3. Data exploration
    if verbose:
        print("\n--- Data Exploration ---")
        print("\nCopurchase Data:")
        print(f"Shape: {copurchase_df.shape}")
        print("Data types:")
        print(copurchase_df.dtypes)
        print("Missing values:")
        print(copurchase_df.isnull().sum())
        
        print("\nProducts Data:")
        print(f"Shape: {products_df.shape}")
        print("Data types:")
        print(products_df.dtypes)
        print("Missing values:")
        print(products_df.isnull().sum())
    
    # 4. Data cleaning for copurchase_df
    if verbose:
        print("\n--- Cleaning Copurchase Data ---")
    
    # Handle missing values
    missing_count = copurchase_df.isnull().sum().sum()
    if missing_count > 0:
        if verbose:
            print(f"Found {missing_count} missing values in copurchase data")
        # For source/target columns, we can't have nulls - remove those rows
        copurchase_df = copurchase_df.dropna(subset=['Source', 'Target'])
        # For weight column, fill with 1 if present and missing
        if 'weight' in copurchase_df.columns:
            copurchase_df['weight'].fillna(1, inplace=True)
    
    # Remove duplicates
    initial_rows = len(copurchase_df)
    copurchase_df = copurchase_df.drop_duplicates()
    if verbose and initial_rows > len(copurchase_df):
        print(f"Removed {initial_rows - len(copurchase_df)} duplicate rows from copurchase data")
    
    # Ensure Source and Target are of the same data type
    copurchase_df['Source'] = copurchase_df['Source'].astype(str)
    copurchase_df['Target'] = copurchase_df['Target'].astype(str)
    
    # 5. Data cleaning for products_df
    if verbose:
        print("\n--- Cleaning Products Data ---")
    
    # Handle missing values in products data
    missing_count = products_df.isnull().sum().sum()
    if missing_count > 0:
        if verbose:
            print(f"Found {missing_count} missing values in products data")
        
        # For ID column, can't have nulls
        if 'id' in products_df.columns:
            products_df = products_df.dropna(subset=['id'])
        elif 'product_id' in products_df.columns:
            products_df = products_df.dropna(subset=['product_id'])
        
        # For other columns, fill missing values appropriately
        if 'price' in products_df.columns:
            products_df['price'].fillna(products_df['price'].median(), inplace=True)
        
        # For text columns, fill with appropriate placeholders
        text_cols = products_df.select_dtypes(include=['object']).columns
        for col in text_cols:
            if col not in ['id', 'product_id'] and products_df[col].isnull().any():
                products_df[col].fillna("Unknown", inplace=True)
    
    # Handle duplicate products
    if 'id' in products_df.columns:
        key_col = 'id'
    elif 'product_id' in products_df.columns:
        key_col = 'product_id'
    else:
        key_col = products_df.columns[0]
        
    initial_rows = len(products_df)
    products_df = products_df.drop_duplicates(subset=[key_col])
    if verbose and initial_rows > len(products_df):
        print(f"Removed {initial_rows - len(products_df)} duplicate product entries")
    
    # 6. Data validation
    if verbose:
        print("\n--- Data Validation ---")
    
    # Check that all Source/Target values in copurchase_df exist in products_df
    if 'id' in products_df.columns:
        product_ids = set(products_df['id'].astype(str))
    elif 'product_id' in products_df.columns:
        product_ids = set(products_df['product_id'].astype(str))
    else:
        product_ids = set()  # Skip this check if no clear ID column
        
    if product_ids:
        # Find sources/targets not in product list
        invalid_sources = set(copurchase_df['Source'].unique()) - product_ids
        invalid_targets = set(copurchase_df['Target'].unique()) - product_ids
        
        if invalid_sources or invalid_targets:
            if verbose:
                print(f"Found {len(invalid_sources)} invalid source products and {len(invalid_targets)} invalid target products")
                
            # Remove invalid entries
            invalid_all = invalid_sources.union(invalid_targets)
            initial_rows = len(copurchase_df)
            copurchase_df = copurchase_df[~copurchase_df['Source'].isin(invalid_all)]
            copurchase_df = copurchase_df[~copurchase_df['Target'].isin(invalid_all)]
            
            if verbose:
                print(f"Removed {initial_rows - len(copurchase_df)} rows with invalid product references")
    
    # 7. Handle self-loops (optional - remove if a product references itself)
    initial_rows = len(copurchase_df)
    copurchase_df = copurchase_df[copurchase_df['Source'] != copurchase_df['Target']]
    if verbose and initial_rows > len(copurchase_df):
        print(f"Removed {initial_rows - len(copurchase_df)} self-loops")
    
    # 8. Outlier detection in weights (if present)
    if 'weight' in copurchase_df.columns:
        # Use IQR method to identify outliers
        Q1 = copurchase_df['weight'].quantile(0.25)
        Q3 = copurchase_df['weight'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = copurchase_df[(copurchase_df['weight'] < lower_bound) | 
                               (copurchase_df['weight'] > upper_bound)]
        
        if len(outliers) > 0 and verbose:
            print(f"Detected {len(outliers)} outliers in weight values")
            print(f"Weight range: [{copurchase_df['weight'].min()}, {copurchase_df['weight'].max()}]")
            print(f"Outlier threshold: [{lower_bound}, {upper_bound}]")
            
            # Option 1: Cap outliers at thresholds
            copurchase_df['weight'] = copurchase_df['weight'].clip(lower=max(0, lower_bound), upper=upper_bound)
            print("Capped outliers at threshold boundaries")
            
            # Alternative: Visualize weight distribution
            plt.figure(figsize=(10, 6))
            sns.histplot(data=copurchase_df, x='weight', bins=50)
            plt.title('Weight Distribution After Capping Outliers')
            plt.savefig('weight_distribution.png')
            if verbose:
                print("Saved weight distribution plot to 'weight_distribution.png'")
    
    # 9. Feature engineering (if needed)
    # Add frequency-based features
    source_counts = copurchase_df['Source'].value_counts().to_dict()
    target_counts = copurchase_df['Target'].value_counts().to_dict()
    
    # Add count information to products_df
    if 'id' in products_df.columns:
        key_col = 'id'
    elif 'product_id' in products_df.columns:
        key_col = 'product_id'
        
    products_df['outgoing_count'] = products_df[key_col].astype(str).map(source_counts).fillna(0)
    products_df['incoming_count'] = products_df[key_col].astype(str).map(target_counts).fillna(0)
    products_df['total_connections'] = products_df['outgoing_count'] + products_df['incoming_count']
    
    if verbose:
        print("\n--- Data Preparation Complete ---")
        print(f"Copurchase data: {len(copurchase_df)} rows")
        print(f"Products data: {len(products_df)} rows")
        print("\n-- Copurchase Data Sample --")
        print(copurchase_df.head())
        print("\n-- Products Data Sample --")
        print(products_df.head())
        
        # Save processing summary
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f"data_prep_summary_{timestamp}.txt", "w") as f:
            f.write(f"Data preparation completed at {datetime.now()}\n")
            f.write(f"Copurchase data: {len(copurchase_df)} rows\n")
            f.write(f"Products data: {len(products_df)} rows\n")
            f.write(f"Features in products data: {', '.join(products_df.columns)}\n")
            f.write(f"Missing values remaining: {products_df.isnull().sum().sum()}\n")
        
        print(f"\nSaved processing summary to data_prep_summary_{timestamp}.txt")
    
    return copurchase_df, products_df

# Original load_data function for backward compatibility
def load_data():
    return load_and_prepare_data(verbose=False)

# Cell: Create Graph
# Fungsi create_graph membangun graph dari data copurchase.
import networkx as nx

def create_graph(copurchase_df):
    """Optimized graph creation using NetworkX's from_pandas_edgelist"""
    if 'weight' in copurchase_df.columns:
        G = nx.from_pandas_edgelist(copurchase_df, 'Source', 'Target', edge_attr='weight')
    else:
        G = nx.from_pandas_edgelist(copurchase_df, 'Source', 'Target')
        nx.set_edge_attributes(G, 1, 'weight')
    
    print(f"Graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    return G

# Cell: Analyze Graph Structure
import numpy as np

def analyze_graph(G):
    """Optimized graph analysis with sampling for large graphs"""
    # Set seed for consistent sampling
    random.seed(42)
    
    node_count = G.number_of_nodes()
    results = {
        'node_count': node_count,
        'edge_count': G.number_of_edges(),
        'density': nx.density(G)
    }
    
    # For large graphs, sample for clustering coefficient
    if node_count > 10000:
        sample_size = min(1000, node_count)
        sampled_nodes = random.sample(list(G.nodes()), sample_size)
        results['avg_clustering'] = nx.average_clustering(G, nodes=sampled_nodes)
        print(f"Clustering coefficient calculated on {sample_size} sampled nodes")
    else:
        results['avg_clustering'] = nx.average_clustering(G)
    
    # Faster component counting for large graphs
    if node_count > 50000:
        # For very large graphs, just check if connected
        results['connected_components'] = "Large graph - checking connectivity"
        results['connected_components'] = 1 if nx.is_connected(G) else ">1"
    else:
        results['connected_components'] = nx.number_connected_components(G)
    
    # Calculate degree statistics (fast operation)
    degrees = [d for _, d in G.degree()]
    results['avg_degree'] = sum(degrees) / len(degrees) if degrees else 0
    results['max_degree'] = max(degrees) if degrees else 0
    
    # For large graphs, use degree as proxy for centrality
    if node_count > 10000:
        sorted_by_degree = sorted(G.degree(), key=lambda x: x[1], reverse=True)
        results['top_5_central_nodes'] = [(node, degree/node_count) for node, degree in sorted_by_degree[:5]]
    else:
        degree_centrality = nx.degree_centrality(G)
        sorted_centrality = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)
        results['top_5_central_nodes'] = sorted_centrality[:5]
    
    return results

# Cell: Community Detection
from collections import Counter

def detect_communities(G):
    """Enhanced community detection using Louvain method with fallback options"""
    from networkx.algorithms import community
    
    # Set seed for community detection algorithms (if they support it)
    random.seed(42)
    
    print("Detecting communities...")
    node_count = G.number_of_nodes()
    is_connected = nx.is_connected(G)
    
    try:
        print("Using Louvain community detection algorithm...")
        communities = list(community.louvain_communities(G))
        print(f"Successfully detected {len(communities)} communities using Louvain method")
    except Exception as e:
        print(f"Louvain algorithm failed: {str(e)}")
        if not is_connected:
            print("Graph is disconnected. Using label propagation algorithm as fallback.")
            communities = list(community.label_propagation_communities(G))
        elif node_count > 5000:
            try:
                print("Large graph. Using fluid communities algorithm as fallback.")
                communities = list(community.asyn_fluidc(G, 10))
            except nx.NetworkXError:
                print("Fluid communities algorithm failed. Using label propagation.")
                communities = list(community.label_propagation_communities(G))
        else:
            print("Using label propagation algorithm as fallback.")
            communities = list(community.label_propagation_communities(G))
    
    partition = {}
    for i, comm in enumerate(communities):
        for node in comm:
            partition[node] = i
    
    community_counts = Counter(partition.values())
    nx.set_node_attributes(G, partition, 'community')
    
    print(f"Found {len(community_counts)} communities")
    return partition, community_counts

# Cell: Visualize Graph
import matplotlib.pyplot as plt

def visualize_graph(G, partition=None, max_nodes=100, colormap='viridis', edge_alpha=0.3):
    """
    Enhanced visualization with better colors and sampling of influential nodes
    """
    # Sample nodes if graph is too large
    if G.number_of_nodes() > max_nodes:
        # Sample the most influential nodes rather than just first N
        degrees = dict(G.degree())
        sorted_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
        sampled_nodes = [node for node, _ in sorted_nodes[:max_nodes]]
        G_sample = G.subgraph(sampled_nodes)
        print(f"Visualizing top {max_nodes} nodes by degree centrality")
    else:
        G_sample = G
    
    # Create a slightly wider figure to accommodate the legend on the right
    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(G_sample, seed=42, k=0.3) # Adjusted k for better spacing
    
    if partition:
        # Get unique community IDs
        communities = list(set(partition.values()))
        cmap = plt.cm.get_cmap(colormap, max(communities)+1)
        
        # Draw nodes with community colors
        for comm_id in communities:
            comm_nodes = [node for node in G_sample.nodes() if node in partition and partition[node] == comm_id]
            if comm_nodes:
                nx.draw_networkx_nodes(
                    G_sample, pos,
                    nodelist=comm_nodes,
                    node_color=[cmap(comm_id)],
                    node_size=80,
                    label=f"Community {comm_id}"
                )
    else:
        # Color nodes by degree centrality for more informative visualization
        node_color = [len(G_sample[node]) for node in G_sample.nodes()]
        nx.draw_networkx_nodes(
            G_sample, pos,
            node_color=node_color,
            cmap=plt.cm.get_cmap(colormap),
            node_size=80
        )
    
    # Draw edges with transparency
    nx.draw_networkx_edges(G_sample, pos, edge_color='gray', alpha=edge_alpha, width=0.5)
    
    plt.title("Product Co-purchasing Network")
    plt.axis('off')
    
    # Add a colorbar legend if using partition - place it outside the plot on the far right
    if partition:
        # Position legend outside the plot on the far right
        plt.legend(
            scatterpoints=1,
            frameon=True,
            labelspacing=1,
            title="Communities",
            loc='center left',
            bbox_to_anchor=(1.0, 0.5)  # Position at the right side
        )
        # Adjust the layout to make room for the legend
        plt.tight_layout()
        plt.subplots_adjust(right=0.85)  # Make room for legend on right
    
    plt.savefig('copurchase_graph.png', dpi=300, bbox_inches='tight')
    plt.show()

# Cell: Most Frequent Co-Purchasing Patterns
def most_frequent_co_purchasing(copurchase_df, top_n=10):
    if 'weight' in copurchase_df.columns:
        df_weight = copurchase_df.groupby(['Source', 'Target'])['weight'].sum().reset_index()
    else:
        df_weight = copurchase_df.groupby(['Source', 'Target']).size().reset_index(name='weight')
    top_pairs = df_weight.sort_values('weight', ascending=False).head(top_n)
    return top_pairs

# Cell: Parallel Processing & Influential Products
import multiprocessing

def process_chunk_for_centrality(args):
    G, node_chunk, node_count = args
    local_results = {}
    for node in node_chunk:
        local_results[node] = len(G[node]) / (node_count - 1)
    return local_results

def find_influential_products_parallel_optimized(G, products_df, top_n=10):
    """Find influential products with parallel processing for large graphs"""
    if G.number_of_nodes() > 10000:
        num_cpus = multiprocessing.cpu_count()
        print(f"Using {num_cpus} CPU cores for processing")
        degree_cent = {}
        all_nodes = list(G.nodes())
        node_count = G.number_of_nodes()
        chunk_size = max(1, len(all_nodes) // num_cpus)
        node_chunks = [all_nodes[i:i+chunk_size] for i in range(0, len(all_nodes), chunk_size)]
        chunk_args = [(G, chunk, node_count) for chunk in node_chunks]
        with multiprocessing.Pool(processes=num_cpus) as pool:
            results = pool.map(process_chunk_for_centrality, chunk_args)
        for result_dict in results:
            degree_cent.update(result_dict)
    else:
        degree_cent = nx.degree_centrality(G)
    
    top_nodes = sorted(degree_cent.items(), key=lambda x: x[1], reverse=True)[:top_n]
    centrality_df = pd.DataFrame([
        {'product_id': node, 'degree_centrality': score} for node, score in top_nodes
    ])
    
    # Handle the data type consistency for merging
    if 'product_id' in products_df.columns:
        # Ensure both columns are of the same type (string)
        centrality_df['product_id'] = centrality_df['product_id'].astype(str)
        products_df_copy = products_df.copy()
        products_df_copy['product_id'] = products_df_copy['product_id'].astype(str)
        influential_products = centrality_df.merge(products_df_copy, on='product_id')
    elif 'id' in products_df.columns:
        # Ensure both columns are of the same type (string)
        centrality_df['product_id'] = centrality_df['product_id'].astype(str)
        products_df_copy = products_df.copy()
        products_df_copy['id'] = products_df_copy['id'].astype(str)
        influential_products = centrality_df.merge(products_df_copy, left_on='product_id', right_on='id')
    else:
        influential_products = centrality_df
    
    return influential_products

# Updated version that uses only degree centrality for better performance
def find_influential_products_with_multiple_metrics(G, products_df, top_n=10, method='degree'):
    """
    Find influential products using centrality measures
    
    Parameters:
    -----------
    G : NetworkX graph
        The product co-purchase network
    products_df : pandas DataFrame
        DataFrame containing product information
    top_n : int
        Number of influential products to return
    method : str
        Always uses 'degree' for better performance
        
    Returns:
    --------
    DataFrame
        DataFrame with influential products and their centrality scores
    """
    print("Finding influential products using degree centrality for better performance")
    
    # Calculate degree centrality (always fast)
    degree_cent = nx.degree_centrality(G)
    centrality = degree_cent
    
    # Get top nodes
    top_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    # Create DataFrame with centrality metrics
    result_data = []
    for node, score in top_nodes:
        node_data = {
            'product_id': node,
            'centrality_score': score,
            'degree_centrality': degree_cent.get(node, 0)
        }
        result_data.append(node_data)
        
    centrality_df = pd.DataFrame(result_data)
    
    # Handle data type consistency for merging
    if 'product_id' in products_df.columns:
        centrality_df['product_id'] = centrality_df['product_id'].astype(str)
        products_df_copy = products_df.copy()
        products_df_copy['product_id'] = products_df_copy['product_id'].astype(str)
        influential_products = centrality_df.merge(products_df_copy, on='product_id')
    elif 'id' in products_df.columns:
        centrality_df['product_id'] = centrality_df['product_id'].astype(str)
        products_df_copy = products_df.copy()
        products_df_copy['id'] = products_df_copy['id'].astype(str)
        influential_products = centrality_df.merge(products_df_copy, left_on='product_id', right_on='id')
    else:
        influential_products = centrality_df
    
    return influential_products

# Cell: Recommendation Functions
def recommend_products_enhanced(G, product_id, top_n=5, partition=None, community_weight=0.3):
    """
    Enhanced product recommendation with community awareness
    
    Parameters:
    -----------
    G : NetworkX graph
        The product co-purchase network
    product_id : str
        The ID of the product for which to generate recommendations
    top_n : int
        Number of recommendations to return
    partition : dict
        Community partition mapping nodes to community IDs
    community_weight : float
        Weight to apply to products in the same community (0.0-1.0)
        
    Returns:
    --------
    list
        List of (product_id, score) tuples for recommended products
    """
    if product_id not in G.nodes():
        return "Product not found in the network"
    
    # Initialize scores dictionary
    scores = {}
    
    # Get first-degree neighbors with edge weights
    for neighbor in G.neighbors(product_id):
        # Use edge weight as base score
        scores[neighbor] = G[product_id][neighbor].get('weight', 1.0)
    
    # Get second-degree neighbors with path-based scores
    second_degree = set()
    for neighbor in G.neighbors(product_id):
        for second_neighbor in G.neighbors(neighbor):
            if second_neighbor != product_id and second_neighbor not in scores:
                second_degree.add((second_neighbor, neighbor))
    
    # Score second-degree neighbors based on connecting paths
    for second_neighbor, intermediate in second_degree:
        # Score is product of edge weights between target→intermediate→second_neighbor
        # If second_neighbor appears multiple times through different paths, take max
        path_score = G[product_id][intermediate].get('weight', 0.5) * G[intermediate][second_neighbor].get('weight', 0.5)
        current_score = scores.get(second_neighbor, 0)
        scores[second_neighbor] = max(current_score, path_score * 0.5)  # Apply dampening
    
    # Boost scores for products in the same community (if partition provided)
    if partition and product_id in partition:
        product_community = partition[product_id]
        for node, score in list(scores.items()):
            if node in partition and partition[node] == product_community:
                scores[node] = score * (1 + community_weight)
    
    # Sort by score and return top_n results
    sorted_recommendations = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return sorted_recommendations

def recommend_products_advanced(G, product_id, products_df, partition=None, top_n=5, 
                               community_weight=0.5, consider_popularity=True):
    """
    Advanced product recommendation with community awareness and popularity consideration
    
    Parameters:
    -----------
    G : NetworkX graph
        The product co-purchase network
    product_id : str
        The ID of the product for which to generate recommendations
    products_df : pandas DataFrame
        DataFrame containing product information
    partition : dict
        Community partition mapping nodes to community IDs
    top_n : int
        Number of recommendations to return
    community_weight : float
        Weight to apply to products in the same community (0.0-1.0)
    consider_popularity : bool
        Whether to consider product popularity in recommendations
        
    Returns:
    --------
    list
        List of recommended product IDs with scores
    """
    if product_id not in G.nodes():
        return "Product not found in the network"
    
    # Calculate recommendations using a weighted scoring system
    recommendations = {}
    
    # Direct neighbors get high scores based on edge weight
    for neighbor in G.neighbors(product_id):
        recommendations[neighbor] = G[product_id][neighbor].get('weight', 1.0) * 2
    
    # Get second-degree neighbors (with optimized approach for large graphs)
    second_neighbors = set()
    for neighbor in G.neighbors(product_id):
        second_neighbors.update(G.neighbors(neighbor))
    
    # Remove the original product and direct neighbors
    second_neighbors.discard(product_id)
    second_neighbors.difference_update(recommendations.keys())
    
    # Score second-degree neighbors
    for second_neighbor in second_neighbors:
        common_neighbors = set(G.neighbors(product_id)) & set(G.neighbors(second_neighbor))
        connection_strength = sum(
            G[product_id][n].get('weight', 0.5) * G[n][second_neighbor].get('weight', 0.5) 
            for n in common_neighbors
        )
        recommendations[second_neighbor] = connection_strength * 0.5
    
    # Enhanced community-based boosting
    if partition and product_id in partition:
        product_community = partition[product_id]
        community_members = [node for node, comm in partition.items() 
                            if comm == product_community and node != product_id]
        
        # Add community members not already in recommendations
        for node in community_members:
            if node not in recommendations:
                # Base score with distance consideration (if possible to calculate)
                try:
                    path_length = nx.shortest_path_length(G, source=product_id, target=node)
                    base_score = 0.5 * (1.0 / path_length)
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    base_score = 0.2  # Default score if no path exists
                
                recommendations[node] = base_score * community_weight
            else:
                # Boost existing recommendations in the same community
                recommendations[node] *= (1 + community_weight)
    
    # Consider product popularity if requested
    if consider_popularity and 'total_connections' in products_df.columns:
        id_col = 'id' if 'id' in products_df.columns else 'product_id'
        popularity_map = dict(zip(products_df[id_col].astype(str), 
                                products_df['total_connections']))
        
        # Adjust scores based on popularity (with normalization)
        max_popularity = products_df['total_connections'].max()
        for node in list(recommendations.keys()):
            popularity = popularity_map.get(str(node), 0)
            popularity_factor = 0.3 * (popularity / max_popularity)  # Reduced impact
            recommendations[node] *= (1 + popularity_factor)
    
    # Get ratings if available
    if 'rating' in products_df.columns:
        id_col = 'id' if 'id' in products_df.columns else 'product_id'
        rating_map = dict(zip(products_df[id_col].astype(str), products_df['rating']))
        
        # Boost products with high ratings
        for node in list(recommendations.keys()):
            rating = rating_map.get(str(node), 0)
            if rating > 0:  # Only boost if rating exists
                rating_factor = 0.2 * (rating / 5.0)  # Assuming 5-star rating system
                recommendations[node] *= (1 + rating_factor)
    
    # Sort all recommendations by score and return top N with scores
    sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return sorted_recommendations

# Cell: Evaluate Recommendations
def evaluate_recommendations(G, test_products, recommend_func, k=5, **recommend_params):
    """
    Evaluate recommendation quality using precision, recall, and other metrics
    
    Parameters:
    -----------
    G : NetworkX graph
        The product co-purchase network
    test_products : list
        List of product IDs to evaluate recommendations for
    recommend_func : function
        Function to generate recommendations
    k : int
        Number of recommendations to evaluate
    **recommend_params : dict
        Parameters to pass to the recommendation function
        
    Returns:
    --------
    dict
        Dictionary with evaluation metrics
    """
    results = {
        'precision': [],
        'recall': [],
        'coverage': set(),
        'total_products': len(G.nodes())
    }
    
    # Track whether each product was evaluated successfully
    successful_evaluations = 0
    
    for product_id in test_products:
        if product_id not in G.nodes():
            continue
            
        # Get actual connections (ground truth)
        actual_connections = set(G.neighbors(product_id))
        if not actual_connections:
            continue
            
        # Get recommendations
        recommendations = recommend_func(G, product_id, **recommend_params)
        if isinstance(recommendations, str):  # Skip if error returned
            continue
            
        # Extract just the product IDs
        recommended_products = [rec[0] for rec in recommendations[:k]]
        
        # Calculate metrics
        relevant_recommendations = set(recommended_products) & actual_connections
        
        # Precision: What fraction of recommended items are relevant
        precision = len(relevant_recommendations) / len(recommended_products) if recommended_products else 0
        
        # Recall: What fraction of relevant items are recommended
        recall = len(relevant_recommendations) / len(actual_connections) if actual_connections else 0
        
        # Update results
        results['precision'].append(precision)
        results['recall'].append(recall)
        results['coverage'].update(recommended_products)
        
        successful_evaluations += 1
    
    # Calculate averages and other metrics
    if successful_evaluations > 0:
        results['avg_precision'] = sum(results['precision']) / successful_evaluations
        results['avg_recall'] = sum(results['recall']) / successful_evaluations
        
        # F1 score (harmonic mean of precision and recall)
        if results['avg_precision'] + results['avg_recall'] > 0:
            results['f1_score'] = 2 * (results['avg_precision'] * results['avg_recall']) / \
                              (results['avg_precision'] + results['avg_recall'])
        else:
            results['f1_score'] = 0
            
        # Coverage ratio - what percentage of the catalog is recommended
        results['coverage_ratio'] = len(results['coverage']) / results['total_products']
    else:
        results['avg_precision'] = 0
        results['avg_recall'] = 0
        results['f1_score'] = 0
        results['coverage_ratio'] = 0
    
    return results

# Cell: Main Function - Add text file output for recommendations
def main():
    # Create output file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"analysis_output_{timestamp}.txt"
    
    # Redirect output to both console and file
    original_stdout = sys.stdout
    sys.stdout = TeeOutput(output_filename)
    
    try:
        # Header with start time
        print(f"Co-Purchase Network Analysis - Started at {datetime.now()}")
        print("=" * 80)
        
        log_section("DATA LOADING AND PREPARATION")
        print("Loading and preparing data...")
        copurchase_df, products_df = load_and_prepare_data(verbose=True)
        
        log_section("GRAPH CREATION")
        print("\nCreating co-purchasing graph...")
        G = create_graph(copurchase_df)
        
        log_section("GRAPH ANALYSIS")
        print("\nAnalyzing graph structure...")
        metrics = analyze_graph(G)
        print("\nGraph Metrics:")
        for metric, value in metrics.items():
            if metric != 'top_5_central_nodes':
                print(f"- {metric}: {value}")
        print("- Top 5 central nodes:")
        for node, score in metrics['top_5_central_nodes']:
            print(f"  * Node {node}: {score:.4f}")
        
        log_section("COMMUNITY DETECTION")
        print("\nDetecting communities...")
        partition, community_counts = detect_communities(G)
        print(f"Found {len(community_counts)} communities")
        print("Top 5 communities by size:")
        for community_id, count in community_counts.most_common(5):
            print(f"- Community {community_id}: {count} nodes")
        
        log_section("CO-PURCHASING PATTERNS")
        print("\nMost Frequent Co-Purchasing Patterns:")
        top_pairs = most_frequent_co_purchasing(copurchase_df)
        print(top_pairs)
        
        log_section("INFLUENTIAL PRODUCTS")
        print("\nIdentifying influential products...")
        influential = find_influential_products_with_multiple_metrics(G, products_df)
        print("Top influential products:")
        if 'title' in influential.columns:
            print(influential[['product_id', 'centrality_score', 'degree_centrality', 'title']].head())
        else:
            print(influential[['product_id', 'centrality_score', 'degree_centrality']].head())
        
        log_section("VISUALIZATION")
        print("\nVisualizing graph with community colors...")
        print("Saving visualization to 'copurchase_graph.png'...")
        visualize_graph(G, partition, colormap='viridis')
        
        # Evaluate recommendation system on sample products
        log_section("RECOMMENDATION EVALUATION")
        print("\nEvaluating recommendation system...")
        # Set seed for consistent sampling
        random.seed(42)
        sample_size = min(50, len(G.nodes()))
        test_products = random.sample(list(G.nodes()), sample_size)
        
        try:
            # Evaluate basic recommendations
            basic_results = evaluate_recommendations(
                G, test_products, recommend_products_enhanced,
                top_n=5, partition=None
            )
            
            # Evaluate community-aware recommendations
            community_results = evaluate_recommendations(
                G, test_products, recommend_products_enhanced,
                top_n=5, partition=partition, community_weight=0.5
            )
            
            # Evaluate advanced recommendations
            advanced_results = evaluate_recommendations(
                G, test_products, recommend_products_advanced,
                top_n=5, products_df=products_df, partition=partition,
                community_weight=0.5, consider_popularity=True
            )
            
            print("\nRecommendation Evaluation Results:")
            print("Basic Recommendations:")
            print(f"- Precision: {basic_results['avg_precision']:.4f}")
            print(f"- Recall: {basic_results['avg_recall']:.4f}")
            print(f"- F1 Score: {basic_results['f1_score']:.4f}")
            print(f"- Coverage: {basic_results['coverage_ratio']:.4f}")
            
            print("\nCommunity-Aware Recommendations:")
            print(f"- Precision: {community_results['avg_precision']:.4f}")
            print(f"- Recall: {community_results['avg_recall']:.4f}")
            print(f"- F1 Score: {community_results['f1_score']:.4f}")
            print(f"- Coverage: {community_results['coverage_ratio']:.4f}")
            
            print("\nAdvanced Recommendations:")
            print(f"- Precision: {advanced_results['avg_precision']:.4f}")
            print(f"- Recall: {advanced_results['avg_recall']:.4f}")
            print(f"- F1 Score: {advanced_results['f1_score']:.4f}")
            print(f"- Coverage: {advanced_results['coverage_ratio']:.4f}")
        except Exception as e:
            print(f"Error evaluating recommendations: {e}")
        
        log_section("PRODUCT RECOMMENDATIONS")
        print("\nGenerating product recommendations...")
        # Create recommendations text file with timestamp
        recommendations_file = f"product_recommendations_{timestamp}.txt"
        
        with open(recommendations_file, "w") as rec_file:
            rec_file.write(f"Product Recommendations Generated at {datetime.now()}\n")
            rec_file.write("=" * 80 + "\n\n")
            
            for idx, row in influential.head(10).iterrows():
                product_id = row['product_id']
                product_info = f"Recommendations for influential product {product_id}"
                if 'title' in row:
                    product_info += f": {row['title']}"
                
                print(product_info)
                rec_file.write(product_info + "\n")
                rec_file.write("-" * 80 + "\n")
                
                # Generate recommendations using the advanced method
                try:
                    recommendation_pairs = recommend_products_advanced(
                        G, product_id, products_df, 
                        partition=partition,
                        community_weight=0.5,
                        consider_popularity=True
                    )
                except Exception as e:
                    print(f"Error with advanced recommendations: {e}")
                    # Fall back to enhanced method
                    recommendation_pairs = recommend_products_enhanced(
                        G, product_id, partition=partition
                    )
                
                if isinstance(recommendation_pairs, str):  # Handle error messages
                    print(recommendation_pairs)
                    rec_file.write(recommendation_pairs + "\n\n")
                    continue
                    
                recommendation_ids = [rec[0] for rec in recommendation_pairs]
                recommendation_scores = [rec[1] for rec in recommendation_pairs]
                    
                if 'id' in products_df.columns:
                    # Convert both sides to strings to ensure matching works
                    recommendations_with_details = products_df[
                        products_df['id'].astype(str).isin([str(r) for r in recommendation_ids])
                    ]
                    
                    if recommendations_with_details.empty:
                        no_match_msg = "No matching products found in database. Raw recommendation IDs:"
                        print(no_match_msg)
                        print(recommendation_ids)
                        rec_file.write(no_match_msg + "\n")
                        rec_file.write(str(recommendation_ids) + "\n\n")
                    else:
                        print(recommendations_with_details)
                        
                        # Write detailed recommendations to file with scores
                        for i, (_, rec) in enumerate(recommendations_with_details.iterrows()):
                            rec_file.write(f"• Product ID: {rec['id']} (Score: {recommendation_scores[i]:.4f})\n")
                            
                            # Include important product information if available
                            if 'title' in rec:
                                rec_file.write(f"  Title: {rec['title']}\n")
                            if 'group' in rec:
                                rec_file.write(f"  Category: {rec['group']}\n")
                            if 'rating' in rec:
                                rec_file.write(f"  Rating: {rec['rating']}\n")
                            if 'total_connections' in rec:
                                rec_file.write(f"  Network Connections: {int(rec['total_connections'])}\n")
                            
                            # Add community information if available
                            if partition and str(rec['id']) in partition:
                                comm_id = partition[str(rec['id'])]
                                rec_file.write(f"  Community: {comm_id}\n")
                            
                            rec_file.write("\n")
                else:
                    # Print raw recommendations with scores
                    for prod_id, score in recommendation_pairs:
                        print(f"- Product {prod_id}: Score {score:.4f}")
                        rec_file.write(f"- Product {prod_id}: Score {score:.4f}\n")
                    rec_file.write("\n")
                
                rec_file.write("\n")
                print()
        
        print(f"\nRecommendations saved to {recommendations_file}")
        
    finally:
        # Close the output file and restore stdout
        if isinstance(sys.stdout, TeeOutput):
            sys.stdout.close()
        sys.stdout = original_stdout
        print(f"Complete analysis log saved to {output_filename}")

if __name__ == "__main__":
    main()