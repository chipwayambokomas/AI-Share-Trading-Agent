import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

class GraphVisualizer:
    """
    A class to handle the visualization of a learned graph structure.
    It takes the final adjacency matrix and a CSV of calculated metrics
    to produce and save a series of informative plots.
    """
    def __init__(self, metrics_csv_path:str,adj_matrix, save_dir: str, percentile_threshold: int = 90):
        """
        Initializes the visualizer by loading data and setting up the graph.

        Args:
            metrics_csv_path (str): Path to the CSV file with graph metrics.
            adj_matrix: The raw adjacency matrix.
            save_dir (str): The directory where visualization plots will be saved.
            percentile_threshold (int): The percentile used for pruning the graph,
                                        to ensure consistency with the metrics calculation.
        """
        print_header("Stage 6: Graph Visualization")
        self.metrics_df = pd.read_csv(metrics_csv_path, index_col=0)
        self.raw_adj_matrix = adj_matrix
        self.save_dir = save_dir
        self.percentile_threshold = percentile_threshold
        self.labels = self.metrics_df.index.tolist()

        self._setup()

    def _setup(self):
        """Prepares directories and the pruned graph object for plotting."""
        # Create save directories if they don't exist
        self.adj_viz_dir = os.path.join(self.save_dir, "adj_viz")
        self.graph_viz_dir = os.path.join(self.save_dir, "graph_viz")
        os.makedirs(self.adj_viz_dir, exist_ok=True)
        os.makedirs(self.graph_viz_dir, exist_ok=True)

        # Re-create the pruned graph exactly as it was for metric calculation
        threshold = np.percentile(self.raw_adj_matrix, self.percentile_threshold)
        pruned_adj = np.copy(self.raw_adj_matrix)
        pruned_adj[pruned_adj < threshold] = 0

        self.G = nx.from_numpy_array(pruned_adj, create_using=nx.DiGraph)
        mapping = {i: name for i, name in enumerate(self.labels)}
        nx.relabel_nodes(self.G, mapping, copy=False)

        # Calculate a stable layout for consistent plotting
        self.pos = nx.spring_layout(self.G, k=0.8, iterations=100, seed=42)
        print(f"GraphVisualizer setup complete. Plots will be saved in '{self.save_dir}'")

    def plot_adjacency_heatmap(self):
        """Plots and saves a heatmap of the raw adjacency matrix."""
        print("Plotting adjacency matrix heatmap...")
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(self.raw_adj_matrix, xticklabels=self.labels, yticklabels=self.labels, cmap='viridis', annot=False, ax=ax)
        ax.set_title("Learned Adjacency Matrix Heatmap", fontsize=16)
        save_path = os.path.join(self.adj_viz_dir, "adjacency_heatmap.png")
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

    def plot_degree_centrality(self):
        """Visualizes the graph with node size representing degree centrality."""
        print("Plotting degree centrality graph...")
        centrality = self.metrics_df['in_degree_centrality'].to_dict()
        node_sizes = [v * 5000 + 100 for v in centrality.values()]

        if 'community_id' in self.metrics_df.columns:
            node_colors = self.metrics_df['community_id'].tolist()
            cmap = plt.cm.jet
        else:
            node_colors = 'skyblue'
            cmap = None

        fig, ax = plt.subplots(figsize=(15, 12))
        nx.draw_networkx_nodes(self.G, self.pos, ax=ax, node_size=node_sizes, node_color=node_colors, cmap=cmap, alpha=0.8)
        nx.draw_networkx_edges(self.G, self.pos, ax=ax, alpha=0.3, arrows=True, arrowstyle='->', arrowsize=10)
        nx.draw_networkx_labels(self.G, self.pos, ax=ax, font_size=10, font_weight='bold')
        ax.set_title("In-Degree Centrality (Node size reflects influence received)", fontsize=20)
        ax.axis('off')
        save_path = os.path.join(self.graph_viz_dir, "degree_centrality_visualization.png")
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

    def plot_betweenness_centrality(self):
        """Visualizes the graph with node size representing betweenness centrality."""
        print("Plotting betweenness centrality graph...")
        centrality = self.metrics_df['betweenness_centrality'].to_dict()
        node_sizes = [v * 15000 + 100 for v in centrality.values()]

        if 'community_id' in self.metrics_df.columns:
            node_colors = self.metrics_df['community_id'].tolist()
            cmap = plt.cm.jet
        else:
            node_colors = 'skyblue'
            cmap = None

        fig, ax = plt.subplots(figsize=(15, 12))
        nx.draw_networkx_nodes(self.G, self.pos, ax=ax, node_size=node_sizes, node_color=node_colors, cmap=cmap, alpha=0.8)
        nx.draw_networkx_edges(self.G, self.pos, ax=ax, alpha=0.3, arrows=True, arrowstyle='->', arrowsize=10)
        nx.draw_networkx_labels(self.G, self.pos, ax=ax, font_size=10, font_weight='bold')
        ax.set_title("Betweenness Centrality (Node size reflects role as a 'bridge')", fontsize=20)
        ax.axis('off')
        save_path = os.path.join(self.graph_viz_dir, "betweenness_centrality_visualization.png")
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
    def plot_eigenvector_centrality(self):
        """Visualizes the graph with node color representing eigenvector centrality."""
        print("Plotting eigenvector centrality graph...")
        centrality_values = self.metrics_df['eigenvector_centrality_or_pagerank'].tolist()

        fig, ax = plt.subplots(figsize=(15, 12))
        nodes = nx.draw_networkx_nodes(self.G, self.pos, ax=ax, node_color=centrality_values, cmap=plt.cm.viridis, alpha=0.9, node_size=800)
        nx.draw_networkx_edges(self.G, self.pos, ax=ax, alpha=0.3, arrows=True, arrowstyle='->', arrowsize=10)
        nx.draw_networkx_labels(self.G, self.pos, ax=ax, font_size=10, font_weight='bold')
        
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=min(centrality_values), vmax=max(centrality_values)))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.5)
        cbar.set_label('Eigenvector Centrality', rotation=270, labelpad=15)
        
        ax.set_title("Eigenvector Centrality (Node color reflects influence)", fontsize=20)
        ax.axis('off')
        save_path = os.path.join(self.graph_viz_dir, "eigenvector_centrality_visualization.png")
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

    def run_all(self):
        """Executes all plotting methods."""
        self.plot_adjacency_heatmap()
        self.plot_degree_centrality()
        self.plot_betweenness_centrality()
        self.plot_eigenvector_centrality()
        print("All visualizations have been generated and saved.")

def print_header(title):
    """Helper function for clean console output."""
    bar = "=" * 80
    print(f"\n{bar}\n## {title.upper()} ##\n{bar}")