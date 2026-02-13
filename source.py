import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import cdist, pdist, squareform
import warnings
import matplotlib.cm as cm

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def acquire_and_prepare_data(tickers, start_date, end_date, interval='1mo'):
    """
    Acquires historical adjusted close prices from Yahoo Finance,
    computes monthly returns, handles missing data, and constructs
    the N x T return matrix. Also computes the N x N correlation matrix.

    Args:
        tickers (list): List of ticker symbols.
        start_date (str): Start date for data acquisition (e.g., 'YYYY-MM-DD').
        end_date (str): End date for data acquisition (e.g., 'YYYY-MM-DD').
        interval (str): Data interval (e.g., '1mo' for monthly, '1wk' for weekly).

    Returns:
        tuple: (return_matrix (pd.DataFrame), returns_df (pd.DataFrame), corr_matrix (pd.DataFrame))
    """
    print(f"Downloading data for {len(tickers)} assets from {start_date} to {end_date} with interval {interval}...")
    # Download adjusted close prices
    data = yf.download(tickers, start=start_date, end=end_date, interval=interval)['Close']

    # Handle missing data: drop assets with >10% missing values
    initial_assets = data.shape[1]
    missing_threshold = 0.10 * data.shape[0]
    data_cleaned = data.dropna(axis=1, thresh=data.shape[0] - missing_threshold)
    dropped_assets = initial_assets - data_cleaned.shape[1]
    if dropped_assets > 0:
        print(f"Dropped {dropped_assets} assets with more than 10% missing return periods.")

    # For remaining gaps, forward-fill (a stock suspended for a month has zero return, consistent with no trading)
    data_filled = data_cleaned.ffill()

    # Compute monthly returns
    returns_df = data_filled.pct_change().dropna()

    # Construct N x T return matrix: N assets (rows) x T return periods (columns)
    return_matrix = returns_df.T

    # Compute N x N correlation matrix
    corr_matrix = returns_df.corr()

    print(f"Return matrix shape: {return_matrix.shape} (N assets x T periods)")
    print(f"Correlation matrix shape: {corr_matrix.shape} (N assets x N assets)")
    return return_matrix, returns_df, corr_matrix

def standardize_and_pca(return_matrix, n_components=5):
    """
    Standardizes the return matrix (assets x periods) and applies PCA
    for dimensionality reduction.

    Args:
        return_matrix (pd.DataFrame): N assets x T time periods return matrix.
        n_components (int): Number of principal components to retain.

    Returns:
        tuple: (R_pca (np.ndarray), pca_model (sklearn.decomposition.PCA))
    """
    print(f"Standardizing return data (mean 0, std 1)...")
    scaler = StandardScaler()
    # StandardScaler expects (n_samples, n_features), where samples are assets (rows)
    # and features are time periods (columns). Our return_matrix is already N x T.
    R_std = scaler.fit_transform(return_matrix)

    print(f"Applying PCA with {n_components} components...")
    pca = PCA(n_components=n_components)
    R_pca = pca.fit_transform(R_std)

    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_explained_variance = np.cumsum(explained_variance_ratio)

    print(f"Explained variance ratio per component: {explained_variance_ratio}")
    print(f"Cumulative explained variance: {cumulative_explained_variance}")

    # Visualization: PCA Scree Plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, n_components + 1), explained_variance_ratio, marker='o', linestyle='--')
    plt.title('PCA Scree Plot: Explained Variance per Component')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.xticks(range(1, n_components + 1))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('pca_scree_plot.png', dpi=150)
    plt.show()

    return R_pca, pca

def find_optimal_k(data_pca, k_range=range(2, 11)):
    """
    Applies K-Means for a range of K values and computes WCSS (inertia)
    and Silhouette Scores to help determine the optimal number of clusters.

    Args:
        data_pca (np.ndarray): PCA-transformed data (N assets x d components).
        k_range (range): Range of K values to test (e.g., range(2, 11)).

    Returns:
        tuple: (inertias (list), silhouettes (list))
    """
    inertias = []
    silhouettes = []

    print(f"Evaluating K-Means for K in {list(k_range)}...")
    for k in k_range:
        kmeans = KMeans(n_clusters=k, n_init=20, random_state=42)
        labels = kmeans.fit_predict(data_pca)
        inertias.append(kmeans.inertia_)
        if k > 1: # Silhouette score requires at least 2 clusters
            silhouettes.append(silhouette_score(data_pca, labels))
        else:
            silhouettes.append(np.nan) # Placeholder for k=1 where silhouette is undefined

    # Visualization: Elbow Plot and Silhouette Score Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    ax1.plot(k_range, inertias, 'b-o')
    ax1.set_xlabel('Number of Clusters (K)')
    ax1.set_ylabel('Within-Cluster SSE (Inertia)')
    ax1.set_title('Elbow Method')
    ax1.grid(True)

    ax2.plot(k_range, silhouettes, 'r-o')
    ax2.set_xlabel('Number of Clusters (K)')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Analysis')
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('kmeans_k_selection.png', dpi=150)
    plt.show()

    return inertias, silhouettes

def perform_hierarchical_clustering(returns_df, K_optimal, tickers, plot_dendrogram=True):
    """
    Performs hierarchical clustering using correlation distance and Ward's linkage.
    Visualizes the result with a dendrogram and cuts it to K_optimal clusters.

    Args:
        returns_df (pd.DataFrame): N assets x T time periods return matrix.
        K_optimal (int): Optimal number of clusters determined from K-Means analysis.
        tickers (list): List of asset tickers for labeling.
        plot_dendrogram (bool): Whether to display the dendrogram.

    Returns:
        np.ndarray: Cluster labels from hierarchical clustering.
    """
    print("Standardizing returns for hierarchical clustering (Euclidean distance for Ward linkage)...")
    scaler_hc = StandardScaler()
    # We transpose returns_df so that assets are rows and time periods are features
    # This is consistent with sklearn's convention for samples x features
    returns_std_T = scaler_hc.fit_transform(returns_df.T) # (N_assets, N_time_periods)

    # Compute Euclidean distances on standardized returns
    euclidean_distances = pdist(returns_std_T, metric='euclidean')

    print("Applying Ward's linkage for hierarchical clustering on Euclidean distances of standardized returns...")
    Z = linkage(euclidean_distances, method='ward')

    if plot_dendrogram:
        # Visualization: Dendrogram
        plt.figure(figsize=(18, 8))
        dendrogram(
            Z,
            labels=tickers,
            leaf_rotation=90,
            leaf_font_size=8,
            color_threshold=0.7 * max(Z[:, 2]), # Highlight clusters above a certain distance
        )
        plt.title('Hierarchical Clustering Dendrogram (Ward Linkage, Euclidean Distance on Standardized Returns)')
        plt.xlabel('Asset')
        plt.ylabel('Distance (Euclidean)')
        plt.tight_layout()
        plt.savefig('dendrogram_assets.png', dpi=150)
        plt.show()

    # Cut dendrogram to get K clusters
    hc_labels = fcluster(Z, t=K_optimal, criterion='maxclust')

    print(f"Hierarchical clustering resulted in {len(np.unique(hc_labels))} clusters.")
    return hc_labels

def interpret_and_visualize_clusters(R_pca, kmeans_labels, available_tickers, sector_map_cleaned, returns_df, pca_model):
    """
    Computes cluster profiles, generates PCA scatter plot with cluster labels,
    and a sorted correlation heatmap.

    Args:
        R_pca (np.ndarray): PCA-transformed data.
        kmeans_labels (np.ndarray): Cluster assignments from K-Means.
        available_tickers (list): List of asset tickers.
        sector_map_cleaned (dict): Mapping of tickers to GICS sectors.
        returns_df (pd.DataFrame): DataFrame of asset returns (Time x Assets).
        pca_model (sklearn.decomposition.PCA): Fitted PCA model.

    Returns:
        tuple: (cluster_profile (pd.DataFrame), asset_df (pd.DataFrame))
    """
    print("Calculating asset-level statistics (annualized return, volatility, beta)...")
    asset_df = pd.DataFrame({
        'ticker': available_tickers,
        'cluster': kmeans_labels,
        'sector': [sector_map_cleaned.get(t, 'Unknown') for t in available_tickers]
    }).set_index('ticker')

    asset_df['ann_return'] = returns_df.mean() * 12
    asset_df['ann_vol']    = returns_df.std() * np.sqrt(12)

    # Calculate beta for each asset against the equal-weighted portfolio of all assets
    market_returns = returns_df.mean(axis=1) # Proxy for market portfolio
    market_variance = market_returns.var()
    asset_df['beta'] = [returns_df[t].cov(market_returns) / market_variance if market_variance > 0 else 0 for t in available_tickers]

    asset_df = asset_df.reset_index() # If ticker needs to be a column later

    print("\nGenerating Cluster Profile Summary...")
    cluster_profile = asset_df.groupby('cluster').agg(
        ann_return=('ann_return', 'mean'),
        ann_vol=('ann_vol', 'mean'),
        beta=('beta', 'mean'),
        dominant_sector=('sector', lambda x: x.mode()[0]), # Most common sector in cluster
        n_assets=('ticker', 'count')
    ).sort_values(by='ann_return', ascending=False)
    print(cluster_profile)

    print("\nVisualizing assets in PCA space, colored by cluster...")
    # Visualization: PCA Scatter Plot
    plt.figure(figsize=(12, 9))
    scatter = plt.scatter(R_pca[:, 0], R_pca[:, 1], c=kmeans_labels, cmap='viridis', s=80, edgecolors='black', alpha=0.8)
    for i, ticker in enumerate(available_tickers):
        plt.annotate(ticker, (R_pca[i, 0], R_pca[i, 1]), fontsize=7, alpha=0.7)

    pc1_explained_var = pca_model.explained_variance_ratio_[0] * 100
    pc2_explained_var = pca_model.explained_variance_ratio_[1] * 100
    plt.xlabel(f'PC1 ({pc1_explained_var:.1f}% variance explained)')
    plt.ylabel(f'PC2 ({pc2_explained_var:.1f}% variance explained)')
    plt.title('Stock Clusters in PCA Space')
    plt.colorbar(scatter, label='Cluster ID')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('pca_cluster_scatter.png', dpi=150)
    plt.show()

    print("\nGenerating Sorted Correlation Heatmap...")
    # Visualization: Sorted Correlation Heatmap
    # Sort the original correlation matrix by cluster assignment
    sort_idx = np.argsort(kmeans_labels)
    sorted_tickers = [available_tickers[i] for i in sort_idx]
    sorted_corr = returns_df.corr().loc[sorted_tickers, sorted_tickers]

    plt.figure(figsize=(14, 12))
    sns.heatmap(sorted_corr, cmap='RdBu_r', center=0,
                xticklabels=False, yticklabels=False, # Tickers can be too many to show
                vmin=-0.5, vmax=1.0)
    plt.title('Correlation Matrix Sorted by Cluster Assignment')
    plt.tight_layout()
    plt.savefig('corr_heatmap_sorted.png', dpi=150)
    plt.show()

    return cluster_profile, asset_df

def evaluate_diversification_and_build_portfolio(
    kmeans_model, R_pca, kmeans_labels, available_tickers, returns_df, K_optimal, cluster_profile_summary, asset_assignment_table
):
    """
    Evaluates diversification quality (intra- vs. inter-cluster correlation),
    constructs a cluster-diversified portfolio, and compares its risk to a random benchmark.

    Args:
        kmeans_model (sklearn.cluster.KMeans): Fitted K-Means model.
        R_pca (np.ndarray): PCA-transformed data (N assets x d components).
        kmeans_labels (np.ndarray): Cluster assignments from K-Means.
        available_tickers (list): List of asset tickers.
        returns_df (pd.DataFrame): DataFrame of asset returns (Time x Assets).
        K_optimal (int): Optimal number of clusters.
        cluster_profile_summary (pd.DataFrame): DataFrame containing cluster profiles.
        asset_assignment_table (pd.DataFrame): DataFrame containing asset assignments to clusters and stats.

    Returns:
        tuple: (intra_corrs_mean (float), inter_corrs_mean (float),
                cluster_portfolio_vol (float), random_portfolio_vol (float),
                volatility_reduction (float))
    """
    print("Calculating intra-cluster and inter-cluster correlations...")
    corr_matrix_full = returns_df.corr()
    intra_corrs = []
    inter_corrs = []

    for i in range(len(available_tickers)):
        for j in range(i + 1, len(available_tickers)): # Only consider unique pairs
            ticker_i = available_tickers[i]
            ticker_j = available_tickers[j]
            corr_val = corr_matrix_full.loc[ticker_i, ticker_j]

            if kmeans_labels[i] == kmeans_labels[j]:
                intra_corrs.append(corr_val)
            else:
                inter_corrs.append(corr_val)

    intra_corrs_mean = np.mean(intra_corrs) if intra_corrs else 0
    inter_corrs_mean = np.mean(inter_corrs) if inter_corrs else 0

    print(f"Average intra-cluster correlation: {intra_corrs_mean:.3f}")
    print(f"Average inter-cluster correlation: {inter_corrs_mean:.3f}")

    diversification_ratio = intra_corrs_mean / inter_corrs_mean if inter_corrs_mean != 0 else np.inf
    print(f"Diversification ratio (Intra/Inter): {diversification_ratio:.2f} (Good clustering if ratio >> 1)")

    print("\nConstructing cluster-diversified portfolio...")
    # Select representative asset from each cluster (closest to centroid in PCA space)
    representatives = []
    cluster_centroids_pca = kmeans_model.cluster_centers_

    for k in range(K_optimal):
        mask = (kmeans_labels == k)
        cluster_points_pca = R_pca[mask]

        if len(cluster_points_pca) == 0:
            print(f"Warning: Cluster {k} is empty. Skipping representative selection.")
            continue

        # Find distances from centroid to all points in this cluster
        distances = cdist([cluster_centroids_pca[k]], cluster_points_pca, 'euclidean')[0]
        nearest_idx_in_cluster = np.argmin(distances)

        # Map back to original tickers
        original_indices_in_cluster = np.where(mask)[0]
        representative_ticker = available_tickers[original_indices_in_cluster[nearest_idx_in_cluster]]
        representatives.append(representative_ticker)

    if not representatives:
        print("Error: No representatives found to build portfolio.")
        return 0, 0, 0, 0, 0

    print(f"Representative assets selected: {representatives}")

    # Equally-weighted portfolio of representatives
    cluster_portfolio_returns = returns_df[representatives].mean(axis=1)
    cluster_portfolio_vol = cluster_portfolio_returns.std() * np.sqrt(12)

    print(f"Cluster-diversified portfolio annualized volatility: {cluster_portfolio_vol:.2%}")

    print("\nConstructing random benchmark portfolio...")
    # Benchmark: random equal-weight portfolio of same size
    np.random.seed(42) # For reproducibility
    # Ensure random picks are from available tickers and are unique
    if len(available_tickers) < K_optimal:
        print("Warning: Not enough available tickers for a random portfolio of size K_optimal. Reducing random portfolio size.")
        num_random_assets = len(available_tickers)
    else:
        num_random_assets = K_optimal

    random_picks = np.random.choice(available_tickers, num_random_assets, replace=False)
    print(f"Randomly selected assets for benchmark: {random_picks.tolist()}")

    random_portfolio_returns = returns_df[list(random_picks)].mean(axis=1)
    random_portfolio_vol = random_portfolio_returns.std() * np.sqrt(12)

    print(f"Random benchmark portfolio annualized volatility: {random_portfolio_vol:.2%}")

    # Calculate Volatility Reduction
    volatility_reduction = (1 - (cluster_portfolio_vol / random_portfolio_vol)) * 100 if random_portfolio_vol > 0 else 0
    print(f"\nVolatility Reduction (Cluster vs. Random): {volatility_reduction:.1f}%")

    # Visualization: Cluster Return/Risk Scatter Plot
    plt.figure(figsize=(10, 7))
    for cluster_id in cluster_profile_summary.index:
        cluster_data = asset_assignment_table[asset_assignment_table['cluster'] == cluster_id]

        # Plot individual assets in the cluster
        plt.scatter(cluster_data['ann_vol'], cluster_data['ann_return'],
                    label=f'Cluster {cluster_id} Assets', alpha=0.6, s=20)

        # Plot cluster centroid
        if cluster_id in cluster_profile_summary.index: # Ensure cluster_id exists in summary
            cluster_avg_vol = cluster_profile_summary.loc[cluster_id, 'ann_vol']
            cluster_avg_return = cluster_profile_summary.loc[cluster_id, 'ann_return']
            cmap = cm.get_cmap('tab10')
            centroid_color = cmap(int(cluster_id) % 10)

            plt.scatter(
                cluster_avg_vol, cluster_avg_return,
                marker='X',
                s=260,                    # bigger X
                c=[centroid_color],       # per-cluster centroid color
                edgecolors='black',
                linewidths=1.8,
                zorder=5,
                label=f'Cluster {cluster_id} Centroid (Avg)'
            )

    plt.title('Cluster Return/Risk Profile (Annualized)')
    plt.xlabel('Annualized Volatility')
    plt.ylabel('Annualized Return')
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('cluster_return_risk_scatter.png', dpi=150)
    plt.show()

    return intra_corrs_mean, inter_corrs_mean, cluster_portfolio_vol, random_portfolio_vol, volatility_reduction


def run_portfolio_clustering_analysis(
    tickers: list,
    start_date: str,
    end_date: str,
    interval: str = '1mo',
    n_components_pca: int = 5,
    k_range_values: range = range(2, 11),
    optimal_k: int = 5, # Explicitly pass optimal_k for automation
    plot_dendrogram: bool = True,
    sector_map_input: dict = None # Optional, or define inside if fixed
):
    """
    Orchestrates the entire portfolio clustering analysis workflow.

    Args:
        tickers (list): List of ticker symbols.
        start_date (str): Start date for data acquisition (e.g., 'YYYY-MM-DD').
        end_date (str): End date for data acquisition (e.g., 'YYYY-MM-DD').
        interval (str): Data interval (e.g., '1mo' for monthly, '1wk' for weekly).
        n_components_pca (int): Number of principal components to retain.
        k_range_values (range): Range of K values to test for K-Means.
        optimal_k (int): The chosen optimal number of clusters for final analysis.
        plot_dendrogram (bool): Whether to display the hierarchical clustering dendrogram.
        sector_map_input (dict, optional): Mapping of tickers to GICS sectors.
                                       If None, a default map will be used.

    Returns:
        dict: A dictionary containing key results of the analysis.
    """
    print("--- Starting Portfolio Clustering Analysis ---")

    # 1. Data Acquisition and Preparation
    return_matrix, returns_df, corr_matrix = acquire_and_prepare_data(tickers, start_date, end_date, interval)

    # Ensure tickers are consistent after data cleaning
    available_tickers = return_matrix.index.tolist()

    # Handle sector_map
    if sector_map_input is None:
        # Define a default sector map if not provided
        sector_map_default = {
            'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOG': 'Technology', 'AMZN': 'Consumer Discretionary', 'META': 'Communication Services',
            'JPM': 'Financials', 'BAC': 'Financials', 'GS': 'Financials', 'MS': 'Financials', 'C': 'Financials',
            'JNJ': 'Healthcare', 'PFE': 'Healthcare', 'UNH': 'Healthcare', 'MRK': 'Healthcare', 'ABBV': 'Healthcare',
            'XOM': 'Energy', 'CVX': 'Energy', 'COP': 'Materials', 'SLB': 'Energy', 'EOG': 'Energy',
            'PG': 'Consumer Staples', 'KO': 'Consumer Staples', 'PEP': 'Consumer Staples', 'WMT': 'Consumer Staples', 'COST': 'Consumer Staples',
            'TSLA': 'Consumer Discretionary', 'GM': 'Consumer Discretionary', 'F': 'Consumer Discretionary', 'NKE': 'Consumer Discretionary', 'SBUX': 'Consumer Discretionary',
            'NEE': 'Utilities', 'DUK': 'Utilities', 'SO': 'Utilities', 'D': 'Utilities', 'AEP': 'Utilities',
            'AMT': 'Real Estate', 'PLD': 'Real Estate', 'CCI': 'Real Estate', 'SPG': 'Real Estate', 'EQIX': 'Real Estate',
            'CAT': 'Industrials', 'DE': 'Industrials', 'HON': 'Industrials', 'MMM': 'Industrials', 'GE': 'Industrials',
            'LIN': 'Materials', 'APD': 'Materials', 'ECL': 'Materials', 'SHW': 'Materials', 'DD': 'Materials'
        }
        sector_map_cleaned = {ticker: sector_map_default.get(ticker, 'Unknown') for ticker in available_tickers}
    else:
        sector_map_cleaned = {ticker: sector_map_input.get(ticker, 'Unknown') for ticker in available_tickers}

    print("\nFirst 5 rows of the return matrix (assets as rows, time periods as columns):")
    print(return_matrix.head())
    print("\nFirst 5 rows and columns of the correlation matrix:")
    print(corr_matrix.iloc[:5, :5])

    # 2. Standardization and PCA
    R_pca, pca_model = standardize_and_pca(return_matrix, n_components=n_components_pca)
    print(f"\nShape of PCA-transformed data: {R_pca.shape} (N assets x {n_components_pca} principal components)")
    print("First 5 rows of PCA-transformed data:")
    print(R_pca[:5])

    # 3. K-Means Clustering: Optimal K Selection
    inertias, silhouettes = find_optimal_k(R_pca, k_range=k_range_values)
    print(f"\nK-Means inertia values for K in {list(k_range_values)}: {inertias}")
    print(f"K-Means silhouette scores for K in {list(k_range_values)}: {silhouettes}")


    # 4. K-Means Clustering: Final Run with Optimal K
    print(f"\nPerforming K-Means with optimal K = {optimal_k} clusters.")
    kmeans = KMeans(n_clusters=optimal_k, n_init=20, random_state=42)
    kmeans_cluster_labels = kmeans.fit_predict(R_pca)

    # 5. Hierarchical Clustering (for comparison/alternative view)
    print("\nPerforming Hierarchical Clustering...")
    hc_cluster_labels = perform_hierarchical_clustering(returns_df, optimal_k, available_tickers, plot_dendrogram)
    print("\nFirst 10 assets with their hierarchical cluster labels:")
    print(pd.Series(hc_cluster_labels, index=available_tickers).head(10))


    # 6. Cluster Interpretation and Visualization
    print("\nInterpreting and Visualizing K-Means Clusters...")
    cluster_profile_summary, asset_assignment_table = interpret_and_visualize_clusters(
        R_pca, kmeans_cluster_labels, available_tickers, sector_map_cleaned, returns_df, pca_model
    )
    print("\nCluster Assignment Table (first 10 assets):")
    print(asset_assignment_table.head(10))
    print("\nFull Cluster Profile Summary:")
    print(cluster_profile_summary)


    # 7. Diversification Evaluation and Portfolio Construction
    print("\nEvaluating Diversification and Building Portfolio...")
    intra_corrs_mean, inter_corrs_mean, cluster_portfolio_vol, random_portfolio_vol, volatility_reduction = \
        evaluate_diversification_and_build_portfolio(
            kmeans, R_pca, kmeans_cluster_labels, available_tickers, returns_df, optimal_k, cluster_profile_summary, asset_assignment_table
        )

    print("\n--- Portfolio Clustering Analysis Complete ---")

    # Collect and return results
    results = {
        "return_matrix": return_matrix,
        "returns_df": returns_df,
        "corr_matrix": corr_matrix,
        "available_tickers": available_tickers,
        "sector_map_cleaned": sector_map_cleaned,
        "R_pca": R_pca,
        "pca_model": pca_model,
        "kmeans_inertias": inertias,
        "kmeans_silhouettes": silhouettes,
        "kmeans_model": kmeans, # return the fitted model
        "kmeans_cluster_labels": kmeans_cluster_labels,
        "hc_cluster_labels": hc_cluster_labels,
        "cluster_profile_summary": cluster_profile_summary,
        "asset_assignment_table": asset_assignment_table,
        "intra_cluster_corr_mean": intra_corrs_mean,
        "inter_cluster_corr_mean": inter_corrs_mean,
        "cluster_portfolio_vol": cluster_portfolio_vol,
        "random_portfolio_vol": random_portfolio_vol,
        "volatility_reduction": volatility_reduction,
    }
    return results

if __name__ == "__main__":
    # Define parameters for data acquisition
    # Selected diverse subset of S&P 500 stocks as per original context
    example_tickers = [
        'AAPL', 'MSFT', 'GOOG', 'AMZN', 'META',  # Tech
        'JPM', 'BAC', 'GS', 'MS', 'C',          # Financials
        'JNJ', 'PFE', 'UNH', 'MRK', 'ABBV',     # Healthcare
        'XOM', 'CVX', 'COP', 'SLB', 'EOG',      # Energy
        'PG', 'KO', 'PEP', 'WMT', 'COST',       # Staples
        'TSLA', 'GM', 'F', 'NKE', 'SBUX',       # Discretionary
        'NEE', 'DUK', 'SO', 'D', 'AEP',         # Utilities
        'AMT', 'PLD', 'CCI', 'SPG', 'EQIX',     # Real Estate
        'CAT', 'DE', 'HON', 'MMM', 'GE',        # Industrials
        'LIN', 'APD', 'ECL', 'SHW', 'DD'        # Materials
    ]
    example_start_date = '2021-01-01'
    example_end_date = '2024-01-01'
    example_optimal_k = 5 # Manually chosen optimal K after reviewing Elbow/Silhouette plots

    # Run the full analysis
    analysis_results = run_portfolio_clustering_analysis(
        tickers=example_tickers,
        start_date=example_start_date,
        end_date=example_end_date,
        optimal_k=example_optimal_k
    )

    # You can now access the results from the dictionary
    print("\n--- Summary of Key Results ---")
    print(f"Optimal K-Means Clusters used: {analysis_results['kmeans_model'].n_clusters}")
    print(f"Cluster Portfolio Annualized Volatility: {analysis_results['cluster_portfolio_vol']:.2%}")
    print(f"Random Portfolio Annualized Volatility: {analysis_results['random_portfolio_vol']:.2%}")
    print(f"Volatility Reduction: {analysis_results['volatility_reduction']:.1f}%")
    print("\nCluster Profile Summary:")
    print(analysis_results['cluster_profile_summary'])
    print("\nFirst 10 assets and their K-Means cluster assignments:")
    print(analysis_results['asset_assignment_table'].head(10))
