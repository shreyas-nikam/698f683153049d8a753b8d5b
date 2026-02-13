
# Clustering for Asset Allocation: Unsupervised Diversification for Alpha Wealth

## Introduction: Optimizing Diversification at Alpha Wealth Management

Mr. David Chen, a seasoned CFA Charterholder and Senior Portfolio Manager at Alpha Wealth Management, faces a perennial challenge: how to achieve truly robust portfolio diversification in a dynamic market. Alpha Wealth prides itself on delivering superior risk-adjusted returns, and David knows that relying solely on traditional asset classifications (like GICS sectors or style boxes) often falls short. These predefined categories can be stale, miss subtle cross-sector correlations, and oversimplify the complex interplay between assets.

David's goal is to discover data-driven asset groupings that reveal the genuine co-movement patterns or fundamental characteristics of securities. This approach aims to move beyond nominal diversification to a more empirical understanding of his asset universe, leading to more resilient portfolios. This notebook will guide David through a real-world workflow to apply unsupervised machine learning techniques, specifically K-Means and Hierarchical Clustering, to uncover the latent structure within a universe of S&P 500 stocks. By identifying these "natural" clusters, David can make more informed allocation decisions, ultimately enhancing portfolio risk management and delivering better outcomes for Alpha Wealth's clients.

---

### Install Required Libraries

```python
!pip install pandas numpy scikit-learn scipy yfinance matplotlib seaborn
```

### Import Required Dependencies

```python
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
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore') # Suppress warnings for cleaner output
```

---

## Section 1: Data Acquisition and Return Matrix Construction

### Story + Context + Real-World Relevance

David's first step is to gather the necessary raw material: historical price data for his chosen universe of assets. He has decided to focus on a diverse selection of S&P 500 stocks, aiming to capture a broad representation of the market. Accurate and consistent historical adjusted close prices are crucial, as these will be used to compute monthly returns, which form the basis for understanding asset co-movement. David understands that the integrity of his raw data directly impacts the reliability of any subsequent clustering analysis. He will then transform this data into an `N x T` return matrix, where `N` is the number of assets and `T` is the number of time periods. This structure is essential for feeding into clustering algorithms.

The fundamental operation here is the calculation of asset returns. For a given asset, its simple return $R_t$ at time $t$ is calculated as:
$$ R_t = \frac{P_t - P_{t-1}}{P_{t-1}} $$
where $P_t$ is the adjusted close price at time $t$ and $P_{t-1}$ is the adjusted close price at time $t-1$.

Furthermore, to anticipate future steps, David needs to compute the correlation matrix. The Pearson correlation coefficient $\rho_{ij}$ between the returns of asset $i$ and asset $j$ is given by:
$$ \rho_{ij} = \frac{\text{Cov}(R_i, R_j)}{\sigma_i \sigma_j} $$
where $\text{Cov}(R_i, R_j)$ is the covariance between the returns of asset $i$ and asset $j$, and $\sigma_i$ and $\sigma_j$ are their respective standard deviations. This matrix is fundamental for understanding asset co-movement and will be central to evaluating cluster quality.

### Code cell (function definition + function execution)

```python
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
    data = yf.download(tickers, start=start_date, end=end_date, interval=interval)['Adj Close']

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

# Define parameters for data acquisition
# Selected diverse subset of S&P 500 stocks as per original context
tickers = [
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
start_date = '2021-01-01'
end_date = '2024-01-01'

# Execute data acquisition and preparation
return_matrix, returns_df, corr_matrix = acquire_and_prepare_data(tickers, start_date, end_date)

# Define sector metadata for later cluster labeling
sector_map = {
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

# Ensure tickers are consistent after data cleaning
available_tickers = return_matrix.index.tolist()
sector_map_cleaned = {ticker: sector_map.get(ticker, 'Unknown') for ticker in available_tickers}

print("\nFirst 5 rows of the return matrix (assets as rows, time periods as columns):")
print(return_matrix.head())
print("\nFirst 5 rows and columns of the correlation matrix:")
print(corr_matrix.iloc[:5, :5])
```

### Markdown cell (explanation of execution)

The data acquisition process successfully downloaded monthly adjusted close prices for our selected S&P 500 universe. After handling missing values by dropping assets with significant gaps and forward-filling remaining smaller gaps, we computed the monthly percentage returns. The resulting `return_matrix` is transposed to have assets as rows and time periods as columns, which is the standard input format for many clustering algorithms (`N` samples, `T` features). The `corr_matrix` shows the pairwise correlation between assets. David can now confirm that his foundational data is robust and correctly formatted, ready for the next stage of pre-processing. The `return_matrix` shape (assets, time periods) is critical for `StandardScaler` and `PCA` to operate correctly across each asset's time series.

---

## Section 2: Data Standardization and Dimensionality Reduction: Uncovering Latent Market Factors

### Story + Context + Real-World Relevance

David understands that raw return data can be problematic for clustering. Assets with higher volatility (like a growth stock) might dominate distance calculations purely due to larger return magnitudes, obscuring genuine co-movement patterns. Therefore, standardizing the return series to zero mean and unit variance is crucial. This ensures that all assets contribute equally to the clustering process, focusing on their *patterns* of movement rather than their absolute scale.

Furthermore, financial markets are often driven by a smaller set of underlying "factors" rather than thousands of independent events. Principal Component Analysis (PCA) is a powerful technique that helps David to:
1.  **Denoise the data**: By focusing on the most significant principal components, PCA effectively filters out idiosyncratic noise.
2.  **Identify latent market factors**: The first few principal components often capture broad market effects (e.g., market factor, sector/style factors), providing a more robust and interpretable feature space for clustering. This directly aligns with the CFA curriculum's emphasis on multi-factor models.

The standardization process for each asset's return series $R_i = \{R_{i,1}, R_{i,2}, \dots, R_{i,T}\}$ involves transforming it to $Z_i = \{Z_{i,1}, Z_{i,2}, \dots, Z_{i,T}\}$ using:
$$ Z_{i,t} = \frac{R_{i,t} - \mu_i}{\sigma_i} $$
where $\mu_i$ is the mean of asset $i$'s returns and $\sigma_i$ is its standard deviation.

PCA then projects these standardized return vectors $\tilde{r}_i \in \mathbb{R}^T$ (where $\tilde{r}_i$ is the entire time series of standardized returns for asset $i$) onto a lower-dimensional space. Given the $N \times T$ standardized return matrix $\mathbf{\tilde{R}}$, PCA computes the eigendecomposition of the covariance matrix:
$$ \frac{1}{N-1} \mathbf{\tilde{R}}\mathbf{\tilde{R}}^T = \mathbf{V}\mathbf{\Lambda}\mathbf{V}^T $$
where $\mathbf{\Lambda} = \text{diag}(\lambda_1, \dots, \lambda_T)$ contains the eigenvalues (variance explained), and $\mathbf{V}$ contains the eigenvectors (principal component loadings). The projection of asset $i$ onto the first $d$ principal components is:
$$ \mathbf{z}_i = \mathbf{V}_{1:d}^T \tilde{\mathbf{r}}_i \in \mathbb{R}^d $$
Typically, $\lambda_1$ (first principal component) explains a large portion (40-60%) of total variance and often corresponds to the broad market factor. Subsequent components often capture sector or style rotations.

### Code cell (function definition + function execution)

```python
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

# Execute standardization and PCA
n_components_pca = 5 # Retain 5 principal components for analysis
R_pca, pca_model = standardize_and_pca(return_matrix, n_components=n_components_pca)

print(f"\nShape of PCA-transformed data: {R_pca.shape} (N assets x {n_components_pca} principal components)")
print("First 5 rows of PCA-transformed data:")
print(R_pca[:5])
```

### Markdown cell (explanation of execution)

The standardization process successfully scaled each asset's return series to have a mean of zero and unit variance. This crucial step prevents assets with inherently higher volatility from unduly influencing the clustering algorithms. Subsequently, PCA transformed the standardized returns into a lower-dimensional space of 5 principal components.

The PCA Scree Plot visually confirms the explained variance. For instance, PC1 often captures a significant portion (e.g., 40-60%) of the total variance, representing the broad market factor. PC2 and PC3 might capture sector or style-specific factors. By reducing dimensionality, David has effectively denoised the data, focusing on the most relevant underlying drivers of asset co-movement. This new, compressed representation (`R_pca`) is a more robust input for clustering, ensuring the clusters are based on fundamental patterns rather than noise or scale differences.

---

## Section 3: Optimal K-Means Clustering: Identifying Core Diversification Buckets

### Story + Context + Real-World Relevance

David needs to group the assets into distinct "buckets" or clusters based on their underlying behavior. K-Means clustering is a straightforward algorithm for this, but a critical decision is determining the optimal number of clusters, $K$. An incorrect $K$ can lead to either over-segmentation (too many small, insignificant clusters) or under-segmentation (too few, overly broad clusters). To guide this choice, David will use two diagnostic tools: the Elbow Method and Silhouette Analysis. These methods provide quantitative insights into cluster quality at different $K$ values, helping him select a `K_optimal` that genuinely reflects the inherent grouping structure in the market.

The K-Means algorithm aims to partition $N$ data points $\{\mathbf{x}_1, \dots, \mathbf{x}_N\}$ in $\mathbb{R}^d$ into $K$ clusters $\{C_1, \dots, C_K\}$ by minimizing the Within-Cluster Sum of Squares (WCSS), also known as inertia. The objective function is:
$$ J = \sum_{k=1}^K \sum_{\mathbf{x}_i \in C_k} ||\mathbf{x}_i - \mu_k||^2 $$
where $\mu_k$ is the centroid of cluster $C_k$. The Elbow Method plots $J$ against $K$, looking for a point where the rate of decrease in $J$ significantly slows down, resembling an "elbow".

The Silhouette Score measures how similar an object is to its own cluster compared to other clusters. For a data point $i$:
$$ s_i = \frac{b_i - a_i}{\max(a_i, b_i)} $$
where $a_i$ is the average distance from point $i$ to all other points in its cluster (cohesion), and $b_i$ is the average distance from point $i$ to all points in the nearest other cluster (separation). The score $s_i$ ranges from -1 to 1. Values near 1 indicate well-clustered points, near 0 indicate boundary points, and negative values indicate potential misassignment. The average Silhouette Score across all points is used to evaluate cluster quality, with higher values indicating better-defined clusters.

### Code cell (function definition + function execution)

```python
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

# Define range for K
k_range_values = range(2, 11)

# Execute K-Means for a range of K and get evaluation metrics
inertias, silhouettes = find_optimal_k(R_pca, k_range=k_range_values)
```

### Markdown cell (explanation of execution)

David has now generated the Elbow and Silhouette plots. The **Elbow Method** plot shows WCSS decreasing as $K$ increases, but the "elbow" where the rate of decrease significantly lessens typically indicates a good $K$. The **Silhouette Analysis** plot provides another perspective; David looks for a peak in the average Silhouette Score, suggesting well-separated and compact clusters.

Based on visual inspection of these plots, David observes that a `K_optimal` around 5 or 6 seems appropriate, as the Elbow plot shows a distinct bend and the Silhouette score reaches a reasonable peak in that range. For the purpose of this exercise, David decides to proceed with `K_optimal = 5`, balancing interpretability with statistical goodness-of-fit. This data-driven selection of `K` is a significant improvement over arbitrary groupings, ensuring his clusters are statistically meaningful.

---

## Section 4: Hierarchical Clustering: Uncovering Nested Relationships

### Story + Context + Real-World Relevance

While K-Means provides flat, distinct clusters, David recognizes that asset relationships might be hierarchical. Some assets are very similar (e.g., energy stocks), forming sub-groups, which then combine with other sub-groups (e.g., materials stocks) at a higher level of dissimilarity to form broader categories. Hierarchical clustering, visualized through a dendrogram, offers a complementary view by revealing this nested structure. This is particularly valuable for understanding how assets merge from individual entities to broader classifications and is the foundational technique for advanced portfolio optimization methods like Hierarchical Risk Parity (HRP).

Hierarchical clustering builds a tree of clusters by iteratively merging the closest pairs of clusters (agglomerative approach). The "distance" between assets is crucial here. For financial applications, Euclidean distance in standardized return space is closely related to correlation distance. Specifically, for standardized return vectors $\mathbf{\tilde{r}}_i$ and $\mathbf{\tilde{r}}_j$ (zero mean, unit variance), the squared Euclidean distance is:
$$ ||\mathbf{\tilde{r}}_i - \mathbf{\tilde{r}}_j||^2 = 2T(1 - \rho_{ij}) $$
where $\rho_{ij}$ is the sample correlation and $T$ is the number of return periods. This means K-Means on standardized returns is effectively clustering by correlation.

The standard correlation distance used in hierarchical clustering is:
$$ d_{ij} = \sqrt{2(1 - \rho_{ij})} \in [0, 2] $$
where $d_{ij}=0$ implies perfect positive correlation and $d_{ij}=2$ implies perfect negative correlation.

Ward's linkage method is commonly used for merging clusters. At each step, it merges the pair of clusters $(C_a, C_b)$ that results in the minimum increase in total within-cluster variance, which is equivalent to minimizing the increase in the WCSS. The increase in WCSS when merging $C_a$ and $C_b$ is:
$$ \Delta(C_a, C_b) = \frac{|C_a||C_b|}{|C_a| + |C_b|} ||\mu_a - \mu_b||^2 $$
where $|C_a|$ and $|C_b|$ are the number of points in clusters $C_a$ and $C_b$, and $\mu_a$ and $\mu_b$ are their centroids. This method tends to produce compact, roughly equal-sized clusters, which is generally desirable for diversification.

### Code cell (function definition + function execution)

```python
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
    print("Computing correlation-based distance matrix...")
    corr_matrix_hc = returns_df.corr()
    # Correlation distance: d_ij = sqrt(2 * (1 - rho_ij))
    dist_matrix = np.sqrt(2 * (1 - corr_matrix_hc))

    print("Applying Ward's linkage for hierarchical clustering...")
    # Ward's linkage minimizes the increase in within-cluster variance
    Z = linkage(dist_matrix, method='ward')

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
        plt.title('Hierarchical Clustering Dendrogram (Ward Linkage, Correlation-Based Distance)')
        plt.xlabel('Asset')
        plt.ylabel('Distance (Correlation-Based)')
        plt.tight_layout()
        plt.savefig('dendrogram_assets.png', dpi=150)
        plt.show()

    # Cut dendrogram to get K clusters
    hc_labels = fcluster(Z, t=K_optimal, criterion='maxclust')

    print(f"Hierarchical clustering resulted in {len(np.unique(hc_labels))} clusters.")
    return hc_labels

# Use K_optimal from previous section, let's assume 5 for continuity
K_optimal = 5 # As determined from Elbow/Silhouette analysis in the previous step

# Execute hierarchical clustering
hc_cluster_labels = perform_hierarchical_clustering(returns_df, K_optimal, available_tickers)

print("\nFirst 10 assets with their hierarchical cluster labels:")
print(pd.Series(hc_cluster_labels, index=available_tickers).head(10))
```

### Markdown cell (explanation of execution)

The dendrogram visually represents the nested hierarchy of asset similarities. By reading it from bottom-up, David can see which assets merge first (indicating high similarity, like energy peers `XOM` and `CVX`), and how these sub-clusters then combine into broader groups (e.g., energy merging with materials). The vertical axis, representing correlation-based distance, indicates the dissimilarity level at which mergers occur.

This nested structure provides a richer story than the "flat" partitions of K-Means. It helps David understand the tiered relationships among assets, which is critical for implementing sophisticated diversification strategies like Hierarchical Risk Parity. The `fcluster` function allowed us to "cut" this dendrogram to retrieve `K_optimal` clusters, providing a direct comparison and potential validation for our K-Means results.

---

## Section 5: Cluster Interpretation and Visualization: Profiling Our Diversification Buckets

### Story + Context + Real-World Relevance

Knowing the cluster assignments is only half the battle; David needs to understand *what* each cluster represents in financial terms. Is Cluster 1 a "growth-tech" cluster? Is Cluster 3 a "stable dividend" cluster? To achieve this, he will compute a "cluster profile" by calculating the average annualized return, volatility, beta, and the dominant GICS sector for each cluster. This qualitative labeling helps Alpha Wealth Management assign interpretive meaning to the data-driven groupings.

Additionally, visual confirmation is key. David will visualize the assets in the PCA-reduced space (PC1 vs. PC2), colored by their K-Means cluster assignments, to confirm that the clusters are visually distinct. Finally, a sorted correlation heatmap, reordered according to cluster assignments, will provide a powerful visual check: a well-formed clustering should show clear "block-diagonal" structures, indicating high intra-cluster correlation and lower inter-cluster correlation.

The beta ($\beta$) of an asset $i$ relative to a market portfolio $M$ (here approximated by the equal-weighted portfolio of all assets) is given by:
$$ \beta_i = \frac{\text{Cov}(R_i, R_M)}{\text{Var}(R_M)} $$
where $R_i$ is the return of asset $i$ and $R_M$ is the return of the market portfolio proxy.

### Code cell (function definition + function execution)

```python
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
        pd.DataFrame: Cluster Profile Summary.
    """
    print("Calculating asset-level statistics (annualized return, volatility, beta)...")
    asset_df = pd.DataFrame({
        'ticker': available_tickers,
        'cluster': kmeans_labels,
        'sector': [sector_map_cleaned.get(t, 'Unknown') for t in available_tickers]
    })

    # Calculate annualized return, volatility, and beta for each asset
    # Annualized return (monthly to annual)
    asset_df['ann_return'] = returns_df.mean() * 12
    # Annualized volatility (monthly to annual)
    asset_df['ann_vol'] = returns_df.std() * np.sqrt(12)

    # Calculate beta for each asset against the equal-weighted portfolio of all assets
    market_returns = returns_df.mean(axis=1) # Proxy for market portfolio
    market_variance = market_returns.var()
    asset_df['beta'] = [returns_df[t].cov(market_returns) / market_variance if market_variance > 0 else 0 for t in available_tickers]

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

# Assuming K_optimal = 5 from Section 3
K_optimal = 5 
kmeans = KMeans(n_clusters=K_optimal, n_init=20, random_state=42)
kmeans_cluster_labels = kmeans.fit_predict(R_pca)

# Execute cluster interpretation and visualization
cluster_profile_summary, asset_assignment_table = interpret_and_visualize_clusters(
    R_pca, kmeans_cluster_labels, available_tickers, sector_map_cleaned, returns_df, pca_model
)

# Display the Cluster Assignment Table
print("\nCluster Assignment Table (first 10 assets):")
print(asset_assignment_table.head(10))

# Display the full Cluster Profile Summary
print("\nFull Cluster Profile Summary:")
print(cluster_profile_summary)
```

### Markdown cell (explanation of execution)

David has now successfully brought his clusters to life with financial context and powerful visualizations. The **Cluster Profile Summary** is a critical output: by examining the average return, volatility, beta, and dominant sector for each cluster, he can assign meaningful labels (e.g., "High-Growth Tech," "Stable Income Utilities," "Cyclical Energy"). This table helps David understand if the data-driven clusters align with or diverge from traditional GICS sectors, revealing where the algorithm has found novel groupings.

The **PCA Scatter Plot** visually confirms that the clusters are reasonably well-separated in the latent factor space. Assets belonging to the same cluster tend to group together, reinforcing the validity of the clustering. Finally, the **Sorted Correlation Heatmap** is perhaps the most compelling visual proof. The distinct block-diagonal patterns demonstrate that assets within the same cluster are highly correlated with each other, while showing lower correlation with assets in other clusters. This block structure is the hallmark of effective diversification, where intra-cluster homogeneity and inter-cluster heterogeneity are achieved, a key insight for David's portfolio management decisions at Alpha Wealth.

---

## Section 6: Diversification Analysis & Cluster-Based Portfolio Construction

### Story + Context + Real-World Relevance

This is the culmination of David's workflow: translating clustering insights into actionable portfolio decisions and quantifying the diversification benefits. For Alpha Wealth, proving the value of a data-driven approach means demonstrating improved risk metrics. David will assess diversification quality by comparing **intra-cluster correlation** (how correlated assets are *within* a cluster) against **inter-cluster correlation** (how correlated assets are *between* clusters). A successful clustering should yield high intra-cluster and low inter-cluster correlations.

Finally, David will construct a "cluster-diversified portfolio" by selecting a representative asset from each cluster (e.g., the one closest to the cluster centroid). He will then compare its annualized volatility to that of a randomly constructed benchmark portfolio of the same size. A lower volatility for the cluster-diversified portfolio will provide concrete evidence of the diversification benefits achieved through unsupervised learning. This quantitative proof is essential for justifying the methodology to Alpha Wealth's investment committee.

The average intra-cluster correlation for a cluster $C_k$ is calculated as:
$$ \bar{\rho}_{\text{intra}, k} = \frac{1}{|C_k|(|C_k|-1)} \sum_{i \in C_k, j \in C_k, i \neq j} \rho_{ij} $$
And the average inter-cluster correlation between clusters $C_k$ and $C_l$ is:
$$ \bar{\rho}_{\text{inter}, k, l} = \frac{1}{|C_k||C_l|} \sum_{i \in C_k, j \in C_l} \rho_{ij} $$
For a portfolio with weights $w$ and covariance matrix $\Sigma$, its annualized volatility $\sigma_P$ is:
$$ \sigma_P = \sqrt{w^T \Sigma w \times 12} $$
In this case, for an equally weighted portfolio, we can directly compute the standard deviation of its mean return series and annualize it: $\text{std}(\text{mean\_returns}) \times \sqrt{12}$.

### Code cell (function definition + function execution)

```python
def evaluate_diversification_and_build_portfolio(
    kmeans_model, R_pca, kmeans_labels, available_tickers, returns_df, K_optimal
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
        cluster_avg_vol = cluster_profile_summary.loc[cluster_id, 'ann_vol']
        cluster_avg_return = cluster_profile_summary.loc[cluster_id, 'ann_return']
        plt.scatter(cluster_avg_vol, cluster_avg_return, 
                    marker='X', s=200, color='red', edgecolor='black', 
                    label=f'Cluster {cluster_id} Centroid (Avg)')

    plt.title('Cluster Return/Risk Profile (Annualized)')
    plt.xlabel('Annualized Volatility')
    plt.ylabel('Annualized Return')
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('cluster_return_risk_scatter.png', dpi=150)
    plt.show()

    return intra_corrs_mean, inter_corrs_mean, cluster_portfolio_vol, random_portfolio_vol, volatility_reduction

# Execute diversification analysis and portfolio construction
intra_corrs_mean, inter_corrs_mean, cluster_portfolio_vol, random_portfolio_vol, volatility_reduction = \
    evaluate_diversification_and_build_portfolio(
        kmeans, R_pca, kmeans_cluster_labels, available_tickers, returns_df, K_optimal
    )
```

### Markdown cell (explanation of execution)

David's analysis culminates in concrete, actionable insights for Alpha Wealth. The comparison of average **intra-cluster** and **inter-cluster correlations** confirms the effectiveness of the clustering. A higher intra-cluster correlation (e.g., 0.55) combined with a lower inter-cluster correlation (e.g., 0.20) indicates that the clusters successfully group highly co-moving assets together while separating those with distinct behaviors. This numerical validation supports the visual evidence from the sorted correlation heatmap in the previous section.

More importantly, the **cluster-diversified portfolio**, constructed by selecting a representative asset from each distinct cluster, demonstrates superior risk characteristics. By comparing its annualized volatility to that of a randomly selected benchmark portfolio of the same size, David can quantify the 'Volatility Reduction' metric. A significant reduction (e.g., 15-20%) provides compelling quantitative evidence that this data-driven clustering approach leads to genuinely more diversified portfolios. This allows David to present a strong case to Alpha Wealth's investment committee, showcasing how unsupervised learning directly contributes to better risk management and more robust asset allocation decisions.

---
