
# Streamlit Application Specification: Unsupervised Diversification for Alpha Wealth

## 1. Application Overview

### Purpose of the Application

This Streamlit application provides a practical, interactive platform for investment professionals, particularly CFA Charterholders and candidates, to apply unsupervised machine learning techniques (K-Means and Hierarchical Clustering, along with PCA) for enhanced portfolio diversification. The app simulates a real-world workflow for David Chen, a Senior Portfolio Manager at Alpha Wealth Management, to discover data-driven asset groupings based on historical return patterns. It aims to demonstrate how to move beyond traditional asset classifications to uncover latent market structures, interpret these structures financially, and quantify their benefits for constructing more resilient, diversified portfolios.

### High-Level Story Flow of the Application

The application guides the user through a six-step workflow:

1.  **Introduction**: Sets the stage, introducing David Chen's challenge at Alpha Wealth and the goals of the analysis.
2.  **Data Acquisition & Preparation**: Users define their asset universe (S&P 500 stocks) and time horizon. The app then fetches historical data, computes returns, and pre-processes it, forming the foundation for clustering.
3.  **Standardization & PCA**: The data is standardized to ensure fair contribution from all assets. Optional Principal Component Analysis (PCA) is applied to reduce dimensionality and identify underlying market factors, which are then used as features for clustering.
4.  **Optimal K-Means Clustering**: The user explores different numbers of clusters (K) using diagnostic tools like the Elbow Method and Silhouette Analysis to determine an optimal `K`. K-Means clustering is then performed with the chosen `K`.
5.  **Hierarchical Clustering**: The app demonstrates an alternative clustering method, Hierarchical Clustering, visualized via a dendrogram, to reveal nested asset relationships and offer complementary insights.
6.  **Cluster Interpretation & Visualization**: The derived clusters are profiled with financial characteristics (return, volatility, beta, dominant sector). Visualizations (PCA scatter plot, sorted correlation heatmap) help interpret cluster separation and internal cohesion.
7.  **Diversification Analysis & Portfolio Construction**: The final stage quantifies diversification benefits by comparing intra-cluster vs. inter-cluster correlations. A cluster-derived diversified portfolio is constructed and its risk is compared against a random benchmark, highlighting the practical value of the methodology.

This sequential flow allows the learner to understand the rationale and impact of each step in building a robust, data-driven diversification strategy.

---

## 2. Code Requirements

### Imports

```python
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from source import * # Import all functions from the source.py file
```

### `st.session_state` Design

The `st.session_state` is crucial for preserving data and processing results across different "pages" (conditional rendering via sidebar selection) and subsequent button clicks within the Streamlit application.

*   **Initialization (at app start if key not present)**:
    *   `st.session_state.page = "Introduction"` (default page)
    *   `st.session_state.tickers = DEFAULT_TICKERS` (pre-defined list in `source.py`)
    *   `st.session_state.start_date = DEFAULT_START_DATE`
    *   `st.session_state.end_date = DEFAULT_END_DATE`
    *   `st.session_state.return_matrix = None`
    *   `st.session_state.returns_df = None`
    *   `st.session_state.corr_matrix = None`
    *   `st.session_state.sector_map_cleaned = None`
    *   `st.session_state.n_components_pca = 5` (default PCA components)
    *   `st.session_state.R_pca = None`
    *   `st.session_state.pca_model = None`
    *   `st.session_state.k_range_values = range(2, 11)` (default K range for analysis)
    *   `st.session_state.inertias = None`
    *   `st.session_state.silhouettes = None`
    *   `st.session_state.K_optimal = 5` (default optimal K)
    *   `st.session_state.kmeans_cluster_labels = None`
    *   `st.session_state.kmeans_model = None`
    *   `st.session_state.hc_cluster_labels = None`
    *   `st.session_state.cluster_profile_summary = None`
    *   `st.session_state.asset_assignment_table = None`
    *   `st.session_state.intra_corrs_mean = None`
    *   `st.session_state.inter_corrs_mean = None`
    *   `st.session_state.cluster_portfolio_vol = None`
    *   `st.session_state.random_portfolio_vol = None`
    *   `st.session_state.volatility_reduction = None`

*   **Update**:
    *   `st.session_state.tickers`, `start_date`, `end_date`, `n_components_pca`, `K_optimal` are updated directly by Streamlit input widgets on user interaction.
    *   The results of function calls (e.g., `return_matrix`, `R_pca`, `kmeans_cluster_labels`) are stored into `st.session_state` variables after the corresponding processing button is clicked.

*   **Read Across Pages**:
    *   All subsequent pages/steps read their required inputs from `st.session_state` variables that were populated by previous steps. For example, "Standardization & PCA" reads `st.session_state.return_matrix`, and "Optimal K-Means Clustering" reads `st.session_state.R_pca`.

### UI Interactions and Function Calls from `source.py`

#### Sidebar Navigation

```python
st.sidebar.title("Navigation")
page_selection = st.sidebar.selectbox(
    "Go to",
    [
        "Introduction",
        "1. Data Acquisition & Preparation",
        "2. Standardization & PCA",
        "3. Optimal K-Means Clustering",
        "4. Hierarchical Clustering",
        "5. Cluster Interpretation & Visualization",
        "6. Diversification Analysis & Portfolio Construction"
    ],
    key="page_selection"
)
st.session_state.page = page_selection
```

#### Page: Introduction

*   No direct `source.py` function calls. This page is purely informational markdown.

#### Page: 1. Data Acquisition & Preparation

```python
st.header("Section 1: Data Acquisition and Return Matrix Construction")
st.markdown(f"") # Intro markdown block
st.markdown(r"$$ R_t = \frac{{P_t - P_{{t-1}}}}{{P_{{t-1}}}} $$")
st.markdown(r"where $P_t$ is the adjusted close price at time $t$ and $P_{{t-1}}$ is the adjusted close price at time $t-1$.")
st.markdown(r"Furthermore, to anticipate future steps, David needs to compute the correlation matrix. The Pearson correlation coefficient $\rho_{{ij}}$ between the returns of asset $i$ and asset $j$ is given by:")
st.markdown(r"$$ \rho_{{ij}} = \frac{{\text{{Cov}}(R_i, R_j)}}{{\sigma_i \sigma_j}} $$")
st.markdown(r"where $\text{{Cov}}(R_i, R_j)$ is the covariance between the returns of asset $i$ and asset $j$, and $\sigma_i$ and $\sigma_j$ are their respective standard deviations. This matrix is fundamental for understanding asset co-movement and will be central to evaluating cluster quality.")

st.subheader("Configuration")
tickers_input = st.text_area(
    "Enter S&P 500 Tickers (comma-separated)",
    value=", ".join(st.session_state.tickers),
    help="e.g., AAPL, MSFT, GOOG"
)
st.session_state.tickers = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]

col1, col2 = st.columns(2)
with col1:
    st.session_state.start_date = st.date_input("Start Date", value=pd.to_datetime(st.session_state.start_date))
with col2:
    st.session_state.end_date = st.date_input("End Date", value=pd.to_datetime(st.session_state.end_date))

if st.button("Acquire & Prepare Data"):
    with st.spinner("Downloading and preparing data..."):
        # Function call from source.py
        # Need to ensure DEFAULT_TICKERS, DEFAULT_START_DATE, DEFAULT_END_DATE are defined in source.py or app.py
        DEFAULT_TICKERS = [
            'AAPL', 'MSFT', 'GOOG', 'AMZN', 'META',
            'JPM', 'BAC', 'GS', 'MS', 'C',
            'JNJ', 'PFE', 'UNH', 'MRK', 'ABBV',
            'XOM', 'CVX', 'COP', 'SLB', 'EOG',
            'PG', 'KO', 'PEP', 'WMT', 'COST',
            'TSLA', 'GM', 'F', 'NKE', 'SBUX',
            'NEE', 'DUK', 'SO', 'D', 'AEP',
            'AMT', 'PLD', 'CCI', 'SPG', 'EQIX',
            'CAT', 'DE', 'HON', 'MMM', 'GE',
            'LIN', 'APD', 'ECL', 'SHW', 'DD'
        ]
        DEFAULT_START_DATE = '2021-01-01'
        DEFAULT_END_DATE = '2024-01-01'

        temp_return_matrix, temp_returns_df, temp_corr_matrix = acquire_and_prepare_data(
            st.session_state.tickers,
            st.session_state.start_date.strftime('%Y-%m-%d'),
            st.session_state.end_date.strftime('%Y-%m-%d')
        )
        
        if not temp_return_matrix.empty:
            st.session_state.return_matrix = temp_return_matrix
            st.session_state.returns_df = temp_returns_df
            st.session_state.corr_matrix = temp_corr_matrix

            # Re-create sector_map_cleaned to ensure it aligns with available_tickers
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
            available_tickers = st.session_state.return_matrix.index.tolist()
            st.session_state.sector_map_cleaned = {ticker: sector_map.get(ticker, 'Unknown') for ticker in available_tickers}

            st.success("Data acquisition and preparation complete!")
            st.write(f"Return matrix shape: {st.session_state.return_matrix.shape}")
            st.write(f"Correlation matrix shape: {st.session_state.corr_matrix.shape}")

            st.subheader("Return Matrix (first 5 assets, first 5 periods)")
            st.dataframe(st.session_state.return_matrix.iloc[:5, :5])

            st.subheader("Correlation Matrix (first 5x5)")
            st.dataframe(st.session_state.corr_matrix.iloc[:5, :5])
        else:
            st.error("Failed to acquire and prepare data. Please check ticker symbols and date range.")

st.markdown(f"") # Concluding markdown block
```

#### Page: 2. Standardization & PCA

```python
st.header("Section 2: Data Standardization and Dimensionality Reduction: Uncovering Latent Market Factors")
st.markdown(f"") # Intro markdown block
st.markdown(r"The standardization process for each asset's return series $R_i = \{{R_{{i,1}}, R_{{i,2}}, \dots, R_{{i,T}}\}}$ involves transforming it to $Z_i = \{{Z_{{i,1}}, Z_{{i,2}}, \dots, Z_{{i,T}}\}\}$ using:")
st.markdown(r"$$ Z_{{i,t}} = \frac{{R_{{i,t}} - \mu_i}}{{\sigma_i}} $$")
st.markdown(r"where $\mu_i$ is the mean of asset $i$'s returns and $\sigma_i$ is its standard deviation.")
st.markdown(r"PCA then projects these standardized return vectors $\tilde{{r}}_i \in \mathbb{{R}}^T$ (where $\tilde{{r}}_i$ is the entire time series of standardized returns for asset $i$) onto a lower-dimensional space. Given the $N \times T$ standardized return matrix $\mathbf{{\tilde{{R}}}}}$, PCA computes the eigendecomposition of the covariance matrix:")
st.markdown(r"$$ \frac{{1}}{{N-1}} \mathbf{{\tilde{{R}}}}\mathbf{{\tilde{{R}}}}^T = \mathbf{{V}}\mathbf{{\Lambda}}\mathbf{{V}}^T $$")
st.markdown(r"where $\mathbf{{\Lambda}} = \text{{diag}}(\lambda_1, \dots, \lambda_T)$ contains the eigenvalues (variance explained), and $\mathbf{{V}}$ contains the eigenvectors (principal component loadings). The projection of asset $i$ onto the first $d$ principal components is:")
st.markdown(r"$$ \mathbf{{z}}_i = \mathbf{{V}}_{{1:d}}^T \tilde{{\mathbf{{r}}}}_i \in \mathbb{{R}}^d $$")
st.markdown(r"Typically, $\lambda_1$ (first principal component) explains a large portion (40-60%) of total variance and often corresponds to the broad market factor. Subsequent components often capture sector or style rotations.")

st.subheader("Configure PCA")
st.session_state.n_components_pca = st.number_input(
    "Number of Principal Components to Retain (d)",
    min_value=1, max_value=min(20, st.session_state.return_matrix.shape[0] if st.session_state.return_matrix is not None else 1, st.session_state.return_matrix.shape[1] if st.session_state.return_matrix is not None else 1),
    value=min(st.session_state.n_components_pca, st.session_state.return_matrix.shape[0] if st.session_state.return_matrix is not None else 1, st.session_state.return_matrix.shape[1] if st.session_state.return_matrix is not None else 1),
    key="n_components_pca_input"
)

if st.button("Perform Standardization & PCA"):
    if st.session_state.return_matrix is not None and not st.session_state.return_matrix.empty:
        with st.spinner("Standardizing data and performing PCA..."):
            # Ensure matplotlib plots are shown in Streamlit
            plt.close('all') # Close previous plots to prevent display issues
            st.session_state.R_pca, st.session_state.pca_model = standardize_and_pca(
                st.session_state.return_matrix, st.session_state.n_components_pca
            )
            
            if st.session_state.pca_model and len(st.session_state.pca_model.explained_variance_ratio_) > 0:
                st.subheader("PCA Scree Plot: Explained Variance per Component")
                # The function `standardize_and_pca` itself generates and saves the plot.
                # We need to capture the plot from Matplotlib's current figure.
                fig_scree = plt.gcf()
                if fig_scree.get_axes():
                    st.pyplot(fig_scree)
                else:
                    st.warning("PCA Scree plot could not be generated. Ensure enough components were requested and data is valid.")
                plt.close(fig_scree) # Close after showing

                st.success("Data standardization and PCA complete!")
                st.write(f"Shape of PCA-transformed data: {st.session_state.R_pca.shape}")
                st.subheader("First 5 rows of PCA-transformed data:")
                st.dataframe(pd.DataFrame(st.session_state.R_pca[:5], columns=[f"PC{i+1}" for i in range(st.session_state.R_pca.shape[1])]))
            else:
                st.error("PCA could not be performed. Check if enough data points and components are available.")
    else:
        st.warning("Please acquire and prepare data first in Section 1.")

st.markdown(f"") # Concluding markdown block
```

#### Page: 3. Optimal K-Means Clustering

```python
st.header("Section 3: Optimal K-Means Clustering: Identifying Core Diversification Buckets")
st.markdown(f"") # Intro markdown block
st.markdown(r"The K-Means algorithm aims to partition $N$ data points $ \{{\mathbf{{x}}_1, \dots, \mathbf{{x}}_N\}}$ in $\mathbb{{R}}^d$ into $K$ clusters $ \{{C_1, \dots, C_K\}}$ by minimizing the Within-Cluster Sum of Squares (WCSS), also known as inertia. The objective function is:")
st.markdown(r"$$ J = \sum_{{k=1}}^K \sum_{{\mathbf{{x}}_i \in C_k}} ||\mathbf{{x}}_i - \mu_k||^2 $$")
st.markdown(r"where $\mu_k$ is the centroid of cluster $C_k$. The Elbow Method plots $J$ against $K$, looking for a point where the rate of decrease in $J$ significantly slows down, resembling an 'elbow'.")
st.markdown(r"The Silhouette Score measures how similar an object is to its own cluster compared to other clusters. For a data point $i$:")
st.markdown(r"$$ s_i = \frac{{b_i - a_i}}{{\max(a_i, b_i)}} $$")
st.markdown(r"where $a_i$ is the average distance from point $i$ to all other points in its cluster (cohesion), and $b_i$ is the average distance from point $i$ to all points in the nearest other cluster (separation). The score $s_i$ ranges from -1 to 1. Values near 1 indicate well-clustered points, near 0 indicate boundary points, and negative values indicate potential misassignment. The average Silhouette Score across all points is used to evaluate cluster quality, with higher values indicating better-defined clusters.")

st.subheader("Determine Optimal K")
min_k = st.number_input("Minimum K for analysis", min_value=2, max_value=2, value=2, key="min_k_input")
max_k = st.number_input("Maximum K for analysis", min_value=3, max_value=min(15, (st.session_state.R_pca.shape[0] - 1) if st.session_state.R_pca is not None else 2), value=10, key="max_k_input")
st.session_state.k_range_values = range(min_k, max_k + 1)

if st.button("Find Optimal K"):
    if st.session_state.R_pca is not None and st.session_state.R_pca.shape[0] >= min_k:
        with st.spinner("Calculating WCSS and Silhouette Scores for different K values..."):
            plt.close('all') # Close previous plots
            st.session_state.inertias, st.session_state.silhouettes = find_optimal_k(
                st.session_state.R_pca, st.session_state.k_range_values
            )

            if any(not np.isnan(i) for i in st.session_state.inertias):
                st.success("Optimal K analysis complete!")
                st.subheader("Elbow Method and Silhouette Analysis")
                fig_kmeans_k_selection = plt.gcf()
                if fig_kmeans_k_selection.get_axes():
                    st.pyplot(fig_kmeans_k_selection)
                else:
                    st.warning("Elbow and Silhouette plots could not be generated.")
                plt.close(fig_kmeans_k_selection)
            else:
                st.error("Could not perform K-Means evaluation. Check if PCA data is valid and enough samples are available.")
    else:
        st.warning("Please perform Standardization & PCA first in Section 2.")

if st.session_state.inertias is not None and any(not np.isnan(i) for i in st.session_state.inertias):
    st.subheader("Select Optimal K")
    st.session_state.K_optimal = st.number_input(
        "Enter your chosen Optimal K based on the plots:",
        min_value=min_k, max_value=max_k, value=st.session_state.K_optimal, key="optimal_k_input"
    )

    if st.button("Perform K-Means Clustering"):
        if st.session_state.R_pca is not None and st.session_state.R_pca.shape[0] >= st.session_state.K_optimal:
            with st.spinner(f"Performing K-Means clustering with K = {st.session_state.K_optimal}..."):
                st.session_state.kmeans_model = KMeans(n_clusters=st.session_state.K_optimal, n_init=20, random_state=42)
                st.session_state.kmeans_cluster_labels = st.session_state.kmeans_model.fit_predict(st.session_state.R_pca)
                st.success(f"K-Means clustering complete with {st.session_state.K_optimal} clusters!")
                st.write(f"First 10 assets with their K-Means cluster labels: {st.session_state.kmeans_cluster_labels[:10]}")
        else:
            st.error(f"Cannot perform K-Means: Not enough samples ({st.session_state.R_pca.shape[0] if st.session_state.R_pca is not None else 0}) for {st.session_state.K_optimal} clusters.")
else:
    st.info("Run 'Find Optimal K' above to generate plots and enable K selection.")

st.markdown(f"") # Concluding markdown block
```

#### Page: 4. Hierarchical Clustering

```python
st.header("Section 4: Hierarchical Clustering: Uncovering Nested Relationships")
st.markdown(f"") # Intro markdown block
st.markdown(r"Hierarchical clustering builds a tree of clusters by iteratively merging the closest pairs of clusters (agglomerative approach). The 'distance' between assets is crucial here. For financial applications, Euclidean distance in standardized return space is closely related to correlation distance. Specifically, for standardized return vectors $\mathbf{{\tilde{{r}}}}_i$ and $\mathbf{{\tilde{{r}}}}_j$ (zero mean, unit variance), the squared Euclidean distance is:")
st.markdown(r"$$ ||\mathbf{{\tilde{{r}}}}_i - \mathbf{{\tilde{{r}}}}_j||^2 = 2T(1 - \rho_{{ij}}) $$")
st.markdown(r"where $\rho_{{ij}}$ is the sample correlation and $T$ is the number of return periods. This means K-Means on standardized returns is effectively clustering by correlation.")
st.markdown(r"The standard correlation distance used in hierarchical clustering is:")
st.markdown(r"$$ d_{{ij}} = \sqrt{{2(1 - \rho_{{ij}})}} \in [0, 2] $$")
st.markdown(r"where $d_{{ij}}=0$ implies perfect positive correlation and $d_{{ij}}=2$ implies perfect negative correlation.")
st.markdown(r"Ward's linkage method is commonly used for merging clusters. At each step, it merges the pair of clusters $(C_a, C_b)$ that results in the minimum increase in total within-cluster variance, which is equivalent to minimizing the increase in the WCSS. The increase in WCSS when merging $C_a$ and $C_b$ is:")
st.markdown(r"$$ \Delta(C_a, C_b) = \frac{{|C_a||C_b|}}{{|C_a| + |C_b|}} ||\mu_a - \mu_b||^2 $$")
st.markdown(r"where $|C_a|$ and $|C_b|$ are the number of points in clusters $C_a$ and $C_b$, and $\mu_a$ and $\mu_b$ are their centroids. This method tends to produce compact, roughly equal-sized clusters, which is generally desirable for diversification.")


if st.button("Perform Hierarchical Clustering"):
    if st.session_state.returns_df is not None and not st.session_state.returns_df.empty and st.session_state.K_optimal > 0:
        with st.spinner(f"Performing Hierarchical Clustering and generating Dendrogram for {st.session_state.K_optimal} clusters..."):
            plt.close('all') # Close previous plots
            available_tickers_for_hc = st.session_state.returns_df.columns.tolist()
            st.session_state.hc_cluster_labels = perform_hierarchical_clustering(
                st.session_state.returns_df, st.session_state.K_optimal, available_tickers_for_hc, plot_dendrogram=True
            )
            
            if st.session_state.hc_cluster_labels is not None and st.session_state.hc_cluster_labels.size > 0:
                st.success("Hierarchical Clustering complete!")
                st.subheader("Hierarchical Clustering Dendrogram")
                fig_dendrogram = plt.gcf()
                if fig_dendrogram.get_axes():
                    st.pyplot(fig_dendrogram)
                else:
                    st.warning("Dendrogram could not be generated. Ensure enough assets and data are available.")
                plt.close(fig_dendrogram)

                st.write("\nFirst 10 assets with their hierarchical cluster labels:")
                if available_tickers_for_hc:
                    st.dataframe(pd.Series(st.session_state.hc_cluster_labels, index=available_tickers_for_hc).head(10))
                else:
                    st.write("No assets to display hierarchical cluster labels for.")
            else:
                st.error("Hierarchical clustering could not be performed. Check if data is valid and enough assets are available.")
    else:
        st.warning("Please acquire data in Section 1 and select Optimal K in Section 3.")

st.markdown(f"") # Concluding markdown block
```

#### Page: 5. Cluster Interpretation & Visualization

```python
st.header("Section 5: Cluster Interpretation and Visualization: Profiling Our Diversification Buckets")
st.markdown(f"") # Intro markdown block
st.markdown(r"The beta ($\beta$) of an asset $i$ relative to a market portfolio $M$ (here approximated by the equal-weighted portfolio of all assets) is given by:")
st.markdown(r"$$ \beta_i = \frac{{\text{{Cov}}(R_i, R_M)}}{{\text{{Var}}(R_M)}} $$")
st.markdown(r"where $R_i$ is the return of asset $i$ and $R_M$ is the return of the market portfolio proxy.")

if st.button("Interpret & Visualize Clusters"):
    if (st.session_state.R_pca is not None and st.session_state.kmeans_cluster_labels is not None and
            st.session_state.returns_df is not None and st.session_state.pca_model is not None and
            st.session_state.sector_map_cleaned is not None and not st.session_state.returns_df.empty):
        with st.spinner("Calculating cluster profiles and generating visualizations..."):
            plt.close('all') # Close previous plots
            available_tickers = st.session_state.returns_df.columns.tolist()

            st.session_state.cluster_profile_summary, st.session_state.asset_assignment_table = interpret_and_visualize_clusters(
                st.session_state.R_pca,
                st.session_state.kmeans_cluster_labels,
                available_tickers,
                st.session_state.sector_map_cleaned,
                st.session_state.returns_df,
                st.session_state.pca_model
            )

            if not st.session_state.cluster_profile_summary.empty:
                st.success("Cluster interpretation and visualizations complete!")
                
                st.subheader("Cluster Profile Summary")
                st.dataframe(st.session_state.cluster_profile_summary)

                st.subheader("Cluster Assignment Table (First 10 Assets)")
                st.dataframe(st.session_state.asset_assignment_table.head(10))

                st.subheader("Stock Clusters in PCA Space")
                # PCA scatter plot is generated by interpret_and_visualize_clusters
                fig_pca_scatter = plt.gcf()
                if fig_pca_scatter.get_axes():
                    st.pyplot(fig_pca_scatter)
                else:
                    st.warning("PCA Cluster Scatter plot could not be generated.")
                plt.close(fig_pca_scatter)

                st.subheader("Correlation Matrix Sorted by Cluster Assignment")
                # Sorted correlation heatmap is generated by interpret_and_visualize_clusters
                fig_corr_heatmap = plt.gcf()
                if fig_corr_heatmap.get_axes():
                    st.pyplot(fig_corr_heatmap)
                else:
                    st.warning("Sorted Correlation Heatmap could not be generated.")
                plt.close(fig_corr_heatmap)

            else:
                st.error("Cluster interpretation and visualization could not be performed. Check if data is valid and clusters were formed.")
    else:
        st.warning("Please complete steps 1, 2, and 3 first to generate PCA data and K-Means labels.")

st.markdown(f"") # Concluding markdown block
```

#### Page: 6. Diversification Analysis & Portfolio Construction

```python
st.header("Section 6: Diversification Analysis & Cluster-Based Portfolio Construction")
st.markdown(f"") # Intro markdown block
st.markdown(r"The average intra-cluster correlation for a cluster $C_k$ is calculated as:")
st.markdown(r"$$ \bar{{\rho}}_{{\text{{intra}}, k}} = \frac{{1}}{{|C_k|(|C_k|-1)}} \sum_{{i \in C_k, j \in C_k, i \neq j}} \rho_{{ij}} $$")
st.markdown(r"And the average inter-cluster correlation between clusters $C_k$ and $C_l$ is:")
st.markdown(r"$$ \bar{{\rho}}_{{\text{{inter}}, k, l}} = \frac{{1}}{{|C_k||C_l|}} \sum_{{i \in C_k, j \in C_l}} \rho_{{ij}} $$")
st.markdown(r"For a portfolio with weights $w$ and covariance matrix $\Sigma$, its annualized volatility $\sigma_P$ is:")
st.markdown(r"$$ \sigma_P = \sqrt{{w^T \Sigma w \times 12}} $$")
st.markdown(r"In this case, for an equally weighted portfolio, we can directly compute the standard deviation of its mean return series and annualize it: $\text{{std}}(\text{{mean\_returns}}) \times \sqrt{{12}}$.")


if st.button("Evaluate Diversification & Build Portfolio"):
    if (st.session_state.kmeans_model is not None and st.session_state.R_pca is not None and
            st.session_state.kmeans_cluster_labels is not None and st.session_state.returns_df is not None and
            st.session_state.K_optimal is not None and st.session_state.cluster_profile_summary is not None and
            st.session_state.asset_assignment_table is not None and not st.session_state.returns_df.empty):
        
        with st.spinner("Analyzing diversification and constructing portfolios..."):
            plt.close('all') # Close previous plots
            available_tickers = st.session_state.returns_df.columns.tolist()

            (st.session_state.intra_corrs_mean, st.session_state.inter_corrs_mean,
             st.session_state.cluster_portfolio_vol, st.session_state.random_portfolio_vol,
             st.session_state.volatility_reduction) = evaluate_diversification_and_build_portfolio(
                st.session_state.kmeans_model,
                st.session_state.R_pca,
                st.session_state.kmeans_cluster_labels,
                available_tickers,
                st.session_state.returns_df,
                st.session_state.K_optimal,
                st.session_state.cluster_profile_summary # Pass this for V6 plotting
            )
            
            if st.session_state.intra_corrs_mean is not None:
                st.success("Diversification analysis and portfolio construction complete!")

                st.subheader("Diversification Metrics")
                st.metric("Average Intra-Cluster Correlation", f"{st.session_state.intra_corrs_mean:.3f}")
                st.metric("Average Inter-Cluster Correlation", f"{st.session_state.inter_corrs_mean:.3f}")
                
                diversification_ratio = st.session_state.intra_corrs_mean / st.session_state.inter_corrs_mean if st.session_state.inter_corrs_mean != 0 else float('inf')
                st.metric("Diversification Ratio (Intra/Inter)", f"{diversification_ratio:.2f} (Target: >> 1)")

                st.subheader("Portfolio Performance Comparison")
                st.metric("Cluster-Diversified Portfolio Annualized Volatility", f"{st.session_state.cluster_portfolio_vol:.2%}")
                st.metric("Random Benchmark Portfolio Annualized Volatility", f"{st.session_state.random_portfolio_vol:.2%}")
                st.metric("Volatility Reduction (Cluster vs. Random)", f"{st.session_state.volatility_reduction:.1f}% (Target: > 15%)")

                st.subheader("Cluster Return/Risk Profile (Annualized)")
                # Cluster Return/Risk Scatter Plot is generated within evaluate_diversification_and_build_portfolio
                fig_return_risk_scatter = plt.gcf()
                if fig_return_risk_scatter.get_axes():
                    st.pyplot(fig_return_risk_scatter)
                else:
                    st.warning("Cluster Return/Risk Scatter plot could not be generated.")
                plt.close(fig_return_risk_scatter)

            else:
                st.error("Diversification analysis could not be performed. Check if all previous steps are complete and data is valid.")
    else:
        st.warning("Please complete all previous steps (1, 2, 3, and 5) to generate necessary data and cluster results.")

st.markdown(f"") # Concluding markdown block
```

### Markdown Content

#### Introduction Page

```python
st.title("Clustering for Asset Allocation: Unsupervised Diversification for Alpha Wealth")

st.markdown("""
## Introduction: Optimizing Diversification at Alpha Wealth Management

Mr. David Chen, a seasoned CFA Charterholder and Senior Portfolio Manager at Alpha Wealth Management, faces a perennial challenge: how to achieve truly robust portfolio diversification in a dynamic market. Alpha Wealth prides itself on delivering superior risk-adjusted returns, and David knows that relying solely on traditional asset classifications (like GICS sectors or style boxes) often falls short. These predefined categories can be stale, miss subtle cross-sector correlations, and oversimplify the complex interplay between assets.

David's goal is to discover data-driven asset groupings that reveal the genuine co-movement patterns or fundamental characteristics of securities. This approach aims to move beyond nominal diversification to a more empirical understanding of his asset universe, leading to more resilient portfolios. This application will guide David through a real-world workflow to apply unsupervised machine learning techniques, specifically K-Means and Hierarchical Clustering, to uncover the latent structure within a universe of S&P 500 stocks. By identifying these "natural" clusters, David can make more informed allocation decisions, ultimately enhancing portfolio risk management and delivering better outcomes for Alpha Wealth's clients.

---

### CFA Curriculum Connection

**Portfolio Management (CFA Levels I–III)**: This case directly operationalizes the CFA curriculum's diversification concepts. At Level I, candidates learn that correlation drives portfolio risk; at Level II, they study multi-factor models and alternative risk decomposition; at Level III, they implement asset allocation strategies. Clustering provides a quantitative tool for all three levels—discovering the correlation structure, mapping it to latent factors, and using it to construct diversified allocations.

**Quantitative Methods**: PCA (principal component analysis) appears in the CFA Level II quantitative methods curriculum as a dimensionality reduction technique. Here it serves a dual purpose: reducing the feature space before clustering and interpreting principal components as latent market factors.

**Connection to Hierarchical Risk Parity (HRP)**: The Lopez de Prado (2020) HRP method—now widely adopted by quantitative allocators—uses hierarchical clustering of the correlation matrix as its core step. This case study introduces the foundational technique underlying HRP, which participants can extend in D5-T1-C2 (AI-Driven Portfolio Optimization).
""")
```

#### Page 1: Data Acquisition & Preparation

```python
st.header("Section 1: Data Acquisition and Return Matrix Construction")
st.markdown("""
David's first step is to gather the necessary raw material: historical price data for his chosen universe of assets. He has decided to focus on a diverse selection of S&P 500 stocks, aiming to capture a broad representation of the market. Accurate and consistent historical adjusted close prices are crucial, as these will be used to compute monthly returns, which form the basis for understanding asset co-movement. David understands that the integrity of his raw data directly impacts the reliability of any subsequent clustering analysis. He will then transform this data into an `N x T` return matrix, where `N` is the number of assets and `T` is the number of time periods. This structure is essential for feeding into clustering algorithms.
""")
st.markdown(r"$$ R_t = \frac{{P_t - P_{{t-1}}}}{{P_{{t-1}}}} $$")
st.markdown(r"where $P_t$ is the adjusted close price at time $t$ and $P_{{t-1}}$ is the adjusted close price at time $t-1$.")
st.markdown(r"Furthermore, to anticipate future steps, David needs to compute the correlation matrix. The Pearson correlation coefficient $\rho_{{ij}}$ between the returns of asset $i$ and asset $j$ is given by:")
st.markdown(r"$$ \rho_{{ij}} = \frac{{\text{{Cov}}(R_i, R_j)}}{{\sigma_i \sigma_j}} $$")
st.markdown(r"where $\text{{Cov}}(R_i, R_j)$ is the covariance between the returns of asset $i$ and asset $j$, and $\sigma_i$ and $\sigma_j$ are their respective standard deviations. This matrix is fundamental for understanding asset co-movement and will be central to evaluating cluster quality.")
# ... (input widgets and function calls as defined above)
st.markdown("""
The data acquisition process successfully downloaded monthly adjusted close prices for our selected S&P 500 universe. After handling missing values by dropping assets with significant gaps and forward-filling remaining smaller gaps, we computed the monthly percentage returns. The resulting `return_matrix` is transposed to have assets as rows and time periods as columns, which is the standard input format for many clustering algorithms (`N` samples, `T` features). The `corr_matrix` shows the pairwise correlation between assets. David can now confirm that his foundational data is robust and correctly formatted, ready for the next stage of pre-processing. The `return_matrix` shape (assets, time periods) is critical for `StandardScaler` and `PCA` to operate correctly across each asset's time series.
""")
```

#### Page 2: Standardization & PCA

```python
st.header("Section 2: Data Standardization and Dimensionality Reduction: Uncovering Latent Market Factors")
st.markdown("""
David understands that raw return data can be problematic for clustering. Assets with higher volatility (like a growth stock) might dominate distance calculations purely due to larger return magnitudes, obscuring genuine co-movement patterns. Therefore, standardizing the return series to zero mean and unit variance is crucial. This ensures that all assets contribute equally to the clustering process, focusing on their *patterns* of movement rather than their absolute scale.

Furthermore, financial markets are often driven by a smaller set of underlying "factors" rather than thousands of independent events. Principal Component Analysis (PCA) is a powerful technique that helps David to:
1.  **Denoise the data**: By focusing on the most significant principal components, PCA effectively filters out idiosyncratic noise.
2.  **Identify latent market factors**: The first few principal components often capture broad market effects (e.g., market factor, sector/style factors), providing a more robust and interpretable feature space for clustering. This directly aligns with the CFA curriculum's emphasis on multi-factor models.
""")
st.markdown(r"The standardization process for each asset's return series $R_i = \{{R_{{i,1}}, R_{{i,2}}, \dots, R_{{i,T}}\}}$ involves transforming it to $Z_i = \{{Z_{{i,1}}, Z_{{i,2}}, \dots, Z_{{i,T}}\}\}$ using:")
st.markdown(r"$$ Z_{{i,t}} = \frac{{R_{{i,t}} - \mu_i}}{{\sigma_i}} $$")
st.markdown(r"where $\mu_i$ is the mean of asset $i$'s returns and $\sigma_i$ is its standard deviation.")
st.markdown(r"PCA then projects these standardized return vectors $\tilde{{r}}_i \in \mathbb{{R}}^T$ (where $\tilde{{r}}_i$ is the entire time series of standardized returns for asset $i$) onto a lower-dimensional space. Given the $N \times T$ standardized return matrix $\mathbf{{\tilde{{R}}}}}$, PCA computes the eigendecomposition of the covariance matrix:")
st.markdown(r"$$ \frac{{1}}{{N-1}} \mathbf{{\tilde{{R}}}}\mathbf{{\tilde{{R}}}}^T = \mathbf{{V}}\mathbf{{\Lambda}}\mathbf{{V}}^T $$")
st.markdown(r"where $\mathbf{{\Lambda}} = \text{{diag}}(\lambda_1, \dots, \lambda_T)$ contains the eigenvalues (variance explained), and $\mathbf{{V}}$ contains the eigenvectors (principal component loadings). The projection of asset $i$ onto the first $d$ principal components is:")
st.markdown(r"$$ \mathbf{{z}}_i = \mathbf{{V}}_{{1:d}}^T \tilde{{\mathbf{{r}}}}_i \in \mathbb{{R}}^d $$")
st.markdown(r"Typically, $\lambda_1$ (first principal component) explains a large portion (40-60%) of total variance and often corresponds to the broad market factor. Subsequent components often capture sector or style rotations.")
# ... (input widgets and function calls as defined above)
st.markdown("""
The standardization process successfully scaled each asset's return series to have a mean of zero and unit variance. This crucial step prevents assets with inherently higher volatility from unduly influencing the clustering algorithms. Subsequently, PCA transformed the standardized returns into a lower-dimensional space of 5 principal components.

The PCA Scree Plot visually confirms the explained variance. For instance, PC1 often captures a significant portion (e.g., 40-60%) of the total variance, representing the broad market factor. PC2 and PC3 might capture sector or style-specific factors. By reducing dimensionality, David has effectively denoised the data, focusing on the most relevant underlying drivers of asset co-movement. This new, compressed representation (`R_pca`) is a more robust input for clustering, ensuring the clusters are based on fundamental patterns rather than noise or scale differences.
""")
```

#### Page 3: Optimal K-Means Clustering

```python
st.header("Section 3: Optimal K-Means Clustering: Identifying Core Diversification Buckets")
st.markdown("""
David needs to group the assets into distinct "buckets" or clusters based on their underlying behavior. K-Means clustering is a straightforward algorithm for this, but a critical decision is determining the optimal number of clusters, $K$. An incorrect $K$ can lead to either over-segmentation (too many small, insignificant clusters) or under-segmentation (too few, overly broad clusters). To guide this choice, David will use two diagnostic tools: the Elbow Method and Silhouette Analysis. These methods provide quantitative insights into cluster quality at different $K$ values, helping him select a `K_optimal` that genuinely reflects the inherent grouping structure in the market.
""")
st.markdown(r"The K-Means algorithm aims to partition $N$ data points $ \{{\mathbf{{x}}_1, \dots, \mathbf{{x}}_N\}}$ in $\mathbb{{R}}^d$ into $K$ clusters $ \{{C_1, \dots, C_K\}}$ by minimizing the Within-Cluster Sum of Squares (WCSS), also known as inertia. The objective function is:")
st.markdown(r"$$ J = \sum_{{k=1}}^K \sum_{{\mathbf{{x}}_i \in C_k}} ||\mathbf{{x}}_i - \mu_k||^2 $$")
st.markdown(r"where $\mu_k$ is the centroid of cluster $C_k$. The Elbow Method plots $J$ against $K$, looking for a point where the rate of decrease in $J$ significantly slows down, resembling an 'elbow'.")
st.markdown(r"The Silhouette Score measures how similar an object is to its own cluster compared to other clusters. For a data point $i$:")
st.markdown(r"$$ s_i = \frac{{b_i - a_i}}{{\max(a_i, b_i)}} $$")
st.markdown(r"where $a_i$ is the average distance from point $i$ to all other points in its cluster (cohesion), and $b_i$ is the average distance from point $i$ to all points in the nearest other cluster (separation). The score $s_i$ ranges from -1 to 1. Values near 1 indicate well-clustered points, near 0 indicate boundary points, and negative values indicate potential misassignment. The average Silhouette Score across all points is used to evaluate cluster quality, with higher values indicating better-defined clusters.")
# ... (input widgets and function calls as defined above)
st.markdown("""
David has now generated the Elbow and Silhouette plots. The **Elbow Method** plot shows WCSS decreasing as $K$ increases, but the "elbow" where the rate of decrease significantly lessens typically indicates a good $K$. The **Silhouette Analysis** plot provides another perspective; David looks for a peak in the average Silhouette Score, suggesting well-separated and compact clusters.

Based on visual inspection of these plots, David observes that a `K_optimal` around 5 or 6 seems appropriate, as the Elbow plot shows a distinct bend and the Silhouette score reaches a reasonable peak in that range. For the purpose of this exercise, David decides to proceed with `K_optimal = 5`, balancing interpretability with statistical goodness-of-fit. This data-driven selection of `K` is a significant improvement over arbitrary groupings, ensuring his clusters are statistically meaningful.
""")
```

#### Page 4: Hierarchical Clustering

```python
st.header("Section 4: Hierarchical Clustering: Uncovering Nested Relationships")
st.markdown("""
While K-Means provides flat, distinct clusters, David recognizes that asset relationships might be hierarchical. Some assets are very similar (e.g., energy stocks), forming sub-groups, which then combine with other sub-groups (e.g., materials stocks) at a higher level of dissimilarity to form broader categories. Hierarchical clustering, visualized through a dendrogram, offers a complementary view by revealing this nested structure. This is particularly valuable for understanding how assets merge from individual entities to broader classifications and is the foundational technique for advanced portfolio optimization methods like Hierarchical Risk Parity (HRP).
""")
st.markdown(r"Hierarchical clustering builds a tree of clusters by iteratively merging the closest pairs of clusters (agglomerative approach). The 'distance' between assets is crucial here. For financial applications, Euclidean distance in standardized return space is closely related to correlation distance. Specifically, for standardized return vectors $\mathbf{{\tilde{{r}}}}_i$ and $\mathbf{{\tilde{{r}}}}_j$ (zero mean, unit variance), the squared Euclidean distance is:")
st.markdown(r"$$ ||\mathbf{{\tilde{{r}}}}_i - \mathbf{{\tilde{{r}}}}_j||^2 = 2T(1 - \rho_{{ij}}) $$")
st.markdown(r"where $\rho_{{ij}}$ is the sample correlation and $T$ is the number of return periods. This means K-Means on standardized returns is effectively clustering by correlation.")
st.markdown(r"The standard correlation distance used in hierarchical clustering is:")
st.markdown(r"$$ d_{{ij}} = \sqrt{{2(1 - \rho_{{ij}})}} \in [0, 2] $$")
st.markdown(r"where $d_{{ij}}=0$ implies perfect positive correlation and $d_{{ij}}=2$ implies perfect negative correlation.")
st.markdown(r"Ward's linkage method is commonly used for merging clusters. At each step, it merges the pair of clusters $(C_a, C_b)$ that results in the minimum increase in total within-cluster variance, which is equivalent to minimizing the increase in the WCSS. The increase in WCSS when merging $C_a$ and $C_b$ is:")
st.markdown(r"$$ \Delta(C_a, C_b) = \frac{{|C_a||C_b|}}{{|C_a| + |C_b|}} ||\mu_a - \mu_b||^2 $$")
st.markdown(r"where $|C_a|$ and $|C_b|$ are the number of points in clusters $C_a$ and $C_b$, and $\mu_a$ and $\mu_b$ are their centroids. This method tends to produce compact, roughly equal-sized clusters, which is generally desirable for diversification.")
# ... (input widgets and function calls as defined above)
st.markdown("""
The dendrogram visually represents the nested hierarchy of asset similarities. By reading it from bottom-up, David can see which assets merge first (indicating high similarity, like energy peers `XOM` and `CVX`), and how these sub-clusters then combine into broader groups (e.g., energy merging with materials). The vertical axis, representing correlation-based distance, indicates the dissimilarity level at which mergers occur.

This nested structure provides a richer story than the "flat" partitions of K-Means. It helps David understand the tiered relationships among assets, which is critical for implementing sophisticated diversification strategies like Hierarchical Risk Parity. The `fcluster` function allowed us to "cut" this dendrogram to retrieve `K_optimal` clusters, providing a direct comparison and potential validation for our K-Means results.
""")
```

#### Page 5: Cluster Interpretation & Visualization

```python
st.header("Section 5: Cluster Interpretation and Visualization: Profiling Our Diversification Buckets")
st.markdown("""
Knowing the cluster assignments is only half the battle; David needs to understand *what* each cluster represents in financial terms. Is Cluster 1 a "growth-tech" cluster? Is Cluster 3 a "stable dividend" cluster? To achieve this, he will compute a "cluster profile" by calculating the average annualized return, volatility, beta, and the dominant GICS sector for each cluster. This qualitative labeling helps Alpha Wealth Management assign interpretive meaning to the data-driven groupings.

Additionally, visual confirmation is key. David will visualize the assets in the PCA-reduced space (PC1 vs. PC2), colored by their K-Means cluster assignments, to confirm that the clusters are visually distinct. Finally, a sorted correlation heatmap, reordered according to cluster assignments, will provide a powerful visual check: a well-formed clustering should show clear "block-diagonal" structures, indicating high intra-cluster correlation and lower inter-cluster correlation.
""")
st.markdown(r"The beta ($\beta$) of an asset $i$ relative to a market portfolio $M$ (here approximated by the equal-weighted portfolio of all assets) is given by:")
st.markdown(r"$$ \beta_i = \frac{{\text{{Cov}}(R_i, R_M)}}{{\text{{Var}}(R_M)}} $$")
st.markdown(r"where $R_i$ is the return of asset $i$ and $R_M$ is the return of the market portfolio proxy.")
# ... (input widgets and function calls as defined above)
st.markdown("""
David has now successfully brought his clusters to life with financial context and powerful visualizations. The **Cluster Profile Summary** is a critical output: by examining the average return, volatility, beta, and dominant sector for each cluster, he can assign meaningful labels (e.g., "High-Growth Tech," "Stable Income Utilities," "Cyclical Energy"). This table helps David understand if the data-driven clusters align with or diverge from traditional GICS sectors, revealing where the algorithm has found novel groupings.

The **PCA Scatter Plot** visually confirms that the clusters are reasonably well-separated in the latent factor space. Assets belonging to the same cluster tend to group together, reinforcing the validity of the clustering. Finally, the **Sorted Correlation Heatmap** is perhaps the most compelling visual proof. The distinct block-diagonal patterns demonstrate that assets within the same cluster are highly correlated with each other, while showing lower correlation with assets in other clusters. This block structure is the hallmark of effective diversification, where intra-cluster homogeneity and inter-cluster heterogeneity are achieved, a key insight for David's portfolio management decisions at Alpha Wealth.
""")
```

#### Page 6: Diversification Analysis & Portfolio Construction

```python
st.header("Section 6: Diversification Analysis & Cluster-Based Portfolio Construction")
st.markdown("""
This is the culmination of David's workflow: translating clustering insights into actionable portfolio decisions and quantifying the diversification benefits. For Alpha Wealth, proving the value of a data-driven approach means demonstrating improved risk metrics. David will assess diversification quality by comparing **intra-cluster correlation** (how correlated assets are *within* a cluster) against **inter-cluster correlation** (how correlated assets are *between* clusters). A successful clustering should yield high intra-cluster and low inter-cluster correlations.

Finally, David will construct a "cluster-diversified portfolio" by selecting a representative asset from each cluster (e.g., the one closest to the cluster centroid). He will then compare its annualized volatility to that of a randomly constructed benchmark portfolio of the same size. A lower volatility for the cluster-diversified portfolio will provide concrete evidence of the diversification benefits achieved through unsupervised learning. This quantitative proof is essential for justifying the methodology to Alpha Wealth's investment committee.
""")
st.markdown(r"The average intra-cluster correlation for a cluster $C_k$ is calculated as:")
st.markdown(r"$$ \bar{{\rho}}_{{\text{{intra}}, k}} = \frac{{1}}{{|C_k|(|C_k|-1)}} \sum_{{i \in C_k, j \in C_k, i \neq j}} \rho_{{ij}} $$")
st.markdown(r"And the average inter-cluster correlation between clusters $C_k$ and $C_l$ is:")
st.markdown(r"$$ \bar{{\rho}}_{{\text{{inter}}, k, l}} = \frac{{1}}{{|C_k||C_l|}} \sum_{{i \in C_k, j \in C_l}} \rho_{{ij}} $$")
st.markdown(r"For a portfolio with weights $w$ and covariance matrix $\Sigma$, its annualized volatility $\sigma_P$ is:")
st.markdown(r"$$ \sigma_P = \sqrt{{w^T \Sigma w \times 12}} $$")
st.markdown(r"In this case, for an equally weighted portfolio, we can directly compute the standard deviation of its mean return series and annualize it: $\text{{std}}(\text{{mean\_returns}}) \times \sqrt{{12}}$.")
# ... (input widgets and function calls as defined above)
st.markdown("""
David's analysis culminates in concrete, actionable insights for Alpha Wealth. The comparison of average **intra-cluster** and **inter-cluster correlations** confirms the effectiveness of the clustering. A higher intra-cluster correlation (e.g., 0.55) combined with a lower inter-cluster correlation (e.g., 0.20) indicates that the clusters successfully group highly co-moving assets together while separating those with distinct behaviors. This numerical validation supports the visual evidence from the sorted correlation heatmap in the previous section.

More importantly, the **cluster-diversified portfolio**, constructed by selecting a representative asset from each distinct cluster, demonstrates superior risk characteristics. By comparing its annualized volatility to that of a randomly selected benchmark portfolio of the same size, David can quantify the 'Volatility Reduction' metric. A significant reduction (e.g., 15-20%) provides compelling quantitative evidence that this data-driven clustering approach leads to genuinely more diversified portfolios. This allows David to present a strong case to Alpha Wealth's investment committee, showcasing how unsupervised learning directly contributes to better risk management and more robust asset allocation decisions.
""")
```

