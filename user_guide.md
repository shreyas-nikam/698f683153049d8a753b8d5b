id: 698f683153049d8a753b8d5b_user_guide
summary: Lab 3: Clustering for Asset Allocation User Guide
feedback link: https://docs.google.com/forms/d/e/1FAIpQLSfWkOK-in_bMMoHSZfcIvAeO58PAH9wrDqcxnJABHaxiDqhSA/viewform?usp=sf_link
environments: Web
status: Published
# QuLab: Clustering for Asset Allocation

## Introduction: Optimizing Diversification at Alpha Wealth Management
Duration: 0:05:00

Mr. David Chen, a seasoned CFA Charterholder and Senior Portfolio Manager at Alpha Wealth Management, faces a perennial challenge: how to achieve truly robust portfolio diversification in a dynamic market. Alpha Wealth prides itself on delivering superior risk-adjusted returns, and David knows that relying solely on traditional asset classifications (like GICS sectors or style boxes) often falls short. These predefined categories can be stale, miss subtle cross-sector correlations, and oversimplify the complex interplay between assets.

David's goal is to discover data-driven asset groupings that reveal the genuine co-movement patterns or fundamental characteristics of securities. This approach aims to move beyond nominal diversification to a more empirical understanding of his asset universe, leading to more resilient portfolios. This application will guide David through a real-world workflow to apply unsupervised machine learning techniques, specifically **K-Means** and **Hierarchical Clustering**, to uncover the latent structure within a universe of S&P 500 stocks. By identifying these "natural" clusters, David can make more informed allocation decisions, ultimately enhancing portfolio risk management and delivering better outcomes for Alpha Wealth's clients.

<aside class="positive">
<b>Key Concept: Unsupervised Learning for Diversification</b>
Traditional diversification relies on predefined categories. This codelab uses unsupervised learning (clustering) to discover hidden, data-driven relationships between assets, aiming for more robust and empirical diversification.
</aside>

### CFA Curriculum Connection

*   **Portfolio Management (CFA Levels I–III)**: This case directly operationalizes the CFA curriculum's diversification concepts. At Level I, candidates learn that correlation drives portfolio risk; at Level II, they study multi-factor models and alternative risk decomposition; at Level III, they implement asset allocation strategies. Clustering provides a quantitative tool for all three levels—discovering the correlation structure, mapping it to latent factors, and using it to construct diversified allocations.
*   **Quantitative Methods**: PCA (principal component analysis) appears in the CFA Level II quantitative methods curriculum as a dimensionality reduction technique. Here it serves a dual purpose: reducing the feature space before clustering and interpreting principal components as latent market factors.
*   **Connection to Hierarchical Risk Parity (HRP)**: The Lopez de Prado (2020) HRP method—now widely adopted by quantitative allocators—uses hierarchical clustering of the correlation matrix as its core step. This case study introduces the foundational technique underlying HRP, which participants can extend in D5-T1-C2 (AI-Driven Portfolio Optimization).

## 1. Data Acquisition & Preparation
Duration: 0:07:00

David's first step is to gather the necessary raw material: historical price data for his chosen universe of assets. He has decided to focus on a diverse selection of S&P 500 stocks, aiming to capture a broad representation of the market. Accurate and consistent historical adjusted close prices are crucial, as these will be used to compute monthly returns, which form the basis for understanding asset co-movement. David understands that the integrity of his raw data directly impacts the reliability of any subsequent clustering analysis. He will then transform this data into an `N x T` return matrix, where `N` is the number of assets and `T` is the number of time periods. This structure is essential for feeding into clustering algorithms.

The return for an asset at time $t$, $R_t$, is calculated as the percentage change from the previous period's price $P_{t-1}$ to the current period's price $P_t$:
$$ R_t = \frac{{P_t - P_{{t-1}}}}{{P_{{t-1}}}} $$

Furthermore, to anticipate future steps, David needs to compute the correlation matrix. The Pearson correlation coefficient $\rho_{{ij}}$ between the returns of asset $i$ and asset $j$ measures the linear relationship between their movements and is given by:
$$ \rho_{{ij}} = \frac{{\text{{Cov}}(R_i, R_j)}}{{\sigma_i \sigma_j}} $$
where $\text{{Cov}}(R_i, R_j)$ is the covariance between the returns of asset $i$ and asset $j$, and $\sigma_i$ and $\sigma_j$ are their respective standard deviations. This matrix is fundamental for understanding asset co-movement and will be central to evaluating cluster quality.

### Configuration

1.  **Enter S&P 500 Tickers**: Use the provided text area to input a comma-separated list of stock tickers. The default list provides a good starting point. You can modify this list to include or exclude specific stocks.
2.  **Start Date** and **End Date**: Select the historical period for which you want to acquire data.
3.  Click the **"Acquire & Prepare Data"** button.

<aside class="positive">
<b>Tip: Data Integrity</b>
Ensure your chosen date range is appropriate and that the tickers are valid. Inaccurate or missing data can significantly impact the quality of your clustering results. The application handles some common data issues like dropping assets with too many missing values.
</aside>

Once the data is processed, you will see confirmation messages and the shapes of the generated `Return Matrix` and `Correlation Matrix`. You'll also see the first few rows/columns of these matrices.

The data acquisition process successfully downloaded monthly adjusted close prices for our selected S&P 500 universe. After handling missing values by dropping assets with significant gaps and forward-filling remaining smaller gaps, we computed the monthly percentage returns. The resulting `return_matrix` is transposed to have assets as rows and time periods as columns, which is the standard input format for many clustering algorithms (`N` samples, `T` features). The `corr_matrix` shows the pairwise correlation between assets. David can now confirm that his foundational data is robust and correctly formatted, ready for the next stage of pre-processing. The `return_matrix` shape (assets, time periods) is critical for `StandardScaler` and `PCA` to operate correctly across each asset's time series.

## 2. Standardization & PCA
Duration: 0:08:00

David understands that raw return data can be problematic for clustering. Assets with higher volatility (like a growth stock) might dominate distance calculations purely due to larger return magnitudes, obscuring genuine co-movement patterns. Therefore, **standardizing** the return series to zero mean and unit variance is crucial. This ensures that all assets contribute equally to the clustering process, focusing on their *patterns* of movement rather than their absolute scale.

Furthermore, financial markets are often driven by a smaller set of underlying "factors" rather than thousands of independent events. **Principal Component Analysis (PCA)** is a powerful technique that helps David to:
1.  **Denoise the data**: By focusing on the most significant principal components, PCA effectively filters out idiosyncratic noise.
2.  **Identify latent market factors**: The first few principal components often capture broad market effects (e.g., market factor, sector/style factors), providing a more robust and interpretable feature space for clustering. This directly aligns with the CFA curriculum's emphasis on multi-factor models.

The standardization process for each asset's return series $R_i = \{{R_{{i,1}}, R_{{i,2}}, \dots, R_{{i,T}}\}}$ involves transforming it to $Z_i = \{{Z_{{i,1}}, Z_{{i,2}}, \dots, Z_{{i,T}}\}\}$ using:
$$ Z_{{i,t}} = \frac{{R_{{i,t}} - \mu_i}}{{\sigma_i}} $$
where $\mu_i$ is the mean of asset $i$'s returns and $\sigma_i$ is its standard deviation.

PCA then projects these standardized return vectors $\tilde{{r}}_i \in \mathbb{{R}}^T$ (where $\tilde{{r}}_i$ is the entire time series of standardized returns for asset $i$) onto a lower-dimensional space. Given the $N \times T$ standardized return matrix $\mathbf{{\tilde{{R}}}}{}$ (assets as rows, time periods as columns), PCA computes the eigendecomposition of its covariance matrix to find principal components. The projection of asset $i$ onto the first $d$ principal components is a new vector $\mathbf{{z}}_i \in \mathbb{{R}}^d$.
Typically, the first principal component explains a large portion (40-60%) of total variance and often corresponds to the broad market factor. Subsequent components often capture sector or style rotations.

### Configure PCA

1.  **Number of Principal Components to Retain (d)**: Use the slider or input box to select how many principal components you want to keep. A common approach is to select components that explain a significant cumulative portion of the variance (e.g., 70-90%). The scree plot, generated next, will help visualize this.
2.  Click the **"Perform Standardization & PCA"** button.

<aside class="negative">
<b>Warning: Data Dependency</b>
This step requires the `return_matrix` to be populated. If you haven't completed Section 1 yet or if data acquisition failed, you will receive a warning to complete that step first.
</aside>

After execution, you will see a **PCA Scree Plot**, which displays the explained variance ratio for each principal component. You'll also see the shape of the PCA-transformed data and its first few rows.

The standardization process successfully scaled each asset's return series to have a mean of zero and unit variance. This crucial step prevents assets with inherently higher volatility from unduly influencing the clustering algorithms. Subsequently, PCA transformed the standardized returns into a lower-dimensional space of your chosen number of principal components.

The PCA Scree Plot visually confirms the explained variance. For instance, PC1 often captures a significant portion (e.g., 40-60%) of the total variance, representing the broad market factor. PC2 and PC3 might capture sector or style-specific factors. By reducing dimensionality, David has effectively denoised the data, focusing on the most relevant underlying drivers of asset co-movement. This new, compressed representation (`R_pca`) is a more robust input for clustering, ensuring the clusters are based on fundamental patterns rather than noise or scale differences.

## 3. Optimal K-Means Clustering
Duration: 0:10:00

David needs to group the assets into distinct "buckets" or clusters based on their underlying behavior. **K-Means clustering** is a straightforward algorithm for this, but a critical decision is determining the optimal number of clusters, $K$. An incorrect $K$ can lead to either over-segmentation (too many small, insignificant clusters) or under-segmentation (too few, overly broad clusters). To guide this choice, David will use two diagnostic tools: the **Elbow Method** and **Silhouette Analysis**. These methods provide quantitative insights into cluster quality at different $K$ values, helping him select a `K_optimal` that genuinely reflects the inherent grouping structure in the market.

The K-Means algorithm aims to partition $N$ data points $ \{{\mathbf{{x}}_1, \dots, \mathbf{{x}}_N\}}$ in $\mathbb{{R}}^d$ into $K$ clusters $ \{{C_1, \dots, C_K\}}$ by minimizing the Within-Cluster Sum of Squares (WCSS), also known as inertia. The objective function is:
$$ J = \sum_{{k=1}}^K \sum_{{\mathbf{{x}}_i \in C_k}} ||\mathbf{{x}}_i - \mu_k||^2 $$
where $\mu_k$ is the centroid (mean) of cluster $C_k$. The Elbow Method plots $J$ against $K$, looking for a point where the rate of decrease in $J$ significantly slows down, resembling an 'elbow'.

The Silhouette Score measures how similar an object is to its own cluster compared to other clusters. For a data point $i$:
$$ s_i = \frac{{b_i - a_i}}{{\max(a_i, b_i)}} $$
where $a_i$ is the average distance from point $i$ to all other points in its cluster (cohesion), and $b_i$ is the average distance from point $i$ to all points in the nearest *other* cluster (separation). The score $s_i$ ranges from -1 to 1. Values near 1 indicate well-clustered points, near 0 indicate boundary points, and negative values indicate potential misassignment. The average Silhouette Score across all points is used to evaluate cluster quality, with higher values indicating better-defined clusters.

### Determine Optimal K

1.  **Minimum K for analysis** and **Maximum K for analysis**: Set the range of cluster numbers you want to evaluate. It's generally good to start from 2 clusters and go up to a reasonable number (e.g., 10-15).
2.  Click the **"Find Optimal K"** button.

This will generate plots for both the **Elbow Method** and **Silhouette Analysis**.

<aside class="positive">
<b>Interpreting Optimal K Plots</b>
For the Elbow Method, look for the "bend" or "elbow" in the curve where the WCSS starts to decrease less rapidly. For Silhouette Analysis, look for the peak in the average Silhouette Score, which indicates a good balance of cohesion and separation. These two plots help you make an informed decision on the best number of clusters.
</aside>

3.  **Enter your chosen Optimal K based on the plots**: After reviewing the plots, input the number of clusters you deem optimal.
4.  Click the **"Perform K-Means Clustering"** button.

<aside class="negative">
<b>Warning: Data Dependency</b>
This step requires the PCA-transformed data from Section 2. If PCA was not performed or failed, you will receive a warning. Ensure your chosen Optimal K is less than the number of assets.
</aside>

Upon completion, you will see a confirmation that K-Means clustering is complete and the first few cluster labels for your assets.

David has now generated the Elbow and Silhouette plots. The **Elbow Method** plot shows WCSS decreasing as $K$ increases, but the "elbow" where the rate of decrease significantly lessens typically indicates a good $K$. The **Silhouette Analysis** plot provides another perspective; David looks for a peak in the average Silhouette Score, suggesting well-separated and compact clusters.

Based on visual inspection of these plots, David observes that a `K_optimal` around 5 or 6 seems appropriate, as the Elbow plot shows a distinct bend and the Silhouette score reaches a reasonable peak in that range. For the purpose of this exercise, David decides to proceed with his selected `K_optimal`, balancing interpretability with statistical goodness-of-fit. This data-driven selection of `K` is a significant improvement over arbitrary groupings, ensuring his clusters are statistically meaningful.

## 4. Hierarchical Clustering
Duration: 0:07:00

While K-Means provides flat, distinct clusters, David recognizes that asset relationships might be hierarchical. Some assets are very similar (e.g., energy stocks), forming sub-groups, which then combine with other sub-groups (e.g., materials stocks) at a higher level of dissimilarity to form broader categories. **Hierarchical clustering**, visualized through a dendrogram, offers a complementary view by revealing this nested structure. This is particularly valuable for understanding how assets merge from individual entities to broader classifications and is the foundational technique for advanced portfolio optimization methods like Hierarchical Risk Parity (HRP).

Hierarchical clustering builds a tree of clusters by iteratively merging the closest pairs of clusters (an agglomerative approach). The 'distance' between assets is crucial here. For financial applications, Euclidean distance in standardized return space is closely related to correlation distance. Specifically, for standardized return vectors $\mathbf{{\tilde{{r}}}}_i$ and $\mathbf{{\tilde{{r}}}}_j$ (zero mean, unit variance), the squared Euclidean distance is:
$$ ||\mathbf{{\tilde{{r}}}}_i - \mathbf{{\tilde{{r}}}}_j||^2 = 2T(1 - \rho_{{ij}}) $$
where $\rho_{{ij}}$ is the sample correlation and $T$ is the number of return periods. This means K-Means on standardized returns is effectively clustering by correlation.

The standard correlation distance used in hierarchical clustering is:
$$ d_{{ij}} = \sqrt{{2(1 - \rho_{{ij}})}} \in [0, 2] $$
where $d_{{ij}}=0$ implies perfect positive correlation and $d_{{ij}}=2$ implies perfect negative correlation.

Ward's linkage method is commonly used for merging clusters. At each step, it merges the pair of clusters $(C_a, C_b)$ that results in the minimum increase in total within-cluster variance, which is equivalent to minimizing the increase in the WCSS. The increase in WCSS when merging $C_a$ and $C_b$ is:
$$ \Delta(C_a, C_b) = \frac{{|C_a||C_b|}}{{|C_a| + |C_b|}} ||\mu_a - \mu_b||^2 $$
where $|C_a|$ and $|C_b|$ are the number of points in clusters $C_a$ and $C_b$, and $\mu_a$ and $\mu_b$ are their centroids. This method tends to produce compact, roughly equal-sized clusters, which is generally desirable for diversification.

### Perform Hierarchical Clustering

1.  Click the **"Perform Hierarchical Clustering"** button.

<aside class="negative">
<b>Warning: Data Dependency</b>
This step requires the `returns_df` from Section 1 and the `K_optimal` value from Section 3. If these are not available or valid, hierarchical clustering cannot be performed.
</aside>

Upon completion, you will see a **Hierarchical Clustering Dendrogram**. You'll also see the first few hierarchical cluster labels for your assets.

The dendrogram visually represents the nested hierarchy of asset similarities. By reading it from bottom-up, David can see which assets merge first (indicating high similarity, like energy peers `XOM` and `CVX`), and how these sub-clusters then combine into broader groups (e.g., energy merging with materials). The vertical axis, representing correlation-based distance, indicates the dissimilarity level at which mergers occur.

This nested structure provides a richer story than the "flat" partitions of K-Means. It helps David understand the tiered relationships among assets, which is critical for implementing sophisticated diversification strategies like Hierarchical Risk Parity. The clustering algorithm "cuts" this dendrogram to retrieve the number of clusters you specified as `K_optimal`, providing a direct comparison and potential validation for your K-Means results.

## 5. Cluster Interpretation & Visualization
Duration: 0:10:00

Knowing the cluster assignments is only half the battle; David needs to understand *what* each cluster represents in financial terms. Is Cluster 1 a "growth-tech" cluster? Is Cluster 3 a "stable dividend" cluster? To achieve this, he will compute a "cluster profile" by calculating the average annualized return, volatility, beta, and the dominant GICS sector for each cluster. This qualitative labeling helps Alpha Wealth Management assign interpretive meaning to the data-driven groupings.

Additionally, visual confirmation is key. David will visualize the assets in the PCA-reduced space (PC1 vs. PC2), colored by their K-Means cluster assignments, to confirm that the clusters are visually distinct. Finally, a sorted correlation heatmap, reordered according to cluster assignments, will provide a powerful visual check: a well-formed clustering should show clear "block-diagonal" structures, indicating high intra-cluster correlation and lower inter-cluster correlation.

The beta ($\beta$) of an asset $i$ relative to a market portfolio $M$ (here approximated by the equal-weighted portfolio of all assets) is given by:
$$ \beta_i = \frac{{\text{{Cov}}(R_i, R_M)}}{{\text{{Var}}(R_M)}} $$
where $R_i$ is the return of asset $i$ and $R_M$ is the return of the market portfolio proxy. A high beta means the asset tends to move more with the market, while a low beta means it moves less.

### Interpret & Visualize Clusters

1.  Click the **"Interpret & Visualize Clusters"** button.

<aside class="negative">
<b>Warning: Data Dependency</b>
This step relies on the PCA-transformed data from Section 2, K-Means cluster labels from Section 3, and the raw returns data and sector map from Section 1. Ensure these prerequisites are met.
</aside>

After processing, you will see:
*   **Cluster Profile Summary**: A table showing average annualized return, volatility, beta, and the most common GICS sector for each cluster.
*   **Cluster Assignment Table**: A sample of assets with their assigned cluster.
*   **Stock Clusters in PCA Space**: A scatter plot of assets in the first two principal components, colored by their K-Means cluster.
*   **Correlation Matrix Sorted by Cluster Assignment**: A heatmap of the correlation matrix, with assets reordered according to their cluster membership.

David has now successfully brought his clusters to life with financial context and powerful visualizations. The **Cluster Profile Summary** is a critical output: by examining the average return, volatility, beta, and dominant sector for each cluster, he can assign meaningful labels (e.g., "High-Growth Tech," "Stable Income Utilities," "Cyclical Energy"). This table helps David understand if the data-driven clusters align with or diverge from traditional GICS sectors, revealing where the algorithm has found novel groupings.

The **PCA Scatter Plot** visually confirms that the clusters are reasonably well-separated in the latent factor space. Assets belonging to the same cluster tend to group together, reinforcing the validity of the clustering. Finally, the **Sorted Correlation Heatmap** is perhaps the most compelling visual proof. The distinct block-diagonal patterns demonstrate that assets within the same cluster are highly correlated with each other, while showing lower correlation with assets in other clusters. This block structure is the hallmark of effective diversification, where intra-cluster homogeneity and inter-cluster heterogeneity are achieved, a key insight for David's portfolio management decisions at Alpha Wealth.

## 6. Diversification Analysis & Portfolio Construction
Duration: 0:08:00

This is the culmination of David's workflow: translating clustering insights into actionable portfolio decisions and quantifying the diversification benefits. For Alpha Wealth, proving the value of a data-driven approach means demonstrating improved risk metrics. David will assess diversification quality by comparing **intra-cluster correlation** (how correlated assets are *within* a cluster) against **inter-cluster correlation** (how correlated assets are *between* clusters). A successful clustering should yield high intra-cluster and low inter-cluster correlations.

Finally, David will construct a "cluster-diversified portfolio" by selecting a representative asset from each cluster (e.g., the one closest to the cluster centroid). He will then compare its annualized volatility to that of a randomly constructed benchmark portfolio of the same size. A lower volatility for the cluster-diversified portfolio will provide concrete evidence of the diversification benefits achieved through unsupervised learning. This quantitative proof is essential for justifying the methodology to Alpha Wealth's investment committee.

The average intra-cluster correlation for a cluster $C_k$ is calculated as the average of all unique pairwise correlations between assets within that cluster:
$$ \bar{{\rho}}_{{\text{{intra}}, k}} = \frac{{1}}{{|C_k|(|C_k|-1)}} \sum_{{i \in C_k, j \in C_k, i \neq j}} \rho_{{ij}} $$
And the average inter-cluster correlation between clusters $C_k$ and $C_l$ is the average of all pairwise correlations between assets in $C_k$ and assets in $C_l$:
$$ \bar{{\rho}}_{{\text{{inter}}, k, l}} = \frac{{1}}{{|C_k||C_l|}} \sum_{{i \in C_k, j \in C_l}} \rho_{{ij}} $$
For a portfolio with weights $w$ and covariance matrix $\Sigma$, its annualized volatility $\sigma_P$ is:
$$ \sigma_P = \sqrt{{w^T \Sigma w \times 12}} $$
In this case, for an equally weighted portfolio, we can directly compute the standard deviation of its mean return series and annualize it: $\text{{std}}(\text{{mean\_returns}}) \times \sqrt{{12}}$.

### Evaluate Diversification & Build Portfolio

1.  Click the **"Evaluate Diversification & Build Portfolio"** button.

<aside class="negative">
<b>Warning: Data Dependency</b>
This final step requires the results from all previous sections: K-Means model and labels, PCA data, raw returns, and the cluster profile summary. Ensure all previous steps are successfully completed.
</aside>

After the analysis runs, you will see:
*   **Diversification Metrics**: The average intra-cluster correlation, average inter-cluster correlation, and a diversification ratio.
*   **Portfolio Performance Comparison**: The annualized volatility of the cluster-diversified portfolio and a randomly chosen benchmark portfolio, along with the percentage volatility reduction.
*   **Cluster Return/Risk Profile (Annualized)**: A scatter plot visualizing the risk-return characteristics of each cluster.

David's analysis culminates in concrete, actionable insights for Alpha Wealth. The comparison of average **intra-cluster** and **inter-cluster correlations** confirms the effectiveness of the clustering. A higher intra-cluster correlation (e.g., 0.55) combined with a lower inter-cluster correlation (e.g., 0.20) indicates that the clusters successfully group highly co-moving assets together while separating those with distinct behaviors. This numerical validation supports the visual evidence from the sorted correlation heatmap in the previous section.

More importantly, the **cluster-diversified portfolio**, constructed by selecting a representative asset from each distinct cluster, demonstrates superior risk characteristics. By comparing its annualized volatility to that of a randomly selected benchmark portfolio of the same size, David can quantify the 'Volatility Reduction' metric. A significant reduction (e.g., 15-20%) provides compelling quantitative evidence that this data-driven clustering approach leads to genuinely more diversified portfolios. This allows David to present a strong case to Alpha Wealth's investment committee, showcasing how unsupervised learning directly contributes to better risk management and more robust asset allocation decisions.
