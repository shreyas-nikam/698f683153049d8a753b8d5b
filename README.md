# QuLab: Lab 3: Clustering for Asset Allocation - Unsupervised Diversification for Alpha Wealth

![QuantUniversity Logo](https://www.quantuniversity.com/assets/img/logo5.jpg)

## Project Title and Description

**QuLab: Lab 3: Clustering for Asset Allocation - Unsupervised Diversification for Alpha Wealth**

This Streamlit application is developed as part of QuLab's Lab 3, focusing on advanced quantitative methods for portfolio management. It provides a practical, interactive platform for financial professionals, like Mr. David Chen, a Senior Portfolio Manager at Alpha Wealth Management, to apply unsupervised machine learning techniques (K-Means and Hierarchical Clustering) to enhance portfolio diversification.

Traditional asset classifications often fall short in capturing the true co-movement patterns of securities. This project addresses this challenge by guiding users through a data-driven workflow to discover latent asset groupings within a universe of S&P 500 stocks. By identifying these "natural" clusters based on historical price returns, portfolio managers can make more informed allocation decisions, leading to more resilient portfolios and improved risk-adjusted returns. The application demonstrates how these methods directly operationalize concepts from the CFA curriculum, including diversification, multi-factor models, and dimensionality reduction (PCA), and lays the groundwork for advanced techniques like Hierarchical Risk Parity.

## Features

This application offers a guided, multi-step workflow for clustering-based asset allocation:

1.  **Data Acquisition & Preparation**:
    *   Input custom lists of S&P 500 tickers and date ranges.
    *   Download historical adjusted close prices for selected assets.
    *   Compute monthly percentage returns.
    *   Generate an `N x T` return matrix and a pairwise correlation matrix.
    *   Basic handling of missing data.

2.  **Standardization & Principal Component Analysis (PCA)**:
    *   Standardize asset return series to ensure fair contribution to clustering.
    *   Perform PCA to reduce dimensionality, denoise data, and uncover latent market factors.
    *   Interactive selection of the number of principal components to retain.
    *   Visualize the explained variance with a PCA Scree Plot.

3.  **Optimal K-Means Clustering**:
    *   Implement K-Means clustering on PCA-transformed data.
    *   Interactive tools (Elbow Method and Silhouette Analysis) to determine the optimal number of clusters (`K`).
    *   Visualize WCSS (Inertia) and Silhouette Scores for a range of `K` values.
    *   Select and apply the optimal `K` for K-Means clustering.

4.  **Hierarchical Clustering**:
    *   Perform agglomerative hierarchical clustering on asset returns.
    *   Visualize the nested relationships between assets using a Dendrogram.
    *   Demonstrate how to cut the dendrogram to achieve a desired number of clusters (e.g., `K_optimal`).

5.  **Cluster Interpretation & Visualization**:
    *   Generate a comprehensive "Cluster Profile Summary" for each K-Means cluster, including average annualized return, volatility, beta, and dominant GICS sector.
    *   Display an asset-to-cluster assignment table.
    *   Visualize asset clusters in the PCA-reduced space (PC1 vs. PC2).
    *   Create a sorted correlation heatmap, reordered by cluster assignment, to visually confirm intra-cluster coherence and inter-cluster separation.

6.  **Diversification Analysis & Portfolio Construction**:
    *   Quantify diversification quality by calculating average intra-cluster and inter-cluster correlations.
    *   Construct a "cluster-diversified portfolio" by selecting a representative asset from each cluster (e.g., closest to centroid).
    *   Compare the annualized volatility of the cluster-diversified portfolio against a randomly constructed benchmark portfolio of the same size.
    *   Calculate and display the percentage volatility reduction.
    *   Visualize the annualized return/risk profile of each cluster.

## Getting Started

Follow these instructions to set up and run the Streamlit application on your local machine.

### Prerequisites

*   Python 3.8+
*   `pip` (Python package installer)

### Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/qu Lab-clustering-asset-allocation.git
    cd qu-lab-clustering-asset-allocation
    ```
    *(Note: Replace `https://github.com/your-username/qu-lab-clustering-asset-allocation.git` with the actual repository URL if available.)*

2.  **Create a virtual environment** (recommended):
    ```bash
    python -m venv venv
    ```

3.  **Activate the virtual environment**:
    *   **On Windows**:
        ```bash
        .\venv\Scripts\activate
        ```
    *   **On macOS/Linux**:
        ```bash
        source venv/bin/activate
        ```

4.  **Install dependencies**:
    Create a `requirements.txt` file in the root of your project with the following content:
    ```
    streamlit
    pandas
    numpy
    matplotlib
    seaborn
    scikit-learn
    scipy
    yfinance
    ```
    Then, install them using pip:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Run the Streamlit application**:
    Ensure your virtual environment is active, then execute:
    ```bash
    streamlit run app.py
    ```
2.  **Access the application**:
    Your web browser should automatically open to the Streamlit application (usually at `http://localhost:8501`). If not, open your browser and navigate to that address.

3.  **Navigate and interact**:
    *   Use the sidebar navigation to move between the different sections of the lab project.
    *   Follow the instructions and interact with the input widgets (text areas, date pickers, number inputs) to configure parameters.
    *   Click the action buttons (e.g., "Acquire & Prepare Data", "Perform Standardization & PCA") to execute each step of the analysis.
    *   Review the generated tables, metrics, and plots to understand the clustering process and its implications.

## Project Structure

```
.
├── app.py                  # Main Streamlit application script
├── source.py               # Contains all core functions for data processing, clustering, and visualization
├── requirements.txt        # List of Python dependencies
└── README.md               # This file
```

## Technology Stack

*   **Frontend**: [Streamlit](https://streamlit.io/) (for building interactive web applications with Python)
*   **Backend Logic**: [Python 3.x](https://www.python.org/)
*   **Data Manipulation**: [Pandas](https://pandas.pydata.org/)
*   **Numerical Operations**: [NumPy](https://numpy.org/)
*   **Statistical Modeling & ML**: [Scikit-learn](https://scikit-learn.org/) (K-Means, PCA, StandardScaler, Silhouette Score)
*   **Hierarchical Clustering**: [SciPy](https://docs.scipy.org/doc/scipy/reference/cluster.html) (for linkage and dendrogram)
*   **Plotting & Visualization**: [Matplotlib](https://matplotlib.org/) and [Seaborn](https://seaborn.pydata.org/)
*   **Market Data**: [Yfinance](https://pypi.org/project/yfinance/) (for downloading historical stock data, implicitly used in `source.py`)

## Contributing

This project is primarily a lab exercise, but contributions are welcome! If you find a bug, have a suggestion for improvement, or want to add a new feature, please:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/AmazingFeature`).
3.  Make your changes.
4.  Commit your changes (`git commit -m 'Add some AmazingFeature'`).
5.  Push to the branch (`git push origin feature/AmazingFeature`).
6.  Open a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details (if you create one, otherwise state it is open source for educational purposes).

## Contact

For questions or inquiries regarding this project, please contact:

*   **QuantUniversity**
*   **Website**: [https://www.quantuniversity.com](https://www.quantuniversity.com)
*   **Email**: info@quantuniversity.com
