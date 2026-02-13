
import pandas as pd
import numpy as np
from streamlit.testing.v1 import AppTest
from unittest.mock import patch, MagicMock
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans # Needed for patching

# Mock the entire source module or individual functions as needed
# For a comprehensive test, it's better to mock functions individually to control their returns.

# Helper function to create dummy data for mocks
def _create_dummy_data(num_assets=10, num_periods=30):
    dates = pd.date_range(start='2021-01-01', periods=num_periods, freq='M')
    tickers = [f'ASSET{i+1}' for i in range(num_assets)]
    
    returns_data = np.random.randn(num_periods, num_assets) * 0.01
    returns_df = pd.DataFrame(returns_data, index=dates, columns=tickers)
    
    # Transposed return matrix (assets as rows, periods as columns)
    return_matrix = returns_df.T
    
    corr_matrix = returns_df.corr()
    
    R_pca = np.random.randn(num_assets, 5) # 5 principal components
    
    pca_model_mock = MagicMock()
    pca_model_mock.explained_variance_ratio_ = np.array([0.5, 0.2, 0.1, 0.05, 0.05])
    
    kmeans_model_mock = MagicMock(spec=KMeans)
    kmeans_model_mock.n_clusters = 3
    
    kmeans_labels = np.random.randint(0, 3, num_assets)
    hc_labels = np.random.randint(0, 3, num_assets)
    
    sector_map = {ticker: f'Sector{np.random.randint(1, 4)}' for ticker in tickers}

    cluster_profile = pd.DataFrame({
        'Cluster': [0, 1, 2],
        'Count': [num_assets//3]*3,
        'Avg Annualized Return': np.random.rand(3) * 0.1 + 0.05,
        'Avg Annualized Volatility': np.random.rand(3) * 0.1 + 0.1,
        'Avg Beta': np.random.rand(3) * 0.5 + 0.8,
        'Dominant Sector': [f'Sector{i+1}' for i in range(3)]
    })

    asset_assignment_table = pd.DataFrame({
        'Asset': tickers,
        'KMeans_Cluster': kmeans_labels,
        'Dominant Sector': [sector_map[t] for t in tickers]
    })

    return (returns_df, return_matrix, corr_matrix, R_pca, pca_model_mock, 
            kmeans_model_mock, kmeans_labels, hc_labels, sector_map,
            cluster_profile, asset_assignment_table)

# Prepare dummy data
(DUMMY_RETURNS_DF, DUMMY_RETURN_MATRIX, DUMMY_CORR_MATRIX, DUMMY_R_PCA, 
 DUMMY_PCA_MODEL, DUMMY_KMEANS_MODEL, DUMMY_KMEANS_LABELS, DUMMY_HC_LABELS, 
 DUMMY_SECTOR_MAP, DUMMY_CLUSTER_PROFILE_SUMMARY, 
 DUMMY_ASSET_ASSIGNMENT_TABLE) = _create_dummy_data()


# Patch the KMeans class directly as it's instantiated in app.py
@patch('app.KMeans')
@patch('source.evaluate_diversification_and_build_portfolio')
@patch('source.interpret_and_visualize_clusters')
@patch('source.perform_hierarchical_clustering')
@patch('source.find_optimal_k')
@patch('source.standardize_and_pca')
@patch('source.acquire_and_prepare_data')
def test_full_app_workflow(
    mock_acquire_and_prepare_data,
    mock_standardize_and_pca,
    mock_find_optimal_k,
    mock_perform_hierarchical_clustering,
    mock_interpret_and_visualize_clusters,
    mock_evaluate_diversification_and_build_portfolio,
    mock_kmeans_class
):
    # Configure mocks for acquire_and_prepare_data
    mock_acquire_and_prepare_data.return_value = (DUMMY_RETURN_MATRIX, DUMMY_RETURNS_DF, DUMMY_CORR_MATRIX)

    # Configure mocks for standardize_and_pca
    mock_standardize_and_pca.return_value = (DUMMY_R_PCA, DUMMY_PCA_MODEL)

    # Configure mocks for find_optimal_k
    dummy_inertias = [100, 50, 20, 10, 5]
    dummy_silhouettes = [0.2, 0.4, 0.6, 0.5, 0.3]
    mock_find_optimal_k.return_value = (dummy_inertias, dummy_silhouettes)

    # Configure mocks for perform_hierarchical_clustering
    mock_perform_hierarchical_clustering.return_value = DUMMY_HC_LABELS

    # Configure mocks for interpret_and_visualize_clusters
    mock_interpret_and_visualize_clusters.return_value = (DUMMY_CLUSTER_PROFILE_SUMMARY, DUMMY_ASSET_ASSIGNMENT_TABLE)

    # Configure mocks for evaluate_diversification_and_build_portfolio
    mock_evaluate_diversification_and_build_portfolio.return_value = (
        0.7, # intra_corrs_mean
        0.2, # inter_corrs_mean
        0.10, # cluster_portfolio_vol
        0.15, # random_portfolio_vol
        33.3 # volatility_reduction ((0.15 - 0.10) / 0.15 * 100)
    )

    # Configure mock for KMeans class and its instance methods
    kmeans_instance_mock = MagicMock()
    kmeans_instance_mock.fit_predict.return_value = DUMMY_KMEANS_LABELS
    mock_kmeans_class.return_value = kmeans_instance_mock

    at = AppTest.from_file("app.py").run()

    # --- Test Introduction Page ---
    assert at.markdown[0].value.startswith("## Introduction: Optimizing Diversification at Alpha Wealth Management")

    # --- Test Section 1: Data Acquisition & Preparation ---
    at.selectbox[0].set_value("1. Data Acquisition & Preparation").run()
    assert at.title[0].value == "QuLab: Lab 3: Clustering for Asset Allocation" # Title from main app
    assert at.header[0].value == "Section 1: Data Acquisition and Return Matrix Construction"

    # Simulate user input for tickers and dates
    at.text_area[0].set_value("AAPL, MSFT").run()
    at.date_input[0].set_value(pd.to_datetime('2022-01-01')).run()
    at.date_input[1].set_value(pd.to_datetime('2023-01-01')).run()

    # Click the "Acquire & Prepare Data" button
    at.button[0].click().run()

    # Verify session state and outputs
    assert "return_matrix" in at.session_state
    assert not at.session_state.return_matrix.empty
    assert at.success[0].value == "Data acquisition and preparation complete!"
    assert "Return matrix shape: (10, 30)" in at.text[1].value # Shape from dummy data
    assert at.dataframe[0].value.shape == (5, 5) # Displaying first 5x5
    mock_acquire_and_prepare_data.assert_called_once()
    assert at.session_state.tickers == ['AAPL', 'MSFT']
    assert at.session_state.start_date == pd.to_datetime('2022-01-01').date() # date_input returns date objects
    assert at.session_state.end_date == pd.to_datetime('2023-01-01').date()


    # --- Test Section 2: Standardization & PCA ---
    at.selectbox[0].set_value("2. Standardization & PCA").run()
    assert at.header[0].value == "Section 2: Data Standardization and Dimensionality Reduction: Uncovering Latent Market Factors"

    # Check initial max_components calculation based on return_matrix shape
    assert at.number_input[0].max_value == min(DUMMY_RETURN_MATRIX.shape[0], DUMMY_RETURN_MATRIX.shape[1], 20)
    
    # Simulate changing n_components_pca
    at.number_input[0].set_value(3).run()

    # Click "Perform Standardization & PCA"
    at.button[0].click().run()

    # Verify session state and outputs
    assert "R_pca" in at.session_state
    assert at.session_state.R_pca.shape == (DUMMY_R_PCA.shape[0], 3) # Should be updated by number_input
    assert at.success[0].value == "Data standardization and PCA complete!"
    assert at.text[1].value == f"Shape of PCA-transformed data: {DUMMY_R_PCA.shape[0], 3}" # n_components is 3 now
    assert at.dataframe[0].value.shape == (5, 3) # Displaying first 5 rows with 3 components
    assert at.pyplot[0].exists
    mock_standardize_and_pca.assert_called_once()

    # --- Test Section 3: Optimal K-Means Clustering ---
    at.selectbox[0].set_value("3. Optimal K-Means Clustering").run()
    assert at.header[0].value == "Section 3: Optimal K-Means Clustering: Identifying Core Diversification Buckets"

    # Simulate setting min_k and max_k
    at.number_input[0].set_value(2).run()
    at.number_input[1].set_value(5).run()

    # Click "Find Optimal K"
    at.button[0].click().run()

    # Verify session state and outputs
    assert "inertias" in at.session_state
    assert "silhouettes" in at.session_state
    assert at.success[0].value == "Optimal K analysis complete!"
    assert at.pyplot[0].exists
    mock_find_optimal_k.assert_called_once()

    # Simulate selecting optimal K
    at.number_input[2].set_value(3).run() # K_optimal = 3

    # Click "Perform K-Means Clustering"
    at.button[1].click().run()

    # Verify session state and outputs
    assert "kmeans_model" in at.session_state
    assert at.session_state.kmeans_model is not None
    assert "kmeans_cluster_labels" in at.session_state
    assert at.session_state.kmeans_cluster_labels.shape == DUMMY_KMEANS_LABELS.shape
    assert at.success[1].value == "K-Means clustering complete with 3 clusters!"
    assert "First 10 assets with their K-Means cluster labels:" in at.text[3].value
    kmeans_instance_mock.fit_predict.assert_called_once_with(DUMMY_R_PCA) # K-means called with DUMMY_R_PCA

    # --- Test Section 4: Hierarchical Clustering ---
    at.selectbox[0].set_value("4. Hierarchical Clustering").run()
    assert at.header[0].value == "Section 4: Hierarchical Clustering: Uncovering Nested Relationships"

    # Click "Perform Hierarchical Clustering"
    at.button[0].click().run()

    # Verify session state and outputs
    assert "hc_cluster_labels" in at.session_state
    assert at.session_state.hc_cluster_labels.shape == DUMMY_HC_LABELS.shape
    assert at.success[0].value == "Hierarchical Clustering complete!"
    assert at.pyplot[0].exists
    assert at.dataframe[0].value.shape[0] == 10 # Displaying first 10 assets
    mock_perform_hierarchical_clustering.assert_called_once()


    # --- Test Section 5: Cluster Interpretation & Visualization ---
    at.selectbox[0].set_value("5. Cluster Interpretation & Visualization").run()
    assert at.header[0].value == "Section 5: Cluster Interpretation and Visualization: Profiling Our Diversification Buckets"

    # Click "Interpret & Visualize Clusters"
    at.button[0].click().run()

    # Verify session state and outputs
    assert "cluster_profile_summary" in at.session_state
    assert not at.session_state.cluster_profile_summary.empty
    assert "asset_assignment_table" in at.session_state
    assert not at.session_state.asset_assignment_table.empty
    assert at.success[0].value == "Cluster interpretation and visualizations complete!"
    assert at.dataframe[0].value.equals(DUMMY_CLUSTER_PROFILE_SUMMARY)
    assert at.dataframe[1].value.head(10).equals(DUMMY_ASSET_ASSIGNMENT_TABLE.head(10)) # Head is taken in app
    assert at.pyplot[0].exists # PCA Scatter plot
    assert at.pyplot[1].exists # Correlation Heatmap
    mock_interpret_and_visualize_clusters.assert_called_once()

    # --- Test Section 6: Diversification Analysis & Portfolio Construction ---
    at.selectbox[0].set_value("6. Diversification Analysis & Portfolio Construction").run()
    assert at.header[0].value == "Section 6: Diversification Analysis & Cluster-Based Portfolio Construction"

    # Click "Evaluate Diversification & Build Portfolio"
    at.button[0].click().run()

    # Verify session state and outputs
    assert "intra_corrs_mean" in at.session_state
    assert at.session_state.intra_corrs_mean == 0.7
    assert "inter_corrs_mean" in at.session_state
    assert at.session_state.inter_corrs_mean == 0.2
    assert "cluster_portfolio_vol" in at.session_state
    assert at.session_state.cluster_portfolio_vol == 0.10
    assert "random_portfolio_vol" in at.session_state
    assert at.session_state.random_portfolio_vol == 0.15
    assert "volatility_reduction" in at.session_state
    assert at.session_state.volatility_reduction == 33.3
    assert at.success[0].value == "Diversification analysis and portfolio construction complete!"
    assert at.metric[0].value == "0.700"
    assert at.metric[1].value == "0.200"
    assert at.metric[2].value == "3.50 (Target: >> 1)"
    assert at.metric[3].value == "10.00%"
    assert at.metric[4].value == "15.00%"
    assert at.metric[5].value == "33.3% (Target: > 15%)"
    assert at.pyplot[0].exists # Cluster Return/Risk Scatter plot
    mock_evaluate_diversification_and_build_portfolio.assert_called_once()

    plt.close('all')

@patch('app.KMeans')
@patch('source.acquire_and_prepare_data')
def test_data_acquisition_error_handling(mock_acquire_and_prepare_data, mock_kmeans_class):
    # Mock acquire_and_prepare_data to return empty dataframes, simulating failure
    mock_acquire_and_prepare_data.return_value = (pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    mock_kmeans_class.return_value = MagicMock() # Mock KMeans to avoid issues if other paths are hit

    at = AppTest.from_file("app.py").run()
    at.selectbox[0].set_value("1. Data Acquisition & Preparation").run()
    at.text_area[0].set_value("INVALID_TICKER").run()
    at.button[0].click().run()

    assert at.error[0].value == "Failed to acquire and prepare data. Please check ticker symbols and date range."
    assert at.session_state.return_matrix.empty
    mock_acquire_and_prepare_data.assert_called_once()
    plt.close('all')

@patch('app.KMeans')
@patch('source.standardize_and_pca')
@patch('source.acquire_and_prepare_data')
def test_pca_error_handling(mock_acquire_and_prepare_data, mock_standardize_and_pca, mock_kmeans_class):
    # Set up session state with some data, but mock PCA to fail or return empty
    mock_acquire_and_prepare_data.return_value = (DUMMY_RETURN_MATRIX, DUMMY_RETURNS_DF, DUMMY_CORR_MATRIX)
    mock_standardize_and_pca.return_value = (np.array([]), None) # Simulate PCA failure
    mock_kmeans_class.return_value = MagicMock()

    at = AppTest.from_file("app.py").run()

    # First, get to section 2 with valid data
    at.selectbox[0].set_value("1. Data Acquisition & Preparation").run()
    at.button[0].click().run() # Run acquire data
    
    at.selectbox[0].set_value("2. Standardization & PCA").run()
    at.button[0].click().run() # Run PCA

    assert at.error[0].value == "PCA could not be performed. Check if enough data points and components are available."
    assert at.session_state.R_pca.size == 0
    mock_standardize_and_pca.assert_called_once()
    plt.close('all')

@patch('app.KMeans')
@patch('source.find_optimal_k')
@patch('source.standardize_and_pca')
@patch('source.acquire_and_prepare_data')
def test_kmeans_optimal_k_error_handling(
    mock_acquire_and_prepare_data,
    mock_standardize_and_pca,
    mock_find_optimal_k,
    mock_kmeans_class
):
    # Set up session state with valid PCA data
    mock_acquire_and_prepare_data.return_value = (DUMMY_RETURN_MATRIX, DUMMY_RETURNS_DF, DUMMY_CORR_MATRIX)
    mock_standardize_and_pca.return_value = (DUMMY_R_PCA, DUMMY_PCA_MODEL)
    mock_find_optimal_k.return_value = (None, None) # Simulate failure in finding optimal K
    mock_kmeans_class.return_value = MagicMock()

    at = AppTest.from_file("app.py").run()

    # Acquire data and perform PCA first
    at.selectbox[0].set_value("1. Data Acquisition & Preparation").run()
    at.button[0].click().run()
    at.selectbox[0].set_value("2. Standardization & PCA").run()
    at.button[0].click().run()
    
    # Go to K-Means section
    at.selectbox[0].set_value("3. Optimal K-Means Clustering").run()
    at.button[0].click().run() # Find Optimal K

    assert at.error[0].value == "Could not perform K-Means evaluation. Check if PCA data is valid and enough samples are available."
    assert at.session_state.inertias is None
    mock_find_optimal_k.assert_called_once()
    
    # Test K-Means clustering with insufficient data (e.g., K_optimal > samples)
    at.number_input[2].set_value(DUMMY_R_PCA.shape[0] + 1).run() # K > number of samples
    at.button[1].click().run() # Perform K-Means Clustering
    assert f"Cannot perform K-Means: Not enough samples ({DUMMY_R_PCA.shape[0]}) for {DUMMY_R_PCA.shape[0] + 1} clusters or K is less than 2." in at.error[1].value
    plt.close('all')

@patch('app.KMeans')
@patch('source.perform_hierarchical_clustering')
@patch('source.find_optimal_k')
@patch('source.standardize_and_pca')
@patch('source.acquire_and_prepare_data')
def test_hierarchical_clustering_error_handling(
    mock_acquire_and_prepare_data,
    mock_standardize_and_pca,
    mock_find_optimal_k,
    mock_perform_hierarchical_clustering,
    mock_kmeans_class
):
    # Set up session state with valid data, but mock HC to fail
    mock_acquire_and_prepare_data.return_value = (DUMMY_RETURN_MATRIX, DUMMY_RETURNS_DF, DUMMY_CORR_MATRIX)
    mock_standardize_and_pca.return_value = (DUMMY_R_PCA, DUMMY_PCA_MODEL)
    mock_find_optimal_k.return_value = ([100, 50], [0.5, 0.6]) # Minimal valid K values
    mock_perform_hierarchical_clustering.return_value = np.array([]) # Simulate HC failure
    mock_kmeans_class.return_value = MagicMock()

    at = AppTest.from_file("app.py").run()

    # Run through initial steps to populate session state
    at.selectbox[0].set_value("1. Data Acquisition & Preparation").run()
    at.button[0].click().run()
    at.selectbox[0].set_value("2. Standardization & PCA").run()
    at.button[0].click().run()
    at.selectbox[0].set_value("3. Optimal K-Means Clustering").run()
    at.number_input[0].set_value(2).run()
    at.number_input[1].set_value(2).run()
    at.button[0].click().run()
    at.number_input[2].set_value(2).run()
    at.button[1].click().run()
    
    # Go to HC section
    at.selectbox[0].set_value("4. Hierarchical Clustering").run()
    at.button[0].click().run()

    assert at.error[0].value == "Hierarchical clustering could not be performed. Check if data is valid and enough assets are available."
    assert at.session_state.hc_cluster_labels.size == 0
    mock_perform_hierarchical_clustering.assert_called_once()
    plt.close('all')
