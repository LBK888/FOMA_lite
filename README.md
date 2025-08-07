# FOMA Data Augmentation and Model Comparison Tool

A comprehensive Streamlit application for comparing different FOMA (First-Order Model Augmentation) implementations in regression tasks. This tool provides an interactive interface for data augmentation, model training, and performance comparison between different FOMA variants.

##

This project was folked and modified from the original [FOMA Repository](https://github.com/azencot-group/FOMA) (Kaufman et al., ICML 2024)

## Features

- **Multiple FOMA Implementations**: Compare No FOMA, Simple FOMA, and Enhanced FOMA approaches
- **Interactive Model Configuration**: Customize neural network architecture with multiple hidden layers
- **Data Quality Analysis**: Comprehensive data quality checks including correlation analysis
- **Advanced Visualization**: PCA and UMAP dimensionality reduction analysis
- **Model Performance Tracking**: Real-time training progress with early stopping
- **Data Augmentation**: Generate synthetic data samples using FOMA transformations
- **Export Capabilities**: Download trained models and augmented datasets

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (optional, for faster training)

### Setup

1. **Clone or download the project files**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**:
   ```bash
   python -c "import torch, streamlit, plotly; print('Installation successful!')"
   ```

## Usage

### Starting the Application

1. **Navigate to the project directory**:
   ```bash
   cd FOMA_lite
   ```

2. **Launch the Streamlit app**:
   ```bash
   streamlit run foma_comparison.py
   ```

3. **Open your web browser** and navigate to the URL shown in the terminal (typically `http://localhost:8501`)

### Using the Application

#### 1. Data Upload
- Upload your Excel file (`.xlsx` format)
- The application will automatically detect columns and allow you to select features and target variables

#### 2. Data Quality Analysis
- Review data statistics and feature correlations
- Examine the correlation matrix heatmap
- Understand your dataset characteristics before training

#### 3. Model Configuration
- **Training Parameters**:
  - Epochs: Number of training iterations (100-5000)
  - Batch Size: Training batch size (8-256)
  - Learning Rate: Optimization learning rate (1e-5 to 1e-2)
  - Patience: Early stopping patience (5-800)

- **Model Architecture**:
  - Number of hidden layers (1-10)
  - Neurons per layer (8-1024)
  - Activation functions: ReLU, LeakyReLU, GELU, SiLU
  - BatchNorm: Optional batch normalization
  - Dropout: Regularization rate (0.0-0.5)

#### 4. FOMA Configuration
- **FOMA Methods**: Choose which FOMA implementations to compare
  - No FOMA: Baseline without augmentation
  - Simple FOMA: Basic FOMA implementation
  - Enhanced FOMA: Advanced FOMA with adaptive parameters

- **FOMA Parameters**:
  - k value: Number of singular values to preserve (1-5)
  - alpha value: Beta distribution parameter (0.1-5.0)
  - Number of augmentations: Synthetic samples to generate (50-1000)

#### 5. Training and Results
- Click "開始訓練" (Start Training) to begin model training
- Monitor real-time training progress with loss and MAPE plots
- Compare performance metrics across different FOMA methods
- Download trained models for later use

#### 6. Data Augmentation Analysis
- Review augmented data quality and statistics
- Visualize data distribution using PCA and UMAP
- Download augmented datasets for further analysis

## FOMA Implementations

### Simple FOMA
- Basic implementation of First-Order Model Augmentation
- Uses fixed alpha parameter for Beta distribution
- Preserves k singular values during SVD transformation

### Enhanced FOMA
- Advanced implementation with adaptive parameters
- Supports multiple modes: 'adaptive' and 'cosine'
- Dynamically adjusts alpha parameter during training
- Improved stability and performance

## Model Architecture

The application uses a configurable neural network with:
- Multiple hidden layers with customizable dimensions
- Various activation functions (ReLU, LeakyReLU, GELU, SiLU)
- Optional batch normalization
- Configurable dropout for regularization
- Single output neuron for regression tasks

## Performance Metrics

- **Loss**: Mean Squared Error (MSE)
- **MAPE**: Mean Absolute Percentage Error
- **R² Score**: Coefficient of determination
- **Training History**: Real-time loss and MAPE tracking

## Data Visualization

### Quality Analysis
- Feature correlation matrices
- Statistical summaries (mean, standard deviation)
- Data distribution analysis

### Dimensionality Reduction
- **PCA Analysis**: Principal Component Analysis for data visualization
- **UMAP Analysis**: Uniform Manifold Approximation and Projection

## Export Features

- **Trained Models**: Download PyTorch model files (`.pth`)
- **Augmented Data**: Export synthetic datasets as CSV files
- **Training History**: Access training progress and metrics

## System Requirements

- **Minimum**: 4GB RAM, CPU-only training
- **Recommended**: 8GB+ RAM, CUDA-compatible GPU
- **Storage**: 1GB free space for model files and datasets

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Reduce batch size
   - Decrease model complexity
   - Use CPU-only training

2. **Slow Training**:
   - Enable GPU acceleration if available
   - Reduce number of augmentations
   - Simplify model architecture

3. **Installation Errors**:
   - Ensure Python 3.8+ is installed
   - Update pip: `pip install --upgrade pip`
   - Install PyTorch separately if needed: `pip install torch`

## License

This project is provided as-is for research and educational purposes.

## Contributing

Feel free to submit issues, feature requests, or pull requests to improve the tool.

## Citation

If you use this tool in your research, please cite the original FOMA paper and this implementation.

---


**Note**: This tool is designed for regression tasks. For classification problems, modifications to the model architecture and loss functions may be required. 
