# Exoplanet Detection System using Python and Machine Learning
This project presents a comprehensive approach to detecting exoplanets—planets orbiting stars beyond our solar system—using both classical machine learning and deep learning techniques. The detection of exoplanets is crucial, as they may host conditions for life. However, due to the highly imbalanced dataset and the subtle patterns in light curves, detecting these exoplanets is challenging. The project combines classical machine learning models (logistic regression, decision trees, XGBoost, and SVM) with deep learning techniques, specifically a Multilayer Perceptron (MLP) neural network. Classical models effectively manage structured data, but as data complexity increased, the project scope expanded to deep learning for enhanced detection capabilities.

### Key Features:
- **Dataset Balancing**: Applied Synthetic Minority Over-sampling Technique (SMOTE) to handle class imbalance and improve model accuracy.
- **Dimensionality Reduction**: Used Principal Component Analysis (PCA) to reduce dataset dimensionality, preserving essential features.
- **Model Evaluation**: Achieved 95% accuracy with Decision Trees and 98.04% accuracy with the MLP neural network. These results highlight that a hybrid approach using classical machine learning and deep learning techniques can create a robust system for exoplanet detection.

## Project Structure
- `dataset/`: Contains the exoplanet training and testing datasets (`exoTrain.csv` and `exoTest.csv`).
- `images/`: Contains images and visualizations generated for the report and README.
- `models/`: Directory where the final trained model is saved (`mlp_exoplanet_classifier.pth`).
- `Exoplanet Detection System.ipynb`: The primary Jupyter notebook containing data preprocessing, model training, evaluation, and model saving/loading steps.
- `Exoplanet Detection System Using Python and Machine Learning.pdf`: Final project report, detailing methodology, results, and conclusions.
- `README.md`: This file, providing an overview and structure of the project.
- `environment.yml`: Conda environment file to set up the required libraries and dependencies.

## Getting Started
To replicate this project locally, follow these steps: 
1. **Clone the repository**:
   ```bash git clone https://github.com/yourusername/exoplanet-detection cd exoplanet-detection ```
2. **Set up the environment**:
   Create a Conda environment from `environment.yml` to install all necessary libraries:
   ```bash conda env create -f environment.yml conda activate exoplanet-detection ```
3. **Run the Jupyter Notebook**:
   Open `Exoplanet Detection System.ipynb` to explore data, preprocess, train, and evaluate models:
   ```bash jupyter notebook Exoplanet\ Detection\ System.ipynb ```

## Code Walkthrough
The Jupyter notebook covers the following steps:

### 1. Importing Libraries
- Imports essential libraries for data handling, visualization, model training, and evaluation.

### 2. Data Import and Exploration
- Loads datasets from the `dataset/` folder.
- Combines data for initial exploration and feature analysis.

### 3. Exploratory Data Analysis (EDA)
- **Class Distribution**: Analyzes the imbalance between exoplanet and non-exoplanet instances.
- **Light Flux Analysis**: Visualizes differences in light curves for exoplanet and non-exoplanet cases.
- **Correlation Analysis**: Examines feature correlations to identify relationships among light flux features.

### 4. Data Preprocessing
- **Outlier Removal**: Detects and removes outliers with Z-score analysis.
- **Dimensionality Reduction with PCA**: Applies PCA to reduce feature dimensions while retaining significant variance.

### 5. Model Selection and Training
- **Classical Machine Learning Models**: Trains Logistic Regression, Decision Tree, XGBoost, and SVM as baselines.
- **Neural Network (MLP)**: Uses PyTorch to implement a Multilayer Perceptron for more complex patterns.
- **Evaluation**: Evaluates models using accuracy, precision, recall, and F1-score.

### 6. Model Saving and Loading
- Saves the trained MLP model in `models/` for reproducibility.
- Demonstrates loading the saved model for predictions on new data.

## Results
- **Decision Tree Model**: Achieved 95% accuracy on the training data.
- **MLP Neural Network**: Achieved 98.04% accuracy, showing strong exoplanet detection capability.

## Conclusion
This project successfully demonstrates a hybrid exoplanet detection system, integrating classical machine learning and deep learning methods. The results underscore the potential for machine learning in astrophysics, especially in identifying exoplanets.

## Future Work
- **Advanced Deep Learning Models**: Explore more complex deep learning architectures such as Convolutional Neural Networks (CNNs) for improved detection accuracy with light curve data.
- **Time Series Analysis**: Investigate the use of Recurrent Neural Networks (RNNs) or Long Short-Term Memory (LSTM) networks to better capture temporal dependencies within light curves.
- **Hyperparameter Tuning**: Implement automated hyperparameter tuning to optimize model parameters and improve performance.
- **Integration with Real-time Data**: Develop pipelines to integrate real-time astronomical data, allowing for continuous exoplanet detection updates.

## License
This project is licensed under the MIT License.
