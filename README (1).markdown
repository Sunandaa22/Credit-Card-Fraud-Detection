# Credit Card Fraud Detection

## Overview
This project implements a machine learning model to detect fraudulent credit card transactions using a Logistic Regression algorithm. The dataset used contains transactions with anonymized features, labeled as fraudulent or non-fraudulent. The goal is to identify patterns that distinguish fraudulent transactions from legitimate ones.

## Dataset
The dataset (`creditcard.csv`) includes:
- **Features**: Time, V1-V28 (anonymized features from PCA transformation), and Amount.
- **Target**: Class (0 for non-fraudulent, 1 for fraudulent).
- The dataset is highly imbalanced, with a small percentage of transactions being fraudulent.

## Prerequisites
To run this project, you need the following installed:
- Python 3.x
- Jupyter Notebook
- Required Python libraries:
  - `numpy`
  - `pandas`
  - `scikit-learn`

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/credit-card-fraud-detection.git
   ```
2. Navigate to the project directory:
   ```bash
   cd credit-card-fraud-detection
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   If you don't have a `requirements.txt` file, install the libraries manually:
   ```bash
   pip install numpy pandas scikit-learn
   ```

## Usage
1. Place the `creditcard.csv` dataset in the project directory.
2. Open the Jupyter Notebook:
   ```bash
   jupyter notebook credit_card_fraud_detection.ipynb
   ```
3. Run the notebook cells sequentially to:
   - Load and preprocess the dataset.
   - Split the data into training and testing sets.
   - Train a Logistic Regression model.
   - Evaluate the model using accuracy scores for both training and test data.

## Project Structure
- `credit_card_fraud_detection.ipynb`: Jupyter Notebook containing the complete code for data loading, preprocessing, model training, and evaluation.
- `creditcard.csv`: Dataset file (not included in the repository; must be sourced separately).
- `README.md`: This file, providing an overview and instructions for the project.

## Methodology
1. **Data Loading**: The dataset is loaded using pandas from `creditcard.csv`.
2. **Data Exploration**: Initial exploration includes viewing the first and last few rows and checking dataset information.
3. **Data Preprocessing**: Features (X) and target (Y) are separated, with the `Class` column as the target.
4. **Data Splitting**: The dataset is split into training (80%) and testing (20%) sets using stratified sampling to maintain class distribution.
5. **Model Training**: A Logistic Regression model is trained on the training data.
6. **Evaluation**: The model’s performance is evaluated using accuracy scores on both training and test datasets.

## Results
- **Training Accuracy**: ~98.28%
- **Test Accuracy**: ~90.00%

Note: The model may require further optimization (e.g., handling imbalanced data with techniques like SMOTE or adjusting the Logistic Regression parameters) to improve performance, especially for the minority class (fraudulent transactions).

## Future Improvements
- Address the class imbalance using techniques like SMOTE or undersampling.
- Experiment with other algorithms (e.g., Random Forest, XGBoost) for better performance.
- Include additional evaluation metrics like precision, recall, and F1-score to better assess performance on imbalanced data.
- Perform feature engineering or selection to improve model accuracy.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- The dataset is sourced from [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud) (ensure you have permission to use it).
- Built using Python, pandas, and scikit-learn.