# Telco Customer Churn Prediction

A comprehensive machine learning project for predicting customer churn in telecommunications companies using an end-to-end ML pipeline with scikit-learn.

## 📋 Project Overview

This project demonstrates a complete machine learning workflow for customer churn prediction in the telecommunications industry. The analysis includes data preprocessing, exploratory data analysis, model training, and evaluation using advanced techniques like SMOTE for handling class imbalance.

## 🎯 Key Features

- **End-to-End ML Pipeline**: Complete workflow from data loading to model deployment
- **Advanced Preprocessing**: Automated handling of categorical and numerical features
- **Class Imbalance Handling**: SMOTE implementation for better model performance
- **Multiple Models**: Comparison of Logistic Regression and Random Forest
- **Hyperparameter Tuning**: GridSearchCV for optimal model parameters
- **Model Persistence**: Export trained models for future predictions

## 📊 Dataset

The project uses the Telco Customer Churn dataset containing:
- **7,043 customers** with 20 features
- **Target variable**: Churn (Yes/No)
- **Features include**: Demographics, service usage, billing information, and contract details

### Key Features:
- **Demographics**: Gender, SeniorCitizen, Partner, Dependents
- **Service Usage**: PhoneService, MultipleLines, InternetService, etc.
- **Billing**: MonthlyCharges, TotalCharges, PaymentMethod
- **Contract**: Contract type, PaperlessBilling

## 🛠️ Installation

### Prerequisites
- Python 3.7+
- Jupyter Notebook or JupyterLab

### Required Packages
```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn joblib
```

## 📈 Methodology

### 1. Data Preprocessing
- **Data Cleaning**: Handle missing values and convert data types
- **Feature Engineering**: Convert categorical variables to numerical
- **Scaling**: Standardize numerical features
- **Encoding**: One-hot encoding for categorical variables

### 2. Exploratory Data Analysis (EDA)
- Distribution analysis of numerical features
- Correlation analysis between features and target
- Visualization of categorical feature relationships with churn
- Outlier detection and analysis

### 3. Model Development
- **Pipeline Architecture**: Integrated preprocessing and modeling
- **Model Selection**: Logistic Regression and Random Forest
- **Hyperparameter Tuning**: GridSearchCV for optimal parameters
- **Class Imbalance**: SMOTE for minority class oversampling

### 4. Model Evaluation
- Classification reports with precision, recall, and F1-score
- Cross-validation for robust performance assessment
- Comparison of different model configurations

## 🚀 Usage

### Running the Notebook
1. Clone or download the project
2. Install required dependencies
3. Open `End_to_End_ML_Pipeline_with_Scikit_learn_Pipeline_API.ipynb`
4. Run all cells sequentially

### Making Predictions
```python
import joblib

# Load the trained pipeline
pipeline = joblib.load('churn_prediction_pipeline.pkl')

# Make predictions on new data
predictions = pipeline.predict(new_data)
```

## 📊 Results

### Model Performance Comparison

| Model | Accuracy | Precision (Churn) | Recall (Churn) | F1-Score (Churn) |
|-------|----------|-------------------|----------------|-------------------|
| Logistic Regression (Original) | 0.80 | 0.66 | 0.56 | 0.60 |
| Random Forest (Original) | 0.80 | 0.66 | 0.53 | 0.59 |
| Logistic Regression (with SMOTE) | 0.74 | 0.51 | 0.78 | 0.62 |
| Random Forest (with SMOTE) | 0.77 | 0.57 | 0.58 | 0.58 |

### Key Findings
- **Class Imbalance**: Original dataset has imbalanced churn classes
- **SMOTE Impact**: Improved recall for churn prediction but slightly reduced precision
- **Best Model**: Logistic Regression with SMOTE shows best balance of metrics
- **Feature Importance**: Contract type, monthly charges, and tenure are key predictors

## 📁 Project Structure

```
TelcoChurn/
├── README.md
├── End_to_End_ML_Pipeline_with_Scikit_learn_Pipeline_API.ipynb
└── churn_prediction_pipeline.pkl (generated after running notebook)
```

## 🔧 Technical Details

### Preprocessing Pipeline
- **Numerical Features**: StandardScaler with mean imputation
- **Categorical Features**: OneHotEncoder with unknown handling
- **Missing Values**: Mean imputation for numerical, mode for categorical

### Model Configuration
- **Logistic Regression**: C parameter optimization (0.1, 1, 10)
- **Random Forest**: n_estimators (100, 200), max_depth (None, 10, 20)
- **Cross-validation**: 5-fold CV for robust evaluation

## 🤝 Contributing

Feel free to contribute to this project by:
- Reporting bugs
- Suggesting new features
- Improving documentation
- Adding new models or techniques

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 📞 Contact

For questions or suggestions, please open an issue in the project repository.

---

**Note**: This project is designed for educational and research purposes. The models and results should be validated with domain experts before deployment in production environments. 