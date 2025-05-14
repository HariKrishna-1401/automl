import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler,  OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_curve
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, load_digits, load_diabetes
from sklearn.preprocessing import label_binarize
import logging
from itertools import product
import time
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import io
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CustomAutoML:
    def __init__(self, cv_folds=5, random_state=42):
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.models = {}
        self.model_scores = {}
        self.feature_names = None
        self.preprocessor = None
        self.label_encoder = LabelEncoder()
        self.class_names = None
        self.task_type = None

        self.numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])
        self.categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        self.classification_configs = {
            'logistic_regression': {
                'model': LogisticRegression,
                'params': {
                    'C': [1.0],
                    'penalty': ['l2'],
                    'max_iter': [1000],
                    'random_state': [self.random_state]
                }
            },
            'random_forest_clf': {
                'model': RandomForestClassifier,
                'params': {
                    'n_estimators': [100],
                    'max_depth': [None],
                    'random_state': [self.random_state]
                }
            },
            'svm_clf': {
                'model': SVC,
                'params': {
                    'C': [1.0],
                    'kernel': ['rbf'],
                    'random_state': [self.random_state],
                    'probability': [True]
                }
            }
        }

        self.regression_configs = {
            'linear_regression': {
                'model': LinearRegression,
                'params': {}
            },
            'random_forest_reg': {
                'model': RandomForestRegressor,
                'params': {
                    'n_estimators': [100],
                    'max_depth': [None],
                    'random_state': [self.random_state]
                }
            },
            'svm_reg': {
                'model': SVR,
                'params': {
                    'C': [1.0],
                    'kernel': ['rbf']
                }
            }
        }

    def determine_task_type(self, y):
        y = y.astype(str)
        try:
            y_numeric = pd.to_numeric(y, errors='raise')
            unique_values = len(np.unique(y))
            logger.debug(f'Unique target values: {unique_values}, Total samples: {len(y)}')
            if unique_values <= 10:
                self.task_type = 'classification'
            else:
                self.task_type = 'regression'
        except (ValueError, TypeError):
            self.task_type = 'classification'
        logger.info(f'Determined task type: {self.task_type}')

    def preprocess_data(self, X, y=None, training=True):
        try:
            if not isinstance(X, pd.DataFrame):
                X = pd.DataFrame(X)
            
            if training:
                self.feature_names = X.columns.tolist()

            numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
            categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
            logger.debug(f'Numeric features: {numeric_features}')
            logger.debug(f'Categorical features: {categorical_features}')

            if training:
                self.preprocessor = ColumnTransformer(
                    transformers=[
                        ('num', self.numeric_transformer, numeric_features),
                        ('cat', self.categorical_transformer, categorical_features)
                    ])
                X_processed = self.preprocessor.fit_transform(X)
                onehot_features = []
                if categorical_features:
                    onehot_encoder = self.preprocessor.named_transformers_['cat'].named_steps['onehot']
                    onehot_categories = onehot_encoder.get_feature_names_out(categorical_features)
                    onehot_features.extend(onehot_categories)
                self.feature_names = numeric_features + onehot_features
                logger.debug(f'Feature names after preprocessing: {self.feature_names}')
            else:
                if self.preprocessor is None:
                    raise ValueError("Preprocessor is not fitted. Call 'fit' method first.")
                X_processed = self.preprocessor.transform(X)

            if y is not None:
                if self.task_type == 'classification':
                    y = y.astype(str)
                    logger.debug(f'Target values before encoding: {y.unique()}')
                    if training:
                        y_encoded = self.label_encoder.fit_transform(y)
                        self.class_names = self.label_encoder.classes_
                    else:
                        y_encoded = self.label_encoder.transform(y)
                    logger.debug(f'Class names: {self.class_names}')
                    y = y_encoded
                else:
                    y = pd.to_numeric(y, errors='coerce')
                    if y.isna().any():
                        raise ValueError("Target variable contains non-numeric values after conversion for regression task.")
                    logger.debug(f'Target values for regression: {y[:5]}')

            return X_processed, y
        except Exception as e:
            logger.error(f'Preprocessing error: {str(e)}')
            raise

    def generate_param_combinations(self, param_grid):
        keys = list(param_grid.keys())
        values = [param_grid[key] for key in keys]
        return [dict(zip(keys, combo)) for combo in product(*values)]

    def fit(self, X, y):
        try:
            self.determine_task_type(y)
            logger.info('Starting data preprocessing')
            X_processed, y = self.preprocess_data(X, y, training=True)
            
            logger.info('Starting model training')
            start_time = time.time()
            
            model_configs = self.classification_configs if self.task_type == 'classification' else self.regression_configs
            scoring_metric = 'accuracy' if self.task_type == 'classification' else 'r2'

            for model_name, config in model_configs.items():
                logger.info(f'Training {model_name}')
                model_class = config['model']
                param_combinations = self.generate_param_combinations(config['params'])
                params = param_combinations[0] if param_combinations else {}
                model = model_class(**params)
                try:
                    model.fit(X_processed, y)
                    self.models[model_name] = model
                    score = cross_val_score(model, X_processed, y, cv=self.cv_folds, scoring=scoring_metric, n_jobs=-1).mean()
                    self.model_scores[model_name] = score
                    logger.info(f'{model_name} CV Score: {score:.4f}')
                except Exception as e:
                    logger.warning(f'Error training {model_name}: {str(e)}')
                    continue
            
            if not self.models:
                raise ValueError('No models trained successfully')
            
            elapsed_time = time.time() - start_time
            logger.info(f'Training completed in {elapsed_time:.2f} seconds')
            
            return self
        except Exception as e:
            logger.error(f'Training error: {str(e)}')
            raise

    def predict(self, X, model_name):
        try:
            if model_name not in self.models:
                raise ValueError(f'Model {model_name} not trained')
            X_processed, _ = self.preprocess_data(X, training=False)
            predictions = self.models[model_name].predict(X_processed)
            return predictions
        except Exception as e:
            logger.error(f'Prediction error: {str(e)}')
            raise

    def evaluate(self, X, y):
        try:
            if self.task_type == 'classification':
                y = y.astype(str)
                _, y = self.preprocess_data(X, y, training=False)
            else:
                y = pd.to_numeric(y, errors='coerce')
                if y.isna().any():
                    raise ValueError("Target variable contains non-numeric values after conversion for regression task.")
                _, y = self.preprocess_data(X, y, training=False)

            results = {}
            best_model_name = max(self.model_scores, key=self.model_scores.get)
            best_model = self.models[best_model_name]
            
            for model_name in self.models:
                logger.info(f'Evaluating {model_name}')
                predictions = self.predict(X, model_name)
                if self.task_type == 'classification':
                    accuracy = accuracy_score(y, predictions)
                    report = classification_report(y, predictions, output_dict=True)
                    cm = confusion_matrix(y, predictions)
                    results[model_name] = {
                        'accuracy': accuracy,
                        'report': report,
                        'cm': cm.tolist()
                    }
                else:
                    mse = mean_squared_error(y, predictions)
                    mae = mean_absolute_error(y, predictions)
                    r2 = r2_score(y, predictions)
                    predictions = predictions.tolist() if isinstance(predictions, np.ndarray) else predictions
                    y = y.tolist() if isinstance(y, np.ndarray) else y
                    results[model_name] = {
                        'mse': float(mse),
                        'mae': float(mae),
                        'r2': float(r2),
                        'predictions': predictions,
                        'actual': y
                    }
            
            logger.info(f'Best Model: {best_model_name}, CV Score: {self.model_scores[best_model_name]:.4f}')
            return results
        except Exception as e:
            logger.error(f'Evaluation error: {str(e)}')
            raise

    def plot_confusion_matrix(self, cm, classes, filename='cm.png'):
        try:
            if self.class_names is not None:
                classes = self.class_names
            plt.figure(figsize=(8, 6))
            try:
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
            except ValueError as e:
                logger.warning(f'Colormap error: {str(e)}. Falling back to viridis.')
                sns.heatmap(cm, annot=True, fmt='d', cmap='viridis', xticklabels=classes, yticklabels=classes)
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.savefig(filename, bbox_inches='tight')
            plt.close()
        except Exception as e:
            logger.error(f'Error plotting confusion matrix: {str(e)}')
            raise

    def plot_precision_recall(self, X, y, model_name, filename='pr.png'):
        try:
            if model_name not in self.models:
                logger.warning(f'Model {model_name} not found for PR curve')
                return None
            model = self.models[model_name]
            if not hasattr(model, 'predict_proba'):
                logger.warning(f'Model {model_name} does not support predict_proba')
                return None
            X_processed, y = self.preprocess_data(X, y, training=False)
            y_pred_proba = model.predict_proba(X_processed)

            if not hasattr(model, 'classes_'):
                logger.warning(f'Model {model_name} does not have classes_ attribute')
                return None
            model_classes = model.classes_
            logger.debug(f'Model {model_name} trained classes: {model_classes}')

            test_classes = np.unique(y)
            logger.debug(f'Test set classes: {test_classes}')
            n_test_classes = len(test_classes)

            if n_test_classes <= 1:
                logger.warning(f'Skipping Precision-Recall curve for {model_name}: only {n_test_classes} class(es) found in test set')
                return None

            class_indices = []
            for cls in test_classes:
                idx = np.where(model_classes == cls)[0]
                if len(idx) == 0:
                    logger.warning(f'Class {cls} in test set not found in model classes')
                    continue
                class_indices.append(idx[0])
            if not class_indices:
                logger.warning(f'No matching classes between test set and model for {model_name}')
                return None

            y_bin = label_binarize(y, classes=test_classes)
            if y_bin.shape[1] == 1:
                y_bin = np.hstack((1 - y_bin, y_bin))

            plt.figure(figsize=(8, 6))
            display_classes = self.class_names if self.class_names is not None else test_classes
            logger.debug(f'Display classes: {display_classes}')
            for idx, test_class_idx in enumerate(class_indices):
                true_binary = y_bin[:, idx]
                proba = y_pred_proba[:, test_class_idx]
                precision, recall, _ = precision_recall_curve(true_binary, proba)
                plt.plot(recall, precision, lw=2, alpha=0.3, label=f'Class {display_classes[idx]}')

            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'Precision-Recall Curve (OvR) - {model_name}')
            plt.legend(loc='lower left')
            plt.savefig(filename, bbox_inches='tight')
            plt.close()
            logger.info(f'Precision-Recall curve generated for {model_name}')
            return filename
        except Exception as e:
            logger.error(f'Error plotting Precision-Recall curve: {str(e)}')
            raise

    def plot_predicted_vs_actual(self, y_true, y_pred, model_name, filename='pred_vs_actual.png'):
        try:
            plt.figure(figsize=(8, 6))
            plt.scatter(y_true, y_pred, alpha=0.5)
            plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
            plt.xlabel('Actual Values')
            plt.ylabel('Predicted Values')
            plt.title(f'Predicted vs Actual - {model_name}')
            plt.savefig(filename, bbox_inches='tight')
            plt.close()
            return filename
        except Exception as e:
            logger.error(f'Error plotting Predicted vs Actual: {str(e)}')
            return None

    def plot_residuals(self, y_true, y_pred, model_name, filename='residuals.png'):
        try:
            residuals = y_true - y_pred
            plt.figure(figsize=(8, 6))
            plt.scatter(y_pred, residuals, alpha=0.5)
            plt.axhline(y=0, color='r', linestyle='--')
            plt.xlabel('Predicted Values')
            plt.ylabel('Residuals')
            plt.title(f'Residual Plot - {model_name}')
            plt.savefig(filename, bbox_inches='tight')
            plt.close()
            return filename
        except Exception as e:
            logger.error(f'Error plotting residuals: {str(e)}')
            return None

    def plot_feature_imp(self, X, model_name, filename='feature_importance.png'):
        try:
            if model_name not in self.models or not hasattr(self.models[model_name], 'feature_importances_'):
                return None
            importances = self.models[model_name].feature_importances_
            plt.figure(figsize=(10, 6))
            indices = np.argsort(importances)[::-1]
            plt.bar(range(len(importances)), importances[indices], align='center')
            plt.xticks(range(len(importances)), [self.feature_names[i] for i in indices], rotation=45)
            plt.title(f'Feature Importance - {model_name}')
            plt.tight_layout()
            plt.savefig(filename, bbox_inches='tight')
            plt.close()
            return filename
        except Exception as e:
            logger.error(f'Error plotting feature importance: {str(e)}')
            return None

def get_dataset_metadata():
    return {
        'iris': {
            'loader': load_iris,
            'description': 'Iris flower dataset, 3 classes, 4 features, classification',
            'link': 'https://scikit-learn.org/stable/datasets/toy_dataset.html#iris-dataset'
        },
        'breast_cancer': {
            'loader': load_breast_cancer,
            'description': 'Breast cancer dataset, 2 classes, 30 features, classification',
            'link': 'https://scikit-learn.org/stable/datasets/toy_dataset.html#breast-cancer-wisconsin-diagnostic-dataset'
        },
        'wine': {
            'loader': load_wine,
            'description': 'Wine recognition dataset, 3 classes, 13 features, classification',
            'link': 'https://scikit-learn.org/stable/datasets/toy_dataset.html#wine-recognition-dataset'
        },
        'digits': {
            'loader': load_digits,
            'description': 'Digits dataset, 10 classes, 64 features, classification',
            'link': 'https://scikit-learn.org/stable/datasets/toy_dataset.html#optical-recognition-of-handwritten-digits-dataset'
        },
        'diabetes': {
            'loader': load_diabetes,
            'description': 'Diabetes dataset, 10 features, regression',
            'link': 'https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset'
        }
    }

def process_dataset(X, y, dataset_name):
    automl = CustomAutoML()
    automl.determine_task_type(y)
    stratify = y if automl.task_type == 'classification' else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify)
    logger.debug(f'Dataset {dataset_name} - Unique classes in y_test: {np.unique(y_test)}')
    logger.info('Data split completed')

    automl.fit(X_train, y_train)
    logger.info('Model training completed')
    results = automl.evaluate(X_test, y_test)
    logger.info('Model evaluation completed')

    visualizations = {}
    for model_name in results:
        vis = {}
        logger.info(f'Generating visualizations for {model_name}')
        if automl.task_type == 'classification':
            classes = np.unique(y).astype(str)
            try:
                automl.plot_confusion_matrix(np.array(results[model_name]['cm']), classes, f'cm_{model_name}.png')
                vis['confusion_matrix'] = f'cm_{model_name}.png'
                logger.info(f'Confusion matrix generated for {model_name}')
            except Exception as e:
                logger.error(f'Failed to generate confusion matrix for {model_name}: {str(e)}')
            
            pr_file = automl.plot_precision_recall(X_test, y_test, model_name, f'pr_{model_name}.png')
            if pr_file:
                vis['precision_recall'] = pr_file
                logger.info(f'Precision-recall curve generated for {model_name}')
        else:
            pred_vs_actual_file = automl.plot_predicted_vs_actual(results[model_name]['actual'], results[model_name]['predictions'], model_name, f'pred_vs_actual_{model_name}.png')
            if pred_vs_actual_file:
                vis['pred_vs_actual'] = pred_vs_actual_file
                logger.info(f'Predicted vs actual plot generated for {model_name}')
            
            residuals_file = automl.plot_residuals(results[model_name]['actual'], results[model_name]['predictions'], model_name, f'residuals_{model_name}.png')
            if residuals_file:
                vis['residuals'] = residuals_file
                logger.info(f'Residuals plot generated for {model_name}')
        
        fi_file = automl.plot_feature_imp(X, model_name, f'fi_{model_name}.png')
        if fi_file:
            vis['feature_importance'] = fi_file
            logger.info(f'Feature importance plot generated for {model_name}')
        
        visualizations[model_name] = vis

    return automl.task_type, results, visualizations

# Streamlit app
st.title("AutoML Web App")

# Sidebar for dataset selection
st.sidebar.header("Dataset Selection")
datasets = get_dataset_metadata()
dataset_names = list(datasets.keys())
selected_dataset = st.sidebar.selectbox("Choose a dataset", dataset_names)

# File upload
uploaded_file = st.sidebar.file_uploader("Or upload your dataset (CSV or Excel)", type=['csv', 'xlsx'])

# Process dataset
if selected_dataset or uploaded_file:
    if uploaded_file:
        st.header("Uploaded Dataset")
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        dataset_name = uploaded_file.name
    else:
        st.header(f"Dataset: {selected_dataset}")
        data = datasets[selected_dataset]['loader']()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target.astype(str)
        dataset_name = selected_dataset

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    with st.spinner('Processing dataset...'):
        task_type, results, visualizations = process_dataset(X, y, dataset_name)

    st.subheader("Task Type")
    st.write(task_type.capitalize())

    st.subheader("Model Results")
    for model_name, result in results.items():
        st.write(f"**{model_name}**")
        if task_type == 'classification':
            st.write(f"Accuracy: {result['accuracy']:.4f}")
            st.write("Classification Report:")
            report_df = pd.DataFrame(result['report']).transpose()
            st.dataframe(report_df)
        else:
            st.write(f"MSE: {result['mse']:.4f}")
            st.write(f"MAE: {result['mae']:.4f}")
            st.write(f"R²: {result['r2']:.4f}")

        st.subheader("Visualizations")
        vis = visualizations.get(model_name, {})
        if task_type == 'classification':
            if 'confusion_matrix' in vis:
                st.image(vis['confusion_matrix'], caption=f"Confusion Matrix - {model_name}")
            if 'precision_recall' in vis:
                st.image(vis['precision_recall'], caption=f"Precision-Recall Curve - {model_name}")
        else:
            if 'pred_vs_actual' in vis:
                st.image(vis['pred_vs_actual'], caption=f"Predicted vs Actual - {model_name}")
            if 'residuals' in vis:
                st.image(vis['residuals'], caption=f"Residuals Plot - {model_name}")
        if 'feature_importance' in vis:
            st.image(vis['feature_importance'], caption=f"Feature Importance - {model_name}")

if __name__ == "__main__":
    st.write("AutoML app is running. Select a dataset or upload your own to begin.")