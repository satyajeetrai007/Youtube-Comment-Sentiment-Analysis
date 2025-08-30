import numpy as np
import pandas as pd
import os
import pickle
import yaml
import logging
import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb

# logging configuration
logger = logging.getLogger('model_building')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('model_building_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        df.fillna('', inplace=True)
        logger.debug('Data loaded and NaNs filled from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise


def apply_tfidf(train_data: pd.DataFrame, max_features: int, ngram_range: tuple) -> tuple:
    """Apply TF-IDF with ngrams to the data."""
    try:
        vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)

        X_train = train_data['clean_comment'].values
        y_train = train_data['category'].values

        # Perform TF-IDF transformation
        X_train_tfidf = vectorizer.fit_transform(X_train)

        logger.debug(f"TF-IDF transformation complete. Train shape: {X_train_tfidf.shape}")

    
        with open(os.path.join(get_root_directory(), 'tfidf_vectorizer.pkl'), 'wb') as f:
            pickle.dump(vectorizer, f)

        logger.debug('TF-IDF applied with trigrams and data transformed')
        return X_train_tfidf, y_train
    except Exception as e:
        logger.error('Error during TF-IDF transformation: %s', e)
        raise


def train_stacking_classifier(X_train: np.ndarray, y_train: np.ndarray, params: dict) -> StackingClassifier:
    """Builds and trains a Stacking Classifier based on parameters."""
    try:
        logger.debug("Configuring Stacking Classifier...")

        lgbm_params = params['base_learners']['lgbm']
        xgb_params = params['base_learners']['xgb']
        svc_params = params['base_learners']['svc']

        base_learners = [
            ('lgb', lgb.LGBMClassifier(**lgbm_params)),
            ('xgb', xgb.XGBClassifier(**xgb_params)),
            ('svc', SVC(**svc_params))
        ]
        logger.debug(f"Base learners configured: {[name for name, _ in base_learners]}")

        knn_params = params['meta_model']['knn']
        meta_model = KNeighborsClassifier(**knn_params)
        logger.debug(f"Meta-model configured: {type(meta_model).__name__}")

        stratified_kfold = StratifiedKFold(
            n_splits=params['n_splits'], shuffle=True, random_state=42
        )

        stacking_clf = StackingClassifier(
            estimators=base_learners,
            final_estimator=meta_model,
            cv=stratified_kfold,
            passthrough=False,
            n_jobs=params.get('n_jobs', 1)
        )

        logger.info("Training Stacking Classifier... This may take a while.")

        stacking_clf.fit(X_train, y_train)
        logger.info("Stacking Classifier training completed successfully.")
        
        return stacking_clf

    except KeyError as e:
        logger.error(f"Missing parameter for Stacking Classifier in params.yaml: {e}")
        raise
    except Exception as e:
        logger.error(f"An error occurred during Stacking Classifier training: {e}")
        raise


def save_model(model, file_path: str) -> None:
    """Save the trained model to a file."""
    try:
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
        logger.debug('Model saved to %s', file_path)
    except Exception as e:
        logger.error('Error occurred while saving the model: %s', e)
        raise


def get_root_directory() -> str:
    """Get the root directory (two levels up from this script's location)."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(current_dir, '../../'))


def main():
    try:
        root_dir = get_root_directory()

        params = load_params(os.path.join(root_dir, 'params.yaml'))

        max_features = params['model_building']['max_features']
        ngram_range = tuple(params['model_building']['ngram_range'])

        stacking_params = params['stacking_classifier']

        train_data = load_data(os.path.join(root_dir, 'data/interim/train_processed.csv'))
        X_train_tfidf, y_train = apply_tfidf(train_data, max_features, ngram_range)

        stacking_model = train_stacking_classifier(X_train_tfidf, y_train, stacking_params)

        save_model(stacking_model, os.path.join(root_dir, 'stacking_model.pkl'))

    except Exception as e:
        logger.error('Failed to complete the model building process: %s', e)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()