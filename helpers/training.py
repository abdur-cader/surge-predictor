import pandas as pd
import numpy as np
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

class ModelTrainer:
    def __init__(self):
        self.df = None
        self.X = None
        self.y = None
        self.y_raw = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.processed_df = None
        self.preprocessor = None
        self.weight_dict = {
            0: 3,    # High Surge
            1: 3,    # Low Surge
            2: 7.0,  # Mild Surge
            3: 0.5   # No Surge
        }
        
    def load_data(self, filepath):
        self.df = pd.read_excel(filepath, sheet_name='All')
        self.df = self.df.drop(columns=['date','fleet_utilization', 'holiday_label'])
        return self.df
        
    def preprocess_data(self):
        self.df.dropna(inplace=True)
        self.X = self.df.drop(columns=['surge_level'])
        self.y_raw = self.df['surge_level']
        
        num = self.X.select_dtypes(include=['float64', 'int64']).columns
        cat = self.X.select_dtypes(include=['object']).columns
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), num),
                ('cat', OneHotEncoder(sparse_output=False), cat) 
            ]
        )
        
        processed = self.preprocessor.fit_transform(self.X)
        
        encoded_cat_cols = self.preprocessor.named_transformers_['cat'].get_feature_names_out(cat)
        column_names = list(num) + list(encoded_cat_cols)
        self.processed_df = pd.DataFrame(processed, columns=column_names)
        
        le = LabelEncoder()
        self.y = le.fit_transform(self.y_raw)
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.processed_df, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        
        sample_weights = np.array([self.weight_dict[y] for y in self.y_train])
        return sample_weights
        
    def train_models(self, sample_weights):
        models = {
            "RandomForest": self._train_random_forest(sample_weights),
            "XGBoost": self._train_xgboost(sample_weights),
            "LightGBM": self._train_lightgbm(sample_weights),
            "CatBoost": self._train_catboost(sample_weights)
        }
        return models
        
    def _train_random_forest(self, sample_weights):
        rf_clf = RandomForestClassifier(random_state=42)
        rf_params = {
            'n_estimators': [100, 150],
            'max_depth': [6, 8, 10],
            'max_features': ['sqrt', 'log2'],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'bootstrap': [True, False],
        }
        rf_gscv = GridSearchCV(
            estimator=rf_clf,
            param_grid=rf_params,
            scoring='accuracy',
            cv=5,
            verbose=2,
            n_jobs=-1
        )
        rf_gscv.fit(self.X_train, self.y_train, sample_weight=sample_weights)
        return rf_gscv.best_estimator_
        
    def _train_xgboost(self, sample_weights):
        xgb_clf = XGBClassifier(
            objective='multi:softprob',
            num_class=4,
            eval_metric='mlogloss',
            random_state=42,
            tree_method='hist',
        )
        xgb_params = {
            'max_depth': [6, 8],
            'eta': [0.01, 0.1, 0.3],
            'subsample': [0.5, 0.8, 1],
            'colsample_bytree': [0.5, 0.8, 1],
            'n_estimators': [100, 150],
            'gamma': [0, 0.1, 0.2]
        }
        xgb_gscv = GridSearchCV(
            estimator=xgb_clf,
            param_grid=xgb_params,
            scoring='accuracy',
            cv=5,
            verbose=2,
            n_jobs=-1
        )
        xgb_gscv.fit(self.X_train, self.y_train, sample_weight=sample_weights)
        return xgb_gscv.best_estimator_
        
    def _train_lightgbm(self, sample_weights):
        lgb_clf = LGBMClassifier(objective='multiclass', num_class=4, random_state=42)
        lgb_param_dist = {
            'max_depth': [6, 8],
            'learning_rate': [0.03, 0.1],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.7, 1.0],
            'n_estimators': [100, 200],
        }
        lgb_rscv = RandomizedSearchCV(
            estimator=lgb_clf,
            param_distributions=lgb_param_dist,
            scoring='accuracy',
            cv=5,
            verbose=1,
            n_jobs=-1
        )
        lgb_rscv.fit(self.X_train, self.y_train, sample_weight=sample_weights)
        return lgb_rscv.best_estimator_
        
    def _train_catboost(self, sample_weights):
        class_weights_cb = [self.weight_dict[i] for i in sorted(self.weight_dict.keys())]
        cb_clf = CatBoostClassifier(
            loss_function='MultiClass',
            random_seed=42,
            verbose=False,
            class_weights=class_weights_cb
        )
        cb_params = {
            'depth': [4, 6, 8],
            'learning_rate': [0.01, 0.1, 0.3],
            'iterations': [300, 500],
            'l2_leaf_reg': [3, 5],
            'bagging_temperature': [0, 1],
        }
        cb_gscv = GridSearchCV(
            n_jobs=-1,
            verbose=2,
            param_grid=cb_params,
            scoring='accuracy',
            estimator=cb_clf
        )
        cb_gscv.fit(self.X_train, self.y_train)
        return cb_gscv.best_estimator_
        
    def evaluate_models(self, models):
        results = {}
        for name, model in models.items():
            preds = model.predict(self.X_test)
            acc = accuracy_score(self.y_test, preds)
            f1 = f1_score(self.y_test, preds, average='weighted')
            results[name] = {"accuracy": acc, "f1_score": f1}
        return results
        
    def save_models(self, models, path='data/models/'):
        for name, model in models.items():
            joblib.dump(model, f'{path}{name.lower()}_model.pkl')
        joblib.dump(self.preprocessor, f'{path}preprocessor.pkl')
        joblib.dump(self.X_test, f'{path}x_test.pkl')
        joblib.dump(self.y_test, f'{path}y_test.pkl')