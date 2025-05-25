import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder

class DataPreprocessing:
    def __init__(self, name="nba2k-full.csv"):
        self.file_name = name
        self.raw_dir = Path("artifacts/raw_data") / self.file_name
        self.preprocessed_dir = Path("artifacts/processed")
        self.preprocessed_dir.mkdir(parents=True, exist_ok=True)
        
    @staticmethod
    def _extract_height_meters(cell) -> float | None:
        if isinstance(cell, str):
            try:
                parts = cell.split('/')
                clean_cell = float(parts[1].replace('m', '').strip())
                return round(clean_cell, 2)
            except ValueError:
                return np.nan
    
    @staticmethod     
    def _extract_weight_kilogram(cell) -> float | None:
        if isinstance(cell, str):
            try:
                parts = cell.split('/')
                clean_cell = float(parts[1].replace('kg.', '').strip())
                return round(clean_cell, 2)
            except ValueError:
                return np.nan
            
    @staticmethod
    def _extract_edition_year(cell) -> int:
        if isinstance(cell, str):
            if cell.endswith(cell):
                return int(2020)
            else:
                return int(2021)
            
    def clean_data(self) -> pd.DataFrame:
        data = pd.read_csv(self.raw_dir)
        
        data['b_day'] = pd.to_datetime(data['b_day'], format='%m/%d/%y')
        data['draft_year'] = pd.to_datetime(data['draft_year'], format='%Y')
        
        cat_cols = ["full_name", "jersey", "position", "college", "country", "draft_round"]
        for col in cat_cols:
            data[col] = data[col].astype("category")
        
        data['height_m'] = data['height'].apply(self._extract_height_meters)
        data['weight_kg'] = data['weight'].apply(self._extract_weight_kilogram)

        data["team"] = data["team"].fillna("NO TEAM")
        data["team"] = data["team"].astype("category")
        
        data['salary'] = data['salary'].astype(str).str.replace('$', '', regex=False)
        data['salary'] = data['salary'].astype("float")
        
        data['country'] = np.where(data['country'] == "USA", "USA", "Not-USA")
        
        data["draft_round"] = data['draft_round'].apply(
            lambda c: "0" if isinstance(c, str) and "Undrafted" in c else c
        )
        return data
    
    def feature_data(self, data: pd.DataFrame) -> pd.DataFrame:
        # Feature Engineering
        data['version'] = data['version'].apply(self._extract_edition_year)
        data['version'] = pd.to_datetime(data['version'], format='%Y')
        
        
        data['age'] = (data['version'] - data['b_day']).dt.days / 365.25
        data['experience'] = (data['version'] - data['draft_year']).dt.days / 365.25
        data['bmi'] = data['weight_kg'] / (data['height_m'] ** 2)
        
        # High Cardinality
        cat_columns = data.select_dtypes(include="category").columns
        high_cardinality_columns = [col for col in cat_columns if data[col].nunique() >= 50]
        data.drop(columns=high_cardinality_columns, axis=1, inplace=True)
        featured_df = data.drop(columns=["weight", "height", "version", "b_day", "draft_year", "weight_kg", "height_m"], axis=1)
        return featured_df
    
    @staticmethod
    def multicol_data(data: pd.DataFrame) -> pd.DataFrame:
        multicols = set()
        target = "salary"
        cor = data.corr(numeric_only=True)
        n_cols = len(cor.columns)
        for i in range(n_cols):
            for j in range(n_cols):
                if i == 1 or j == 1 or i == j:
                    continue
                corr_value = cor.iloc[j, i]
                if abs(corr_value) >= 0.5:
                    col1 = cor.columns[i]
                    col2 = cor.columns[j]
                    cor_col1 = abs(cor.loc[col1, target])
                    cor_col2 = abs(cor.loc[col2, target])
                    if cor_col1 >= cor_col2:
                        multicols.add(col2)
                    else:
                        multicols.add(col1)
        print(multicols)
        data.drop(columns=multicols, axis=1, inplace=True)
        return data
    
    def transform_data(self, data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        X = data.drop(columns=["salary"], axis=1)
        y = data["salary"]
        
        num_cols = X.select_dtypes("number").columns.to_list()
        cat_cols = X.select_dtypes("category").columns.to_list()
        
        scaler = StandardScaler()
        scaled_numerical = scaler.fit_transform(X[num_cols])
        X_num = pd.DataFrame(scaled_numerical, columns=num_cols)
        
        encoder = OneHotEncoder(sparse_output=False)
        encoded_categorical = encoder.fit_transform(X[cat_cols])
        # cat_features = []
        # for categories in encoder.categories_:
        #     cat_features.extend([str(cat) for cat in categories])
        cat_features = encoder.get_feature_names_out(cat_cols)
        X_cat = pd.DataFrame(encoded_categorical, columns=cat_features)
        
        X = pd.concat([X_num, X_cat], axis=1)
        
        return X, y
        
    def save_data(self, X: pd.DataFrame, y: pd.Series) -> None:
        data = pd.concat([X, y], axis=1)
        data.to_csv(self.preprocessed_dir / self.file_name, index=False)
        
    def run(self):
        clean_data = self.clean_data()
        feature_data = self.feature_data(clean_data)
        data = self.multicol_data(feature_data)
        X, y = self.transform_data(data)
        self.save_data(X, y)
        
