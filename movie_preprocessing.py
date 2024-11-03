from feature_engine.encoding import RareLabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# Remove colunas específicas
class RemoveColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns if columns is not None else []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=self.columns)

# Reduz várias colunas de data em uma única coluna categorizada por décadas
class DateProcessing(BaseEstimator, TransformerMixin):
    def __init__(self, date_cols=None, priority_order=None):
        self.date_cols = date_cols if date_cols is not None else []
        self.priority_order = priority_order if priority_order is not None else date_cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Converte as colunas para datetime
        for col in self.date_cols:
            X[col] = pd.to_datetime(X[col], errors='coerce')

        # Cria a coluna 'releaseDate' baseado na regra de prioridade
        X['releaseDate'] = pd.concat([X[col] for col in self.priority_order], axis=1).bfill(axis=1).iloc[:, 0]
        X['releaseDecade'] = (X['releaseDate'].dt.year // 10) * 10

        return X.drop(columns=self.date_cols + ['releaseDate'])

# Classifica a coluna de tempo de execução do filme em grupos de 20 minutos
class RuntimeProcessing(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X['runtimeMinutes'] = pd.to_numeric(X['runtimeMinutes'], errors='coerce')
        bins = list(range(0, int(X['runtimeMinutes'].max()) + 20, 20))
        labels = [f"{i}_{i+19}" for i in bins[:-1]]
        X['runtime'] = pd.cut(X['runtimeMinutes'], bins=bins, labels=labels, include_lowest=True)
        return X.drop(columns='runtimeMinutes')

# Remove as linhas com valores faltantes em determinadas colunas
class RemoveMissingValues(BaseEstimator, TransformerMixin):
    def __init__(self, subset=None):
        self.subset = subset if subset is not None else []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.dropna(subset=self.subset)

# Agrupa categorias mais raras em uma única categoria
class RareLabelEncoderStep(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None, tols=None):
        self.columns = columns if columns is not None else []
        self.tols = tols if tols is not None else []

    def fit(self, X, y=None):
        self.encoders = {}
        for i, col in enumerate(self.columns):
            X[col] = X[col].astype('category')
            encoder = RareLabelEncoder(n_categories=1, max_n_categories=50, replace_with='Other', tol=self.tols[i]/X.shape[0])
            encoder.fit(X[[col]])
            self.encoders[col] = encoder
        return self

    def transform(self, X):
        for col in self.columns:
            X[col] = self.encoders[col].transform(X[[col]])
            # Remove as categorias sem registros
            X[col] = X[col].cat.remove_unused_categories()
        return X

# Vetoriza as colunas categóricas em várias colunas binárias
class VectorizeColumnStep(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns if columns is not None else []

    def fit(self, X, y=None):
        self.vectorizers = {}
        for col in self.columns:
            ll = X[col].fillna('Other').astype(str).str.split(', ').to_list()
            ll = [' '.join([i.replace('\'', '_').replace('&', '_').replace('.', '_').replace('-', '_').replace(' ', '_').replace('+', '_') for i in item]) for item in ll]
            vectorizer = CountVectorizer(min_df=25, lowercase=False)
            vectorizer.fit(ll)
            self.vectorizers[col] = vectorizer
        return self

    def transform(self, X):
        for col in self.columns:
            ll = X[col].fillna('Other').astype(str).str.split(', ').to_list()
            ll = [' '.join([i.replace('\'', '_').replace('&', '_').replace('.', '_').replace('-', '_').replace(' ', '_').replace('+', '_') for i in item]) for item in ll]
            vector = self.vectorizers[col].transform(ll)
            voc = self.vectorizers[col].vocabulary_
            voc_inv = {v: col+'_'+k for k, v in voc.items()}
            tt = pd.DataFrame(vector.toarray(), columns=[voc_inv[i] for i in range(len(voc_inv))])
            X = pd.concat([X.reset_index(drop=True), tt.reset_index(drop=True)], axis=1).drop([col], axis=1)
        return X

# Constrói a pipeline para pré-processamento do dataframe de filmes
def preprocess_movies_data(movies_data, remove_cols=None, date_cols=None, priority_order=None, remove_missing_cols=None, rare_label_cols=None, rare_label_tols=None, vectorize_cols=None):
    pipeline = Pipeline([
        ('remove_columns', RemoveColumns(columns=remove_cols)),
        ('process_dates', DateProcessing(date_cols=date_cols, priority_order=priority_order)),
        ('process_runtime', RuntimeProcessing()),
        ('remove_missing', RemoveMissingValues(subset=remove_missing_cols)),
        ('encode_rare_labels', RareLabelEncoderStep(columns=rare_label_cols, tols=rare_label_tols)),
        ('vectorize_columns', VectorizeColumnStep(columns=vectorize_cols))
    ])

    return pipeline.fit_transform(movies_data)