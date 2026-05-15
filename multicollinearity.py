import csv
import math
from typing import Dict, List, Tuple

class Dataset:
    """Estrutura tabular simples."""
    def __init__(self, data: Dict[str, List[float]]):
        self.data = data
        self.columns = list(data.keys())

        self._validate()

    def _validate(self):
        if not self.data:
            raise ValueError("Datset vazio")
        
        lengths = [len(v) for v in self.data.values()]

        if len(set(lengths)) != 1:
            raise ValueError("Todas as colunas devem possuir o mesmo tamanho.")
        
        if lengths[0] < 2:
            raise ValueError("Dataset deve possuir pelo menos 2 linhas.")

    
    @classmethod
    def from_csv(cls, filepath: str):
        with open(filepath, "r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            data = {}

            for row in reader:
                for key, value in row.items():
                    data.setdefault(key, []).append(float(value))

        return cls(data)
        
    def get_column(self, column_name: str) -> List[float]:
        return self.data[column_name]

class Statistics:
    """
    Estatística básica implementada manualmente.
    """

    @staticmethod
    def mean(values: List[float]) -> float:
        return sum(values) / len(values)

    @staticmethod
    def variance(values: List[float]) -> float:
        mean = Statistics.mean(values)

        return sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    
    @staticmethod
    def std(values: List[float]) -> float:
        return math.sqrt(Statistics.variance(values))

    @staticmethod
    def covariance(x: List[float], y: List[float]) -> float:
        mean_x = Statistics.mean(x)
        mean_y = Statistics.mean(y)

        covariance_sum = 0.0

        for xi, yi in zip(x, y):
            covariance_sum += (xi - mean_x) * (yi - mean_y)

        return covariance_sum / (len(x) - 1)
    
    @staticmethod
    def pearson_correlation(x: List[float], y: List[float]) -> float:
        cov = Statistics.covariance(x, y)
        std_x = Statistics.std(x)
        std_y = Statistics.std(y)

        if std_x == 0 or std_y == 0:
            raise ValueError("Desvio padrão zero, correlação indefinida.")

        return cov / (std_x * std_y)
    
    @staticmethod
    def rank(values: List[float]) -> List[float]:
        sorted_values = sorted((v, i) for i, v in enumerate(values))
        ranks = [0.0] * len(values)
        i = 0
        while i < len(sorted_values):
            j = i
            while j < len(sorted_values) and sorted_values[j][0] == sorted_values[i][0]:
                j += 1
            avg_rank = (i + j + 1) / 2  # média dos postos
            for k in range(i, j):
                ranks[sorted_values[k][1]] = avg_rank
            i = j
        return ranks
    
    @staticmethod
    def spearman_correlation(x: List[float], y: List[float]) -> float:
        rank_x = Statistics.rank(x)
        rank_y = Statistics.rank(y)

        return Statistics.pearson_correlation(rank_x, rank_y)
    

class Matrix:
    """ Algebra linear básica implementada manualmente."""

    @staticmethod
    def transpose(matrix: List[List[float]]) -> List[List[float]]:
        return [list(row) for row in zip(*matrix)]
    
    @staticmethod
    def multiply(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
        if len(a[0]) != len(b):
            raise ValueError("Número de colunas de A deve ser igual ao número de linhas de B.")

        result = [[0.0 for _ in range(len(b[0]))] for _ in range(len(a))]

        for i in range(len(a)):
            for j in range(len(b[0])):
                for k in range(len(b)):
                    result[i][j] += a[i][k] * b[k][j]

        return result

    @staticmethod
    def inverse(matrix: List[List[float]]) -> List[List[float]]: 
        """Inversão Gauss-Jordan"""
        n = len(matrix)

        if any(len(row) != n for row in matrix):
            raise ValueError("A matriz deve ser quadrada.")

        augmented = [row[:] + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(matrix)]

        for i in range(n):
            pivot_row = i
            while pivot_row < n and abs(augmented[pivot_row][i]) < 1e-12:
                pivot_row += 1

            if pivot_row == n:
                raise ValueError("Pivô zero encontrado, matriz singular.")

            if pivot_row != i:
                augmented[i], augmented[pivot_row] = augmented[pivot_row], augmented[i]

            pivot = augmented[i][i]
            augmented[i] = [x / pivot for x in augmented[i]]

            for j in range(n):
                if j == i:
                    continue
                factor = augmented[j][i]
                augmented[j] = [xj - factor * xi for xj, xi in zip(augmented[j], augmented[i])]

        return [row[n:] for row in augmented]

class LinearRegression:
    """Regressão linear simples implementada manualmente."""
    def fit(self, X: List[List[float]], y: List[float]):
        
        ones = [[1.0] for _ in range(len(X))]
        x_design = [ones[i] + X[i] for i in range(len(X))]
        xt = Matrix.transpose(x_design)
        xtx = Matrix.multiply(xt, x_design)
        xtx_inv = Matrix.inverse(xtx)
        y_matrix = [[yi] for yi in y]
        xty = Matrix.multiply(xt, y_matrix)
        beta = Matrix.multiply(xtx_inv, xty)
        self.coefficients = [b[0] for b in beta]

        self.x_design = x_design
        self.y = y
    
    def predict(self, X: List[List[float]]) -> List[float]:
        predictions = []

        for row in X:
            rows_with_bias = [1.0] + row

            prediction = sum(coef * val for coef, val in zip(self.coefficients, rows_with_bias))
            predictions.append(prediction)
        return predictions

    def r_squared(self) -> float:
        predictions = self.predict([row[1:] for row in self.x_design])
        mean_y = Statistics.mean(self.y)

        ss_total = sum((yi - mean_y) ** 2 for yi in self.y)
        sa_residual = sum((yi - pred) ** 2 for yi, pred in zip(self.y, predictions))

        return 1 - (sa_residual / ss_total)

class MulticollinearityDetector:
    """Detecção de multicolinearidade usando VIF."""
    
    def __init__(self, dataset: Dataset):
        self.dataset = dataset
    
    def pearson_matrix(self) -> Dict[str, Dict[str, float]]:
        result = {}

        for col1 in self.dataset.columns:
            result[col1] = {}
             
            for col2 in self.dataset.columns:
                corr = Statistics.pearson_correlation(
                    self.dataset.get_column(col1), 
                    self.dataset.get_column(col2))
                result[col1][col2] = corr
        
        return result
    
    def spearman_matrix(self) -> Dict[str, Dict[str, float]]:
        result = {}

        for col1 in self.dataset.columns:
            result[col1] = {}
             
            for col2 in self.dataset.columns:
                corr = Statistics.spearman_correlation(
                    self.dataset.get_column(col1), 
                    self.dataset.get_column(col2))
                result[col1][col2] = corr
        
        return result

    def vif(self) -> Dict[str, Dict[str, float]]:
        result = {}

        columns = self.dataset.columns

        for target_column in columns:

            y = self.dataset.get_column(target_column)

            predictor_columns = [
                col
                for col in columns
                if col != target_column
            ]

            X = []

            num_rows = len(y)

            for row_index in range(num_rows):
                row = []

                for predictor in predictor_columns:
                    row.append(
                        self.dataset.get_column(predictor)[row_index]
                    )

                X.append(row)

            model = LinearRegression()
            try:
                model.fit(X, y)
                r2 = model.r_squared()
            except ValueError as error:
                message = str(error).lower()
                if "singular" in message or "pivô zero" in message:
                    r2 = 1.0
                else:
                    raise

            tolerance = 1.0 - r2

            if tolerance <= 0:
                vif = float("inf")
            else:
                vif = 1.0 / tolerance

            result[target_column] = {
                "r_squared": r2,
                "tolerance": tolerance,
                "vif": vif,
            }

        return result


if __name__ == "__main__":

    dataset = Dataset.from_csv("dataset.csv")

    detector = MulticollinearityDetector(dataset)

    print("\n=== MATRIZ PEARSON ===")
    pearson = detector.pearson_matrix()

    for col, values in pearson.items():
        print(col, values)

    print("\n=== MATRIZ SPEARMAN ===")
    spearman = detector.spearman_matrix()

    for col, values in spearman.items():
        print(col, values)

    print("\n=== VIF E TOLERÂNCIA ===")
    vif = detector.vif()

    for variable, diagnostics in vif.items():
        print(variable, diagnostics)
    







