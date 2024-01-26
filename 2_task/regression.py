from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR
import numpy as np


class Regression:
    def __init__(self):
        self.model = linear_model

    def fit(self, x_train: np.ndarray, y_train: np.ndarray):
        """
        Обучение регриссонной модели
        Parameters
        ----------
        x_train : Тренировочные значения
        y_train : Тренировочные целевые значения
        """
        self.model.fit(x_train, y_train)

    def predict(self, x_test: np.ndarray):
        """
        Выполнение регререссии на тестовых данных
        Parameters
        ----------
        x_test :Тестотовые значения

        Returns
        -------
        y_pred ; Предсказаннные целевые значения

        """
        y_pred = self.model.predict(x_test)
        return y_pred

    @staticmethod
    def calculate_metrics(y_test: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
        """
        Вычисление метрик оценки регриссий
        Parameters
        ----------
        y_test : Тестотовые целевые значения
        y_pred : Предсказаннные целевые значения

        Returns
        -------
        mse : Среднеквадратическая ошибка
        r2  ; Коэффициент детерминации
        """
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return mse, r2

    @staticmethod
    def print_metrics(mse: float, r2: float) -> None:
        """
        Вывод метрик в консоль
        Parameters
        ----------
        mse : Среднеквадратическая ошибка
        r2  ; Коэффициент детерминации
        """
        print("mean squared error = %.2f" % mse)
        print("r2_score = %.2f" % r2)


class LinearRegression(Regression):
    """
    Класс линейной регрессии
    """

    def __init__(self):
        self.model = linear_model.LinearRegression()


class PolynomialRegression(Regression):
    """
    Класс полиномиальной регрессии
    """

    def __init__(self):
        self.model = linear_model.LinearRegression()

    @staticmethod
    def polynomialize_features(x_train: np.ndarray, x_test: np.ndarray, degree: int):
        """
        Метод для приведения данных к полиномиальному виду
        Parameters
        ----------
        x_train :Тренировочные заначения
        x_test :Тестотовые значения
        degree :Уровень полинома

        Returns
        -------
        poly_x_train  Полиномиальные тренировочные заначения
        poly_x_test : Полиномиальные тестотовые значения
        """
        poly = PolynomialFeatures(degree=degree)
        poly_x_train = poly.fit_transform(x_train)
        poly_x_test = poly.fit_transform(x_test)
        return poly_x_train, poly_x_test


class SVMRegression(Regression):
    """Класс SVM регрессии"""

    def __init__(self):
        self.model = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))
