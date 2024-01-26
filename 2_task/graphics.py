import matplotlib.pyplot as plt
import numpy as np


def graphics_decorator(func):
    """
    Функция обертки для настройки графиков
    Parameters
    ----------
    func : Оборачиваемая функция
    """

    def wrapper(*args, **kwargs):
        plt.figure(figsize=(15, 8))
        func(*args, **kwargs)
        plt.xlabel('Время, часы')
        plt.ylabel('Кол-во сообщений')
        plt.show()

    return wrapper


@graphics_decorator
def plot_distribution(x: np.ndarray, y: np.ndarray) -> None:
    """
    Функция отрисовки графиков распределения
    Parameters
    ----------
    x : Заначения x
    y : Целевые значения y
    """
    plt.scatter(x, y)


@graphics_decorator
def plot_linear_regr_line_with_distr(x_test: np.ndarray, y_test: np.ndarray, y_pred: np.ndarray) -> None:
    """
    Функция отрисовки регрессионной линии на фоне распределения данных
    Parameters
    ----------
    x_test : Тестотовые значения
    y_test : Тестотовые целевые значения
    y_pred : Предсказанные целевые значения
    """
    plt.scatter(x_test, y_test)
    plt.plot(x_test, y_pred, 'r-')


@graphics_decorator
def plot_poly_regr_line_with_distr(x_test: np.ndarray, y_test: np.ndarray,
                                   sorted_x_test: np.ndarray,y_pred: np.ndarray) -> None:
    """
    Функция отрисовки регрессионной линии на фоне распределения данных
    Parameters
    ----------
    x_test : Тестотовые значения
    y_test : Тестотовые целевые значения
    sorted_x_test : Отсортированные тестовые значения
    y_pred : Предсказанные целевые значения
    """
    plt.scatter(x_test, y_test)
    plt.plot(sorted_x_test, y_pred, 'r-')
