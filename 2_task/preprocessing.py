from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import re


def df_typing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Приведение столбцов датафрейма к правильным типам данных
    Parameters
    ----------
    df :Целевой датафрейм

    Returns
    -------
    df ; Дтафрейм после установки типов данных
    """

    df['num_of_messages'] = df['num_of_messages'].astype(int)
    df['time'] = pd.to_datetime(df['time'])
    return df


def time_conversion(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавление в датафрейм нового столбца, который содержит значения показывающие количество секунд с начала дня
    Parameters
    ----------
    df : Целевой датафрейм

    Returns
    -------
    df : Обработанный датафрейм
    """
    pattern = '\d{0,9}:\d{0,9}:\d{0,9}'
    sec = df.index.map(lambda x: re.split(':', re.search(pattern, str(x)).group()))
    df.insert(loc=0, column='time_in_sec', value=sec.map(lambda x: (int(x[0]) * 3600) + (int(x[1]) * 60) + int(x[2])))
    return df


def split_df_to_train_test_sets(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Разбиение датафрема на тренировочные и тестовые выборки данных
    Parameters
    ----------
    df : Целевой датафрейм

    Returns
    -------
    x_train ; Тренировочная выборка данных
    x_test : Тестовая выборка данных
    y_train : Тренировочная выборка данных целевых значений
    y_test : Тестовая выборка данных целевых значений
    """
    x = df['time_in_sec'].to_numpy().reshape(-1, 1)
    y = df['num_of_messages'].to_numpy().reshape(-1, 1)

    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, random_state=42)
    return x_train, x_test, y_train, y_test


def normalize_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Отсечение 5% данных сверху и снизу
    Parameters
    ----------
    df : Целевой датафрейм
    Returns
    -------
    df : Обработанный датафрейм
    """
    df = df.sort_values(by='num_of_messages', ascending=True)
    num_of_rows = df.shape[0]

    df = df[int(num_of_rows * 0.05):int(num_of_rows * 0.95)]
    df = df.sort_values(by='time_in_sec', ascending=True)

    return df


def sampling_df(df: pd.DataFrame, sec_sample: str = '5s') -> pd.DataFrame:
    """
    Разбиение данных за промежуток времени и выдача среднее за промежуток
    Parameters
    ----------
    df : Целевой датафрейм
    sec_sample : Отрезок времени, по которому будет производиться сэмплирование данных
    Returns
    -------
    df_for_sampling; Датафрейм разбитый по времени
    """
    df_for_sampling = df.copy()
    df_for_sampling = df_for_sampling.resample(sec_sample, label='left')
    return df_for_sampling.mean()


# def clear_columns_from_outliers(df: pd.Series, columns: list[str]):
#     """
#     Функция очистки выбросов данных с помощью расчета квартилей
#     Parameters
#     ----------
#     df : Целевой датафрейм
#     columns : Список названий столбцов для очистки
#     Returns
#     -------
#     df : Обработанный датафрейм
#     """
#     for col in columns:
#         # Рассчет квантилей столбца
#         q1 = df[col].quantile(q=.15)
#         q3 = df[col].quantile(q=.75)
#
#         # print(q1, q3)
#         # Рассчет межквартильного размаха столбца
#         iqr = q3 - q1
#         # Нахождение минимума и максимума значений в столбце
#         minimum = q1 - iqr * 1.5
#         maximum = q3 + iqr * 1.5
#         # Замена на NA данных не совпадающих с условием
#         df[col] = df[col].apply(lambda x: pd.NA if (x < minimum) or (x > maximum) else x)
#     return df

# def fill_nan_values_by_mean(df, cols):
#     """
#     Функция заполнения очищенных данных средним значением датафрейма
#     Parameters
#     ----------
#     df : Целевой датафрейм
#     cols : Список названий столбцов для заполнения
#     Returns
#     -------
#     df : Обработанный датафрейм
#     """
#     for col in cols:
#         df[col] = df[col].fillna(df[col].mean)
#     return df
