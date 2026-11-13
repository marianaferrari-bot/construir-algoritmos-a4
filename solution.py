import numpy as np
import pandas as pd


def regressao_linear(arquivo):
    df = pd.read_csv(arquivo)

    area = df["area_util_m2"].to_list()
    quartos = df["quantidade_quartos"].to_list()
    preco = df["preco_venda"].to_list()

    X = np.column_stack((np.ones(len(area)), area, quartos))
    y = np.array(preco)

    beta = np.linalg.solve(X.T @ X, X.T @ y)

    return beta.tolist()