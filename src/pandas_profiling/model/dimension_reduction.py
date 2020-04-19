"""Correlations between variables."""
import itertools
import warnings
from contextlib import suppress
from functools import partial
from typing import Callable, Dict, List, Optional, Any

import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


from pandas.core.base import DataError
from scipy import stats
from tqdm.auto import tqdm

from pandas_profiling.config import config
from pandas_profiling.model.base import Variable

from pandas_profiling.visualisation.plot import scatter_2d_plot_by_label


def encode_variables(df: pd.DataFrame, variables: dict) -> Optional[pd.DataFrame]:
    # todo: add one hot encoding to categorical variables
    # todo: add filtering other types of variables
    df_encoded = pd.DataFrame()
    continuous_variables = [
        column for column, type in variables.items() if type == Variable.TYPE_NUM and column != target_col
    ]
    for column, type in variables.items():
        # if type == Variable.TYPE_NUM:
        #     df_encoded[column] = df[column]
        # elif type == Variable.TYPE_CAT:
        #
        pass


def calculate_dim_reduction(df: pd.DataFrame, variables: dict) -> dict:
    """Calculate the correlation coefficients between variables for the correlation types selected in the config
    (pearson, spearman, kendall, phi_k, cramers).

    Args:
        variables: A dict with column names and variable types.
        df: The DataFrame with variables.

    Returns:
        A dictionary containing the correlation matrices for each of the active correlation measures.
    """
    dim_reductions: Dict[str, np.ndarray] = {}
    dim_reductions

    disable_progress_bar = not config["progress_bar"].get(bool)

    dim_reduction = [
        dim_reduction_name
        for dim_reduction_name in [
            "pca",
            "t-sne",
            "lda"
        ]
        #if config["correlations"][correlation_name]["calculate"].get(bool)
    ]

    if len(dim_reduction) > 0:
        with tqdm(
            total=len(dim_reduction),
            desc="dim_reduction",
            disable=disable_progress_bar,
        ) as pbar:
            target_col = config['target_col'].get(str)
            for dim_reduction_name in dim_reduction:
                pbar.set_description_str(f"dim reduction [{dim_reduction_name}]")

                if dim_reduction_name == "pca":
                    embedding = PCA(n_components=2).fit_transform(df)
                    if target_col is None:
                        embedding_plot = scatter_2d_plot_by_label(embedding, target_col)
                    else:
                        embedding_plot = scatter_2d_plot_by_label(embedding, df[target_col])
                    dim_reductions[dim_reduction_name] = embedding_plot

                elif dim_reduction_name == "t-sne":
                    embedding = TSNE(n_components=2).fit_transform(df)
                    if target_col is None:
                        embedding_plot = scatter_2d_plot_by_label(embedding, target_col)
                    else:
                        embedding_plot = scatter_2d_plot_by_label(embedding, df[target_col])
                    dim_reductions[dim_reduction_name] = embedding_plot

                elif dim_reduction_name == "lda":
                    if target_col is None:
                        # todo: add message that can't calculate lda without target col
                        pass
                    elif df[target_col].nunique() == 2:
                        embedding = LDA(n_components=1).fit_transform(df.drop(target_col, axis=1), df[target_col])
                        embedding_2d = np.concatenate((np.array(range(len(df)), ndmin=2).T, embedding), axis=1)
                        embedding_plot = scatter_2d_plot_by_label(embedding_2d, df[target_col])
                        dim_reductions[dim_reduction_name] = embedding_plot
                    else:
                        embedding = LDA(n_components=2).fit_transform(df.drop(target_col, axis=1), df[target_col])
                        embedding_plot = scatter_2d_plot_by_label(embedding, df[target_col])
                        dim_reductions[dim_reduction_name] = embedding_plot

                if dim_reduction_name in dim_reductions:
                    # Drop rows and columns with NaNs
                    # todo: add tests for the result
                    pass
                pbar.update()

    return dim_reductions




