import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple, Dict
from lifelines import KaplanMeierFitter, CoxPHFitter, NelsonAalenFitter
from sklearn.model_selection import KFold
from scipy.stats import gaussian_kde, norm
from sklearn.model_selection import train_test_split

pd.set_option('display.max_columns', None)

from rpy2.robjects import r
from rpy2.robjects.conversion import localconverter
from rpy2.robjects import default_converter
from rpy2.robjects import pandas2ri, conversion
from rpy2.robjects.packages import importr
import rpy2.robjects as ro
from rpy2.robjects import Formula
from rpy2.robjects.vectors import FloatVector, IntVector, ListVector, Vector, StrVector
from rpy2.robjects import DataFrame, IntVector

rstpm2 = importr("rstpm2")
survival = importr("survival")
ggplot2 = importr("ggplot2")
graphics = importr("graphics")
stats = importr("stats")
lmtest = importr("lmtest")
from .fp_model_cv_aic_for_model import fp_model_cv_aic_for_model

def perform_fp_model_cv_aic_for_model(train_df, covariates, df_candidates, n_splits=5):
    current_vars = covariates.copy()
    best_score = np.inf
    best_vars = current_vars
    best_df = None

    while True:
        candidates = []

        for df_spline in df_candidates:
            score = fp_model_cv_aic_for_model(train_df, current_vars, df_spline, n_splits)
            candidates.append((score, current_vars, df_spline))

        for var in current_vars:
            test_vars = [v for v in current_vars if v != var]
            for df_spline in df_candidates:
                score = fp_model_cv_aic_for_model(train_df, test_vars, df_spline, n_splits)
                candidates.append((score, test_vars, df_spline))

        best_candidate = min(candidates, key=lambda x: x[0])

        cand_score, cand_vars, cand_df = best_candidate

        if cand_score >= best_score:
            break

        best_score = cand_score
        best_vars = cand_vars
        best_df = cand_df
        current_vars = cand_vars

    print("\nSelected model by CV-AIC:")
    print("Variables:", best_vars)
    print("Spline df:", best_df)
    print("CV-AIC:", best_score)








# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from typing import Optional, Tuple, Dict
# from lifelines import KaplanMeierFitter, CoxPHFitter, NelsonAalenFitter
# from sklearn.model_selection import KFold
# from scipy.stats import gaussian_kde, norm
# from sklearn.model_selection import train_test_split

# pd.set_option('display.max_columns', None)

# from rpy2.robjects import r
# from rpy2.robjects.conversion import localconverter
# from rpy2.robjects import default_converter
# from rpy2.robjects import pandas2ri, conversion
# from rpy2.robjects.packages import importr
# import rpy2.robjects as ro
# from rpy2.robjects import Formula
# from rpy2.robjects.vectors import FloatVector, IntVector, ListVector, Vector, StrVector
# from rpy2.robjects import DataFrame, IntVector


# from .fp_model_cv_aic_for_model import fp_model_cv_aic_for_model


# rstpm2 = importr("rstpm2")
# survival = importr("survival")
# ggplot2 = importr("ggplot2")
# graphics = importr("graphics")
# stats = importr("stats")
# lmtest = importr("lmtest")


# def perform_fp_model_cv_aic_for_model(train_df,covariates, df_candidates,n_splits=5):

#     n_splits = n_splits

#     current_vars = covariates.copy()
#     best_score = np.inf
#     best_vars = current_vars
#     best_df = None

#     while True:
#         candidates = []

#         # Try current model with different dfs
#         for df_spline in df_candidates:
#             score = fp_model_cv_aic_for_model(train_df, current_vars, df_spline, n_splits)
#             candidates.append((score, current_vars, df_spline))

#         # Try removing each variable
#         for var in current_vars:
#             test_vars = [v for v in current_vars if v != var]
#             for df_spline in df_candidates:
#                 score = fp_model_cv_aic_for_model(train_df, test_vars, df_spline, n_splits)
#                 candidates.append((score, test_vars, df_spline))

#         # Pick best candidate
#         best_candidate = min(candidates, key=lambda x: x[0])

#         cand_score, cand_vars, cand_df = best_candidate

#         # Stop if no improvement
#         if cand_score >= best_score:
#             break

#         best_score = cand_score
#         best_vars = cand_vars
#         best_df = cand_df
#         current_vars = cand_vars

#     print("\n Selected model by CV-AIC:")
#     print("Variables:", best_vars)
#     print("Spline df:", best_df)
#     print("CV-AIC:", best_score)

# # perform_fp_model_cv_aic_for_model(train_df, covariates, df_candidates, n_splits=5)
