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

def fp_model_cv_aic_for_model(df_python, vars_list, spline_df, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    aics = []

    rhs = " + ".join(vars_list) if vars_list else "1"
    formula_str = f"Surv(Disease_Duration, Event==1) ~ {rhs}"
    formula = Formula(formula_str)

    for train_idx, test_idx in kf.split(df_python):
        train_df = df_python.iloc[train_idx]

        with localconverter(default_converter + pandas2ri.converter):
            r_train = pandas2ri.py2rpy(train_df)

        try:
            model = rstpm2.stpm2(formula, data=r_train, df=spline_df)
            aic = stats.AIC(model)[0]
            aics.append(aic)
        except Exception as e:
            print(f"Model fitting failed in fold: {e}")
            aics.append(np.inf)

    return np.mean(aics)













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

# rstpm2 = importr("rstpm2")
# survival = importr("survival")
# ggplot2 = importr("ggplot2")
# graphics = importr("graphics")
# stats = importr("stats")
# lmtest = importr("lmtest")

# def fp_model_cv_aic_for_model(df_python, vars_list, spline_df, n_splits=5):
#     """
#     Compute mean cross-validated AIC for a given variable set and spline df
#     """
#     kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
#     aics = []

#     rhs = " + ".join(vars_list) if vars_list else "1"
#     formula_str = f"Surv(Disease_Duration, Event==1) ~ {rhs}"
#     formula = Formula(formula_str)

#     for train_idx, test_idx in kf.split(df_python):
#         train_df = df_python.iloc[train_idx]

#         # Convert fold to R
#         with localconverter(default_converter + pandas2ri.converter):
#             r_train = pandas2ri.py2rpy(train_df)

#         try:
#             model = rstpm2.stpm2(formula, data=r_train, df=spline_df)
#             aic = stats.AIC(model)[0]
#             aics.append(aic)
#         except Exception:
#             aics.append(np.inf)

#     return np.mean(aics)
