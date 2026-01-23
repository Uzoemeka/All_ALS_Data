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


def test_harrels_fp_model_with_cross_validation(df_python, vars_list, spline_df, n_splits=5):
    """
    Test a survival model using cross-validation.
    Returns average AIC and average C-index.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    aics = []
    c_indices = []

    rhs = " + ".join(vars_list) if vars_list else "1"
    formula_str = f"Surv(Disease_Duration, Event==1) ~ {rhs}"
    formula = Formula(formula_str)


    for train_idx, test_idx in kf.split(df_python):
        train_df = df_python.iloc[train_idx]
        test_df = df_python.iloc[test_idx]

        with localconverter(default_converter + pandas2ri.converter):
            r_train = pandas2ri.py2rpy(train_df)
            r_test = pandas2ri.py2rpy(test_df)

        try:
            # Fit model on training data
            model = rstpm2.stpm2(formula, data=r_train, df=spline_df)

            # Calculate AIC on training data
            aic = stats.AIC(model)[0]
            aics.append(aic)

            # Calculate C-index on test data
            # Get predictions for test set
            r_predict = ro.r['predict']
            predictions = r_predict(model, newdata=r_test, type="link")
            # print(ro.r("is.null")(predictions))
            # print(ro.r("length")(predictions))
            # print(ro.r("class")(predictions))

            # Calculate concordance using survival package
            r_test = ro.r("data.frame")(r_test)
            # print(r_test.names)        # Column names
            # print(r["nrow"](r_test))

            r_test = ro.r("transform")(r_test, predicted=predictions)
            # print(r_test.names)
            # print(r["tail"](r_test))

            # print(len(predictions), r["nrow"](r_test)[0])

            # r_test.rx2['predicted'] = predictions
            c_index = survival.concordance(
                ro.Formula(f"Surv(Disease_Duration, Event==1) ~ predicted"),
                data=r_test
            )[0]  # First element is the C-index

            c_indices.append(c_index)

        except Exception as e:
            print(f"Model fitting failed in fold: {e}")
            aics.append(np.inf)
            c_indices.append(0)  # Worst possible C-index

    return np.mean(aics), np.mean(c_indices)
