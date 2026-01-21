import numpy as np


from rpy2.robjects import default_converter
from rpy2.robjects import pandas2ri, conversion
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.vectors import FloatVector, IntVector
from rpy2.robjects import DataFrame 

def make_newdata_category_for_fp_model_prediction(df, means, modes, time_col, cat_col):
    times = np.linspace(0, df[time_col].max(), 100)

    n = len(times)

    values = {c: df[c].mean() for c in means}
    values.update({c: df[c].mode()[0] for c in modes})

    # Choose the two covariate levels you want to compare
    level0 = df[cat_col].unique()[0]
    level1 = df[cat_col].unique()[1]

    newdata_0 = {}
    newdata_0.update({c: FloatVector([values[c]] * n) for c in means})
    newdata_0.update({c: IntVector([values[c]] * n) for c in modes})
    newdata_0[cat_col] = IntVector([level0] * n)
    newdata_0[time_col] = FloatVector(times)

    newdata_1 = {}
    newdata_1.update({c: FloatVector([values[c]] * n) for c in means})
    newdata_1.update({c: IntVector([values[c]] * n) for c in modes})
    newdata_1[cat_col] = IntVector([level1] * n)
    newdata_1[time_col] = FloatVector(times)

    with localconverter(default_converter + pandas2ri.converter):
        return DataFrame(newdata_0), DataFrame(newdata_1)
