import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rpy2.robjects import r
from rpy2.robjects.conversion import localconverter
from rpy2.robjects import default_converter
from rpy2.robjects import pandas2ri, conversion
import rpy2.robjects as ro


def prediction_hazard_for_average_patient_cat_cov_fp_model_train_test(df_train,
                                                                        df_test,
                                                                        df_valid,
                                                                        fp_model,
                                                                        newdata_0_train=None,
                                                                        newdata_1_train=None,
                                                                        newdata_0_test=None,
                                                                        newdata_1_test=None,
                                                                        newdata_0_valid=None,
                                                                        newdata_1_valid=None,
                                                                        time_col=None,
                                                                        title_suffix="",
                                                                        Train_set="",
                                                                        Test_set="",
                                                                        Validation_set=""
                                                                    ):
    r_predict = ro.r['predict']

    plt.figure(figsize=(7, 5))

    # Time grid
    times_train = np.linspace(0, df_train[time_col].max(), 100)
    times_test = np.linspace(0, df_test[time_col].max(), 100)
    times_valid = np.linspace(0, df_valid[time_col].max(), 100)
    # times = np.linspace(0, np.max(df_train.rx2(time_col)), 100)


    # --- Train ---
    if newdata_0_train is not None:
        S0_train = r_predict(fp_model, newdata=newdata_0_train, type="hazard")
        with localconverter(default_converter + pandas2ri.converter):
            py_S0_train = np.array(S0_train)
        plt.plot(times_train, py_S0_train, label=f"{Train_set} (average patient)", color="blue")
            
    if newdata_1_train is not None:
        S1_train = r_predict(fp_model, newdata=newdata_1_train, type="hazard")
        with localconverter(default_converter + pandas2ri.converter):
            py_S1_train = np.array(S1_train)
        plt.plot(times_train, py_S1_train, label=f"{Train_set} (average patient)", color="blue")

    # --- Test ---
    if newdata_0_test is not None:
        S0_test = r_predict(fp_model, newdata=newdata_0_test, type="hazard")
        with localconverter(default_converter + pandas2ri.converter):
            py_S0_test = np.array(S0_test)
        plt.plot(times_test, py_S0_test, label=f"{Test_set} (average patient)", color="red", linestyle="--")

    if newdata_1_test is not None:
        S1_test = r_predict(fp_model, newdata=newdata_1_test, type="hazard")
        with localconverter(default_converter + pandas2ri.converter):
            py_S1_test = np.array(S1_test)
        plt.plot(times_test, py_S1_test, label=f"{Test_set} (average patient)", color="red", linestyle="--")

    # --- Validation ---
    if newdata_0_valid is not None:
        S0_valid = r_predict(fp_model, newdata=newdata_0_valid, type="hazard")
        with localconverter(default_converter + pandas2ri.converter):
            py_S0_valid = np.array(S0_valid)
        plt.plot(times_valid, py_S0_valid, label=f"{Validation_set} (average patient)", color="green", linestyle=":")

    if newdata_1_valid is not None:
        S1_valid = r_predict(fp_model, newdata=newdata_1_valid, type="hazard")
        with localconverter(default_converter + pandas2ri.converter):
            py_S1_valid = np.array(S1_valid)
        plt.plot(times_valid, py_S1_valid, label=f"{Validation_set} (average patient)", color="green", linestyle=":")

    plt.xlabel("Disease Duration")
    plt.ylabel("Survival Probability")
    plt.ylim(0, .5)
    plt.title(f"Royston–Parmar Flexible Parametric Model {title_suffix}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()