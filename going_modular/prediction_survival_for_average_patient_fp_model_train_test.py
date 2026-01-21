import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rpy2.robjects import r
from rpy2.robjects.conversion import localconverter
from rpy2.robjects import default_converter
from rpy2.robjects import pandas2ri, conversion
import rpy2.robjects as ro

def prediction_survival_for_average_patient_fp_model_train_test(
    df_train,
    df_test,
    df_valid,
    fp_model,
    newdata_train=None,
    newdata_test=None,
    newdata_valid=None,
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
    if newdata_train is not None:
        S_train = r_predict(fp_model, newdata=newdata_train, type="surv")
        with localconverter(default_converter + pandas2ri.converter):
            py_S_train = np.array(S_train)
        plt.plot(times_train, py_S_train, label=f"{Train_set} (average patient)", color="blue")

    # --- Test ---
    if newdata_test is not None:
        S_test = r_predict(fp_model, newdata=newdata_test, type="surv")
        with localconverter(default_converter + pandas2ri.converter):
            py_S_test = np.array(S_test)
        plt.plot(times_test, py_S_test, label=f"{Test_set} (average patient)", color="red", linestyle="--")

    # --- Validation ---
    if newdata_valid is not None:
        S_valid = r_predict(fp_model, newdata=newdata_valid, type="surv")
        with localconverter(default_converter + pandas2ri.converter):
            py_S_valid = np.array(S_valid)
        plt.plot(times_valid, py_S_valid, label=f"{Validation_set} (average patient)",
                 color="green", linestyle=":")

    plt.xlabel("Disease Duration")
    plt.ylabel("Survival Probability")
    plt.ylim(0, 1.05)
    plt.title(f"Royston–Parmar Flexible Parametric Model {title_suffix}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# def prediction_survival_for_average_patient_fp_model_train_test(
#     df,
#     fp_model,
#     newdata_train=None,
#     newdata_test=None,
#     newdata_valid=None,
#     time_col=None,
#     title_suffix="",
#     Train_set="",
#     Test_set="",
#     Validation_set=""
# ):
#     r_predict = ro.r['predict']

#     plt.figure(figsize=(7, 5))

#     # Time grid
#     # times = np.linspace(0, df[time_col].max(), 100)
#     times = np.linspace(0, np.max(df.rx2(time_col)), 100)


#     # --- Train ---
#     if newdata_train is not None:
#         S_train = r_predict(fp_model, newdata=newdata_train, type="surv")
#         with localconverter(default_converter + pandas2ri.converter):
#             py_S_train = np.array(S_train)
#         plt.plot(times, py_S_train, label=f"{Train_set} (average patient)", color="blue")

#     # --- Test ---
#     if newdata_test is not None:
#         S_test = r_predict(fp_model, newdata=newdata_test, type="surv")
#         with localconverter(default_converter + pandas2ri.converter):
#             py_S_test = np.array(S_test)
#         plt.plot(times, py_S_test, label=f"{Test_set} (average patient)", color="red", linestyle="--")

#     # --- Validation ---
#     if newdata_valid is not None:
#         S_valid = r_predict(fp_model, newdata=newdata_valid, type="surv")
#         with localconverter(default_converter + pandas2ri.converter):
#             py_S_valid = np.array(S_valid)
#         plt.plot(times, py_S_valid, label=f"{Validation_set} (average patient)",
#                  color="green", linestyle=":")

#     plt.xlabel("Disease Duration")
#     plt.ylabel("Survival Probability")
#     plt.ylim(0, 1.05)
#     plt.title(f"Royston–Parmar Flexible Parametric Model {title_suffix}")
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()




# def prediction_survival_for_average_patient_fp_model_train_test(df, fp_model, newdata_train=None, newdata_test=None, newdata_valid=None, time_col=None):
#     # Predict survival probabilities using the fitted model
#     r_predict = ro.r['predict']

#     S_train = r_predict(fp_model, newdata=newdata_train, type="surv")
#     S_test = r_predict(fp_model, newdata=newdata_test, type="surv")

#     with localconverter(default_converter + pandas2ri.converter):
#         py_S_train = np.array(S_train)
#         py_S_test = np.array(S_test)

#     # Plot
#     times = np.linspace(0, df[time_col].max(), 100)
#     plt.figure(figsize=(7,5))
#     plt.plot(times, py_S_train, label="Train set (average patient)", color='blue')
#     plt.plot(times, py_S_test, label="Test set (average patient)", color='red', linestyle='--')

#     plt.xlabel("Disease Duration")
#     plt.ylabel("Survival Probability")
#     plt.ylim(0, 1.05)
#     plt.title("Royston–Parmar Flexible Parametric Model - Spline df=0")
#     plt.legend()
#     plt.grid(True)
#     plt.show()

# prediction_survival_for_average_patient_fp_model_train_test(df, fp_model, newdata_train, newdata_test, 'Disease_Duration')
