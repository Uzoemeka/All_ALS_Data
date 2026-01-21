def prediction_hazard_for_average_patient_cat_cov_fp_model_train_test(df,
                                                                        fp_model,
                                                                        newdata_0_train=None,
                                                                        newdata_1_train=None,
                                                                        newdata_0_test=None,
                                                                        newdata_1_test=None,
                                                                        time_col=None,
                                                                        cat_col_0=None,
                                                                        cat_col_1=None):
    # Predict survival probabilities using the fitted model
    r_predict = ro.r['predict']
    times = np.linspace(0, df[time_col].max(), 100)

    S0_train = r_predict(fp_model, newdata=newdata_0_train, type="hazard")
    S1_train = r_predict(fp_model, newdata=newdata_1_train, type="hazard")

    S0_test = r_predict(fp_model, newdata=newdata_0_test, type="hazard")
    S1_test = r_predict(fp_model, newdata=newdata_1_test, type="hazard")

    with localconverter(default_converter + pandas2ri.converter):
        py_S0_train = np.array(S0_train)
        py_S1_train = np.array(S1_train)
        py_S0_test = np.array(S0_test)
        py_S1_test = np.array(S1_test)

    #_______________________________________________________
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8,5))
    plt.plot(times, py_S0_train, label=f"{cat_col_0}")
    plt.plot(times, py_S1_train, label=f"{cat_col_1}")
    plt.plot(times, py_S0_test, label=f"{cat_col_0} (Test)", linestyle='--')
    plt.plot(times, py_S1_test, label=f"{cat_col_1} (Test)", linestyle='--')
    plt.xlabel("Disease Duration")
    plt.ylabel("Hazard")
    plt.title("Royston–Parmar Flexible Parametric Model (rstpm2)")
    plt.legend()
    plt.show()

# prediction_hazard_for_average_patient_cat_cov_fp_model_train_test(df,
#                                                                     fp_model,
#                                                                     newdata_0_train=newdata_0_train,
#                                                                     newdata_1_train=newdata_1_train,
#                                                                     newdata_0_test=newdata_0_test,
#                                                                     newdata_1_test=newdata_1_test,
#                                                                     time_col='Disease_Duration',
#                                                                     cat_col_0='LICALS',
#                                                                     cat_col_1='MIROCALS')





def prediction_hazard_for_average_patient_fp_model_train_test(df, fp_model, newdata_train=None, newdata_test=None, time_col=None):
    # Predict survival probabilities using the fitted model
    r_predict = ro.r['predict']
    times = np.linspace(0, df[time_col].max(), 100)

    S_train = r_predict(fp_model, newdata=newdata_train, type="hazard")
    S_test = r_predict(fp_model, newdata=newdata_test, type="hazard")

    with localconverter(default_converter + pandas2ri.converter):
        py_S_train = np.array(S_train)
        py_S_test = np.array(S_test)

    # Plot
    plt.figure(figsize=(7,5))
    plt.plot(times, py_S_train, label="Train set (average patient)", color='blue')
    plt.plot(times, py_S_test, label="Test set (average patient)", color='red', linestyle='--')

    plt.xlabel("Disease Duration")
    plt.ylabel("Hazard")
    plt.ylim(-0.005, 0.3)
    plt.title("Royston–Parmar Flexible Parametric Model - Spline df=0")
    plt.legend()
    plt.grid(True)
    plt.show()

# prediction_hazard_for_average_patient_fp_model_train_test(df, fp_model, newdata_train=newdata_train, newdata_test=newdata_test, time_col='Disease_Duration')


def prediction_survival_for_average_patient_cat_cov_fp_model_train_test(df,
                                                                        fp_model,
                                                                        newdata_0_train=None,
                                                                        newdata_1_train=None,
                                                                        newdata_0_test=None,
                                                                        newdata_1_test=None,
                                                                        time_col=None,
                                                                        cat_col_0=None,
                                                                        cat_col_1=None):
    # Predict survival probabilities using the fitted model
    r_predict = ro.r['predict']
    times = np.linspace(0, df[time_col].max(), 100)


    S0_train = r_predict(fp_model, newdata=newdata_0_train, type="surv")
    S1_train = r_predict(fp_model, newdata=newdata_1_train, type="surv")

    S0_test = r_predict(fp_model, newdata=newdata_0_test, type="surv")
    S1_test = r_predict(fp_model, newdata=newdata_1_test, type="surv")

    with localconverter(default_converter + pandas2ri.converter):
        py_S0_train = np.array(S0_train)
        py_S1_train = np.array(S1_train)
        py_S0_test = np.array(S0_test)
        py_S1_test = np.array(S1_test)

    plt.figure(figsize=(8,5))
    plt.plot(times, py_S0_train, label=f"{cat_col_0}")
    plt.plot(times, py_S1_train, label=f"{cat_col_1}")
    plt.plot(times, py_S0_test, label=f"{cat_col_0} (Test)", linestyle='--')
    plt.plot(times, py_S1_test, label=f"{cat_col_1} (Test)", linestyle='--')
    plt.xlabel("Disease Duration")
    plt.ylabel("Survival probability")
    plt.title("Royston–Parmar Flexible Parametric Model (rstpm2)")
    plt.legend()
    plt.show()

# prediction_survival_for_average_patient_cat_cov_fp_model_train_test(df,
#                                                                     fp_model,
#                                                                     newdata_0_train=newdata_0_train,
#                                                                     newdata_1_train=newdata_1_train,
#                                                                     newdata_0_test=newdata_0_test,
#                                                                     newdata_1_test=newdata_1_test,
#                                                                     time_col='Disease_Duration',
#                                                                     cat_col_0='LICALS',
#                                                                     cat_col_1='MIROCALS')


def prediction_survival_for_average_patient_fp_model_train_test(df, fp_model, newdata_train, newdata_test, time_col):
    # Predict survival probabilities using the fitted model
    r_predict = ro.r['predict']
    times = np.linspace(0, df[time_col].max(), 100)

    S_train = r_predict(fp_model, newdata=newdata_train, type="hazard")
    S_test = r_predict(fp_model, newdata=newdata_test, type="hazard")

    with localconverter(default_converter + pandas2ri.converter):
        py_S_train = np.array(S_train)
        py_S_test = np.array(S_test)

    # Plot
    plt.figure(figsize=(7,5))
    plt.plot(times, py_S_train, label="Train set (average patient)", color='blue')
    plt.plot(times, py_S_test, label="Test set (average patient)", color='red', linestyle='--')

    plt.xlabel("Disease Duration")
    plt.ylabel("Hazard")
    plt.ylim(0, 0.3)
    plt.title("Royston–Parmar Flexible Parametric Model - Spline df=0")
    plt.legend()
    plt.grid(True)
    plt.show()

# prediction_survival_for_average_patient_fp_model_train_test(df, fp_model, newdata_train, newdata_test, 'Disease_Duration')
