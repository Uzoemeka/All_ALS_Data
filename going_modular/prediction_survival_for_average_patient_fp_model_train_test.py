def prediction_survival_for_average_patient_fp_model_train_test(df, fp_model, newdata_train, newdata_test, time_col):
    # Predict survival probabilities using the fitted model
    r_predict = ro.r['predict']

    S_train = r_predict(fp_model, newdata=newdata_train, type="surv")
    S_test = r_predict(fp_model, newdata=newdata_test, type="surv")

    with localconverter(default_converter + pandas2ri.converter):
        py_S_train = np.array(S_train)
        py_S_test = np.array(S_test)

    # Plot
    times = np.linspace(0, df[time_col].max(), 100)
    plt.figure(figsize=(7,5))
    plt.plot(times, py_S_train, label="Train set (average patient)", color='blue')
    plt.plot(times, py_S_test, label="Test set (average patient)", color='red', linestyle='--')

    plt.xlabel("Disease Duration")
    plt.ylabel("Survival Probability")
    plt.ylim(0, 1.05)
    plt.title("Royston–Parmar Flexible Parametric Model - Spline df=0")
    plt.legend()
    plt.grid(True)
    plt.show()

# prediction_survival_for_average_patient_fp_model_train_test(df, fp_model, newdata_train, newdata_test, 'Disease_Duration')
