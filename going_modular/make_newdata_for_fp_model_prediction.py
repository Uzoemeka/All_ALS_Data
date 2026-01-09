def make_newdata_for_fp_model_prediction(df, means, modes, time_col):
    times = np.linspace(0, df[time_col].max(), 100)

    n = len(times)

    values = {c: df[c].mean() for c in means}
    values.update({c: df[c].mode()[0] for c in modes})

    newdata_r = {}
    newdata_r.update({c: FloatVector([values[c]] * n) for c in means})
    newdata_r.update({c: IntVector([values[c]] * n) for c in modes})
    newdata_r[time_col] = FloatVector(times)

    with localconverter(default_converter + pandas2ri.converter):
        return DataFrame(newdata_r)

# newdata = make_newdata_for_fp_model_prediction(train_df, means, modes, 'Disease_Duration')
