def prognostic_score(df, fp_model, groups):
    # prognostic score (linear predictor) with standard error
    r_predict = r['predict']

    df["prognostic_score"] = r_predict(fp_model, type = "link")

    if groups == 2:
        df['risk_group'] = pd.qcut(df['prognostic_score'], groups, labels=['Low','High'])
    elif groups == 3:
        df['risk_group'] = pd.qcut(df['prognostic_score'], groups, labels=['Low','Medium','High'])

    return df
