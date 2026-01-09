def extract_coefficients_from_fp_model_haz_scale(fp_model):
    # Get coefficients directly
    summary = ro.r.summary(fp_model)

    # Try using slotNames to see what's available
    print("Available slots:", ro.r.slotNames(summary))

    # Extract using the correct slot name (usually 'coef' or 'coefficients')
    coefficients = summary.slots['coef']

    # Convert to pandas DataFrame
    df_coef = pd.DataFrame(
        np.array(coefficients),
        columns=['Estimate', 'Std. Error', 'z value', 'Pr(z)'],
        index=list(coefficients.rownames)
    )

    df_coef = df_coef.iloc[1: -1]
    df_coef = df_coef.reset_index().rename(columns={"index": "Variable"})

    #--------------------------------------
    z = 1.96  # 95% CI

    # Convert to hazard ratios
    df_coef['HR'] = np.exp(df_coef['Estimate'])
    df_coef['HR_lower'] = np.exp(df_coef['Estimate'] - z * df_coef['Std. Error'])
    df_coef['HR_upper'] = np.exp(df_coef['Estimate'] + z * df_coef['Std. Error'])
    df_coef['Significant'] = df_coef['Pr(z)'] < 0.05
    return df_coef
