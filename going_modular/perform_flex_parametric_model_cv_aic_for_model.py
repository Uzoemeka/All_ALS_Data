def perform_flex_parametric_model_cv_aic_for_model(train_df, n_splits):

    n_splits = n_splits

    current_vars = covariates.copy()
    best_score = np.inf
    best_vars = current_vars
    best_df = None

    while True:
        candidates = []

        # Try current model with different dfs
        for df_spline in df_candidates:
            score = flex_parametric_model_cv_aic_for_model(train_df, current_vars, df_spline, n_splits)
            candidates.append((score, current_vars, df_spline))

        # Try removing each variable
        for var in current_vars:
            test_vars = [v for v in current_vars if v != var]
            for df_spline in df_candidates:
                score = flex_parametric_model_cv_aic_for_model(train_df, test_vars, df_spline, n_splits)
                candidates.append((score, test_vars, df_spline))

        # Pick best candidate
        best_candidate = min(candidates, key=lambda x: x[0])

        cand_score, cand_vars, cand_df = best_candidate

        # Stop if no improvement
        if cand_score >= best_score:
            break

        best_score = cand_score
        best_vars = cand_vars
        best_df = cand_df
        current_vars = cand_vars

    print("\n Selected model by CV-AIC:")
    print("Variables:", best_vars)
    print("Spline df:", best_df)
    print("CV-AIC:", best_score)

# perform_flex_parametric_model_cv_aic_for_model(train_df, n_splits=5)
