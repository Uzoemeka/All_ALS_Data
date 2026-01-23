import numpy as np
from typing import List, Tuple
from going_modular.test_harrels_fp_model_with_cross_validation import test_harrels_fp_model_with_cross_validation

def find_best_harrels_fp_model_by_backward_elimination(train_df, covariates, df_candidates, n_splits=5):
    """
    Find the best model by testing different variable combinations.
    Uses both AIC (lower is better) and C-index (higher is better).
    """
    current_vars = covariates.copy()
    best_aic = np.inf
    best_c_index = 0
    best_vars = current_vars.copy()
    best_df = None

    while len(current_vars) > 0:
        candidates = []

        # Test current model with different spline df values
        for df_spline in df_candidates:
            aic, c_idx = test_harrels_fp_model_with_cross_validation(train_df, current_vars, df_spline, n_splits)
            candidates.append((aic, c_idx, current_vars.copy(), df_spline))

        # Test removing each variable
        for var in current_vars:
            test_vars = [v for v in current_vars if v != var]
            if len(test_vars) == 0:
                continue
            for df_spline in df_candidates:
                aic, c_idx = test_harrels_fp_model_with_cross_validation(train_df, test_vars, df_spline, n_splits)
                candidates.append((aic, c_idx, test_vars.copy(), df_spline))

        # Find best candidate (minimum AIC)
        # best_candidate = min(candidates, key=lambda x: x[0])

        best_candidate = max(candidates, key=lambda x: x[1])  # x[1] is C-index
        # best_candidate = max(candidates, key=lambda x: (x[1], -x[0]))  # Maximize C-index, minimize AIC
        cand_aic, cand_c_idx, cand_vars, cand_df = best_candidate

        # Stop if no improvement
        # if cand_aic >= best_aic:
        #     break
        if cand_c_idx <= best_c_index:  # Stop if C-index doesn't improve
            break

        # Update best model
        best_aic = cand_aic
        best_c_index = cand_c_idx
        best_vars = cand_vars.copy()
        best_df = cand_df
        current_vars = cand_vars.copy()

    print("\nSelected model by Harrel's C-Index:")
    print("Variables:", best_vars)
    print("Spline df:", best_df)
    print("CV-AIC:", best_aic)
    print("CV C-index:", best_c_index)

    return best_vars, best_df, best_aic, best_c_index
