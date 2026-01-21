import numpy as np


def sample_survival_times_from_empirical_baseline(U, H0_df, linear_pred):
    """
    Inverse transform sampling using an empirical baseline cumulative hazard.
    """
    H0_t = H0_df["time"].values
    H0_vals = H0_df["cumhaz"].values

    v = -np.log(U) / np.exp(linear_pred)

    # Interpolate inverse of H0
    T = np.interp(v, H0_vals, H0_t, left=H0_t[0], right=H0_t[-1])
    return T
