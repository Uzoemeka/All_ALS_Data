import pandas as pd
from lifelines import NelsonAalenFitter

def estimate_baseline_cumhaz(original_df, time_col='time', event_col='event'):
    """
    Estimate baseline cumulative hazard from real data using Nelson–Aalen.
    """
    naf = NelsonAalenFitter()
    naf.fit(original_df[time_col], event_observed=original_df[event_col])

    H0 = naf.cumulative_hazard_.reset_index()
    H0.columns = ['time', 'cumhaz']
    return H0

# H0_df = estimate_baseline_cumhaz(train_df, time_col='Disease_Duration', event_col='Event')
# H0_df

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

def estimate_baseline_cumhaz_breslow(placebo_df: pd.DataFrame,
                                     time_col: str = "Disease_Duration",
                                     event_col: str = "Event",
                                     covariate_cols: list[str] = None) -> pd.DataFrame:
    """
    Fit Cox proportional hazards on the real placebo arm and return the
    Breslow baseline cumulative hazard H0(t) (baseline = covariates = 0).

    Why: This matches the Cox proportional hazards data-generating equation
    used for inverse transform simulation.
    """
    if covariate_cols is None:
        raise ValueError("Provide covariate_cols (your 5 baseline covariates).")

    df_fit = placebo_df[[time_col, event_col] + covariate_cols].copy()

    cph = CoxPHFitter()
    cph.fit(df_fit, duration_col=time_col, event_col=event_col)

    # lifelines stores baseline cumulative hazard as a DataFrame indexed by time
    H0 = cph.baseline_cumulative_hazard_.reset_index()
    H0.columns = ["time", "cumhaz"]
    return H0, cph
