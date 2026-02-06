import numpy as np
import pandas as pd
from scipy.stats import norm
from .gaussian_copula_samples import gaussian_copula_samples
from .sample_survival_times_from_empirical_baseline import sample_survival_times_from_empirical_baseline

def sample_censoring_times_from_censored_only(
    placebo_df: pd.DataFrame,
    n: int,
    time_col: str,
    event_col: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Empirical resampling of censoring times using only censored participants.
    Assumes event_col == 1 indicates event, 0 indicates censored.
    """
    cens_times = placebo_df.loc[placebo_df[event_col] == 0, time_col].to_numpy(dtype=float)

    if len(cens_times) == 0:
        raise ValueError("No censored observations found; cannot resample censoring times.")

    return rng.choice(cens_times, size=n, replace=True)




def sample_followup_times_empirical(
    placebo_df: pd.DataFrame,
    n: int,
    time_col: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Resample observed follow-up times (regardless of event status).
    """
    times = placebo_df[time_col].to_numpy(dtype=float)
    return rng.choice(times, size=n, replace=True)


def sample_censoring_times_administrative(
    placebo_df: pd.DataFrame,
    n: int,
    time_col: str,
) -> np.ndarray:
    """
    Administrative censoring at study end.
    Everyone has censoring time equal to the maximum observed follow-up.
    """
    t_admin = float(placebo_df[time_col].max())
    return np.full(n, t_admin, dtype=float)

def sample_censoring_times_mixture_admin_dropout(
    placebo_df: pd.DataFrame,
    n: int,
    time_col: str,
    event_col: str,
    rng: np.random.Generator,
    admin_frac: float,
    admin_tol: float = 1e-6,
) -> np.ndarray:
    """
    Mixture censoring model:
      - with probability admin_frac: administrative censoring at t_admin
      - otherwise: resample dropout censoring times among censored participants
        that are NOT at the administrative time.

    Why relevant: your real placebo has ~80% administrative censoring.
    """
    t_admin = float(placebo_df[time_col].max())

    cens_times = placebo_df.loc[placebo_df[event_col] == 0, time_col].to_numpy(dtype=float)
    if len(cens_times) == 0:
        return np.full(n, t_admin, dtype=float)

    # Dropout censoring times = censored times not essentially at the admin cutoff
    dropout_times = cens_times[np.abs(cens_times - t_admin) > admin_tol]
    if len(dropout_times) == 0:
        return np.full(n, t_admin, dtype=float)

    u = rng.uniform(size=n)
    admin_mask = (u < admin_frac)

    C = np.empty(n, dtype=float)
    C[admin_mask] = t_admin
    C[~admin_mask] = rng.choice(dropout_times, size=(~admin_mask).sum(), replace=True)
    return C




def simulate_cox_dataset_with_correlation(
    n,
    baseline="empirical",
    H0_df=None,
    censoring="empirical_censored_only",   # new
    placebo_df_for_censor=None,            # new
    time_col_censor="Disease_Duration",    # new
    event_col_censor="Event",              # new
    lam=0.1,
    rho=1.0,
    censor_rate=0.3,
    seed=None,
    var_specs=None,
    corr=None,
    admin_frac = 0.8
):
    rng = np.random.default_rng(seed)
    if var_specs is None:
        raise ValueError("var_specs must be provided.")
    p = len(var_specs)

    # ----- STEP 1: correlated uniforms -----
    if corr is None:
        U = None
    else:
        corr = np.asarray(corr)
        if corr.shape != (p, p):
            raise ValueError("corr must be p x p with p = len(var_specs)")
        U = gaussian_copula_samples(n, corr)

    df = pd.DataFrame(index=np.arange(n))
    linear_pred = np.zeros(n)

    # ----- STEP 2: covariates -----
    for j, spec in enumerate(var_specs):
        name = spec["name"]
        typ = spec["type"].lower()
        uj = None if U is None else U[:, j]

        if typ == "continuous":
            dist = spec.get("dist", {"kind":"normal", "mean":0, "sd":1})
            mu = dist.get("mean", 0)
            sd = dist.get("sd", 1)
            if uj is None:
                x = rng.normal(mu, sd, size=n)
            else:
                x = mu + sd * norm.ppf(uj)
            df[name] = x
            linear_pred += float(spec.get("coef", 0.0)) * x

        elif typ == "binary":
            p_bin = float(spec.get("prob", 0.5))
            x = rng.binomial(1, p_bin, size=n) if uj is None else (uj < p_bin).astype(int)
            df[name] = x
            linear_pred += float(spec.get("coef", 0.0)) * x

        else:
            raise ValueError(f"Unsupported variable type: {typ}")

    # ----- STEP 3: event times -----
    U_time = rng.uniform(size=n)

    if baseline == "empirical":
        if H0_df is None:
            raise ValueError("You must supply H0_df when baseline='empirical'.")
        T = sample_survival_times_from_empirical_baseline(U_time, H0_df, linear_pred)

    elif baseline == "exponential" or rho == 1.0:
        T = -np.log(U_time) / (lam * np.exp(linear_pred))

    elif baseline == "weibull":
        T = (-np.log(U_time)) ** (1.0 / rho) / (lam * np.exp(linear_pred)) ** (1.0 / rho)

    else:
        raise ValueError("baseline must be 'empirical', 'exponential', or 'weibull'")

        # ----- STEP 4: censoring times -----
    if censoring == "administrative":
        if placebo_df_for_censor is None:
            raise ValueError("placebo_df_for_censor must be provided for administrative censoring.")
        C = sample_censoring_times_administrative(
            placebo_df_for_censor, n, time_col_censor
        )

    elif censoring == "mixture_admin_dropout":
        if placebo_df_for_censor is None:
            raise ValueError("placebo_df_for_censor must be provided for mixture censoring.")
        C = sample_censoring_times_mixture_admin_dropout(
            placebo_df_for_censor, n,
            time_col=time_col_censor,
            event_col=event_col_censor,
            rng=rng,
            admin_frac=admin_frac  # you told me 80%
        )

    elif censoring.startswith("empirical"):
        if placebo_df_for_censor is None:
            raise ValueError("placebo_df_for_censor must be provided for empirical censoring.")

        if censoring == "empirical_censored_only":
            C = sample_censoring_times_from_censored_only(
                placebo_df_for_censor, n, time_col_censor, event_col_censor, rng
            )
        elif censoring == "empirical_followup":
            C = sample_followup_times_empirical(
                placebo_df_for_censor, n, time_col_censor, rng
            )
        else:
            raise ValueError("Unknown empirical censoring mode.")

    else:
        # fallback heuristic (kept for compatibility)
        scale_c = max(1e-6, T.mean() * censor_rate / (1 - censor_rate + 1e-9))
        C = rng.exponential(scale=scale_c, size=n)


    observed_time = np.minimum(T, C)
    event = (T <= C).astype(int)

    df["time"] = observed_time
    df["event"] = event
    df["true_survival_time"] = T
    df["linear_predictor"] = linear_pred

    cols = ["time", "event", "true_survival_time", "linear_predictor"]
    cols += [c for c in df.columns if c not in cols]
    return df[cols]




# def simulate_cox_dataset_with_correlation(placebo_df: pd.DataFrame,
#                                           n,
#                                           baseline='weibull',
#                                             H0_df=None,
#                                             lam=0.1,
#                                             rho=1.0,
#                                             censor_rate=0.3,
#                                             seed=None,
#                                             var_specs=None,
#                                             corr=None):
#     """
#     Simulate a Cox proportional hazards dataset WITH correlation among covariates
#     using a Gaussian copula approach.

#     Parameters
#     ----------
#     n : int
#         Sample size.
#     baseline : {'weibull','exponential'}
#     lam : float
#         Baseline hazard parameter.
#     rho : float
#         Weibull shape parameter.
#     censor_rate : float
#         Desired censoring proportion (approx).
#     seed : int or None
#     var_specs : list of dicts
#         Each dict must include:
#             name : str
#             type : {'continuous','binary','categorical'}
#             and distribution details.
#             coef : float or dict (for categorical)
#     corr : array-like or None
#         Correlation matrix among the variables in var_specs order.
#         If None: variables are generated independently.

#     Returns
#     -------
#     DataFrame with:
#         time, event, true_survival_time, linear_predictor, `<covariates...>`
#     """

#     rng = np.random.default_rng(seed)

#     if var_specs is None:
#         raise ValueError("var_specs must be provided.")

#     p = len(var_specs)

#     # ----- STEP 1: Generate correlated uniforms -----
#     if corr is None:
#         U = None
#     else:
#         corr = np.asarray(corr)
#         if corr.shape != (p, p):
#             raise ValueError("corr must be p x p with p = len(var_specs)")
#         U = gaussian_copula_samples(n, corr)

#     # storage
#     df = pd.DataFrame(index=np.arange(n))
#     linear_pred = np.zeros(n)

#     # ----- STEP 2: Generate covariates with correct marginals but correlated -----
#     for j, spec in enumerate(var_specs):
#         name = spec['name']
#         typ = spec['type'].lower()

#         # Uniform samples for this variable
#         uj = None if U is None else U[:, j]

#         # ============================================================
#         # CONTINUOUS
#         # ============================================================
#         if typ == 'continuous':
#             dist = spec.get('dist', {'kind': 'normal', 'mean': 0, 'sd': 1})

#             if dist['kind'] == 'normal':
#                 mu = dist.get('mean', 0)
#                 sd = dist.get('sd', 1)

#                 if uj is None:
#                     x = rng.normal(mu, sd, size=n)
#                 else:
#                     z = norm.ppf(uj)
#                     x = mu + sd * z

#             elif dist['kind'] == 'uniform':
#                 lo = dist.get('low', 0)
#                 hi = dist.get('high', 1)
#                 if uj is None:
#                     x = rng.uniform(lo, hi, size=n)
#                 else:
#                     x = lo + uj * (hi - lo)

#             else:
#                 raise ValueError(f"Unsupported continuous distribution: {dist['kind']}")

#             df[name] = x
#             coef = float(spec.get('coef', 0.0))
#             linear_pred += coef * x

#         # ============================================================
#         # BINARY
#         # ============================================================
#         elif typ == 'binary':
#             p_bin = float(spec.get('prob', 0.5))

#             if uj is None:
#                 x = rng.binomial(1, p_bin, size=n)
#             else:
#                 x = (uj < p_bin).astype(int)

#             df[name] = x
#             coef = float(spec.get('coef', 0.0))
#             linear_pred += coef * x

#         # ============================================================
#         # CATEGORICAL
#         # ============================================================
#         elif typ == 'categorical':
#             levels = list(spec['levels'])
#             probs = np.asarray(spec.get('probs', [1 / len(levels)] * len(levels)))

#             # boundaries for inverse-CDF categories
#             cum = np.cumsum(probs)

#             if uj is None:
#                 cats = rng.choice(levels, p=probs, size=n)
#             else:
#                 # Assign category by uniform bins
#                 cats = np.empty(n, dtype=object)
#                 for k, lvl in enumerate(levels):
#                     if k == 0:
#                         mask = uj <= cum[k]
#                     else:
#                         mask = (uj > cum[k - 1]) & (uj <= cum[k])
#                     cats[mask] = lvl

#             df[name] = pd.Categorical(cats, categories=levels)

#             # Add to linear predictor
#             coef_map = spec.get('coef', {})
#             ref = spec.get('ref', levels[0])
#             for lvl in levels:
#                 if lvl == ref:
#                     continue
#                 coef_lvl = float(coef_map.get(lvl, 0.0))
#                 linear_pred += coef_lvl * (cats == lvl)

#         else:
#             raise ValueError(f"Unsupported variable type: {typ}")

#     # ----- STEP 3: Generate survival times -----
#     U_time = rng.uniform(size=n)

#     if baseline == "empirical":
#         if H0_df is None:
#             raise ValueError("You must supply H0_df when using empirical baseline.")
#         T = sample_survival_times_from_empirical_baseline(U_time, H0_df, linear_pred)

#     elif baseline == 'exponential' or rho == 1.0:
#         T = -np.log(U_time) / (lam * np.exp(linear_pred))

#     elif baseline == 'weibull':
#         T = (-np.log(U_time)) ** (1.0 / rho) / (lam * np.exp(linear_pred)) ** (1.0 / rho)
#     else:
#         raise ValueError("baseline must be 'exponential' or 'weibull'")

#     # ----- STEP 4: Generate censoring -----
#     # heuristic to reach the target censoring proportion
#     # scale_c = max(1e-6, T.mean() * censor_rate / (1 - censor_rate + 1e-9))
#     # C = rng.exponential(scale=scale_c, size=n)

#     # observed_time = np.minimum(T, C)
#     # event = (T <= C).astype(int)

#     C = sample_censoring_times_from_censored_only(
#         placebo_df=placebo_df,
#         n=n,
#         time_col="Disease_Duration",
#         event_col="Event",
#         rng=rng,
#         )
#     observed_time = np.minimum(T, C)
#     event = (T <= C).astype(int)


#     # ----- STEP 5: Build output -----
#     df['time'] = observed_time
#     df['event'] = event
#     df['true_survival_time'] = T
#     df['linear_predictor'] = linear_pred

#     cols = ['time', 'event', 'true_survival_time', 'linear_predictor']
#     cols += [c for c in df.columns if c not in cols]
#     df = df[cols]

#     return df

