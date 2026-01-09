import numpy as np
def log_transform_columns(df, cols):
    for col in cols:
        df[col + "_l"] = np.log1p(df[col])  # ln(1 + x)
    return df
# df = log_transform_columns(df, ['Disease_Duration', 'Diagnostic_Delay', 'Vital_capacity'])
# df
