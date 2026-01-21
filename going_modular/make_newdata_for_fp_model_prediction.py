import pandas as pd
from rpy2.robjects import FloatVector, IntVector, DataFrame
from rpy2.robjects import pandas2ri, default_converter
from rpy2.robjects.conversion import localconverter
from typing import List, Dict, Optional


def make_newdata_for_fp_model_prediction(df, times, mean_cols, mode_cols, time_col_name="Disease_Duration"):

    # Compute column summaries
    mean_cols = {col: df[col].mean() for col in mean_cols}
    mode_cols = {col: df[col].mode()[0] for col in mode_cols}

    # Build dictionary for R DataFrame
    r_dict = {}
    for col, mean in mean_cols.items():
        r_dict[col] = FloatVector([mean] * len(times))
    for col, mode in mode_cols.items():
        r_dict[col] = IntVector([mode] * len(times))

    # Add the time column
    r_dict[time_col_name] = FloatVector(times)

    with localconverter(default_converter + pandas2ri.converter):
        newdata_r = DataFrame(r_dict)

    return newdata_r


# def make_newdata_for_fp_model_prediction(
#     df: pd.DataFrame,
#     mean_cols: Optional[List[str]] = None,
#     mode_cols: Optional[List[str]] = None,
#     time_col_name: str = "Disease_Duration",
#     times: Optional[List[float]] = None
# ) -> DataFrame:
#     """
#     Create an R DataFrame with aggregated statistics from a pandas DataFrame.
    
#     Parameters
#     ----------
#     df : pd.DataFrame
#         Input pandas DataFrame containing the source data
#     mean_cols : List[str], optional
#         Column names to aggregate using mean. If None, uses default columns.
#     mode_cols : List[str], optional
#         Column names to aggregate using mode. If None, uses default columns.
#     time_col_name : str, default="Disease_Duration"
#         Name for the time column in the output DataFrame
#     times : List[float], optional
#         List of time points to use for the time column (numeric values).
#         If None, uses unique values from df[time_col_name]
    
#     Returns
#     -------
#     DataFrame
#         R DataFrame with aggregated statistics repeated for each time point
    
#     Examples
#     --------
#     >>> df = pd.DataFrame({
#     ...     'Age': [45, 50, 55],
#     ...     'TRICALS': [40, 45, 50],
#     ...     'Sex_Male': [1, 0, 1],
#     ...     'Disease_Duration': [0, 6, 12]
#     ... })
#     >>> # Using column names for aggregation
#     >>> result = make_newdata(df, mean_cols=['Age', 'TRICALS'], 
#     ...                       mode_cols=['Sex_Male'])
#     >>> # Or specify custom times
#     >>> result = make_newdata(df, ['Age'], ['Sex_Male'], times=[0, 6, 12, 18])
#     """
#     # Default columns if not specified
#     if mean_cols is None:
#         mean_cols = ["Age", "TRICALS", "Vital_capacity", "Diagnostic_Delay"]
    
#     if mode_cols is None:
#         mode_cols = ["Onset_Limb", "Sex_Male", "Study_Arm_Placebo"]
    
#     # Get times from dataframe if not provided
#     if times is None:
#         if time_col_name in df.columns:
#             times = sorted(df[time_col_name].unique())
#         else:
#             raise ValueError(f"Column '{time_col_name}' not found in DataFrame and times not provided")
    
#     # Convert times to list of floats to ensure compatibility
#     times_list = [float(t) for t in times]
#     n_times = len(times_list)
#     data_dict = {}
    
#     # Calculate means for specified columns
#     for col in mean_cols:
#         if col in df.columns:
#             mean_val = float(df[col].mean())
#             data_dict[col] = FloatVector([mean_val] * n_times)
#         else:
#             print(f"Warning: Column '{col}' not found in DataFrame. Skipping.")
    
#     # Calculate modes for specified columns
#     for col in mode_cols:
#         if col in df.columns:
#             mode_val = df[col].mode()[0] if not df[col].mode().empty else 0
#             data_dict[col] = IntVector([int(mode_val)] * n_times)
#         else:
#             print(f"Warning: Column '{col}' not found in DataFrame. Skipping.")
    
#     # Add time column - ensure it's a list of floats
#     data_dict[time_col_name] = FloatVector(times_list)
    
#     # Convert to R DataFrame
#     with localconverter(default_converter + pandas2ri.converter):
#         newdata_r = DataFrame(data_dict)
    
#     return newdata_r


# def make_newdata_custom(
#     df: pd.DataFrame,
#     times: List[float],
#     column_config: Dict[str, Dict],
#     time_col_name: str = "Disease_Duration"
# ) -> DataFrame:
#     """
#     Create an R DataFrame with custom aggregation functions.
    
#     Parameters
#     ----------
#     df : pd.DataFrame
#         Input pandas DataFrame
#     times : List[float]
#         List of time points (numeric values)
#     column_config : Dict[str, Dict]
#         Configuration dictionary with format:
#         {
#             'column_name': {
#                 'agg': 'mean' | 'mode' | 'median' | callable,
#                 'dtype': 'float' | 'int'
#             }
#         }
#     time_col_name : str
#         Name for the time column
    
#     Returns
#     -------
#     DataFrame
#         R DataFrame with custom aggregated statistics
    
#     Examples
#     --------
#     >>> config = {
#     ...     'Age': {'agg': 'mean', 'dtype': 'float'},
#     ...     'Sex_Male': {'agg': 'mode', 'dtype': 'int'},
#     ...     'TRICALS': {'agg': 'median', 'dtype': 'float'}
#     ... }
#     >>> times = [0, 6, 12, 18]  # Numeric time points
#     >>> result = make_newdata_custom(df, times, config)
#     """
#     # Convert times to list of floats to ensure compatibility
#     times_list = [float(t) for t in times]
#     n_times = len(times_list)
#     data_dict = {}
    
#     for col, config in column_config.items():
#         if col not in df.columns:
#             print(f"Warning: Column '{col}' not found. Skipping.")
#             continue
        
#         agg_func = config.get('agg', 'mean')
#         dtype = config.get('dtype', 'float')
        
#         # Calculate aggregated value
#         if agg_func == 'mean':
#             value = df[col].mean()
#         elif agg_func == 'mode':
#             value = df[col].mode()[0] if not df[col].mode().empty else 0
#         elif agg_func == 'median':
#             value = df[col].median()
#         elif callable(agg_func):
#             value = agg_func(df[col])
#         else:
#             raise ValueError(f"Unknown aggregation function: {agg_func}")
        
#         # Convert to appropriate R vector type
#         if dtype == 'int':
#             data_dict[col] = IntVector([int(value)] * n_times)
#         else:
#             data_dict[col] = FloatVector([float(value)] * n_times)
    
#     # Add time column - ensure it's a list of floats
#     data_dict[time_col_name] = FloatVector(times_list)
    
#     with localconverter(default_converter + pandas2ri.converter):
#         newdata_r = DataFrame(data_dict)
    
#     return newdata_r
