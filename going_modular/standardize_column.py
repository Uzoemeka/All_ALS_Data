def standardize_column(df):
    for col in df.select_dtypes(include=['float']).columns:
        df[col + '_std'] = (df[col] - df[col].mean()) / df[col].std()
    return df

# df = standardize_column(df)
# df.head(3)
