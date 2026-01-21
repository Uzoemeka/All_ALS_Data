def make_columns_int(df, columns):
    df = df.copy()
    for col in columns:
        df[col] = df[col].fillna(0).astype(int)
    return df
