def make_columns_object(df, cols):
    for col in cols:
        df[col] = df[col].astype('object').astype(int)
    return df

# df = make_columns_object(df, ['European', 'Event'])
# df.info()
