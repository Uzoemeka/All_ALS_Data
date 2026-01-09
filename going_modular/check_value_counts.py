def check_value_counts(df):
    for col in df.select_dtypes(include=['category', 'object']).columns:
        print(f"{col}: {df[col].value_counts().to_dict()}\n")

# check_value_counts(df)
