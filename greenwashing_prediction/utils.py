def get_X_y_from_dataset(df):
    X = df[[col for col in df.columns if col.startswith("dim_")]]
    y = df["greenwash_score"].values
    return X, y