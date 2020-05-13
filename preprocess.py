def prep_data(df):

    X = df[["Height", "Width", "Length1", "Length2", "Length3"]].values
    y = df["Weight"].values

    #df = df.assign(hw=df["Height"] * df["Width"])
    #X = df[["Height", "Width", "hw"]].values
    #y = df["Weight"].values

    return X, y
