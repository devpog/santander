def read_file(dir, type, low_memory=True, dtype={"sexo":str, "ind_nuevo":str, "ult_fec_cli_1t":str, "indext":str}):
    """
    Read a file with accordance to its type provided:
    train = regex(train)
    test = regex(test)
    :param dir:
    :param type:
    :return:
    """
    import os
    import re
    import pandas as pd

    try:
        file = [os.path.join(dir, f) for f in os.listdir(dir) if re.match('.*{}.*'.format(type), f)].pop()
    except IndexError as err:
        print('File {} not found\n{}'.format(file, err))
        return 1

    try:
        raw = pd.read_csv(file, dtype=dtype, low_memory=low_memory)
    except Exception as err:
        print('Read error\n{}'.format(err))
        return 1

    return raw

def change_column_names(df):
    """
    Create a hash to use short names such as X1..XN for convenience
    and rename train set accordingly
    """
    col_map = dict((i, x) for (i, x) in enumerate(df.columns))
    col_dict = dict((v, 'X'+str(k+1)) for (k, v) in col_map.items())
    name_map = dict((c, col_dict.pop(c)) for c in df.columns)
    df.rename(columns=name_map, inplace=True)

    return df, name_map

def select_type(df, dtype, return_df=False):
    """
    Method to select columns with specified data types only
    """
    cols_to_return = []
    types = df.dtypes

    if isinstance(dtype, list):
        for d in dtype:
            cols_to_return.append([ind for ind in types.index if str(types[ind]) == d])

    if not return_df:
        return [i for l1 in cols_to_return for i in l1]
    else:
        return df.loc[:, [i for l1 in cols_to_return for i in l1]]

def nan_share(df):
    """
    Show the percentage of missing values across all rows
    """
    import pandas as pd
    nans = dict()
    for c in df.columns:
        s = df[c]
        stf = pd.Series(pd.isnull(s).values).value_counts()
        try:
            if len(stf) > 1:
                nans[c] = stf[1]/len(s)
        except IndexError as err:
            print(err)
    return nans
