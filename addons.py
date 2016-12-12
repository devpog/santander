def read_file(dir, type, low_memory=True, dtype={"sexo":str, "ind_nuevo":str, "ult_fec_cli_1t":str, "indext":str}, limit=False):
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

    if limit:
        num_rows = 700000

    try:
        file = [os.path.join(dir, f) for f in os.listdir(dir) if re.match('.*{}.*'.format(type), f)].pop()
    except IndexError as err:
        print('File {} not found\n{}'.format(file, err))
        return 1

    try:
        if type == 'train':
            if limit:
                raw = pd.read_csv(file, dtype=dtype, low_memory=False, nrows=num_rows)
            else:
                raw = pd.read_csv(file, dtype=dtype, low_memory=False)
        elif type == 'test':
            raw = pd.read_csv(file, dtype=dtype)
        elif type == 'reservoir':
            raw = pd.read_csv(file, dtype=dtype)
        else:
            raise Exception
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


def change_dtype(df, columns, type):
    import pandas as pd

    for c in columns:
        print('Casting {} into {}'.format(c, type))
        try:
            df.loc[:, c] = df.loc[:, c].astype(type)
        except ValueError:
            nan = pd.notnull(df.loc[:, c])
            df.loc[nan, c] = df.loc[nan, c].astype(type)
    return df


def get_random_sample(df, id):
    """
    Return a random 10% sample of the original data set
    """
    import pandas as pd

    size = int((0.1*len(df)))

    ids = pd.Series(df.loc[:, id].unique())
    u_id = ids.sample(n=size)
    return df[df.loc[:, id].isin(u_id)]


def reservoir_sampling(dir, k, type='train', reservoir='reservoir.csv'):
    import os
    import re
    import random

    try:
        file = [os.path.join(dir, f) for f in os.listdir(dir) if re.match('.*{}.*'.format(type), f)].pop()
    except IndexError as err:
        print('File {} not found\n{}'.format(file, err))
        return 1

    try:
        result = dict()
        with open(file) as f:
            for i, line in enumerate(f):
                if i < k:
                    result[i] = line
                else:
                    seed = random.randint(1, i)
                    if seed < k:
                        result[seed] = line

        with open(dir + '/' + reservoir, 'w') as fw:
            for (k, v) in result.items():
                fw.write(v)
            fw.close()

        return reservoir.split('.')[0]
    except Exception as err:
        print('Read error\n{}'.format(err))
        return 1


