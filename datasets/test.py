
Q = pd.read_csv('datafiles/Q matrix.csv')
resp = pd.read_csv('datafiles/response.csv')
RT = pd.read_csv('datafiles/RTs.csv')
actions = pd.read_csv('datafiles/the number of actions.csv')


user_ids = np.arange(resp.shape[0], dtype=np.int64)
item_ids = np.arange(resp.shape[1], dtype=np.int64)

item_names = list(resp.columns)
RT.columns