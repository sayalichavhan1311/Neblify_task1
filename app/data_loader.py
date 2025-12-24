import pandas as pd

USERS_PATH = "data/users.csv"
TX_PATH = "data/transactions.csv"

users_df = pd.read_csv(USERS_PATH)
transactions_df = pd.read_csv(TX_PATH)
