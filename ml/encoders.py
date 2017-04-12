import pandas as pd

def ordinal(data):
    # {column_name: {value_1: 0, value_2: 1, value_3: 2}}
    # encodings['column_name'][value]
    encodings = {}
    columns = data.columns.values
    count = 0
    for index, row in data.iterrows():
        for column in columns:
            if column not in encodings:
                encodings[column] = {}
            value = row[column]
            if value not in encodings[column]:
                encodings[column][value] = len(encodings[column])
    return encodings
        