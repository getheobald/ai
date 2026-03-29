import pickle

file_path = 'Q_table_1000_0.99.pickle'

# Open the file in binary mode and load the data
with open(file_path, 'rb') as file:
    data = pickle.load(file) # deserialized python object

print(data)
