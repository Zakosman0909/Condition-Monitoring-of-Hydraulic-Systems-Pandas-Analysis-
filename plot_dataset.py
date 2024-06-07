import pandas as pd

file_path = 'PS1.txt' 

data = pd.read_csv(file_path, sep='\t', header=None)


print(data)