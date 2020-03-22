from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
import pickle

data_input = pd.read_csv('data_input_normalize.csv').values
data_output = pd.read_csv('data_output_normalize.csv').values
model = MultiOutputRegressor(GradientBoostingRegressor(random_state=0)).fit(data_input, data_output)
# model = MultiOutputRegressor(RandomForestRegressor(max_depth=30, random_state=0)).fit(data_input, data_output)
filename = 'network_model.sav'
pickle.dump(model, open(filename, 'wb'))
