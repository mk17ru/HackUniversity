import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

input_file = open("data_input.txt")
input_file_csv = open("data_input.csv", "w")

output_file = open("data_output.txt")
output_file_csv = open("data_output.csv", "w")

for string in input_file:
    input_file_csv.write(string.replace(' ', ','))
for string in output_file:
    output_file_csv.write(string.replace(' ', ','))

input_file.close()
input_file_csv.close()
output_file.close()
output_file_csv.close()

data_input = pd.read_csv('data_input.csv')
data_output = pd.read_csv('data_output.csv')

parameters_input = ['number', 'phone type', 'version of OS', 'application 1', 'application 2', 'application 3',
                    'gender', 'age', 'region', 'LTV', 'trips abroad', 'traffic volume', 'mobile rate', 'service 1',
                    'service 2']
parameters_output = ['bcg_color', 'icon_color', 'music', 'hot_news']
data_input.columns = parameters_input
data_output.columns = parameters_output

for par_in in parameters_input:
    data_input[par_in] = [str(string).__hash__() for string in data_input[par_in]]
# for par_out in parameters_output:
#     data_output[par_out] = [str(string).__hash__() for string in data_output[par_out]]

sc_x = StandardScaler()
np.savetxt("data_input_normalize.csv", sc_x.fit_transform(data_input), delimiter=',')
np.savetxt("data_output_normalize.csv", data_output, delimiter=',')
