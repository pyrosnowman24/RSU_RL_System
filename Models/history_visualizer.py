from cmath import nan
from enum import unique
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

directory_path = "/home/acelab/Dissertation/RSU_RL_Placement/trained_models/"
model_name = "25_Epochs_30_Intersections_Scaled"
model_directory = os.path.join(directory_path,model_name+'/')
history_path = os.path.join(model_directory,"model_history.csv")

history_df = pd.read_csv(history_path, on_bad_lines='skip', engine = 'python')

# funky_boi = history_df[history_df['loss'] != 0]

# print(funky_boi.info)

loss = history_df['loss'].to_numpy()

x = np.arange(0,len(loss)+1,step = 100)
average = []

for i in range(len(x)-1):
    average.append(np.mean(loss[x[i]:x[i+1]]))
print(average)

fig,ax = plt.subplots(1)
ax.plot(np.arange(len(loss)),loss)
ax.plot(x[1:],average)
plt.show()