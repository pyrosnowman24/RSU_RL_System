import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

directory_path = "/home/acelab/Dissertation/RSU_RL_Placement/trained_models/"
model_name = "50_epochs_50_intersections"
model_directory = os.path.join(directory_path,model_name+'/')
history_path = os.path.join(model_directory,"model_history.csv")

history_df = pd.read_csv(history_path, on_bad_lines='skip', engine = 'python')

loss = history_df['loss'].to_numpy()

fig,ax = plt.subplots(1)
ax.plot(loss)
plt.show()