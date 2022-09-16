from cmath import nan
from enum import unique
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, re

directory_path = "/home/acelab/Dissertation/RSU_RL_Placement/trained_models/"
model_name = "Training_for_Reward"
model_directory = os.path.join(directory_path,model_name+'/')
history_path = os.path.join(model_directory,"model_history.csv")

history_df = pd.read_csv(history_path, on_bad_lines='skip', engine = 'python')

rsu_history = history_df['rsu_network']
intersections_history = history_df['intersection_idx']
loss_history = history_df['loss']
avg_rsu = 0

for rsu_network in rsu_history:
    results = re.findall(r"\d+",rsu_network)
    avg_rsu += len(results)
    avg_rsu /= 2
print(avg_rsu)

loss = history_df['loss']

loss = loss.mask(loss > loss.quantile(0.90)) # Removes outlies from loss, some are >10,000 when most are ~200 - 1000

loss = loss.to_numpy()

x = np.arange(0,len(loss)+1,step = 100)
average = []

for i in range(len(x)-1):
    average.append(np.nanmean(loss[x[i]:x[i+1]]))
print(average)

fig,ax = plt.subplots(1)
ax.plot(np.arange(len(loss)),loss)
ax.plot(x[1:],average)
plt.show()

# len_rsu_net = np.zeros(shape = rsu_history.shape)
# for i in range(len(rsu_history)):
#     results = re.findall(r"\d+",rsu_history[i])
#     len_rsu_net[i] = len(results)

# len_intersections = np.zeros(shape = intersections_history.shape)
# for i in range(len(intersections_history)):
#     results = np.array(re.findall(r"\d+",intersections_history[i])).astype(int)
#     len_intersections[i] = len(results[results>0])

# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')

# ax.scatter(len_intersections,len_rsu_net,loss_history)
# ax.set_xlabel("Number of Intersections")
# ax.set_ylabel("Number of RSU")
# ax.set_zlabel("Loss")
# plt.show()