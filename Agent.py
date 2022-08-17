from audioop import avg
from importlib.resources import path
import torch
import numpy as np
from numpy.random import default_rng
import os
import subprocess
import re
import pyproj
import pandas as pd

class Agent:
    # This class will be responsible for keeping track of the current simulation and the environment for the simulation (sumo stuff)
    def __init__(self):
        self.setup_paths()
        self.setup_simulation(index=0)
        self.network_intersections = torch.Tensor(self.prepare_intersections()) # The coordinates of all intersections

    def setup_paths(self):
        self.parent_dir ="/home/acelab/veins_sim/"
        self.simulation_dir = os.path.join(self.parent_dir,"veins/examples/veins/")
        self.simulation_info = os.path.join(self.simulation_dir,"simulation_info.csv")
        self.simulation_variables_db = pd.read_csv(self.simulation_info)
        self.logs_dir = os.path.join(self.parent_dir,"logs/")
        self.omnet_ini = os.path.join(self.simulation_dir,"omnetpp.ini")
        self.scenario_ned = os.path.join(self.simulation_dir,'RSUExampleScenario.ned')
        self.omnet_results_file = os.path.join(self.simulation_dir,"results/General-#0.sca")

    def setup_simulation(self,index=0):
        # This will setup similar things to the cnn setup, stuff like the world image, the transforms, the size of the samples, ect.
        #
        new_simulation_variables_db = self.simulation_variables_db.iloc[index,:]
        
        self.sumo_scenario_dir = os.path.join(self.simulation_dir,new_simulation_variables_db.loc["sim_path"])
        self.sumo_net_xml = os.path.join(self.sumo_scenario_dir,new_simulation_variables_db.loc["net_name"])
        self.sumo_launchd_file = os.path.join(self.sumo_scenario_dir,new_simulation_variables_db.loc["launchd_name"])
        playground_size = new_simulation_variables_db.loc[["playground_x","playground_y","playground_z"]].to_numpy()
        self.omnet_dimensions = new_simulation_variables_db.loc[["min_omnet_x","min_omnet_y","max_omnet_x","max_omnet_y"]].to_numpy()
        simulation_time = new_simulation_variables_db[["start_time","stop_time"]].to_numpy()

        self.update_omnet_playground(playground_size)
        self.update_omnet_simulation_time(simulation_time)
        self.update_omnet_launchd()


# Functions to interact with the simulation.

    def simulation_step(self,rsu_network_idx,sim_idx,W=[.5,.5],model = 'Actor Critic'):
        # Receive new action, add it to the state, then gather intersections in RSU network
        try: subprocess.Popen("kill $(cat sumo-launchd.pid)",cwd=self.logs_dir,shell=True)
        except: pass
        
        rsu_network = self.get_simulation_rsu_network(rsu_network_idx,sim_idx)
        self.place_rsu_network(rsu_network)
        process1 = subprocess.Popen("./run.sh -d",cwd=self.parent_dir,shell=True)
        process2 = subprocess.Popen("./run -u Cmdenv",cwd=self.simulation_dir,shell=True)
        # process2 = subprocess.Popen("./run",cwd=self.simulation_dir,shell=True)
        process2.wait()
        process3 = subprocess.Popen("kill $(cat sumo-launchd.pid)",cwd=self.logs_dir,shell=True)

        # features = self.collect_all_results(desired_features = ["recvPower_dBm:mean","TotalLostPackets"])
        features = self.collect_rsu_results(rsu_network_idx,desired_features = ["recvPower_dBm:mean","ReceivedBroadcasts"])
        if model == "Policy Gradient":
            reward = self.reward_pg(features,W)
        else:
            reward = self.reward(features,W)

        return reward

    def reward(self,features,W):
        avg_features = np.nanmean(features,axis=1)
        avg_features[0] = -(300/avg_features[0])
        avg_features[1] = 200/avg_features[1,:]
        reward = np.multiply(avg_features,W)
        reward = np.sum(reward)
        return reward

    def reward_pg(self,features,W):
        features = np.nan_to_num(features,nan = -10000)
        # print(features)
        features[0] = np.power(100/(features[0,:]+30),2)
        features[1] = .025*features[1]
        features = np.sum(features,axis=0)
        return features

    def kill_sumo_env(self):
        print('\n',os.path.exists("sumo-launchd.pid"),'\n')
        if os.path.exists("sumo-launchd.pid"):
            print("killing")
            subprocess.Popen("kill $(cat sumo-launchd.pid)",cwd=self.logs_dir,shell=True)
        else: print("lasdjhfapsdncjkvnawuinbiufah")

    def display_environment(self,rsu_network_idx,sim_idx):
        rsu_network = self.get_simulation_rsu_network(rsu_network_idx,sim_idx)
        self.place_rsu_network(rsu_network)
        process2 = subprocess.Popen("./run",cwd=self.simulation_dir,shell=True)

# Functions to update the simulation

    def place_rsu_network(self,intersections):
        self.update_omnet_ini(intersections)
        self.update_scenario_params(len(intersections))
    
    def update_omnet_ini(self,intersections):
        f = open(self.omnet_ini,"r+")
        lines = f.readlines()

        start_index,end_index = self.find_rsu_text(lines)

        text_begining = lines[:start_index]
        text_end = lines[end_index:]

        rsu_text = []

        for i,intersection in enumerate(intersections):
            self.add_RSU(i,intersection,rsu_text)

        new_lines = text_begining+rsu_text+text_end

        f.seek(0)
        for line in new_lines:
            f.write(line)
        f.truncate()

    def update_scenario_params(self,num_rsus):
        f = open(self.scenario_ned,"r+")
        desired_string = ": RSU {\n"
        lines = f.readlines()
        for i,line in enumerate(lines):
            if line.find(desired_string) != -1:
                line_index = i
                break
        text_begining = lines[:line_index]
        text_end = lines[line_index+1:]

        new_text = [f"        rsu[{num_rsus}]: RSU {{\n"]
        
        new_lines = text_begining+new_text+text_end

        f.seek(0)
        for line in new_lines:
            f.write(line)
        f.truncate()

    def update_omnet_playground(self,playground_size):
        f = open(self.omnet_ini,"r+")
        lines = f.readlines()

        start_index,end_index = self.find_playground_text(lines)

        text_begining = lines[:start_index-1]
        text_end = lines[end_index:]

        playground_text = []

        playground_text.append("*.playgroundSizeX = {}m\n".format(playground_size[0]))
        playground_text.append("*.playgroundSizeY = {}m\n".format(playground_size[1]))
        playground_text.append("*.playgroundSizeZ = {}m\n".format(playground_size[2]))

        new_lines = text_begining+playground_text+text_end

        f.seek(0)
        for line in new_lines:
            f.write(line)
        f.truncate()

    def update_omnet_simulation_time(self,simulation_time):
        f = open(self.omnet_ini,"r+")
        lines = f.readlines()

        start_index,end_index = self.find_simulation_time_text(lines)

        text_begining = lines[:start_index-1]
        text_end = lines[end_index:]

        sim_time_text = []

        sim_time_text.append("*.manager.firstStepAt = {}s\n".format(simulation_time[0]))
        sim_time_text.append("sim-time-limit = {}s\n".format(simulation_time[1]))

        new_lines = text_begining+sim_time_text+text_end

        f.seek(0)
        for line in new_lines:
            f.write(line)
        f.truncate()

    def update_omnet_launchd(self):
        f = open(self.omnet_ini,"r+")
        lines = f.readlines()
        desired_string = "*.manager.launchConfig ="
        for i, line in enumerate(lines):
            if desired_string in line:
                index = i + 1
                break

        text_begining = lines[:index-1]
        text_end = lines[index:]

        sim_time_text = []

        sim_time_text.append("*.manager.launchConfig = xmldoc(\"{}\")\n".format(self.sumo_launchd_file))

        new_lines = text_begining+sim_time_text+text_end

        f.seek(0)
        for line in new_lines:
            f.write(line)
        f.truncate()

    def add_RSU(self,index,rsu_coords,rsu_text):
        rsu_coord_string = lambda index, coord, position : "*.rsu[{}].mobility.{} = {}\n".format(index,coord,position)
        rsu_text.append(rsu_coord_string(index,'x',rsu_coords[0]))
        rsu_text.append(rsu_coord_string(index,'y',rsu_coords[1]))
        rsu_text.append(rsu_coord_string(index,'z',rsu_coords[2]))

    def find_rsu_text(self,lines):
        start_string = "#                       RSU SETTINGS                     #"
        end_string = "*.rsu[*].applType = \"TraCIDemoRSU11p\""
        start_index = 0
        end_index = 0

        for i, line in enumerate(lines):

            if start_string in line:
                start_index = i + 1
            if end_string in line:
                end_index = i + 1
                break
        start_index += 3 # Adds three to get past the header in the file for the RSU section
        end_index -=2 # Subtracts two to go above the end line, which shouldnt be deleted
        return start_index,end_index

    def find_playground_text(self,lines):
        start_string = "*.playgroundSizeX ="
        end_string = "*.playgroundSizeZ ="
        start_index = 0
        end_index = 0

        for i, line in enumerate(lines):

            if start_string in line:
                start_index = i + 1
            if end_string in line:
                end_index = i + 1
                break
        return start_index,end_index

    def find_simulation_time_text(self,lines):
        start_string = "*.manager.firstStepAt ="
        end_string = "sim-time-limit ="
        start_index = 0
        end_index = 0

        for i, line in enumerate(lines):

            if start_string in line:
                start_index = i + 1
            if end_string in line:
                end_index = i + 1
                break
        return start_index,end_index

# Functions to collect the results of a simulation

    def collect_all_results(self,desired_features = ["recvPower_dBm:count","recvPower_dBm:mean","TotalLostPackets"]):
        features = []
        for current_feature in desired_features:
            results = subprocess.run(["scavetool","q","-f",f"{current_feature}","-l","-g",self.omnet_results_file], capture_output=True, text=True).stdout
            results = re.findall(rf"{current_feature} (.*)",results)
            results = [float(i) for i in results]
            features.append(results)
        features = np.asarray(features)
        return features

    def collect_rsu_results(self,rsu_network_idx,desired_features = ["recvPower_dBm:count","recvPower_dBm:mean","TotalLostPackets"]):
        features = np.empty((len(desired_features),len(rsu_network_idx)))
        for j,current_feature in enumerate(desired_features):
            current_feature_results = []
            for i in range(len(rsu_network_idx)):
                current_rsu = "RSUExampleScenario.rsu[{}].nic.mac1609_4".format(i)
                current_filter = "name({}) AND module({})".format(current_feature,current_rsu)
                results = subprocess.run(["scavetool","q","-f",f"{current_filter}","-l","-g",self.omnet_results_file], capture_output=True, text=True).stdout
                results = re.findall(rf"{current_feature} (.*)",results)
                results = [float(i) for i in results]
                current_feature_results.append(results)
            features[j,:] = np.asarray(current_feature_results)[:,0]
        features = np.asarray(features)
        return features
    
    def collect_vehicle_results(self,desired_features = ["recvPower_dBm:count","recvPower_dBm:mean","TotalLostPackets"]):
        features = []
        for current_feature in desired_features:
            car_filter = "RSUExampleScenario.node[..].nic.mac1609_4"
            current_filter = "name({}) AND module({})".format(current_feature,car_filter)
            results = subprocess.run(["scavetool","q","-f",f"{current_filter}","-l","-g",self.omnet_results_file], capture_output=True, text=True).stdout
            results = re.findall(rf"{current_feature} (.*)",results)
            results = [float(i) for i in results]
            features.append(results)
        features = np.asarray(features)
        return features

# Functions to get information from the simulation

    def prepare_intersections(self):
        f = open(self.sumo_net_xml,"r")
        lines = f.readlines()
        self.create_projection(lines)
        intersections = self.find_junctions(lines)
        return intersections

    def find_junctions(self,lines):
        intersections = np.empty((0,3))
        for i,line in enumerate(lines):
            if "<junction " in line:
                x = re.search('x=\"(.*?)\" ', line)
                if x is not None: x = float(x.group(1))
                else: continue
                y = re.search('y=\"(.*?)\" ', line)
                if y is not None: y = float(y.group(1))
                else: continue
                z = re.search('z=\"(.*?)\" ', line)
                if z is not None: z = float(z.group(1))
                else: z = 0
                coords = np.array((x,y,z))
                coords = np.reshape(coords,(1,3))
                coords[0,:2] = self.traci2omnet(coords[0,0],coords[0,1])
                intersections = np.append(intersections, coords, axis=0)
        return intersections

    def create_projection(self,lines):
        for line in lines:
            if "<location " in line:
                projection_vars = re.search('projParameter=\"(.*?)\"', line).group(1)
                offset = re.search('netOffset=\"(.*?)\"', line).group(1).split(',')
                boundry = re.search('convBoundary=\"(.*?)\"', line).group(1).split(',')
                break
        self.offset = [float(x) for x in offset]
        self.projection = pyproj.Proj(projection_vars)
        self.traciBoundry = [float(x) for x in boundry]

    def traci2geospatial(self,x,y):
        x -= self.offset[0]
        y -= self.offset[1]
        return self.projection(x,y,inverse=True)

    def traci2omnet(self,x,y):
        x = x - self.traciBoundry[0]
        x = x + self.omnet_dimensions[0]
        y = y - self.traciBoundry[1]
        y = self.omnet_dimensions[3] - y + self.omnet_dimensions[1]
        return [x,y]

    def get_simulation_rsu_network(self,
                                   rsu_net_idx,
                                   sim_idx):
        """Given the RSU network IDs and the IDs of the available intersections in that simulation,
           returns the coordinates of the intersections in that RSU network.

        Args:
            rsu_net_idx (numpy.ndArray): Indices of the RSU network intersections in the simulated subset of intersections.
            sim_idx (numpy.ndArray): Indices of the simulated subset of intersections in the complete list of intersections.

        Returns:
            numpy.ndArray: Array of coordinates for the RSU network.
        """
        intersections = self.network_intersections[sim_idx[rsu_net_idx],:]
        return intersections

    def get_simulated_intersections(self,idx):
        """Given an array of indices for a subset of intersections, 
           returns an array of coordinates for the subset of intersections.

        Args:
            idx (numpy.ndArray): Array of indices for the subset of intersections in the complete list of intersections.

        Returns:
            numpy.ndArray: Array of coordinates for the subset of intersections.
        """
        intersections = self.network_intersections[idx,:]
        return intersections

# sim_rsu_place = Agent()

# intersection_ids = np.random.choice(sim_rsu_place.network_intersections.shape[0],size = 10,replace=False)
# rsu_ids = np.random.choice(intersection_ids.shape[0],size = 5,replace=False)

# reward = sim_rsu_place.simulation_step(rsu_ids,intersection_ids)