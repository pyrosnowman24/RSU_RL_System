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
    def __init__(self,rsu_limit = 0):
        self.setup_paths()
        self.setup_simulation(index=0)
        self.intersections = torch.Tensor(self.prepare_intersections())
        self.state = torch.ones(1,self.intersections.shape[0],dtype=torch.int) # Mask of values in RSU network, 1 is not included, 0 is included
        self.rsu_limit = rsu_limit

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

    def simulation_step(self,action,W=[.5,.5]):
        # Receive new action, add it to the state, then gather intersections in RSU network
        done = False
        new_state = torch.bitwise_and(self.state,action)
        rsu_network = torch.Tensor(0,3)
        bool_mask = torch.where(new_state>0.,False,True)
        for i,row_mask in enumerate(bool_mask[0,:]):
            if row_mask:
                rsu_network = torch.cat((rsu_network,self.intersections[i,:][None,:]),0)
        self.place_rsu_network(rsu_network)

        self.state = new_state.clone()
        process1 = subprocess.Popen("./run.sh -d",cwd=self.parent_dir,shell=True)
        process2 = subprocess.Popen("./run -u Cmdenv",cwd=self.simulation_dir,shell=True)
        # process2 = subprocess.Popen("./run",cwd=self.simulation_dir,shell=True)
        process2.wait()
        process3 = subprocess.Popen("kill $(cat sumo-launchd.pid)",cwd=self.logs_dir,shell=True)

        features = self.collect_all_results(desired_features = ["recvPower_dBm:mean","TotalLostPackets"])
        reward = self.reward(features,W)

        if torch.sum(new_state) == self.rsu_limit: # Checks if the limit on RSUs has been reached
            done = True

        return new_state, reward, done
    
    def reset(self):
        "Resets the simulation environment"
        # Selects a new simulation from list of potential environments then sets up the environment 
        new_sim_index = np.random.randint(0,self.simulation_variables_db.shape[0])
        self.setup_simulation(new_sim_index)

        # Identifies the intersections in the new environment and resets state for new environment
        self.intersections = torch.Tensor(self.prepare_intersections())
        self.state = torch.ones(1,self.intersections.shape[0],dtype=torch.int)

        return self.state

    def reward(self,features,W):
        avg_features = np.nanmean(features,axis=1)
        avg_features[0] = 1/avg_features[0]*-1000
        avg_features[1] = avg_features[1]/10
        reward = np.multiply(avg_features,W)
        reward = np.sum(reward)
        return reward

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

    def collect_rsu_results(self,desired_features = ["recvPower_dBm:count","recvPower_dBm:mean","TotalLostPackets"]):
        features = []
        for current_feature in desired_features:
            results = subprocess.run(["scavetool","q","-f",f"{current_feature}","-l","-g",self.omnet_results_file], capture_output=True, text=True).stdout
            results = re.findall(rf"{current_feature} (.*)",results)
            results = [float(i) for i in results]
            features.append(results)
        features = np.asarray(features)
        return features
    
    def collect_vehicle_results(self,desired_features = ["recvPower_dBm:count","recvPower_dBm:mean","TotalLostPackets"]):
        features = []
        for current_feature in desired_features:
            results = subprocess.run(["scavetool","q","-f",f"{current_feature}","-l","-g",self.omnet_results_file], capture_output=True, text=True).stdout
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

intersections = np.random.rand(5,3)
sim_rsu_place = Agent()
action = sim_rsu_place.state
action[:,:4] = 0

sim_rsu_place.simulation_step(action)
