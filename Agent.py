from importlib.resources import path
import torch
import numpy as np
from numpy.random import default_rng
import os
import subprocess
import re
import pyproj

class Agent:
    # This class will be responsible for keeping track of the current simulation and the environment for the simulation (sumo stuff)
    def __init__(self):
        self.setup_paths()
        self.setup_environment()
        self.intersections = self.prepare_intersections()
        self.state = torch.ones(1,self.intersections.shape[0],dtype=torch.int)
        self.place_rsu_network(self.intersections)

    def setup_paths(self):
        self.parent_dir ="/home/acelab/veins_sim/"
        self.simulation_dir ="/home/acelab/veins_sim/veins/examples/veins/"
        self.sumo_scenario_dir ="/home/acelab/veins_sim/veins/examples/veins/scenario/"
        self.logs_dir = "/home/acelab/veins_sim/logs/"
        self.omnet_ini = os.path.join(self.simulation_dir,"omnetpp.ini")
        self.scenario_ned = os.path.join(self.simulation_dir,'RSUExampleScenario.ned')
        self.omnet_results_file = os.path.join(self.simulation_dir,"results/General-#0.sca")
        self.sumo_net_xml = os.path.join(self.sumo_scenario_dir,"net.net.xml")

    def setup_environment(self):
        # This will setup similar things to the cnn setup, stuff like the world image, the transforms, the size of the samples, ect.
        #
        #The omnet dimensions are the actual boundries for the network, the playground boundries are larger then the actual area.
        self.omnet_dimensions = [22.67,9.0,10005,6387.01]

# Functions to interact with the simulation.

    def simulation_step(self,action,w1,w2):
        # new_state = torch.bitwise_and(self.state,action)

        process1 = subprocess.Popen("./run.sh ",cwd=self.parent_dir,shell=True)
        process2 = subprocess.Popen("./run -u Cmdenv",cwd=self.simulation_dir,shell=True)
        # process2 = subprocess.Popen("./run",cwd="/home/acelab/veins_sim/veins/examples/veins/",shell=True)
        process2.wait()
        process3 = subprocess.Popen("kill $(cat sumo-launchd.pid)",cwd=self.logs_dir,shell=True)

        features = self.collect_all_results()
        print(features)
        # return new_state, reward, done
    
    def reset(self):
        "Resets the simulation environment"
        self.state = torch.ones(1,self.intersections.shape[2],dtype=torch.int)
        return self.state


# Functions to add RSU network to necessary files

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
        iter_start_index = True
        iter_end_index = True

        for i, line in enumerate(lines):

            if start_string in line:
                start_index = i + 1
            if end_string in line:
                end_index = i + 1
                break
        start_index += 3 # Adds three to get past the header in the file for the RSU section
        end_index -=2 # Subtracts two to go above the end line, which shouldnt be deleted
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
                else: continue
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
        x = x + self.omnet_dimensions[0]
        y = y / self.traciBoundry[3]
        y = y * self.omnet_dimensions[3]
        y = self.omnet_dimensions[3] - y + self.omnet_dimensions[1]
        return [x,y]


intersections = np.random.rand(5,3)
sim_rsu_place = Agent()
