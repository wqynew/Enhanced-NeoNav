import collections
import copy
import json
import os
import time
import gym
from gym.envs.registration import register
import gym.spaces
from gym import spaces
import networkx as nx 
import numpy as np 
import scipy.io as sio 
from absl import logging
import torch
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import random
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
#import visualization_utils as vis_util 
import pickle

TRAIN_WORLDS = [
    'Home_014_1'
]
_MAX_DEPTH_VALUE = 12102
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
register(id='active-vision-env-v0',entry_point='cognitive_planning.envs.active_vision_dataset_env:ActiveVisionDatasetEnv',)


_Graph=collections.namedtuple('_Graph',['graph','id_to_index','index_to_id'])

with open ('./jsonfile/timagelist', 'rb') as ft:
    imagelist=pickle.load(ft)
def _get_image_folder(root,world):
    return os.path.join(root, world,'jpg_rgb')

def _get_json_path(root, world):
    return os.path.join(root, world, 'annotations.json')

def _get_image_path(root, world, image_id):
    return  os.path.join(_get_image_folder(root, world),image_id+'.jpg')


def read_all_poses(world):
    """reads all the poses for each world
    Args:
    dataset_root: the path to the root of the dataset.
    world: string, name of the world
    Returns:
    dictonary of poses for all the images in each world. The key is the image id of each view
    and the values are tuple of (x,z,R, scale).
    where x and z are the first and third coordinate of translation. R is the 3X3 rotation matrix
    and scale is a float scalar that indicates the scale that needs to be multipled to x and z in order to get the real world coordicates.
    """
    dataset_root='./jsonfile'
    path = os.path.join(dataset_root, world, 'image_structs.mat')
    data = sio.loadmat(path)

    xyz = data['image_structs']['world_pos']
    image_names = data['image_structs']['image_name'][0]

    dire=data['image_structs']['direction']
    #rot = data['image_structs']['R'][0]
    #scale = data['scale'][0][0]

    n = xyz.shape[1]
    x = [xyz[0][i][0][0] for i in range(n)]
    y = [0 for i in range(n)]
    z = [xyz[0][i][2][0] for i in range(n)]
    px= [dire[0][i][0][0] for i in range(n)]
    py= [0 for i in range(n)]
    pz= [dire[0][i][2][0] for i in range(n)]
#============================================================
    names = [name[0][:-4] for name in image_names]
    if len(names) != len(x):
        raise ValueError('number of image names are not equal to the number of '
                        'poses {} != {}'.format(len(names), len(x)))
    output = {}
    for i in range(n):
        output[names[i]] = [x[i], z[i],  px[i], pz[i]]
    return output

ACTIONS=['right', 'rotate_cw', 'rotate_ccw', 'forward', 'left', 'backward','stop']
worlds= ['Home_011_1','Home_013_1','Home_016_1']

import random 

class ActiveVisionDatasetEnv():
    """simulates the environment from ActiveVisionDataset."""
    cached_data=None
    def __init__(self,dataset_root='./jsonfile', actions=ACTIONS):
        with open ('featuredes', 'rb') as ft:
            self.depth_data=pickle.load(ft)
        print(len(self.depth_data))
        self.action_space =spaces.Discrete(7)#len(ACTIONS)
        self.observation_space=np.zeros([2,64,64])
        self._dataset_root=dataset_root
        self._actions=ACTIONS   
        self.worlds=worlds

  
    def reset(self,world,startid,endid):
        # randomize initial state
        self.mypath=[]
        self.preim1=False
        self.preim2=False
        self.preim3=False

        self.frame=0
        widx=random.randint(0,7)
        gidx=random.randint(0,4)
        #==========================================
        self._cur_world=world
        self._world_id_dict={}
        self._world_id_dict[self._cur_world]=imagelist[self._cur_world]
        self.pos=read_all_poses(self._cur_world)
        self._all_graph = {}
        with open(_get_json_path(self._dataset_root, self._cur_world), 'r') as f:
            file_content = f.read()
            file_content = file_content.replace('.jpg', '')
            io = StringIO(file_content)
            self._all_graph[self._cur_world] = json.load(io)

        self.graph=nx.DiGraph()
        self.id_to_index={}
        self.index_to_id={}
        self.image_image_action={}
        image_list=self._world_id_dict[self._cur_world]
        #print(image_list)
        for image_id in image_list[self._cur_world]:
            self.image_image_action[image_id]={} 
            for action in self._actions:
                if action=='stop':
                    self.image_image_action[image_id][image_id]=action
                    continue
                next_image=self._all_graph[self._cur_world][image_id][action]
                if next_image:
                    self.image_image_action[image_id][next_image]=action

        for i, image_id in enumerate(image_list[self._cur_world]):
            self.id_to_index[image_id]=i
            self.index_to_id[i]=image_id
            self.graph.add_node(i)
        for image_id in image_list[self._cur_world]:
            for action in self._actions:
                if action=='stop':
                    continue
                next_image=self._all_graph[self._cur_world][image_id][action]

                if next_image:
                    self.graph.add_edge(self.id_to_index[image_id],self.id_to_index[next_image],action=action)
        self.n_locations=self.graph.number_of_nodes()
        #=====================================================================================
        #======================================================================================
        self.goal_image_id=endid
        self.goal_vertex=self.id_to_index[self.goal_image_id]
        self._cur_image_id=startid
        self.current_vertex=self.id_to_index[self._cur_image_id]
        self._steps_taken=0
        self.reward   = 0
        self.collided = False
        self.done = False
        self.cimg=self.depth_data[self._cur_world+'/'+self._cur_image_id]#torch.FloatTensor(np.float32(self.depth_data[self._cur_world][self._cur_image_id].transpose(2, 0, 1)))
        self.cimg=self.cimg.to(device)
        self.gimg=self.depth_data[self._cur_world+'/'+self.goal_image_id]#torch.FloatTensor(np.float32(self.depth_data[self._cur_world][self.goal_image_id].transpose(2, 0, 1)))
        self.gimg=self.gimg.to(device)
#==============================================
        action = self._actions[2]
        imid=self._cur_image_id
        self.mypath.append(self._cur_image_id)
        reco=[]
        for i in range(10):
            if i%3==0:
                reco.append(imid)
            imid = self._all_graph[self._cur_world][imid][action]
#=========================================================
        pim1=self.depth_data[self._cur_world+'/'+reco[1]]#torch.FloatTensor(np.float32(self.depth_data[self._cur_world][reco[1]].transpose(2, 0, 1)))
        pim1=pim1.to(device)
        pim2=self.depth_data[self._cur_world+'/'+reco[2]]#torch.FloatTensor(np.float32(self.depth_data[self._cur_world][reco[2]].transpose(2, 0, 1)))
        pim2=pim2.to(device)
        pim3=self.depth_data[self._cur_world+'/'+reco[3]]#torch.FloatTensor(np.float32(self.depth_data[self._cur_world][reco[3]].transpose(2, 0, 1)))
        pim3=pim3.to(device)
        t=[0.,0.,0.,0.,0.,0.,0.,1.,0.]
        self.pre_action=torch.from_numpy(np.array(t,dtype=np.float32))
        self.pre_action=self.pre_action.to(device)
        return pim1,pim2,pim3, self.cimg, self.gimg,self.pre_action


  
    def step(self, action):#action is a digit
        #assert not self.terminal, 'step() called in terminal state'
        self.pre_action=[0.,0.,0.,0.,0.,0.,0.,0.,0.]
        self.pre_action[action]=1.
        self.preim1=self.preim2
        self.preim2=self.preim3
        self.preim3=self._cur_image_id
        gt_action=np.array(action)
        self.frame+=1
        action = self._actions[action]
        if action=='stop':
            next_image_id=self._cur_image_id
        else:
            next_image_id = self._all_graph[self._cur_world][self._cur_image_id][action]
        self._steps_taken += 1
        self.done = False
        self.success = True
        #pre_path=self.shortest_path(self.current_vertex,self.goal_vertex)
        #pre_len=len(pre_path)
        if not next_image_id:
            self.pre_action[-1]=1.0 #collision
            self.success = False
            self.collided =True
            self.reward=-0.2
        else:
            self.pre_action[-2]=1.0 #no collision
            self._cur_image_id = next_image_id
            self.current_vertex=self.id_to_index[self._cur_image_id]
            #potential_path=self.shortest_path(self.current_vertex,self.goal_vertex)
            #path_len=len(potential_path)
            #self.reward=0.1*(pre_len-path_len)-0.001
            pos1=self.pos[self._cur_image_id]
            pos2=self.pos[self.goal_image_id]
            path_len=(pos1[0]-pos2[0])**2+(pos1[1]-pos2[1])**2                    
            l1=np.sqrt(pos1[2]*pos1[2]+pos1[3]*pos1[3])
            l2=np.sqrt(pos2[2]*pos2[2]+pos2[3]*pos2[3])
            v=(pos1[2]*pos2[2]+pos1[3]*pos2[3])/(l1*l2)
            if v>1:
                v=1
            if v<-1:
                v=-1
            path_ang=np.arccos(v)*180/3.14159
            if path_len<=1.0 and path_ang<61 and action=='stop':
                    self.done=True
                    self.reward=10
            elif action=='stop':
                self.done=True
                self.reward=-10.0

            self.cimg=self.depth_data[self._cur_world+'/'+self._cur_image_id]#torch.FloatTensor(np.float32(self.depth_data[self._cur_world][self._cur_image_id].transpose(2, 0, 1)))
            self.mypath.append(self._cur_image_id)
            self.cimg=self.cimg.to(device)
        #=====================================================================================
        action = self._actions[2]
        imid=self._cur_image_id
        reco=[]
        for i in range(10):
            if i%3==0:
                reco.append(imid)
            imid = self._all_graph[self._cur_world][imid][action]
        pim1=self.depth_data[self._cur_world+'/'+reco[1]]#torch.FloatTensor(np.float32(self.depth_data[self._cur_world][reco[1]].transpose(2, 0, 1)))
        pim1=pim1.to(device)
        pim2=self.depth_data[self._cur_world+'/'+reco[2]]#torch.FloatTensor(np.float32(self.depth_data[self._cur_world][reco[2]].transpose(2, 0, 1)))
        pim2=pim2.to(device)
        pim3=self.depth_data[self._cur_world+'/'+reco[3]]#torch.FloatTensor(np.float32(self.depth_data[self._cur_world][reco[3]].transpose(2, 0, 1)))
        pim3=pim3.to(device)
        #==========================================================================
        self.pre_action=torch.from_numpy(np.array(self.pre_action,dtype=np.float32))
        self.pre_action=self.pre_action.to(device)
        return pim1, pim2, pim3, self.cimg,self.gimg, self.reward,self.done, 0, 0,self.pre_action

    def observation(self):
        return (self.reward,self.done,
            self.depth_data[self._cur_world+'/'+self._cur_image_id])
        #torch.FloatTensor(np.float32(self.depth_data[self._cur_world][self._cur_image_id].transpose(1,2,0))))

    def shortest_path(self,vertex,goal):
        path=nx.shortest_path(self.graph, vertex, goal)
        return path
    def all_shortest_path(self,start, end):
        path=nx.all_shortest_paths(self.graph, start, end)
        return path

    

if __name__ == "__main__":
    env = ActiveVisionDatasetEnv()
    print("end")



    







   



        












        




























