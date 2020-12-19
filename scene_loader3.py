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
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import random
import torch
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

register(id='active-vision-env-v0',entry_point='cognitive_planning.envs.active_vision_dataset_env:ActiveVisionDatasetEnv',)


_Graph=collections.namedtuple('_Graph',['graph','id_to_index','index_to_id'])

with open ('./jsonfile/imagelist', 'rb') as ft:
    imagelist=pickle.load(ft)
def _get_image_folder(root,world):
    return os.path.join(root, world,'jpg_rgb')

def _get_json_path(root, world):
    return os.path.join(root, world, 'annotations.json')

def _get_image_path(root, world, image_id):
    return  os.path.join(_get_image_folder(root, world),image_id+'.jpg')

def _get_image_list(path, world):
    """builds a dictionary for all the worlds.
    Args:
    path: the path to the dataset on cns.
    worlds: list of the worlds.
    returns:
    dictionary where the key is the world names and
    the values are the image_ids of that world
    """
    world_id_dict={}
    #files=[t[:-4] for t in tf.gfile.ListDirectory(_get_image_folder(path, world))]
    files=[t[:-4] for t in os.listdir(_get_image_folder(path, world))]

    world_id_dict[world]=files
    return world_id_dict
def read_cached_data(output_size):
    load_start = time.time()
    result_data = {}

    depth_image_path = './depth_imgs.npy'
    logging.info('loading depth: %s', depth_image_path)

    # with tf.gfile.Open(depth_image_path) as f:
    #     depth_data = np.load(f,encoding="latin1").item()
    depth_data = dict(np.load(depth_image_path,encoding="latin1",allow_pickle=True).item())

    #print(depth_data.keys())

    logging.info('processing depth')
    for home_id in depth_data:
        #print(home_id)
        images = depth_data[home_id]
        for image_id in images:
            depth = images[image_id]
            #print("depth:", depth.shape)
            
            depth = cv2.resize(
                depth / _MAX_DEPTH_VALUE, (output_size, output_size),
                interpolation=cv2.INTER_NEAREST)
            depth_mask = (depth > 0).astype(np.float32)
            depth = np.dstack((depth, depth_mask))
            images[image_id] = depth
    result_data['depth'] = depth_data

    return result_data


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
    dataset_root='./jsonfile/'
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
worlds= ['Home_001_1','Home_002_1','Home_003_1','Home_004_1','Home_006_1','Home_010_1', 'Home_014_1', 'Home_015_1']

goallist={'Home_001_1':['000110000370101','000110000530101','000110001980101','000110000700101','000110005180101',
'000110001980101','000110003160101','000110003880101','000110004350101','000110004950101',
'000110008100101','000110008650101','000110013190101','000110013260101','000110013350101'],
'Home_002_1':['000210012160101','000210006730101','000210001300101','000210007170101','000210009820101',
'000210010790101','000210010960101','000210007900101','000210008360101','000210011470101',
'000210011610101','000210011740101','000210012010101','000210002890101','000210004850101'],
'Home_003_1':['000310001270101','000310004320101','000310012620101','000310012590101','000310014300101',
'000310014470101','000310014550101','000310009630101','000310014030101','000310012700101',
'000310004320101','000310003960101','000310002270101','000310013190101','000310012930101'],
'Home_004_1':['000410002730101','000410003040101','000410004830101','000410004930101','000410010460101',
'000410002730101','000410003040101','000410004830101','000410004930101','000410010460101',
'000410002730101','000410003040101','000410004830101','000410004930101','000410010460101',],
'Home_006_1':['000610023850101','000610000140101','000610002720101','000610006810101','000610020720101',
'000610022130101','000610022200101','000610022500101','000610019080101','000610021030101',
'000610023790101','000610000630101','000610002040101','000610007260101','000610007440101',
'000610010470101','000610012690101','000610012750101','000610012900101'],
'Home_010_1':['001010004350101','001010001120101','001010001320101','001010009000101','001010009030101',
'001010011860101','001010012300101','001010012960101','001010012990101','001010000410101',
'001010000630101','001010001180101','001010003780101','001010004680101','001010007620101',],
'Home_014_1':['001410001830101','001410004530101','001410005180101','001410006480101','001410000680101',
'001410006240101','001410006540101','001410006570101','001410005100101','001410001970101',],
'Home_015_1':['001510000240101','001510001350101',    '001510002630101','001510003470101','001510007260101',
'001510000280101','001510002130101','001510003240101','001510006420101','001510006900101']}

import random 

class ActiveVisionDatasetEnv():
    """simulates the environment from ActiveVisionDataset."""
    cached_data=None
    def __init__(self,dataset_root='./jsonfile', actions=ACTIONS):
        self.depth_data=read_cached_data(64)['depth']#'rgb'
        self.action_space =spaces.Discrete(7)#len(ACTIONS)
        self.observation_space=np.zeros([2,64,64])
        self._dataset_root=dataset_root
        self._actions=ACTIONS   
        self.worlds=worlds
        self.goallist=goallist 
        self.reset()
  
    def reset(self):
        # randomize initial state
        self.frame=0
        widx=random.randint(0,7)
        self._cur_world=self.worlds[widx]
        gnum=len(self.goallist[self._cur_world])-1
        gidx=random.randint(0,gnum)
        self.pos=read_all_poses(self._cur_world)
        #==========================================
        self._world_id_dict={}
        self._world_id_dict[self._cur_world]=imagelist[self._cur_world]

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
        x=random.randrange(7)
        if x<1:
            k = random.randrange(self.n_locations)
            self.goal_vertex=k
            self.goal_image_id=self.index_to_id[self.goal_vertex]
            while True:
                k = random.randrange(self.n_locations)
                min_d = np.inf
                path = nx.shortest_path(self.graph,k,self.goal_vertex)
                min_d=min(min_d,len(path))
                if min_d<4:
                    break
            self.current_vertex=k
            self._cur_image_id=self.index_to_id[self.current_vertex]
            self._steps_taken=0
            self.reward   = 0
            self.collided = False
            self.done = False
            self.cimg=self.depth_data[self._cur_world][self._cur_image_id].transpose(2, 0, 1)
            self.gimg=self.depth_data[self._cur_world][self.goal_image_id].transpose(2, 0, 1)
            action = self._actions[2]
            imid=self._cur_image_id
            reco=[]
            for i in range(10):
                if i%3==0:
                    reco.append(imid)
                imid = self._all_graph[self._cur_world][imid][action]
            pim1=self.depth_data[self._cur_world][reco[1]].transpose(2, 0, 1)
            pim2=self.depth_data[self._cur_world][reco[2]].transpose(2, 0, 1)
            pim3=self.depth_data[self._cur_world][reco[3]].transpose(2, 0, 1)
            ti=random.randint(0,6)
            t=[0.,0.,0.,0.,0.,0.,0.,1.,0.]#no collision
            t[ti]=1
            self.pre_action=torch.from_numpy(np.array(t,dtype=np.float32))
            self.colli_info=[0.,0.,0.,0.,0.,0.,0.]
            for i in range(6):
                a=self._actions[i]
                n= self._all_graph[self._cur_world][self._cur_image_id][a]
                if n:
                    self.colli_info[i]=1.
            self.colli_info=torch.from_numpy(np.array(self.colli_info,dtype=np.float32))
            return pim1,pim2,pim3, self.cimg, self.gimg,1,self.pre_action,self.colli_info
        else:
        #======================================================================================
            self.goal_image_id=self.goallist[self.worlds[widx]][gidx]
            self.goal_vertex=self.id_to_index[self.goal_image_id]
            while True:
                k = random.randrange(self.n_locations)
                min_d = np.inf
                path = nx.shortest_path(self.graph,k,self.goal_vertex)
                min_d=min(min_d,len(path))
                if min_d>2:
                    break
            self.current_vertex=k
            self._cur_image_id=self.index_to_id[self.current_vertex]
            self._steps_taken=0
            self.reward   = 0
            self.collided = False
            self.done = False
            self.cimg=self.depth_data[self._cur_world][self._cur_image_id].transpose(2, 0, 1)
            self.gimg=self.depth_data[self._cur_world][self.goal_image_id].transpose(2, 0, 1)
#==============================================
            action = self._actions[2]
            imid=self._cur_image_id
            reco=[]
            for i in range(10):
                if i%3==0:
                    reco.append(imid)
                imid = self._all_graph[self._cur_world][imid][action]
#=========================================================
            pim1=self.depth_data[self._cur_world][reco[1]].transpose(2, 0, 1)
            pim2=self.depth_data[self._cur_world][reco[2]].transpose(2, 0, 1)
            pim3=self.depth_data[self._cur_world][reco[3]].transpose(2, 0, 1)
            t=[0.,0.,0.,0.,0.,0.,0.,1.,0.]
            self.pre_action=torch.from_numpy(np.array(t,dtype=np.float32))
            self.colli_info=[0.,0.,0.,0.,0.,0.,0.]
            for i in range(6):
                a=self._actions[i]
                n= self._all_graph[self._cur_world][self._cur_image_id][a]
                if n:
                    self.colli_info[i]=1.
            self.colli_info=torch.from_numpy(np.array(self.colli_info,dtype=np.float32))
            return pim1,pim2,pim3, self.cimg, self.gimg, len(path), self.pre_action, self.colli_info

    def start(self):
        """Starts a new episode."""
        self.frame=0
        self.reward=0
        pim1,pim2,pim3, self.cimg, self.gimg,shortest=self.reset()
        
        return pim1,pim2,pim3, self.cimg, self.gimg,shortest
  
    def step(self, action):#action is a digit
        #assert not self.terminal, 'step() called in terminal state'
        self.done = False
        self.success = True
        self.pre_action=[0.,0.,0.,0.,0.,0.,0.,0.,0.]
        self.pre_action[action]=1.
        gt_action=np.array(action)
        self.frame+=1
        action = self._actions[action]
        pre_path=self.shortest_path(self.current_vertex,self.goal_vertex)
        pre_len=len(pre_path)
        if action=='stop':
            next_image_id=self._cur_image_id
            self.done=True
        else:
            next_image_id = self._all_graph[self._cur_world][self._cur_image_id][action]

        if not next_image_id:
            self.pre_action[-1]=1.0 #collision
            self.success = False
            self.collided =True
            self.reward=-0.2
        else:
            self.pre_action[-2]=1.0#no collision
            self._cur_image_id = next_image_id
            self.current_vertex=self.id_to_index[self._cur_image_id]
            potential_path=self.shortest_path(self.current_vertex,self.goal_vertex)
            path_len=len(potential_path)
            self.reward=0.1*(pre_len-path_len)-0.001
            #====================================================================
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
            if path_len<1.0 and path_ang<60 and action=='stop':
                self.done=True
                self.reward=10
            if self.frame>101:
                self.done=True
            self.cimg=self.depth_data[self._cur_world][self._cur_image_id].transpose(2, 0, 1)
            gt_state=self.cimg
        if self.reward<0:
            if pre_len>1:
                gt_action=self.image_image_action[self.index_to_id[pre_path[0]]][self.index_to_id[pre_path[1]]]
                imid=self.index_to_id[pre_path[1]]
                gt_state=self.depth_data[self._cur_world][imid].transpose(2, 0, 1)
            else:
                gt_action='stop'
                gt_state=self.cimg
            gt_action=np.array(self._actions.index(gt_action))
        action = self._actions[2]
        imid=self._cur_image_id
        reco=[]
        for i in range(10):
            if i%3==0:
                reco.append(imid)
            imid = self._all_graph[self._cur_world][imid][action]
        pim1=self.depth_data[self._cur_world][reco[1]].transpose(2, 0, 1)
        pim2=self.depth_data[self._cur_world][reco[2]].transpose(2, 0, 1)
        pim3=self.depth_data[self._cur_world][reco[3]].transpose(2, 0, 1)
        self.pre_action=torch.from_numpy(np.array(self.pre_action,dtype=np.float32))
        #==============================
        self.colli_info=[0.,0.,0.,0.,0.,0.,0.]
        for i in range(6):
            a=self._actions[i]
            n= self._all_graph[self._cur_world][self._cur_image_id][a]
            if n:
                self.colli_info[i]=1.
        self.colli_info=torch.from_numpy(np.array(self.colli_info,dtype=np.float32))
        return pim1, pim2, pim3, self.cimg,self.gimg, self.reward,self.done, gt_action,gt_state, pre_len,self.pre_action,self.colli_info


    def observation(self):
        return (self.reward,self.done,
        self.depth_data[self._cur_world][self._cur_image_id].transpose(1,2,0))

    def shortest_path(self,vertex,goal):
        path=nx.shortest_path(self.graph, vertex, goal)
        return path
    def all_shortest_path(self,start, end):
        path=nx.all_shortest_paths(self.graph, start, end)
        return path

    

if __name__ == "__main__":
    env = ActiveVisionDatasetEnv(world="Home_001_1")
    print("end")



    







   



        












        




























