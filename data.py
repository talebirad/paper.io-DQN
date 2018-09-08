import torch
import torch.multiprocessing as mp
import sys
import numpy as np
import torch.nn.functional as F
import random
import math
from net import DQN
import time

sys.path.append('paper.io.sessdsa') 
import AI.AI_simple_goround as p2
from match_core import match
from match_interface import save_match_log
from time import perf_counter as pf

device = "cpu"

def changNoneToZeros(array):
    return [[0 if x is None else x for x in lines] for lines in array]
def changeArrayToBooleanTensor(array,num):
    return torch.tensor(torch.tensor(changNoneToZeros(array)) == num, dtype = torch.float)
def rotateTensor(tensor,k):
    for i in range(k):
        tensor = tensor.flip((1,)).transpose(0,1)
    return tensor
def padTensor(tensor):
    return F.pad(tensor, (5,6,5,5), "constant", 0)
def rotatePad(array,rot):
    return rotateTensor(padTensor(array),rot)
def rotatePadChange(array,num,rot):
    return rotateTensor(padTensor(changeArrayToBooleanTensor(array,num)),rot)
def oneTensor(shape, pos):
    tensor = torch.zeros(*shape)
    tensor[pos[0]][pos[1]]=1
    return tensor

def getFeatureMap(stat):
    me = stat['now']['me']['id']
    rot = stat['now']['me']['direction'] + 1
    myFields = rotatePadChange(stat['now']['fields'], me, rot)
    enemyFields = rotatePadChange(stat['now']['fields'], 3 - me, rot)
    myBands = rotatePadChange(stat['now']['bands'], me, rot)
    enemyBands = rotatePadChange(stat['now']['bands'], 3 - me, rot)
    myPosition = rotatePad(oneTensor((102,101), (stat['now']['me']['x'],stat['now']['me']['y'])), rot)
    enemyPosition = rotatePad(oneTensor((102,101), (stat['now']['enemy']['x'],stat['now']['enemy']['y'])), rot)
    area = rotatePad(torch.ones(102,101), rot)
    tensor = torch.stack([myFields,enemyFields,myBands,enemyBands,myPosition,enemyPosition,area])
    return tensor.unsqueeze(0), torch.sum(myFields) - torch.sum(enemyFields)

def worker(id, net, data_queue, save = False):
    EPS_START = 0.9
    EPS_END = 0.2
    EPS_DECAY = 5000
    class player1:
        def load(self, stat, storage):
            storage['field_size'] = 0
            if 'tensor' in storage:
                del storage['tensor']
            
        def play(self, stat, storage):
            tensor, field_size = getFeatureMap(stat)
            reward = field_size - storage['field_size']
            if 'tensor' in storage:
                data_queue.put((storage['tensor'], storage['action'], tensor, reward))
            storage['tensor'] = tensor
            storage['field_size'] = field_size
            if 'steps_done' in storage:
                storage['steps_done'] += 1
            else:
                storage['steps_done'] = 1
            #print(tensor.shape)
            sample = random.random()
            eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                math.exp(-1. * storage['steps_done'] / EPS_DECAY)
            #print('EPS: ', storage['steps_done'], eps_threshold)
            choices = ['l','s','r']
            if sample > eps_threshold:
                #print('Using Net')
                with torch.no_grad():
                    res = net(tensor).max(1)[1]
                ret = choices[res]
            else:
                ret = random.choice(['l', 'r', 's'])
            storage['action'] = ret
            #print(ret)
            return ret
        
        def summary(self, result, stat, storage):
            tensor, field_size = getFeatureMap(stat)
            eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                math.exp(-1. * storage['steps_done'] / EPS_DECAY)
            if id == 0:
                print('EPS: ', storage['steps_done'], eps_threshold, "Size: ", tensor[0][0].sum())
            if result[0] is not None and result[1] >= 0:
                if result[0] == stat['log'][0]['me']['id'] - 1:
                    reward = 0
                else:
                    reward = -0
                data_queue.put((storage['tensor'], storage['action'], None, reward))


    class player2:
        def load(self, stat, storage):
            return p2.load(stat, storage)
        def play(self, stat, storage):
            return "r"
            return p2.play(stat, storage)
    gameCnt = 1
    while True:
        t1 = pf()
        res = match((player1(), player2()))
        t2 = pf()
        if id == 0:
            print("Time: ", t2 - t1)
            print(gameCnt, ' Match Done')
        if save:
            save_match_log(res, 'saves/' + str(gameCnt % 100) + '.zlog')
        gameCnt += 1
        if 'DEBUG_TRACEBACK' in dir(match):
            print(match.DEBUG_TRACEBACK)
            exit()
def start(num = 3):
    mp.set_start_method('spawn')
    global net, p, q
    net = DQN().to(device)
    net.share_memory()
    q = mp.Queue()
    p = [mp.Process(target=worker, args=(id, net, q, id == 0)) for id in range(num)]
    for i in p:
        i.start()

def stop():
    for i in p:
        i.terminate()
        
def getData():
    ret = []
    while not q.empty():
        ret.append(q.get())
    return ret
    
def updateNet(stat):
    net.load_state_dict(stat)

if __name__ == '__main__':
    start(1)
    for i in range(5):
        time.sleep(1)
    stop()
    print(len(getData()))
    
