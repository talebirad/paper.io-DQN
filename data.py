import torch
import torch.multiprocessing as mp
import sys

sys.path.append('paper.io.sessdsa') 
from match_core import match
from match_interface import save_match_log
from time import perf_counter as pf

def getFeatureMap(stat, me):
    meFields = stat['now']['fields'] == me
    enemyFields = stat['now']['fields'] == 3 - me
    meBands = stat['now']['bands'] == me
    enemyBands = stat['now']['bands'] == 3 - me
    mePosition = 

def worker(id, net, data_queue, save = False):
    class player1:
        def play(self, stat, storage):
            return 'r'
            
    class player2:
        def play(self, stat, storage):
            return 'l'

    t1 = pf()
    res = match((player1(), player2()))
    t2 = pf()
    print(t2 - t1)
    
    if save:
        save_match_log(res, 'saves/1.zlog')
        
def start():
    pass

def getData():
    pass

if __name__ == '__main__':
    mp.set_start_method('spawn')
    q = mp.Queue()
    p = mp.Process(target=worker, args=(1, 0, q, True))
    p.start()
    p.join()
    
    
