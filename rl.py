import torch
import torch.multiprocessing as mp
import sys

sys.path.append('paper.io.sessdsa') 
from match_core import match
from match_interface import save_match_log
from time import perf_counter as pf

def worker(id, net, data_queue):
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
    save_match_log(res, 'saves/1.zlog')

if __name__ == '__main__':
    mp.set_start_method('spawn')
    q = mp.Queue()
    p = mp.Process(target=worker, args=(1, 0, q))
    p.start()
    p.join()
    
    
