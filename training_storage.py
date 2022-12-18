import os
import numpy as np
from threading import Lock


# TODO: The storage could be way more efficient by pre-allocating the arrays once the shape is known, instead of concatenate-growing each sample
# TODO: Store structure along with storage, might be able to account for structure changes (e.g. inserting new choices, more options in Set?)
class TrainingStorage:
    def __init__(self, basename):
        self.basename = basename
        if os.path.exists(basename + '_x.npy'):
            self.x = np.load(basename + '_x.npy')
            self.y = np.load(basename + '_y.npy')
        else:
            self.x = None
            self.y = None
        
        self.lock = Lock()
            
    def clear(self):
        with self.lock:
            self.x = None
            self.y = None
    
    def save(self):
        # TODO: Detect if there's any changes to the data
        if not self.x is None:
            np.save(self.basename + '_x.npy', self.x)
            np.save(self.basename + '_y.npy', self.y)
        elif os.path.exists(self.basename + '_x.npy'):
            os.remove(self.basename + '_x.npy')
            os.remove(self.basename + '_y.npy')
             
    def index(self, x):
        if self.x is None:
            return None
        
        r = np.nonzero((abs(self.x - x[None,:]) < 1e-8).all(axis=1))[0]
        if len(r) == 0:
            return None
        else:
            assert len(r) == 1, 'Multiple of the same x'
            return r[0]
    
    def store(self, x, y):
        with self.lock:
            if self.x is None:
                self.x = x[None,:].copy()
                self.y = y[None,:].copy()
                return
            
            i = self.index(x)
            if i == None:
                self.x = np.concatenate((self.x, x[None,:]), axis=0)
                self.y = np.concatenate((self.y, y[None,:]), axis=0)
            else:
                self.x[i] = x
                self.y[i] = y

    def store_many(self, x, y):
        if x is None:
            return
        
        with self.lock:
            if self.x is None:
                u, i = np.unique(x, return_index=True, axis=0)
    
                self.x = x[i].copy()
                self.y = y[i].copy()
                return
            
            # Concatenate new values first to update y if it overwrites anything (unique will find them first)
            x = np.concatenate((x, self.x), axis=0)
            y = np.concatenate((y, self.y), axis=0)
            
            u, i = np.unique(x, return_index=True, axis=0)
            i = sorted(i)
    
            self.x = x[i]
            self.y = y[i]
        
    def remove(self, x):
        with self.lock:
            i = self.index(x)
            if i != None:
                self.x = np.concatenate((self.x[:i,:], self.x[i+1:,:]), axis=0)
                self.y = np.concatenate((self.y[:i,:], self.y[i+1:,:]), axis=0)
                if self.x.size == 0:
                    self.x = None
                    self.y = None
    
    def get_data(self):
        return self.x, self.y
    
    def get_y(self, x):
        i = self.index(x)
        if i == None:
            return None
        return self.y[i]
