import numpy as np
import json
import os
from threading import Thread, Lock, Event
from copy import copy, deepcopy
import time

from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.exceptions import NotFittedError

from annotation import String, StringLiteral, List, Choice, Reference, Set, Bool, Dictionary

from training_storage import TrainingStorage

class StorageCopy:
    x = None
    y = None
    
    def get_data(self):
        return self.x, self.y

def worker_thread(classifier):
    # Worker operates on shallow copy of classifier
    c = copy(classifier)
    
    print('Copying classifier')
    with c.lock:
        if type(c) == SetClassifier and c.multi_classifier:
            c.classifiers = [deepcopy(x) for x in c.classifiers]
        else:
            c.classifier = deepcopy(c.classifier)
    
    c.storage = StorageCopy()
    
    while True:
        
        # print('Waiting for train event')
        if not c.train_event.wait(timeout=1):
            # print('Timeout')
            continue
        c.train_event.clear()
        
        if classifier.storage.x is None:
            continue
        
        # print('Acquiring storage lock')
        with classifier.storage.lock:
            c.storage.x = classifier.storage.x.copy()
            c.storage.y = classifier.storage.y.copy()
        
        t0 = time.time()
        # print('Training')
        if c._train():
            # print('Copying classifier')
            with c.lock:
                if type(c) == SetClassifier and c.multi_classifier:
                    classifier.classifiers = [deepcopy(x) for x in c.classifiers]
                else:
                    classifier.classifier = deepcopy(c.classifier)
        print('Train took:', time.time() - t0)
    



class Classifier:
    def __init__(self, name, template, storage_dir):
        self.name = name
        self.template = template
        template.classifier = self
        self.storage = TrainingStorage(storage_dir + '.'.join(name))
        self.needs_retrain = True
        
        self.lock = Lock()
        self.train_event = Event()
        self.worker_thread = Thread(target=worker_thread, args=(self,), daemon=True)
        self.worker_thread.start()
        
    def _train(self):
        raise NotImplementedError('Cannot call abstract _train() method')
    
    def train(self):
        if not self.needs_retrain:
            return
        # TODO: Check train output for non-convergence, errors, etc.
        
        # if self._train():
        self.train_event.set()
        self.needs_retrain = False
    
    # TODO: Inefficient, every classifier is reading all json files
    def load_targets(self, image_names, clips, structure, annotation_extension='.json'):
        valid = []
        x = None
        y = None
        
        for i, (name, clip) in enumerate(zip(image_names, clips)):
            filename_json = os.path.join(os.path.splitext(name)[0] + annotation_extension)
            if os.path.exists(filename_json):
                tmp = json.load(open(filename_json, 'r'))
                
                if tmp != None and 'annotations' in tmp:
                    if 'annotations_meta' in tmp and 'acknowledged' in tmp['annotations_meta']:
                        acknowledged = tmp['annotations_meta']['acknowledged']
                    else:
                        acknowledged = False
                    if not acknowledged:
                        continue

                    t = load_target_from_annotation(self.name, structure, tmp['annotations'])
                    if t is not None:
                        if x is None:
                            x = np.empty((len(image_names), clip.shape[0]), dtype=clip.dtype)
                            y = np.empty((len(image_names), t.shape[0]), dtype=t.dtype)
                        valid.append(i)
                        x[i] = clip
                        y[i] = t
        if x is not None:
            x = x[valid]
            y = y[valid]
            
            self.storage.store_many(x, y)
    
class ChoiceClassifier(Classifier):
    def __init__(self, name, template, storage_dir):
        self.classifier = MLPClassifier((64,16), alpha=0.01, activation='tanh', solver='lbfgs', warm_start=True)
        super().__init__(name, template, storage_dir)
        
    def _train(self):
        print('train', self.name)
        c = self.classifier
        x,y = self.storage.get_data()

        if x is None:
            print(self.name, 'not fitting: no data')
            return False
        
        if len(np.unique(y)) == 1:
            print(self.name, 'not fitting: single class', set(np.unique(y)))
            return False
        
        if hasattr(c, 'warm_start'):
            prev_warm_start = c.warm_start
            
            if hasattr(c, 'classes_') and set(np.unique(y)) != set(c.classes_):
                print(self.name, 'number of classes changed, disabling warm_start')
                c.warm_start = False

        c.fit(x, y[:,0])
        if hasattr(c, 'warm_start'):
            c.warm_start = prev_warm_start
        return True
    
    def predict_entropy(self, x):
        with self.lock:
            v = self.classifier.predict_proba(x).clip(1e-6,1)
        entropy = (np.log(v) * v).sum(axis=1)
        return entropy 
    
    def predict(self, x):
        with self.lock:
            probs = self.classifier.predict_proba(x[None,:])[0]
        return int(self.classifier.classes_[probs.argmax()]), list(zip(self.classifier.classes_, probs.tolist()))
    
    def predict_many(self, x):
        with self.lock:
            probs = self.classifier.predict_proba(x)
        return self.classifier.classes_[probs.argmax(axis=1)], probs

class SetClassifier(Classifier):
    def __init__(self, name, template, storage_dir):
        self.classifier = MLPClassifier((64,16), alpha=0.01, activation='tanh', solver='lbfgs', warm_start=True)
        self.multi_classifier = False
        
        super().__init__(name, template, storage_dir)
        # self.classifiers = [SVC(C=5.0, probability=True) for x in template.options]
        # self.multi_classifier = True
        
    def _train(self):
        print('train', self.name)
        if self.multi_classifier:
            x,y = self.storage.get_data()
            if x is None:
                print(self.name, 'not fitting: no data')
                return False
                
            for i,c in enumerate(self.classifiers):
                if len(np.unique(y[:,i])) == 1:
                    print(self.name, self.template.options[i], 'not fitting: single class', set(np.unique(y[:,i])))
                else:
                    c.fit(x, y[:,i])
            return True
        else:
            c = self.classifier
            x,y = self.storage.get_data()
    
            if x is None:
                print(self.name, 'not fitting: no data')
                return False
            
            c.fit(x, y)
            return True
    
    def predict_entropy(self, x):
        if self.multi_classifier:
            sum_entropy = 0
            with self.lock:
                for i,c in enumerate(self.classifiers):
                    v = c.predict_proba(x).clip(1e-6,1)
                    sum_entropy += (np.log(v) * v).sum(axis=1)
            return sum_entropy / len(self.classifiers)
        else:
            with self.lock:
                v = self.classifier.predict_proba(x).clip(1e-6,1)
            entropy = (np.log(v) * v).sum(axis=1)
            return entropy
    
    def predict(self, x):
        if self.multi_classifier:
            r = set()
            probs = []
            n_notfitted = 0
            with self.lock:
                for i,c in enumerate(self.classifiers):
                    try:
                        prob = c.predict_proba(x[None,:])[0,1]
                        v = bool(prob > 0.5)
                    except NotFittedError:
                        prob = 0.5
                        v = False
                        n_notfitted += 1
                    
                    if v:
                        r.add(self.template.options[i])
                    probs.append(prob)
            
            if n_notfitted == len(self.classifiers):
                raise NotFittedError('No classifier fitted')
            return r, probs
        else:
            with self.lock:
                probs = self.classifier.predict_proba(x[None,:])[0]
            assert self.classifier.n_outputs_ == len(self.template.options), (self.classifier.n_outputs_, len(self.template.options))
            v = set()
            for i in range(self.classifier.n_outputs_):
                if probs[i] > 0.5:
                    v.add(self.template.options[i])
            return v, probs.tolist()
            
    def predict_many(self, x):
        if self.multi_classifier:
            raise NotImplementedError()
        else:
            with self.lock:
                probs = self.classifier.predict_proba(x)
            return probs > 0.5, probs



class BoolClassifier(Classifier):
    def __init__(self, name, template, storage_dir):
        self.classifier = MLPClassifier((64,16), alpha=0.01, activation='tanh', solver='lbfgs', warm_start=True)
        # self.classifier = SVC(C=5.0, probability=True)
        super().__init__(name, template, storage_dir)
        
    def _train(self):
        print('train', self.name)

        c = self.classifier
        x,y = self.storage.get_data()
        
        if x is None:
            print(self.name, 'not fitting: no data')
            return False
        
        if len(np.unique(y)) == 1:
            print(self.name, 'not fitting: single class', set(np.unique(y)))
            return False

        c.fit(x, y[:,0])
        return True

    def predict_entropy(self, x):
        with self.lock:
            v = self.classifier.predict_proba(x).clip(1e-6,1)
        entropy = (np.log(v) * v).sum(axis=1)
        return entropy

    def predict(self, x):
        with self.lock:
            prob = self.classifier.predict_proba(x[None,:])[0,1]
        v = bool(prob > 0.5)
        
        return v, prob
        
    def predict_many(self, x):
        with self.lock:
            probs = self.classifier.predict_proba(x)[:,1]
        return probs > 0.5, probs

def find_classifiers(template, storage_dir, name=None):
    if name == None:
        name = []
    
    it = type(template)
    
    if it == Dictionary:
        r = []
        for x in template.dict_items:
            c = find_classifiers(template.dict_items[x], storage_dir, name + [x])
            if c:
                r.extend(c)
        return r
    elif it == List:
        return find_classifiers(template.item_type, storage_dir, name)
    elif it == Choice:
        r = []
        for x in template.options:
            r.extend(find_classifiers(x, storage_dir, name + [x.name]))
        
        if template.train_classifier:
            r.extend([ChoiceClassifier(name, template, storage_dir)])
        return r
    elif it == Set:
        if template.train_classifier:
            return [SetClassifier(name, template, storage_dir)]
        return []
    elif it == Bool:
        if template.train_classifier:
            return [BoolClassifier(name, template, storage_dir)]
        return []
    elif it == Reference:
        return find_classifiers(template.structure[template.name], storage_dir, name)
    elif it == String or it == StringLiteral:
        return []
    else:
        raise ValueError('Invalid type', it)


def load_target_from_annotation(name, template, annotations):
    if annotations == None:
        return None
    
    it = type(template)
    if it == Dictionary:
        if 'dictionary_class' not in annotations or annotations['dictionary_class'] != template.name:
            return None
        if len(name) > 0:
            if name[0] in annotations:
                return load_target_from_annotation(name[1:], template.dict_items[name[0]], annotations[name[0]])
            else:
                return None
        else:
            return None
    elif it == List:
        # List returns first target matching name
        for x in annotations:
            r = load_target_from_annotation(name, template.item_type, x)
            if r != None:
                return r
        return None
    elif it == Bool:
        if len(name) > 0:
            return None
        return np.array([annotations])
    elif it == Set:
        if len(name) > 0:
            return None
        if 'set_class' not in annotations:
            return None
        if 'value' not in annotations:
            return None
        
        v = []
        for o in template.options:
            v.append(o in annotations['value'])
        return np.array(v)
    elif it == Choice:
        if 'choice_class' not in annotations:
            return None
        
        if len(name) > 0:
            if annotations == None:
                return None
            if 'value_index' not in annotations or 'value' not in annotations:
                return None

            if template.options[annotations['value_index']].name == name[0]:
                return load_target_from_annotation(name[1:], template.options[annotations['value_index']], annotations['value'])
            else:
                return None
        else:
            if 'value_index' not in annotations:
                return None
            return np.array([annotations['value_index']])
    elif it == Reference:
        return load_target_from_annotation(name, template.structure[template.name], annotations)
    else:
        # TODO: Strings?
        return None
    
