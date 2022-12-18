import sys
import os
import time
import yaml
import json
import numpy as np
import random
import zipfile
import argparse

from PyQt5.QtWidgets import QApplication, QFileDialog, QMessageBox, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QSplitter, QLabel
from PyQt5.QtCore import Qt, QUrl, QSettings, pyqtSignal as Signal
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtNetwork import QNetworkAccessManager, QNetworkReply, QNetworkRequest

import widgets
from widgets import ImageWidget, AnnotationTreeWidget, DatasetWidget
from annotation_instance import instances_from_dict, create_instance_from_template
from annotation import parse_annotation_structure
from classification import find_classifiers, load_target_from_annotation, ChoiceClassifier, SetClassifier, BoolClassifier

from sklearn.exceptions import NotFittedError
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix

keymap = {}
for key, value in vars(Qt).items():
    if isinstance(value, Qt.Key):
        keymap[value] = key.partition('_')[2]
        
# TODO: Should this be QMainWindow? Complains about layout
class AnnotationWidget(QWidget):
    shortcutPressed = Signal(int)
    
    def __init__(self, structure, annotator='', subject='', parent=None):
        super().__init__(parent=parent)
        self.treeWidget = AnnotationTreeWidget(parent=self)
        self.imageWidget = ImageWidget(self)
        self.imageWidget.setMinimumWidth(800)
        self.nam = QNetworkAccessManager()
        self.nam.finished.connect(self.nam_finished)
        
        self.vlayout = QVBoxLayout(self)
        
        layout = QHBoxLayout()
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.treeWidget)
        splitter.addWidget(self.imageWidget)
        
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 1)
        splitter.setCollapsible(0, False)
        splitter.setCollapsible(1, False)
        
        layout.addWidget(splitter)
                        
        self.metadata = None
        self.annotations = None
        self.current_clip_value = None
        self.structure = structure
        self.classifiers = []

        self.datasetWidget = DatasetWidget(self)
        self.datasetWidget.setMaximumHeight(60)
       
        self.vlayout.addWidget(self.datasetWidget)
        self.vlayout.addLayout(layout)
        self.undoButton = QPushButton('Undo')
        self.redoButton = QPushButton('Redo')
        self.balancedButton = QPushButton('Next (balanced)')
        self.uncertainButton = QPushButton('Next (uncertain)')
        self.evaluateButton = QPushButton('Evaluate classifier')
        self.packageButton = QPushButton('Package results')
        self.annotationCounter = QLabel()
        
        bottom_layout = QHBoxLayout()
        self.annotationCounter.setMaximumHeight(60)
        bottom_layout.addWidget(self.undoButton)
        bottom_layout.addWidget(self.redoButton)
        bottom_layout.addWidget(self.balancedButton)
        bottom_layout.addWidget(self.uncertainButton)
        bottom_layout.addWidget(self.evaluateButton)
        bottom_layout.addWidget(self.packageButton)
        bottom_layout.addWidget(self.annotationCounter)
        
        self.vlayout.addLayout(bottom_layout)
        
        self.undoButton.pressed.connect(self.undo)
        self.redoButton.pressed.connect(self.redo)
        self.balancedButton.pressed.connect(self.next_balanced)
        self.uncertainButton.pressed.connect(self.next_most_uncertain)
        self.evaluateButton.pressed.connect(self.evaluate_classifier)
        self.packageButton.pressed.connect(self.package_results)
        self.datasetWidget.datasetUnload.connect(self.dataset_unloaded)
        self.datasetWidget.imageUnload.connect(self.image_unloaded)
        self.datasetWidget.datasetChanged.connect(self.dataset_changed)
        self.datasetWidget.indexChanged.connect(self.update_image_index)
        self.treeWidget.annotationChanged.connect(self.annotation_changed)
        
        self.treeWidget.setFocusPolicy(Qt.NoFocus)
        self.grabKeyboard()
        
        self.annotation_extension = '.json'
        self.storage_dir = './storage/'
        
        self.subject = subject
        self.annotator = annotator
        self.total_annotated = 0
        self.annotated = set()
        
        self.undid = False
        self.previous_index_list = []
        self.next_index_list = []
        
        self.shortcuts = [Qt.Key_1,Qt.Key_2,Qt.Key_3,Qt.Key_4,Qt.Key_5,Qt.Key_6,Qt.Key_7,Qt.Key_8,Qt.Key_9,Qt.Key_0]
        self.shortcut_names = [keymap[x] for x in self.shortcuts]

        
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Right:
            self.datasetWidget.next_image()
        elif event.key() == Qt.Key_Left:
            self.datasetWidget.prev_image()
        elif event.key() == Qt.Key_N:
            self.next_most_uncertain()
        elif event.key() == Qt.Key_B:
            self.next_balanced()        
        elif event.key() == Qt.Key_U:
            self.undo()
        elif event.key() == Qt.Key_R:
            self.redo()
        elif event.key() in self.shortcuts:
            self.shortcutPressed.emit(self.shortcuts.index(event.key()))

    def image_unloaded(self, idx):
        if not self.annotations['annotations_meta']['acknowledged']:
            self.annotated.add(idx)
            self.total_annotated += 1
            self.annotationCounter.setText(f'Annotated: {self.total_annotated}')
            
        self.annotations['annotations_meta']['acknowledged'] = True

        for x in self.classifiers:
            x.train()
            
        self.save_annotations()
        self.current_clip_value = None
        
        if not self.undid:
            self.previous_index_list.append(idx)
    
    def save_annotations(self):
        annotations_filename = os.path.splitext(self.datasetWidget.get_image(True))[0] + self.annotation_extension
        json.dump(self.annotations, open(annotations_filename, 'w'), indent=4)
    
    def save_metadata(self):
        yaml.dump(self.metadata, open(self.metadata_filename, 'w'), sort_keys=False)
    
    def load_annotations(self):
        annotations_filename = os.path.splitext(self.datasetWidget.get_image(True))[0] + self.annotation_extension

        if os.path.exists(annotations_filename):
            self.annotations = json.load(open(annotations_filename, 'r'))
        else:
            self.annotations = {}

        if 'annotations' not in self.annotations:
            self.annotations['annotations'] = {}
        if 'annotations_meta' not in self.annotations:
            self.annotations['annotations_meta'] = {'acknowledged': False, 'annotator':self.annotator}
        else:
            if 'acknowledged' not in self.annotations['annotations_meta']:
                self.annotations['annotations_meta']['acknowledged'] = False
            if 'annotator' not in self.annotations['annotations_meta']:
                self.annotations['annotations_meta']['annotator'] = self.annotator
        
    def update_image_index(self, idx):
        self.update_image()
        self.metadata['current_image'] = self.datasetWidget.get_image()
        self.save_metadata()
        
        self.load_annotations()
        self.current_clip_value = self.datasetWidget.clips[self.datasetWidget.current_image_index]
        
        if self.annotations['annotations'] != {}:
            valid, instance_root = instances_from_dict(self.annotations['annotations'], self.structure)
            if not valid:
                valid, instance_root = instances_from_dict(self.annotations['annotations'], self.structure, strict=False)
                # TODO: Save copy of annotations, in case something gets lost?
                if valid:
                    print('Warning: Stored data did not match structure, resolved using less strict matching')
                else:
                    print('Warning: Stored data did not match structure')
                    instance_root = create_instance_from_template(self.structure)
        else:
            instance_root = create_instance_from_template(self.structure)

        self.treeWidget.set_root(instance_root)
            
    def closeEvent(self, event):
        self.annotations['annotations_meta']['acknowledged'] = True
        self.save_annotations()
        for x in self.classifiers:
            x.storage.save()

    def nam_finished(self, reply):
        if reply.error() != QNetworkReply.NoError:
            print("Network error: ", reply.errorString())
            return
        
        pixmap = QPixmap()
        pixmap.loadFromData(reply.readAll())
        # pixmap = pixmap.scaled(QSize(512,512), Qt.KeepAspectRatio)
        self.imageWidget.setPixmap(pixmap)

    def update_image(self):
        self.nam.get(QNetworkRequest(QUrl(self.datasetWidget.get_url())))
        self.imageWidget.setPixmap(QPixmap(1,1))
        
    
    def dataset_unloaded(self, directory):
        self.save_annotations()
        self.annotations = {}
        self.current_clip_value = None
        for x in self.classifiers:
            x.storage.save()
        
    def dataset_changed(self, directory):
        self.load_classifiers()
        
        self.metadata_filename = os.path.join('./output', 'metadata.yaml')
        self.metadata = {'annotator': self.annotator, 'last_change':time.time()}
        self.annotations = {}
        if os.path.exists(self.metadata_filename):
            self.metadata = yaml.safe_load(open(self.metadata_filename, 'r'))
            self.datasetWidget.set_image(self.metadata['current_image'])
        else:
            self.datasetWidget.set_index(0)

        self.total_annotated = 0
        self.annotated = set()
        for i, img_name in enumerate(self.datasetWidget.image_names_internal):
            filename = os.path.splitext(img_name)[0] + self.annotation_extension
            if not os.path.exists(filename):
                continue
                        
            annotations = json.load(open(filename, 'r'))
            if 'annotations_meta' not in annotations:
                continue
            
            if 'acknowledged' not in annotations['annotations_meta']:
                continue
            
            if not annotations['annotations_meta']['acknowledged']:
                continue

            self.annotated.add(i)
            self.total_annotated += 1
            
        self.annotationCounter.setText(f'Annotated: {self.total_annotated}')

        self.metadata['current_image'] = self.datasetWidget.get_image()
        
        clips = self.datasetWidget.clips
        clips_index = list(range(len(clips)))

        self.clips = np.stack(clips, axis=0) if clips != [] else []
        self.clips_index = clips_index
        
    def annotation_changed(self):
        self.annotations['annotations'] = self.treeWidget.get_value()
        self.metadata['last_change'] = time.time()
        self.annotations['annotations_meta']['time'] = time.strftime('%Y/%m/%d %H:%M')
        self.save_annotations()
        self.save_metadata()
        
        if self.current_clip_value is None:
            return
        
        for x in self.classifiers:
            current_target = x.storage.get_y(self.current_clip_value)
            t = load_target_from_annotation(x.name, self.structure, self.annotations['annotations'])
            if current_target is not None:
                if t is None:
                    x.storage.remove(self.current_clip_value)
                    x.needs_retrain = True
                elif any(current_target != t):
                    x.storage.store(self.current_clip_value, t)
                    x.needs_retrain = True
            else:
                if t is not None:
                    x.storage.store(self.current_clip_value, t)
                    x.needs_retrain = True
    
    def current_clip(self):
        return self.current_clip_value
    
    def load_classifiers(self):
        self.classifiers = find_classifiers(self.structure, self.storage_dir)
        for x in self.classifiers:
            x.template.app = self
            # TODO: Only load targets if not already done so?
            x.load_targets(self.datasetWidget.image_names_internal, self.datasetWidget.clips, self.structure, annotation_extension=self.annotation_extension)
            x.train()
    
    def next_most_uncertain(self):
        for x in self.classifiers:
            x.train()
        
        sum_entropy = 0
        for c in self.classifiers:
            try:
                sum_entropy += c.predict_entropy(self.clips)
            except NotFittedError:
                pass
        if type(sum_entropy) == int:
            return

        while not (sum_entropy == 1e10).all():
            index = sum_entropy.argmin()
            if index in self.annotated or index == self.datasetWidget.current_image_index:
                sum_entropy[index] = 1e10
            else:
                break


        if not (sum_entropy == 1e10).all():
            self.datasetWidget.set_index(self.clips_index[index])
        
            
    def next_balanced(self):
        sum_score = 0
        for c in self.classifiers:
            c.train()
        
            try:
                if isinstance(c, ChoiceClassifier):
                    labels, counts = np.unique(c.storage.y, return_counts=True)
                    
                    y, probs = c.predict_many(self.clips)
                    for i,cl in enumerate(c.classifier.classes_):
                        w = np.where(labels == cl)[0]
                        if w.size == 0:
                            continue

                        sum_score += (np.clip(probs[:,i], 0.0, 0.8)) * -(1/counts[w])
                elif isinstance(c, SetClassifier):
                    pos_counts = c.storage.y.sum(axis=0)
                    neg_counts = c.storage.y.shape[0] - pos_counts
                    
                    y, probs = c.predict_many(self.clips)
                    for i in range(probs.shape[1]):
                        sum_score += (np.clip(probs[:,i], 0.0, 0.8)) * -(1/pos_counts[i])
                        sum_score += (np.clip(1-probs[:,i], 0.0, 0.8)) * -(1/neg_counts[i])
                elif isinstance(c, BoolClassifier):
                    pos_count = c.storage.y.sum()
                    neg_count = c.storage.y.shape[0] - pos_count
                    
                    y, probs = c.predict_many(self.clips)
                    sum_score += (np.clip(probs, 0.0, 0.8)) * -(1/pos_count)
                    sum_score += (np.clip(1-probs, 0.0, 0.8)) * -(1/neg_count)
            except NotFittedError:
                pass
                
        if type(sum_score) == int:
            return
            
        # Ignore already fully tagged items
        while not (sum_score == 1e10).all():
            index = sum_score.argmin()
            if index in self.annotated or index == self.datasetWidget.current_image_index:
                sum_score[index] = 1e10
            else:
                break

        if not (sum_score == 1e10).all():
            self.datasetWidget.set_index(self.clips_index[index])
        

    def evaluate_classifier(self):
        msgbox = QMessageBox()
        msgbox.setWindowTitle('Evaluation')
        msg = ''
        for ci,c in enumerate(self.classifiers):
            if ci > 0:
                msg += '------------\n'
            try:
                cname = '.'.join(c.name)
                x = c.storage.x
                y = c.storage.y
                
                if isinstance(c, ChoiceClassifier):
                    cm = 0
                    loo = KFold(5)
                    labels = list(range(len(c.template.options)))
                    for train_index, test_index in loo.split(x):
                        cl = MLPClassifier((64,16), alpha=0.01, activation='tanh', solver='lbfgs', max_iter=500, verbose=True)
                        cl.fit(x[train_index], y[train_index,0])
                        y_pred = cl.predict(x[test_index])
                        cm += confusion_matrix(y[test_index], y_pred, labels=labels)
                    
                    
                    accuracy = np.diag(cm) / cm.sum(axis=1)
                    
                    msg += f'{cname} - Classification accuracy:\n'
                    for name, acc in zip(c.template.options, accuracy):
                        msg += f'  {str(name): <15} {acc:.3f}\n'
                    
                    msg += f'\nConfusion matrix:\n{cm}\n'
                elif isinstance(c, SetClassifier):
                    no = len(c.template.options)
                    tp = np.zeros(no, int)
                    fp = np.zeros(no, int)
                    fn = np.zeros(no, int)
                    tn = np.zeros(no, int)

                    loo = KFold(5)
                    for train_index, test_index in loo.split(x):
                        cl = MLPClassifier((64,16), alpha=0.01, activation='tanh', solver='lbfgs', max_iter=500, verbose=True)
                        cl.fit(x[train_index], y[train_index])
                        y_pred = cl.predict(x[test_index])
                        for i in range(no):
                            tp[i] += (y_pred[y[test_index,i] == 1,i] == 1).sum()
                            fp[i] += (y_pred[y[test_index,i] == 0,i] == 1).sum()
                            fn[i] += (y_pred[y[test_index,i] == 1,i] == 0).sum()
                            tn[i] += (y_pred[y[test_index,i] == 0,i] == 0).sum()
                    
                    accuracy = (tp + tn) / y.shape[0]
                    dice = 2*tp / (2*tp + fp + fn)
                    msgbox = QMessageBox()
                    msgbox.setWindowTitle('Evaluation')
                    msg += f'{cname} - Classification metrics:\n'
                    for name, tpn, fpn, tnn, fnn, d, acc in zip(c.template.options, tp, fp, tn, fn, dice, accuracy):
                        msg += f'  {str(name): <15} Accuracy: {acc:.3f}     Dice: {d:.3f}\n'
                        msg += f'  {str(name): <15} TP: {tpn:4}, FP: {fpn:4}, TN: {tnn:4}, FN: {fnn:4}\n'
                elif isinstance(c, BoolClassifier):
                    tp = 0
                    fp = 0
                    fn = 0
                    tn = 0
                    
                    loo = KFold(5)
                    for train_index, test_index in loo.split(x):
                        cl = MLPClassifier((64,16), alpha=0.01, activation='tanh', solver='lbfgs', max_iter=500, verbose=True)
                        cl.fit(x[train_index], y[train_index, 0])
                        y_pred = cl.predict(x[test_index])
                        tp += (y_pred[y[test_index,0] == 1] == 1).sum()
                        fp += (y_pred[y[test_index,0] == 0] == 1).sum()
                        fn += (y_pred[y[test_index,0] == 1] == 0).sum()
                        tn += (y_pred[y[test_index,0] == 0] == 0).sum()
                    
                    accuracy = (tp + tn) / y.shape[0]
                    dice = 2*tp / (2*tp + fp + fn)
                    
                    msg += f'{cname} - Classification metrics:\n'
                    msg += f'  Accuracy: {accuracy:.3f}     Dice: {dice:.3f}\n'
                    msg += f'  TP: {tp:4}, FP: {fp:4}, TN: {tn:4}, FN: {fn:4}\n'

            except Exception as e:
                msg += f'{cname} - Exception occured:\n' + str(e)

                
        msgbox.setFont(QFont('Courier'))
        msgbox.setText(msg)
        msgbox.setStyleSheet('QLabel{min-height: 300px; min-width: 1200px}')
        msgbox.exec()
        
    def undo(self):
        if len(self.previous_index_list) == 0:
            return
        
        self.next_index_list.append(self.datasetWidget.current_image_index)
        self.undid = True
        self.datasetWidget.set_index(self.previous_index_list.pop())
        self.undid = False

    def redo(self):
        if len(self.next_index_list) == 0:
            return
        
        self.datasetWidget.set_index(self.next_index_list.pop())

    def package_results(self):
        with zipfile.ZipFile(f'./result_{self.subject}.zip', 'w', zipfile.ZIP_DEFLATED) as zip_file:
            
            for i in self.datasetWidget.image_names_internal:
                filename = os.path.splitext(i)[0] + self.annotation_extension
                if os.path.exists(filename):
                    zip_file.write(filename)
            
            for c in self.classifiers:
                c.storage.save()
                if os.path.exists(c.storage.basename + '_x.npy'):
                    zip_file.write(c.storage.basename + '_x.npy')
                    zip_file.write(c.storage.basename + '_y.npy')
            
        msgbox = QMessageBox()
        msgbox.setWindowTitle('Packaged')
        msgbox.setText(f'Results package saved to: ./result_{self.subject}.zip')
        
        msgbox.exec()


if __name__ == '__main__':
    settings = QSettings('SmirkingFace', 'Annotator')
    if settings.value('id') == None:
        settings.setValue('id', random.randint(0,1000000000))
    
    os.makedirs('./output', exist_ok=True)
    os.makedirs('./storage', exist_ok=True)
    
    app = QApplication(sys.argv)
    filename = QFileDialog.getOpenFileName(QWidget(), 'Open file', '.', 'Configuration files (*.yaml)')[0]

    if filename == '':
        sys.exit(0)

    structure = yaml.safe_load(open(filename, 'r'))
    structure = parse_annotation_structure(structure)
    ext = os.path.splitext(os.path.basename(filename))[0]

    view = AnnotationWidget(structure['Scene'], annotator=settings.value('id'), subject=ext)
    widgets.mainWidget = view
    view.annotation_extension = f'_{ext}.json'
    view.storage_dir = './storage/'
    view.setGeometry(100, 100, 1800, 900)
    view.setWindowTitle('Annotator')
    view.datasetWidget.load_dataset('./dataset.parquet')
    view.show()
    sys.exit(app.exec_())

