import os
import random
import pandas as pd

from PyQt5.QtCore import Qt, QPoint, QPointF, QRect, QMargins, QSize, pyqtSignal as Signal
from PyQt5.QtWidgets import QWidget, QTreeWidget, QLayout, QHBoxLayout, QVBoxLayout, QSizePolicy, QRadioButton, QCheckBox, QComboBox, QLabel, QPushButton, QFileDialog
from PyQt5.QtGui import QPixmap, QPainter

mainWidget = None

class AnnotationTreeWidget(QWidget):
    annotationChanged = Signal()
    
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.tree = QTreeWidget(self) # TODO: Could inherit from QTreeWidget?
        
        self.tree.setStyleSheet('''
QTreeView::branch:has-siblings:!adjoins-item {
    border-image: url(vline.png) 0;
}

QTreeView::branch:has-siblings:adjoins-item {
    border-image: url(branch-more.png) 0;
}

QTreeView::branch:!has-children:!has-siblings:adjoins-item {
    border-image: url(branch-end.png) 0;
}

QTreeView::branch:has-children:!has-siblings:closed,
QTreeView::branch:closed:has-children:has-siblings {
        border-image: none;
        image: url(branch-closed.png);
}

QTreeView::branch:open:has-children:!has-siblings,
QTreeView::branch:open:has-children:has-siblings  {
        border-image: none;
        image: url(branch-open.png);
}
                                ''')
        
        self.tree.annotationWidget = self
        self.tree.setColumnCount(2)
        self.tree.setColumnWidth(0,300)
        self.tree.setHeaderHidden(True)
        
        self.instance_root = None
        
        self.layout = QHBoxLayout(self)
        self.layout.addWidget(self.tree)
    
    def set_root(self, instance_root):
        if self.instance_root is not None:
            self.tree.clear()
            # self.instance_root.remove()
            
        self.instance_root = instance_root
        import annotation_instance
        annotation_instance.shortcut_counter = 0
        self.instance_root.create_items(instance_root.template.name, self.tree)
        self.annotationChanged.emit()
    
    def get_value(self):
        return self.instance_root.get_value() if self.instance_root else ''

class ImageWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.image = QPixmap()
        self.resize(self.image.width(), self.image.height())
        self.zoom = 1

    def setPixmap(self, pixmap):
        self.image = pixmap
        self.resize(self.image.width(), self.image.height())
        self.updateGeometry()
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        
        if self.image.size().width() == 0:
            return
        
        rw, rh = self.rect().width(), self.rect().height()
        iw, ih = self.image.size().width(), self.image.size().height()
        
        rar = rw / rh
        iar = iw / ih
        
        if rar > iar:
            h = rh
            w = iar * h
            y = self.rect().y()
            x = self.rect().x() + rw/2 - w/2
        else:
            w = rw
            h = w / iar
            x = self.rect().x()
            y = self.rect().y() + rh/2 - h/2
        self.imageRect = QRect(round(x),round(y),round(w),round(h))
        
        painter.drawPixmap(self.imageRect, self.image)

    def getRelativeCoordinate(self, p):
        r = self.imageRect
        x0, y0 = r.x(), r.y()
        x1, y1 = x0 + r.width(), y0 + r.height()
        x_relative = (p.x() - x0) / (x1 - x0)
        y_relative = (p.y() - y0) / (y1 - y0)
        
        return QPointF(x_relative * self.image.size().width(), y_relative * self.image.size().height())

    def wheelEvent(self, event):
        if event.angleDelta().y() > 0:
            self.zoom *= 1.1
        else:
            self.zoom /= 1.1

class HorizontalRadioWidget(QWidget):
    currentIndexChanged = Signal(int)
    
    def __init__(self, options, parent=None, shortcutIndex=0):
        super().__init__(parent=parent)
        layout = QVBoxLayout(self)
        
        self.shortcutIndex = shortcutIndex
        
        shortcut_names = [mainWidget.shortcut_names[i] if i < len(mainWidget.shortcut_names) else None for i in range(shortcutIndex, shortcutIndex+len(options))]
        
        self.buttons = [QRadioButton((f'({s}) ' if s else '') + x) for s,x in zip(shortcut_names,options)]
        self.buttons[0].setChecked(True)
        for b in self.buttons:
            layout.addWidget(b, alignment=Qt.AlignLeft)
            b.toggled.connect(self.button_toggled)
        layout.addStretch(1)
        
        self.currentIndex = 0

        mainWidget.shortcutPressed.connect(self.shortcut)
    
    def shortcut(self, idx):
        if idx >= self.shortcutIndex and idx < self.shortcutIndex + len(self.buttons):
            self.setCurrentIndex(idx - self.shortcutIndex)
    
    def setCurrentIndex(self, idx):
        self.buttons[idx].setChecked(True)
        self.currentIndex = idx
        self.currentIndexChanged.emit(idx)
    
    def button_toggled(self):
        idx = [b.isChecked() for b in self.buttons].index(True)
        if idx != self.currentIndex:
            self.currentIndex = idx
            self.currentIndexChanged.emit(idx)


class HorizontalCheckBoxWidget(QWidget):
    itemChanged = Signal(int, bool)
    
    def __init__(self, names, parent=None, shortcutIndex=0):
        super().__init__(parent=parent)
        layout = QVBoxLayout(self)
        
        self.shortcutIndex = shortcutIndex
        shortcut_names = [mainWidget.shortcut_names[i] if i < len(mainWidget.shortcut_names) else None for i in range(shortcutIndex, shortcutIndex+len(names))]

        self.buttons = [QCheckBox((f'({s}) ' if s else '') + x) for s,x in zip(shortcut_names,names)]
        for i,b in enumerate(self.buttons):
            layout.addWidget(b, alignment=Qt.AlignLeft)
            b.stateChanged.connect(lambda x, i=i:self.itemChanged.emit(i, x))
        layout.addStretch(1)
        
        mainWidget.shortcutPressed.connect(self.shortcut)

    def shortcut(self, idx):
        if idx >= self.shortcutIndex and idx < self.shortcutIndex + len(self.buttons):
            idx -= self.shortcutIndex
            
            if self.buttons[idx].checkState() == Qt.Checked:
                self.setChecked(idx, False)
            else:
                self.setChecked(idx, True)
    

    def setChecked(self, idx, value):
        v = self.buttons[idx].checkState() == Qt.Checked
        if value != v:
            self.buttons[idx].setCheckState(Qt.Checked if value else Qt.Unchecked)




class DatasetWidget(QWidget):
    datasetChanged = Signal(str)
    datasetUnload = Signal(str)
    indexChanged = Signal(int)
    imageUnload = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        
        self.dataset_directory = None
        self.metadata = None
        self.image_names = []
        self.current_image_index = None
        
        layout = QHBoxLayout(self)
        self.imageList = QComboBox(self)
        self.datasetLabel = QLabel(self)
        self.prevButton = QPushButton('<-', parent=self)
        self.nextButton = QPushButton('->', parent=self)
        self.randomCheckbox = QCheckBox('Random', parent=self)
        self.datasetLabel.setText('No dataset loaded')
        
        layout.addWidget(self.datasetLabel)
        layout.addWidget(self.imageList)
        layout.addWidget(self.prevButton)
        layout.addWidget(self.nextButton)
        layout.addWidget(self.randomCheckbox)
        
        self.prevButton.pressed.connect(self.prev_image)
        self.nextButton.pressed.connect(self.next_image)
        self.imageList.currentIndexChanged.connect(self.select_image)
    
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Right:
            self.next_image()
        elif event.key() == Qt.Key_Left:
            self.prev_image()
    
    def is_randomized(self):
        return self.randomCheckbox.checkState() == Qt.Checked
    
    def get_directory(self):
        return self.dataset_directory
    
    def get_index(self):
        return self.current_image_index
    
    def get_image(self, full_path=False):
        return self.image_names_internal[self.current_image_index]
    
    def get_url(self, full_path=False):
        return self.urls[self.current_image_index]
    
    def select_image(self, idx):
        self.set_index(idx)

    def next_image(self):
        if self.is_randomized():
            self.set_index(self.random_next[self.current_image_index])
        else:
            self.set_index(min(self.current_image_index + 1, len(self.image_names)-1))

    def prev_image(self):
        if self.is_randomized():
            self.set_index(self.random_prev[self.current_image_index])
        else:
            self.set_index(max(self.current_image_index - 1, 0))
    
    def random_image(self):
        self.set_index(random.randint(0, len(self.image_names)-1))
    
    def set_image(self, name):
        if name in self.image_names_internal:
            self.set_index(self.image_names_internal.index(name))
    
    def set_index(self, idx):
        assert idx >= 0 and idx < len(self.image_names), 'Invalid image index'
        
        prev_index = self.current_image_index

        if prev_index != None and prev_index != idx:
            self.imageUnload.emit(prev_index)
        self.current_image_index = idx
        
        self.imageList.blockSignals(True)
        self.imageList.setCurrentIndex(self.current_image_index)
        self.imageList.blockSignals(False)
        
        if prev_index != self.current_image_index:
            self.indexChanged.emit(self.current_image_index)

    def load_dataset(self, filename):
        if not os.path.exists(filename):
            self.current_image_index = None
            return

        df = pd.read_parquet(filename)

        self.current_image_index = None
        self.urls = df['url'].tolist()
        self.clips = df['clip'].tolist()
        self.image_names = [str(x) for x in df['id'].tolist()]
        self.image_names_internal = [f'./output/{i:09}.jpg' for i in df['id'].tolist()]
        if len(self.image_names) == 0:
            print('No images found in directory')
            self.current_image_index = None
            return
        
        # Predictable random order, to allow forward/backward to work as expected
        rng_state = random.getstate()
        random.seed(0)
        random_order = list(range(len(self.image_names)))
        random.shuffle(random_order)
        
        self.random_next = dict(zip(random_order[:-1], random_order[1:]))
        self.random_prev = dict(zip(random_order[1:], random_order[:-1]))
        
        random.setstate(rng_state)
        
        # Update widgets
        self.imageList.blockSignals(True)
        self.imageList.clear()
        self.imageList.addItems([x for x in self.image_names])
        self.imageList.blockSignals(False)
        
        self.datasetLabel.setText(filename)
        
        # Emit signals
        self.datasetChanged.emit(filename)
