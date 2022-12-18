from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QPushButton, QTreeWidgetItem
from PyQt5.QtGui import QColor

from widgets import HorizontalRadioWidget, HorizontalCheckBoxWidget
from annotation import String, StringLiteral, List, Choice, Reference, Set, Bool, Dictionary
from sklearn.exceptions import NotFittedError

cmap = [(int(x[1:3],16), int(x[3:5], 16), int(x[5:7],16)) for x in '#F3B94D,#E1C03B,#C9C82D,#AAD129,#81D835,#3CDF4B'.split(',')]

shortcut_counter = 0

def classification_color(value):
    value *= len(cmap)-1
    index = int(value)
    alpha = value - index
    
    if index < 0:
        index = 0
        alpha = 0
    
    if index >= len(cmap)-1:
        index = len(cmap)-2
        alpha = 1

    return [(1-alpha)*c1 + alpha*c2 for c1,c2 in zip(cmap[index], cmap[index+1])]


def create_instance_from_template(template):
    it = type(template)
    
    if it == Dictionary:
        return DictionaryInstance(template)
    elif it == List:
        return ListInstance(template)
    elif it == String:
        return StringInstance(template)
    elif it == StringLiteral:
        return StringLiteralInstance(template)
    elif it == Choice:
        if template.train_classifier:
            try:
                clip = template.app.current_clip()
                if clip is None:
                    return ChoiceInstance(template)
                value_index, probs = template.classifier.predict(clip)

                new_value = template.options[value_index]
                new_value = create_instance_from_template(new_value)
                return ChoiceInstance(template, value=new_value, value_index=value_index, probabilities=probs)
            except NotFittedError:
                return ChoiceInstance(template)
        else:
            return ChoiceInstance(template)
    elif it == Set:
        if template.train_classifier:
            try:
                clip = template.app.current_clip()
                if clip is None:
                    return SetInstance(template)
                v, probs = template.classifier.predict(clip)
                return SetInstance(template, value=v, probabilities=probs)
            except NotFittedError:
                return SetInstance(template)
        else:
            return SetInstance(template)
    elif it == Reference:
        return create_instance_from_template(template.structure[template.name])
    elif it == Bool:
        if template.train_classifier:
            try:
                clip = template.app.current_clip()
                if clip is None:
                    return BoolInstance(template)
                v, prob = template.classifier.predict(template.app.current_clip())
                return BoolInstance(template, value=v, probability=prob)
            except NotFittedError:
                return BoolInstance(template)
        else:
            return BoolInstance(template)
    else:
        raise ValueError('Invalid template type:', template)

def instances_from_dict(value, template, strict=True):
    st = type(template)
    vt = type(value)

    if value == None or value == {}:
        return True, create_instance_from_template(template)
    
    if type(template) == Reference:
        template = template.structure[template.name]
        st = type(template)

    if st == StringLiteral and vt == str:
        if not strict or value == template.name:
            return True, StringLiteralInstance(template)
        else:
            return False, None
    elif st == String and vt == str:
        return True, StringInstance(template, value)
    elif st == Bool and vt == bool:
        return True, BoolInstance(template, value)
    elif st == List and vt == list:
        r = [instances_from_dict(x, template.item_type) for x in value]
        if any(x[0] == False for x in r):
            if strict:
                return False, None
            else:
                r = filter(lambda x:x[0], r)
        return True, ListInstance(template, value=[x[1] for x in r])
    elif st == Dictionary and vt == dict and 'dictionary_class' in value:
        if value['dictionary_class'] != template.name:
            return False, None
        r = {}
        # TODO: Maybe strict=True should check if items in value are not in dict_items?
        for x in template.dict_items:
            if x not in value:
                if strict:
                    return False, None
                else:
                    v = create_instance_from_template(template.dict_items[x])
            else:
                valid, v = instances_from_dict(value[x], template.dict_items[x], strict=strict)
                if not valid:
                    if strict:
                        return False, None
                    else:
                        v = create_instance_from_template(template.dict_items[x])
            r[x] = v
        return True, DictionaryInstance(template, value=r)
    elif st == Set and vt == dict and ('set_class' in value or not strict):
        if strict and value['set_class'] != template.name:
            return False, None
        
        if 'value' not in value and strict:
            return False, None
        
        r = set()
        for x in value['value']:
            if x not in template.options:
                if strict:
                    return False, None
                else:
                    continue
            r.add(x)
        return True, SetInstance(template, value=r)
    elif st == Choice and vt == dict and ('choice_class' in value or not strict):        
        if 'value' not in value or 'value_index' not in value:
            return False, None
        
        valid,v = instances_from_dict(value['value'], template.options[value['value_index']], strict=True)
        if valid:
            return True, ChoiceInstance(template, value=v, value_index=value['value_index'])
        
        if not strict:
            valid,v = instances_from_dict(value['value'], template.options[value['value_index']], strict=False)
            if valid:
                return True, ChoiceInstance(template, value=v, value_index=value['value_index'])
        
        if not valid and strict:
            return False, None

        for i,x in enumerate(template.options):
            valid,v = instances_from_dict(value['value'], x, strict=True)
            if valid:
                return True, ChoiceInstance(template, value=v, value_index=i)

        if not strict:
            for i,x in enumerate(template.options):
                valid,v = instances_from_dict(value['value'], x, strict=False)
                if valid:
                    return True, ChoiceInstance(template, value=v, value_index=i)
        return False, None
    else:
        return False, None



class StringInstance:
    def __init__(self, template, value=None):
        self.template = template
        self.value = value
        if value == None:
            self.value = template.default_value
        self.item = None
    
    def item_doubleclicked(self, item, column):
        if self.item != item:
            return
        if column != 1:
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
        else:
            item.setFlags(item.flags() | Qt.ItemIsEditable)
    
    def item_changed(self, item, column):
        if self.item != item or column != 1:
            return

        self.value = self.item.text(1)

        if self.value == '' and self.template.allow_undefined:
            self.value = None
            self.ignore_itemchanged = True
            self.item.treeWidget().blockSignals(True)
            self.item.setText(1, '<undefined>')
            self.item.treeWidget().blockSignals(False)
        
        self.item.treeWidget().annotationWidget.annotationChanged.emit()

    def create_items(self, name, parent):
        item = QTreeWidgetItem(parent)
        item.setText(0, name)
        if self.value == None:
            assert(self.template.allow_undefined)
            item.setText(1, '<undefined>')
        else:
            item.setText(1, self.value)
        self.item = item
        self.item.treeWidget().itemChanged.connect(self.item_changed)
        self.item.treeWidget().itemDoubleClicked.connect(self.item_doubleclicked)
    
    def remove(self):
        self.item.parent().removeChild(self.item)

    def get_value(self):
        return self.value
    
    def set_value(self, name, value):
        assert(len(name) == 0)
        assert(type(value) == str or (value == None and self.template.allow_undefined))
        
        self.item.setText(1, value if value else '')

class BoolInstance:
    def __init__(self, template, value=None, probability=None):
        self.template = template
        self.value = value
        if value == None:
            self.value = self.template.default_value
        self.item = None
        self.probability = probability
    
    def item_changed(self, item, column):
        if self.item != item or column != 1:
            return

        self.value = self.item.checkState(1) == Qt.Checked       
        self.item.treeWidget().annotationWidget.annotationChanged.emit()

    def create_items(self, name, parent):
        global shortcut_counter
        item = QTreeWidgetItem(parent)
        
        w = item.treeWidget().annotationWidget.parent().parent()
        shortcut_name = w.shortcut_names[shortcut_counter] if shortcut_counter < len(w.shortcut_names) else None
        self.shortcut_index = shortcut_counter
        shortcut_counter += 1

        item.setText(0, name)
        if shortcut_name:
            item.setText(1, f'({shortcut_name})')
        if not self.value:
            item.setCheckState(1, Qt.Unchecked)
        else:
            item.setCheckState(1, Qt.Checked)
        
        if self.probability != None:
            r,g,b = classification_color(self.probability)
            item.setData(1, Qt.BackgroundColorRole, QColor(int(r),int(g),int(b)))
        
        self.item = item
        self.item.treeWidget().itemChanged.connect(self.item_changed)
        w.shortcutPressed.connect(self.shortcut)
    
    def shortcut(self, index):
        if index == self.shortcut_index:
            if self.item.checkState(1) == Qt.Checked:
                self.item.setCheckState(1, Qt.Unchecked)
            else:
                self.item.setCheckState(1, Qt.Checked)
    
    def remove(self):
        self.item.parent().removeChild(self.item)

    def get_value(self):
        return self.value
    
    def set_value(self, name, value):
        assert(len(name) == 0)
        assert(type(value) == bool)
        
        if value:
            self.item.setCheckState(1, Qt.Unchecked)
        else:
            self.item.setCheckState(1, Qt.Checked)


class StringLiteralInstance:
    def __init__(self, template):
        self.template = template
        self.value = template.name
        self.item = None
    
    def create_items(self, name, parent):
        item = QTreeWidgetItem(parent)
        item.setText(0, name)
        item.setText(1, self.value)
        self.item = item
    
    def remove(self):
        self.item.parent().removeChild(self.item)

    def get_value(self):
        return self.value
    
    def set_value(self, name, value):
        raise ValueError('Trying to set String literal value')

class ChoiceInstance:
    def __init__(self, template, value=None, value_index=0, probabilities=None):
        self.template = template
        self.value_index = value_index
        self.value = value
        
        if value == None:
            if self.template.default_option_index != None:
                self.value_index = self.template.default_option_index
            self.value = self.create_new_value(value_index)
        
        self.item = None
        self.combobox = None
        self.probabilities = probabilities
    
    def create_new_value(self, index):
        new_value = self.template.options[self.value_index]
        return create_instance_from_template(new_value)
    
    def combobox_changed(self, index):
        if self.value is not None:
            if type(self.value) != StringLiteralInstance:
                self.value.remove()
        
        self.value_index = index
        new_value = self.create_new_value(self.value_index)

        if type(new_value) == DictionaryInstance:
            new_value.create_items(None, self.item)
        elif type(new_value) != StringLiteralInstance:
            new_value.create_items('Value', self.item)
        self.item.setExpanded(True)
        
        self.value = new_value
        self.item.treeWidget().annotationWidget.annotationChanged.emit()

    def create_items(self, name, parent):
        global shortcut_counter
        if not name:
            name = self.template.name

        item = QTreeWidgetItem(parent)
        item.setText(0, name)
        
        self.item = item
        self.combobox = HorizontalRadioWidget([str(x) for x in self.template.options], parent=item.treeWidget(), shortcutIndex=shortcut_counter)
        shortcut_counter += len(self.template.options)
        
        if self.probabilities != None:
            for i,x in self.probabilities:
                r,g,b = classification_color(x)
                self.combobox.buttons[i].setStyleSheet(f'QRadioButton{{ background-color: rgb({r},{g},{b});}}')
        
        if self.value != None:
            # Maybe connect changed signal and just set index?
            if type(self.value) == DictionaryInstance:
                self.value.create_items(None, self.item)
            elif type(self.value) != StringLiteralInstance and self.value != None:
                self.value.create_items('Value', self.item)
            self.combobox.setCurrentIndex(self.value_index)
        self.item.setExpanded(True)
        self.combobox.currentIndexChanged.connect(self.combobox_changed)
        item.treeWidget().setItemWidget(item, 1, self.combobox)
        
    def remove(self):
        if self.value is not None:
            if type(self.value) != StringLiteralInstance:
                self.value.remove()

        if self.item.parent():
            self.item.parent().removeChild(self.item)
    
    def get_value(self):
        return {'choice_class':self.template.name, 'value':self.value.get_value(), 'value_index':self.value_index}
    
    def set_value(self, name, value):
        assert len(name) == 0, name
        assert type(value) == int, type(value) # TODO: Allow actual values, and setting subvalues
        assert value >= 0 and value < len(self.template.options), value
        self.combobox.setCurrentIndex(value)

class SetInstance:
    def __init__(self, template, value=None, probabilities=None):
        self.template = template
        self.value = value
        if value == None:
            self.value = set(self.template.default_value)

        self.item = None
        self.items = []
        self.probabilities = probabilities
    
    def item_changed(self, idx, value):
        if value:
            self.value.add(self.template.options[idx])
        else:
            self.value.remove(self.template.options[idx])
        self.item.treeWidget().annotationWidget.annotationChanged.emit()
    
    def create_items(self, name, parent):
        global shortcut_counter

        if not name:
            name = self.template.name
        
        item = QTreeWidgetItem(parent)
        item.setText(0, name)
        item.setText(1, '')
        self.item = item
        
        self.checkbox = HorizontalCheckBoxWidget(self.template.options, parent=item.treeWidget(), shortcutIndex=shortcut_counter)
        shortcut_counter += len(self.template.options)
        for x in self.value:
            self.checkbox.setChecked(self.template.options.index(x), True)
        
        if self.probabilities != None:
            for i,x in enumerate(self.probabilities):
                r,g,b = classification_color(x)
                self.checkbox.buttons[i].setStyleSheet(f'QCheckBox{{ background-color: rgb({r},{g},{b});}}')
        
        
        item.treeWidget().setItemWidget(item, 1, self.checkbox)
        self.checkbox.itemChanged.connect(self.item_changed)


    def remove(self):
        self.item.parent().removeChild(self.item)
    
    def get_value(self):
        return {'set_class':self.template.name, 'value':list(self.value)}

    def set_value(self, name, value):
        assert(len(name) == 1)
        assert(name[0] in self.template.options)
        assert(type(value) == bool) # TODO: Allow set or list
        
        self.item_changed(self.template.options.index(name[0]), value)
        
    
class ListInstance:
    def __init__(self, template, value=None):
        self.template = template
        self.value = value
        self.item = None
        self.add_item = None
    
    def add_new_item(self):
        instance_name = f'{self.name}[{len(self.value)+1}]'
        instance = create_instance_from_template(self.template.item_type)

        self.value.append(instance)
        self.item.removeChild(self.add_item)
        instance.create_items(instance_name, self.item)
        self.item.addChild(self.add_item)
        self.add_item.setText(0, f'{self.name}[{len(self.value)+1}]')
        
        # Button disappears if item gets moved.. Just create a new one to work around this
        self.button = QPushButton('Add', parent=self.item.treeWidget())
        self.item.treeWidget().setItemWidget(self.add_item, 1, self.button)
        self.button.pressed.connect(self.add_new_item)
        
        self.item.treeWidget().annotationWidget.annotationChanged.emit()

    def remove_item(self, index):
        pass

    def create_items(self, name, parent):
        item = QTreeWidgetItem(parent)
        item.setText(0, name)
        self.item = item
        self.name = name
        
        # TODO: Maybe make an option to allow/disallow empty values
        if self.value == None:
            self.value = []

        for i,x in enumerate(self.value):
            x.create_items(f'{name}[{i+1}]', self.item)
        self.add_item = QTreeWidgetItem(self.item)
        self.add_item.setText(0, f'{name}[{len(self.value)+1}]')
        self.item.setExpanded(True)
        
        self.button = QPushButton('Add', parent=self.item.treeWidget())
        self.item.treeWidget().setItemWidget(self.add_item, 1, self.button)
        self.button.pressed.connect(self.add_new_item)

    def remove(self):
        if self.value is not None:
            for x in self.value:
                x.remove()

        self.item.parent().removeChild(self.item)
    
    def get_value(self):
        if self.value is None:
            return None
        
        r = []
        for x in self.value:
            v = x.get_value()
            r.append(v)
        return r
    
    def set_value(self, name, value):
        if len(name) == 0:
            raise NotImplementedError('ListInstance set_value with entire list value is not supported yet')
        else:
            assert(type(name[0]) == int)
            assert(name[0] < len(self.value))
            self.value[name[0]].set_value(name[1:], value)

class DictionaryInstance:
    def __init__(self, template, value=None):
        self.template = template
        self.value = value
        if value == None:
            self.value = {}
        self.item = None
        
    def create_items(self, name, parent):
        if name == '':
            name = self.template.name
        
        if name != None:
            item = QTreeWidgetItem(parent)
            item.setText(0, name)
            self.item = item
            parent = item

        for x in self.template.dict_items:
            if x not in self.value:
                self.value[x] = create_instance_from_template(self.template.dict_items[x])
            self.value[x].create_items(x, parent)

        parent.setExpanded(True)

    def remove(self):
        if self.value is not None:
            for x in self.value:
                self.value[x].remove()
        if self.item:
            self.item.parent().removeChild(self.item)
    
    def get_value(self):
        if self.value is None:
            return None
        r = {'dictionary_class': self.template.name}
        for x in self.value:
            v = self.value[x].get_value()
            r[x] = v
        return r
    
    def set_value(self, name, value):
        if len(name) == 0:
            raise NotImplementedError('DictionaryInstance set_value with entire dict value is not supported yet')
        elif name[0] in self.value:
            print('set_value', name[0])
            self.value[name[0]].set_value(name[1:], value)