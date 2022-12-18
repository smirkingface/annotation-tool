class String:
    def __init__(self, name='', default_value=None, allow_undefined=True):
        self.name = name
        self.default_value = default_value
        self.allow_undefined = allow_undefined
        if not allow_undefined and default_value == None:
            self.default_value = ''

    def __str__(self):
        return self.name if self.name else 'Custom string'

class StringLiteral:
    def __init__(self, name=''):
        self.name = name
    
    def __str__(self):
        return self.name
    
class Bool:
    def __init__(self, name='', default_value=False, train_classifier=False):
        self.name = name
        self.default_value = default_value
        self.train_classifier = train_classifier
    
    def __str__(self):
        return self.name

class Choice:
    def __init__(self, name='', options=[], default_option_index=None, train_classifier=False):
        self.name = name
        self.options = options
        self.default_option_index = default_option_index
        if default_option_index == None:
            self.default_option_index = 0
        self.train_classifier = train_classifier
    
    def __str__(self):
        return self.name if self.name else '[' + ' or '.join(str(x) for x in self.options) + ']'

class List:
    def __init__(self, name='', item_type=None):
        self.name = name
        self.item_type = item_type
    
    def __str__(self):
        return f'List({self.name})' if self.name else f'List of {self.item_type}'

class Set:
    def __init__(self, name='', options=[], default_value=[], train_classifier=False):
        self.name = name
        self.options = options
        self.default_value = set(default_value)
        self.train_classifier = train_classifier
    
    def __str__(self):
        return f'Set({self.name})' if self.name else 'Set(' + ','.join(str(x) for x in self.options) + ')'

class Dictionary:
    def __init__(self, name='', dict_items={}):
        self.name = name
        self.dict_items = dict_items
    
    def __str__(self):
        return f'{self.name}' if self.name else 'Dict(' + ','.join(self.dict_items) + ')'

class Reference:
    def __init__(self, name='', structure=None):
        self.name = name
        self.structure = structure
        
    def __str__(self):
        return str(self.structure[self.name])


def parse_annotation_template(item, name='', structure={}): 
    if type(item) == dict:
        if 'type' in item:
            if item['type'] == 'Choice':
                return Choice(options=[parse_annotation_template(x, structure=structure) for x in item['options']], **{x:item[x] for x in item if x not in ['type','options']})
            elif item['type'] == 'Set':
                return Set(**{x:item[x] for x in item if x != 'type'})
            elif item['type'] == 'List':
                return List(item_type=parse_annotation_template(item['item_type'], structure=structure), **{x:item[x] for x in item if x not in ['type','item_type']})
            elif item['type'] == 'String':
                return String(**{x:item[x] for x in item if x != 'type'})
            elif item['type'] == 'Bool':
                return Bool(**{x:item[x] for x in item if x != 'type'})
            else:
                raise ValueError('Invalid item type in structure:', item['type'])
        else:
            r = {}
            for x in item:
                r[x] = parse_annotation_template(item[x], structure=structure)
            return Dictionary(name, r)
    elif type(item) == str:
        if item in structure:
            return Reference(item, structure)
        elif item.startswith('String'):
            s = item.split(' ')
            if len(s) > 0:
                return String(' '.join(s[1:]))
            else:
                return String()
        elif item.startswith('Bool'):
            s = item.split(' ')
            if len(s) > 0:
                return Bool(' '.join(s[1:]))
            else:
                return Bool()
        else:
            return StringLiteral(item)
    elif type(item) == list:
        return Set(name, item)
    else:
        raise ValueError(f'Incorrect type in template: {type(item)}')
        

def parse_annotation_structure(structure):
    result = {x:None for x in structure}
    for x in structure:
        result[x] = parse_annotation_template(structure[x], name=x, structure=result)
    return result