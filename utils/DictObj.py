class DictObj:
    '''
    This class can convert a dict into a python class.
    Then we can access the attributes via ".keyname" instead of "[keyname]"
    '''
    def __init__(self, in_dict:dict):
        assert isinstance(in_dict, dict)
        for key, val in in_dict.items():
            if isinstance(val, (list, tuple)):
               setattr(self, key, [DictObj(x) if isinstance(x, dict) else x for x in val]) #学习这里递归的调用方法
            else:
               setattr(self, key, DictObj(val) if isinstance(val, dict) else val)