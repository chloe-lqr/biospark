from itertools import chain

__all__ = ['DictDiffer']

class DictDiffer(object):
    '''
    Calculate the difference between two dictionaries as:
    (1) items added
    (2) items removed
    (3) keys same in both but changed values
    (4) keys same in both and unchanged values

    modified from
    https://raw.githubusercontent.com/hughdbrown/dictdiffer/master/dictdiffer/__init__.py
    '''
    def __init__(self, current_dict, past_dict):
        self.current_dict, self.past_dict = current_dict, past_dict
        self.current_keys, self.past_keys = [
            set(d.keys()) for d in (current_dict, past_dict)
            ]
        self.intersect = self.current_keys.intersection(self.past_keys)

    def __call__(self):
        return self.getDiff()

    def added(self):
        return self.current_keys - self.intersect

    def removed(self):
        return self.past_keys - self.intersect

    def changed(self):
        return set(o for o in self.intersect
                   if self.past_dict[o] != self.current_dict[o])

    def unchanged(self):
        return set(o for o in self.intersect
                   if self.past_dict[o] == self.current_dict[o])

    def getDiff(self):
        '''
        return a new dict with elements in .current_dict that are not in past_dict
        '''
        retDict = {}
        for key in chain(self.added(), self.changed()):
            retDict[key] = self.current_dict[key]
        return retDict