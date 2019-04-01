def IsContainer(x):
    '''
    tests if x is an instance of one of the builtin container types
    '''
    return isinstance(x, (dict, frozenset, list, set, tuple))

def Tupify(x):
    '''
    if x is an instance of a builtin container, convert it to a tuple. Otherwise, place x into a tuple
    '''
    if IsContainer(x):
        return tuple(x)
    else:
        return (x,)