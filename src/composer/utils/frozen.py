class FrozenDict(dict):
    def __init__(self, *args, **kwargs):
        super(FrozenDict, self).__init__(*args, **kwargs)

    def __setitem__(self, key, value):
        raise TypeError("This dictionary is immutable and cannot be modified.")

    def __delitem__(self, key):
        raise TypeError("This dictionary is immutable and cannot be modified.")

    def clear(self):
        raise TypeError("This dictionary is immutable and cannot be modified.")

    def pop(self, key, default=None):
        raise TypeError("This dictionary is immutable and cannot be modified.")

    def popitem(self):
        raise TypeError("This dictionary is immutable and cannot be modified.")

    def setdefault(self, key, default=None):
        raise TypeError("This dictionary is immutable and cannot be modified.")

    def update(self, *args, **kwargs):
        raise TypeError("This dictionary is immutable and cannot be modified.")

    def __repr__(self):
        return f"{self.__class__.__name__}({super().__repr__()})"
