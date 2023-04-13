# prototype

If an object offers a cloning method, it is a **Prototype**. Note the difference between shallow copy and deep copy.

```python
import copy

class Prototype():
    def __init__(self, nested_anything):
        self.nested_anything = nested_anything

    def __copy__(self):
        # create new object and inserts references from original
        nested_anything = copy.copy(self.nested_anything)

        new = self.__class__(nested_anything)
        # update writable attributes
        new.__dict__ = copy.copy(self.__dict__) # or use __dict__.update()

        return new

    def __deepcopy__(self, memo):
        # create new object and recursively insert copies
        nested_anything = copy.deepcopy(self.nested_anything, memo)

        new = self.__class__(nested_anything)
        new.__dict__ = copy.deepcopy(self.__dict__, memo)

        return new

```
