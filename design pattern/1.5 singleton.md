# singleton

Singleton pattern ensures that a class has only **one** instance and a global access point is provided. This is an antipattern sometimes though.

```python
# this is a thread-safe implementation
from threading import Lock

class Singleton():
    _instance = None
    _lock = Lock()

    def __new__(cls):
        # this could work, but acquiring a lock is expensive, so do it only when necessary
        #
        # with cls._lock:
        #     if _instance is None:
        #         cls._instance = super().new(cls)

        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

def main():
    assert Singleton() == Singleton()
```
