import time

from collections import defaultdict

class Timer:
    """A Timer class registering time spent on each task
    Timer class is defined as a singleton to guarantee every time the user calls a new timing task,
    ther results are written to a defaultdict
    You can probably achieve the same goal with class variables, which might or might not be cleaner..."""
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, category: str, enabled: bool=True) -> None:
        super().__init__()
        self.enabled = enabled
        self._category = category
        if not hasattr(self, "_category_sec_avg"):
            self._category_sec_avg = defaultdict(lambda: [0., 0., 0])   # A bucket of [total_secs, latest_start, num_calls]

    def __enter__(self) -> object:
        """The returned value is bounded to the variable after the 'as' keyward in the with statement."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end()
        return
    
    def start(self):
        if self.enabled:
            stat = self._category_sec_avg[self._category]
            if stat is None:
                raise ValueError(f"The timer's category {self._category} is not recognized!")
            stat[1] = time.perf_counter()
            stat[2] += 1

    def end(self):
        if self.enabled:
            stat = self._category_sec_avg[self._category]
            if stat is None:
                raise ValueError(f"The timer's category {self._category} is not recognized!")
            stat[0] += time.perf_counter() - stat[1]

    def print_stat(self):
        if self.enabled:
            print("Printing timer stats...")
            for key, val in self._category_sec_avg.items():
                if val[2]>0:
                    print(f":> category {key}, total {val[0]:.1f}s, num {val[2]}, avg {val[0] / val[2]:.2E}s")

    def reset_stat(self) -> None:
        if self.enabled:
            print("Resetting timer stats...")
            for val in self._category_sec_avg.values():
                val[0], val[1], val[2] = 0, 0, 0