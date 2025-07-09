# m3gnet/callbacks.py
import os

class ManualStop:
    """
    A simple class to check for a 'STOP' file to halt training.
    In PyTorch, this is checked manually within the training loop.
    """
    def __init__(self, file_path: str = "STOP"):
        self.file_path = file_path

    def should_stop(self) -> bool:
        """Returns True if the stop file exists."""
        if os.path.isfile(self.file_path):
            print(f"Stop file '{self.file_path}' found. Halting training.")
            os.remove(self.file_path) # 移除文件以免影响下次训练
            return True
        return False