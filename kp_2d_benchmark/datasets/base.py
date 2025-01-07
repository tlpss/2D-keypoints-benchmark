

# base class for all datasets.

class DatasetContainer:
    json_train_path: str
    json_val_path: str
    json_test_path: str

    def download(override: bool = False):
        raise NotImplementedError

    def __repr__(self):
        return f"{self.__class__.__name__}({self.json_train_path})"
