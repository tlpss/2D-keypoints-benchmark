

import json


# base class for all datasets.
class DatasetContainer:
    json_train_path: str
    json_val_path: str
    json_test_path: str

    category_name: str = None

    def download(override: bool = False):
        raise NotImplementedError
    
    @property 
    def num_keypoints(self):
        with open(self.json_train_path) as f:
            data = json.load(f)
            for category in data["categories"]:
                if category["name"] == self.category_name:
                    return len(category["keypoints"])

    def __repr__(self):
        return f"{self.__class__.__name__}({self.json_train_path})"
