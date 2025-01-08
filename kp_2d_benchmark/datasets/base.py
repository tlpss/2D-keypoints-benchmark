

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

    def size(self, split="train"):
        if split == "train":
            json_path = self.json_train_path
        elif split == "val":
            json_path = self.json_val_path
        elif split == "test":
            json_path = self.json_test_path
        else:
            raise ValueError(f"split must be one of ['train', 'val', 'test'], got {split}")
        
        with open(json_path) as f:
            data = json.load(f)
            return len(data["images"])

    def __repr__(self):
        return f"{self.__class__.__name__}"
