import json

from kp_2d_benchmark.eval.coco_results import CocoKeypointsDataset


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
        raise ValueError(f"Category {self.category_name} not found in dataset")

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

    def get_split(self, split="train"):
        if split == "train":
            json_path = self.json_train_path
        elif split == "val":
            json_path = self.json_val_path
        elif split == "test":
            json_path = self.json_test_path
        else:
            raise ValueError(f"split must be one of ['train', 'val', 'test'], got {split}")

        return CocoKeypointsDataset(**json.load(open(json_path)))

    def __repr__(self):
        return f"{self.__class__.__name__}"
