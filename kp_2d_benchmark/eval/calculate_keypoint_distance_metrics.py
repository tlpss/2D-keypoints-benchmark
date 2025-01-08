from kp_2d_benchmark.eval.coco_results import COCOKeypointResults, CocoKeypointsDataset


def calculate_keypoint_distances(coco_dataset: CocoKeypointsDataset, coco_results: COCOKeypointResults):

    # create dict with categories and keypoint ids

    distance_dict = {}
    for category in coco_dataset.categories:
        keypoint_dict = {}
        for i, keypoint in enumerate(category.keypoints):
            keypoint_dict[keypoint] = []
        distance_dict[category.id] = keypoint_dict

    category_id_to_category = {category.id: category for category in coco_dataset.categories}

    # first check if for each image, there is max one annotation
    image_id_to_annotations = {}
    for annotation in coco_dataset.annotations:
        if annotation.image_id in image_id_to_annotations:
            raise ValueError(f"Image {annotation.image_id} has multiple annotations.")
        image_id_to_annotations[annotation.image_id] = annotation

    image_id_to_predictions = {}
    for prediction in coco_results:
        if prediction.image_id in image_id_to_predictions:
            raise ValueError(f"Image {prediction.image_id} has multiple predictions.")
        image_id_to_predictions[prediction.image_id] = prediction

    # then for each annotation, find the prediction

    for image_id, annotation in image_id_to_annotations.items():
        if image_id not in image_id_to_predictions:
            # TODO: take image center?
            # for now pass
            continue
        prediction = image_id_to_predictions[image_id]

        predicted_keypoints = prediction.keypoints
        predicted_keypoints = [predicted_keypoints[i : i + 3] for i in range(0, len(predicted_keypoints), 3)]
        annotated_keypoints = annotation.keypoints
        annotated_keypoints = [annotated_keypoints[i : i + 3] for i in range(0, len(annotated_keypoints), 3)]

        category_id = annotation.category_id
        for i, name in enumerate(category_id_to_category[category_id].keypoints):
            # calculate the distance between the predicted and annotated keypoint
            # add the distance to the list
            predicted_keypoint = predicted_keypoints[i]
            annotated_keypoint = annotated_keypoints[i]

            distance = (
                (predicted_keypoint[0] - annotated_keypoint[0]) ** 2
                + (predicted_keypoint[1] - annotated_keypoint[1]) ** 2
            ) ** 0.5
            distance_dict[category_id][name].append(distance)
        # calculate the distance for each keypoint, and add them to the list

    # if there is no prediction, predict all keypoints to the center of the image.

    # calculate the distance for each keypoint, and add them to the list

    # if there are remaining predictions, for which there was no annotation,
    # TODO: what to to with these? cannot take FP into account in distance metric?.
    # for now, just ignore them.

    # if there are categories for which we have not a single prediction, we should also ignore these categories
    new_distance_dict = {}
    for category_id, keypoint_dict in distance_dict.items():
        if all(len(distances) == 0 for distances in keypoint_dict.values()):
            continue
        new_distance_dict[category_id] = keypoint_dict
    return new_distance_dict


def calculate_average_distances(distance_dict):
    # calculate the average distance for each keypoint
    average_distance_dict = {}
    for category_id, keypoint_dict in distance_dict.items():
        average_distance_dict[category_id] = {}
        for keypoint_id, distances in keypoint_dict.items():
            average_distance_dict[category_id][keypoint_id] = sum(distances) / len(distances)
    return average_distance_dict


def calculate_median_distances(distance_dict):
    # calculate the average distance for each keypoint
    median_distance_dict = {}
    for category_id, keypoint_dict in distance_dict.items():
        median_distance_dict[category_id] = {}
        for keypoint_id, distances in keypoint_dict.items():
            median_distance_dict[category_id][keypoint_id] = sorted(distances)[len(distances) // 2]
    return median_distance_dict


def calculate_std_deviation(distance_dict):
    # calculate the average distance for each keypoint
    std_deviation_dict = {}
    for category_id, keypoint_dict in distance_dict.items():
        std_deviation_dict[category_id] = {}
        for keypoint_id, distances in keypoint_dict.items():
            mean = sum(distances) / len(distances)
            std_deviation_dict[category_id][keypoint_id] = (
                sum([(distance - mean) ** 2 for distance in distances]) / len(distances)
            ) ** 0.5
    return std_deviation_dict


if __name__ == "__main__":
    dataset_path = "/home/tlips/Code/2D-keypoints-benchmark/test/data/dummy_keypoints.json"
    coco_results_path = "/home/tlips/Code/2D-keypoints-benchmark/test/data/dummy_keypoint_results.json"
    import json

    with open(dataset_path, "r") as f:
        coco_dataset = CocoKeypointsDataset.parse_obj(json.load(f))
    with open(coco_results_path, "r") as f:
        coco_results = COCOKeypointResults.parse_obj(json.load(f))
    distance_dict = calculate_keypoint_distances(coco_dataset, coco_results)
    average_distance_dict = calculate_average_distances(distance_dict)
    median_distance_dict = calculate_median_distances(distance_dict)
    std_deviation_dict = calculate_std_deviation(distance_dict)
    print(average_distance_dict)
    # print(median_distance_dict)
    # print(std_deviation_dict)
