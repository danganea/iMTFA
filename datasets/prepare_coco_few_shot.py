import argparse
import json
import os
import random


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 10],
                        help="Range of seeds")
    parser.add_argument('--dataset', type=str, default='coco', choices = ['coco'])
    parser.add_argument('--save-dir', type=str, default='')
    args = parser.parse_args()
    return args


def generate_seeds(args):
    if args.dataset == 'coco':
        data_path = 'datasets/cocosplit/datasplit/trainvalno5k.json'
    else:
        raise ValueError

    dataset_root = f'datasets/{args.dataset}split'
    json_dataset = json.load(open(data_path))

    new_all_cats = []
    for cat in json_dataset['categories']:
        new_all_cats.append(cat)

    id2img = {}
    for i in json_dataset['images']:
        id2img[i['id']] = i

    # Store a dictionary of type dict[class_id] -> List[annotations].
    # This way we can get all the annotations for one particular class
    classid_to_annolist_dict = {i: [] for i in ID2CLASS.keys()}
    for anno in json_dataset['annotations']:
        if anno['iscrowd'] == 1:
            continue
        classid_to_annolist_dict[anno['category_id']].append(anno)

    for seed_index in range(args.seeds[0], args.seeds[1]):  # Generate splits for multiple seeds
        random.seed(seed_index)
        for curr_cls_id in ID2CLASS.keys():  # Loop over all possible category ids
            imgid_to_annolist_dict = {}
            for anno in classid_to_annolist_dict[curr_cls_id]:  # For all annotations that have that category_id, add the image_id to imgid_to_annolist_dict
                if anno['image_id'] in imgid_to_annolist_dict:
                    imgid_to_annolist_dict[anno['image_id']].append(anno)
                else:
                    imgid_to_annolist_dict[anno['image_id']] = [anno]
            # At this point imgid_to_annolist_dict contains all image ids that have annotations with our class 'curr_cls_id' as keys,
            # and as values it contains all the annotations themselves.
            sample_shots = []
            sample_imgs = []
            # for shots in [1, 2, 3, 5, 10, 30]:  # Generate splits for multiple number of shots
            for shots in [1, 5, 10]:  # Generate splits for multiple number of shots
                while True:  # Loop till we find all the shots we need.
                    # Sample 'shots' number of imgid_to_annolist_dict into 'imgs'
                    imgs = random.sample(list(imgid_to_annolist_dict.keys()), shots)

                    # Go through all the sampled images, with the goal being to add it + the sample shots into our list.
                    for img in imgs:
                        skip = False
                        # We might have added the same shots previously, if we sampled the same imgid_to_annolist_dict in the
                        # 'while True' loop. So if our image is in our sampled_shots already, skip it.
                        for s in sample_shots:
                            if img == s['image_id']:
                                skip = True
                                break
                        if skip:
                            continue
                        # We have a count of 'sample_shots' shots previously, check that adding the shots in imgid_to_annolist_dict
                        # does not go overboard on the number of shots.
                        if len(imgid_to_annolist_dict[img]) + len(sample_shots) > shots:
                            continue
                        # Add the shots to our sample_shots array and the image to our sample_imgs array.
                        sample_shots.extend(imgid_to_annolist_dict[img])
                        sample_imgs.append(id2img[img])
                        # Finish if we have an equal amount of shots to the one we need.
                        if len(sample_shots) == shots:
                            break
                    # End 'while True' loop when we've gathered enough shots.
                    if len(sample_shots) == shots:
                        break
                # We can now add the full JSON file for this category_id and this number of shots.
                new_data = {
                    'info': json_dataset['info'],  # Info remains the same as main JSON.
                    'licenses': json_dataset['licenses'],  # Licenses remain the same as main JSON.
                    'images': sample_imgs,
                    # Images now are just the sample_imgs which contain the sample_shots.
                    'annotations': sample_shots,
                    # Annotations just contain the sample shots for this category_id
                }

                save_path = get_save_path_seeds(dataset_root, args.save_dir, ID2CLASS[curr_cls_id], shots, seed_index)
                new_data['categories'] = new_all_cats
                # print(save_path)
                with open(save_path, 'w') as f:
                    json.dump(new_data, f)


def get_save_path_seeds(dataset_root, save_dir, cls, shots, seed):
    prefix = 'full_box_{}shot_{}_trainval'.format(shots, cls)
    save_dir = os.path.join(dataset_root, save_dir, 'seed' + str(seed))
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, prefix + '.json')
    return save_path


if __name__ == '__main__':
    args = parse_args()
    if args.dataset == 'coco':
        ID2CLASS = {
            1: "person",
            2: "bicycle",
            3: "car",
            4: "motorcycle",
            5: "airplane",
            6: "bus",
            7: "train",
            8: "truck",
            9: "boat",
            10: "traffic light",
            11: "fire hydrant",
            13: "stop sign",
            14: "parking meter",
            15: "bench",
            16: "bird",
            17: "cat",
            18: "dog",
            19: "horse",
            20: "sheep",
            21: "cow",
            22: "elephant",
            23: "bear",
            24: "zebra",
            25: "giraffe",
            27: "backpack",
            28: "umbrella",
            31: "handbag",
            32: "tie",
            33: "suitcase",
            34: "frisbee",
            35: "skis",
            36: "snowboard",
            37: "sports ball",
            38: "kite",
            39: "baseball bat",
            40: "baseball glove",
            41: "skateboard",
            42: "surfboard",
            43: "tennis racket",
            44: "bottle",
            46: "wine glass",
            47: "cup",
            48: "fork",
            49: "knife",
            50: "spoon",
            51: "bowl",
            52: "banana",
            53: "apple",
            54: "sandwich",
            55: "orange",
            56: "broccoli",
            57: "carrot",
            58: "hot dog",
            59: "pizza",
            60: "donut",
            61: "cake",
            62: "chair",
            63: "couch",
            64: "potted plant",
            65: "bed",
            67: "dining table",
            70: "toilet",
            72: "tv",
            73: "laptop",
            74: "mouse",
            75: "remote",
            76: "keyboard",
            77: "cell phone",
            78: "microwave",
            79: "oven",
            80: "toaster",
            81: "sink",
            82: "refrigerator",
            84: "book",
            85: "clock",
            86: "vase",
            87: "scissors",
            88: "teddy bear",
            89: "hair drier",
            90: "toothbrush",
        }
    else:
        raise ValueError

    CLASS2ID = {v: k for k, v in ID2CLASS.items()}

    generate_seeds(args)
