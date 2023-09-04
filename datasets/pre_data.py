import os
import random

# Mapping of class IDs to class names
class_names = {
    "02691156": "airplane",
    "02747177": "trash bin",
    "02773838": "bag",
    "02801938": "basket",
    "02808440": "bathtub",
    "02818832": "bed",
    "02828884": "bench",
    "02843684": "birdhouse",
    "02871439": "bookshelf",
    "02876657": "bottle",
    "02880940": "bowl",
    "02924116": "bus",
    "02933112": "cabinet",
    "02942699": "camera",
    "02946921": "can",
    "02954340": "cap",
    "02958343": "car",
    "02992529": "cellphone",
    "03001627": "chair",
    "03046257": "clock",
    "03085013": "keyboard",
    "03207941": "dishwasher",
    "03211117": "display",
    "03261776": "earphone",
    "03325088": "faucet",
    "03337140": "file cabinet",
    "03467517": "guitar",
    "03513137": "helmet",
    "03593526": "jar",
    "03624134": "knife",
    "03636649": "lamp",
    "03642806": "laptop",
    "03691459": "loudspeaker",
    "03710193": "mailbox",
    "03759954": "microphone",
    "03761084": "microwaves",
    "03790512": "motorbike",
    "03797390": "mug",
    "03928116": "piano",
    "03938244": "pillow",
    "03948459": "pistol",
    "03991062": "flowerpot",
    "04004475": "printer",
    "04074963": "remote",
    "04090263": "rifle",
    "04099429": "rocket",
    "04225987": "skateboard",
    "04256520": "sofa",
    "04330267": "stove",
    "04379243": "table",
    "04401088": "telephone",
    "04460130": "tower",
    "04468005": "train",
    "04530566": "watercraft",
    "04554684": "washer"
}

# Reverse mapping of class names to class IDs
class_ids = {v: i for i, v in enumerate(class_names.values())}

# Save the class mapping to class_mapping.txt
with open("datasets/shapenet/class_mapping.txt", "w") as f:
    for class_name, class_label in class_ids.items():
        f.write(f"{class_label}: {class_name}\n")

# Path to the folder containing the shuffled files
folder_path = "/home/uceeuam/graduation_project/datasets/shapenet/ShapeNet55/shapenet_pc"

# Initialize dictionaries for train, val, and test filenames
train_filenames = {}
val_filenames = {}
test_filenames = {}

# Loop over the files in the folder
for filename in os.listdir(folder_path):
    class_id = filename[:8]  # Extract the class ID from the filename
    class_name = class_names.get(class_id)  # Get the corresponding class name

    # Skip the file if the class ID is not in the class_names dictionary
    if class_name is None:
        continue

    # Map the class name to the class ID in the range 0-54
    class_label = class_ids[class_name]

    # Randomly assign the file to train, val, or test set
    rand = random.random()
    if rand < 0.8:
        filenames_dict = train_filenames
    elif rand < 0.9:
        filenames_dict = val_filenames
    else:
        filenames_dict = test_filenames

    # Append the filename and class label to the corresponding list in the dictionary
    filenames_dict.setdefault(class_label, []).append(filename)

# Save the filenames to train.txt, val.txt, and test.txt
with open("datasets/shapenet/train.txt", "w") as f:
    for class_label, filenames in train_filenames.items():
        for filename in filenames:
            f.write(f"{class_label}/{filename}\n")

with open("datasets/shapenet/val.txt", "w") as f:
    for class_label, filenames in val_filenames.items():
        for filename in filenames:
            f.write(f"{class_label}/{filename}\n")

with open("datasets/shapenet/test.txt", "w") as f:
    for class_label, filenames in test_filenames.items():
        for filename in filenames:
            f.write(f"{class_label}/{filename}\n")