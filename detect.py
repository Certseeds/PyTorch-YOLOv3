from __future__ import division

from pathlib import PurePath

import cv2

from models.models import *
from utils.utils import *
from utils.datasets import *
from utils.transforms import *

import os
import time
import datetime
import argparse

from PIL import Image

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable

# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
from matplotlib.ticker import NullLocator


def builds(opt):
    os.makedirs("output", exist_ok=True)
    opt.output_save_path = increment_path(Path("output") / opt.name, exist_ok=opt.exist_ok | opt.evolve)
    opt.save_dir = Path(opt.output_save_path)
    opt.save_dir.mkdir(parents=True, exist_ok=True)
    opt.label_dir = opt.save_dir / "labels"
    opt.label_dir.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="data/samples", help="path to dataset")
    parser.add_argument("--imglist_file", type=str, help="file that store img_file_list")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")

    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')

    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    builds(opt)

    # Set up model
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)

    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))

    model.eval()  # Set in evaluation mode
    if opt.imglist_file is not None:
        dataset = ImageList(opt.imglist_file, transform= \
            transforms.Compose([DEFAULT_TRANSFORMS, Resize(opt.img_size)]))
    else:
        dataset = ImageFolder(opt.image_folder, transform= \
            transforms.Compose([DEFAULT_TRANSFORMS, Resize(opt.img_size)]))

    dataloader = DataLoader(dataset,
                            batch_size=opt.batch_size,
                            shuffle=False,
                            num_workers=opt.n_cpu,
                            )

    classes = load_classes(opt.class_path)  # Extracts class labels from file

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    imgs = []  # Stores image paths
    img_detections = []  # Stores detections for each image index

    print("\nPerforming object detection:")
    for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
        prev_time = time.time()
        # Configure input
        input_imgs = Variable(input_imgs.type(Tensor))

        # Get detections
        with torch.no_grad():
            detections = model(input_imgs)
            detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)

        # Log progress
        current_time = time.time()
        inference_time = datetime.timedelta(seconds=current_time - prev_time)
        print("\t+ Batch %d, Inference Time: %s" % (batch_i, inference_time))
        print("\t+ picture {}, interface time: {}".format(img_paths, inference_time))
        # Save image and detections
        imgs.extend(img_paths)
        img_detections.extend(detections)

    # Bounding-box colors
    # cmap = plt.get_cmap("tab20b")
    # colors = [cmap(i) for i in np.linspace(0, 1, 20)]
    print("\nSaving images:")
    # Iterate through images and save plot of detections
    for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):

        print("(%d) Image: '%s'" % (img_i, path))

        # Create plot
        img = cv2.imread(path)
        cv2.imshow(path, img)
        filename = PurePath(path).name.split(".")[0]

        # Draw bounding boxes and labels of detections
        if detections is not None:
            # Rescale boxes to original image
            detections = rescale_boxes(detections, opt.img_size, (img.shape[1], img.shape[0]))
            unique_labels = detections[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)
            bbox_colors = random.sample(colors, n_cls_preds)
            for y1, x1, y2, x2, conf, cls_conf, cls_pred in detections:
                print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))
                x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)
                box_w, box_h = x2 - x1, y2 - y1
                color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(img, classes[int(cls_pred)], (x1, y1), cv2.FONT_HERSHEY_COMPLEX_SMALL, 6,
                            (0, 0, 255), 10)
                with open(opt.label_dir / f"{filename}.txt", 'a') as result_txt:
                    x_middle, y_middle = min(1, (x1 + x2) / 2 / img.shape[1]), min(1, (y1 + y2) / 2 / img.shape[0])
                    x_length, y_length = box_w / img.shape[1], box_h / img.shape[0]
                    result_txt.write(f"0 {x_middle} {y_middle} {x_length} {y_length}\n")
                    # result_txt.write(f"0 {x1} {x2} {y1} {y2}")
        else:
            emptyfile = opt.label_dir / f"{filename}.txt"
            emptyfile.touch()
        # Save generated image with detections
        output_path = os.path.join(opt.output_save_path, f"{filename}.jpg")
        cv2.imwrite(output_path, img)
