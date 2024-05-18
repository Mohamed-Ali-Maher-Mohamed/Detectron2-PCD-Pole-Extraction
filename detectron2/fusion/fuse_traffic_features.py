import io
import os
import cv2
import torch
import pickle

from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.data.detection_utils import read_image

from detectron2.projects.panopticfcn import add_panopticfcn_config
from densepose.config import add_densepose_head_config


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)


def setup_cfg(config_file, opts):
    # load config from file and command-line arguments
    cfg = get_cfg()
    # To use demo for Panoptic-DeepLab, please uncomment the following two lines.
    # from detectron2.projects.panoptic_deeplab import add_panoptic_deeplab_config  # noqa
    # add_panoptic_deeplab_config(cfg)

    add_panopticfcn_config(cfg)
    add_densepose_head_config(cfg)

    cfg.merge_from_file(config_file)
    cfg.merge_from_list(opts)
    
    cfg.freeze()
    return cfg


if __name__ == "__main__":
    instance_id = '1651752750486380409'

    coco_pred_path = os.path.join("/Users/Mohamed/Documents/hsd/detectron2/output/siemens_inference", instance_id, instance_id + "_coco.pkl")
    cityscapes_pred_path = os.path.join("/Users/Mohamed/Documents/hsd/detectron2/output/siemens_inference", instance_id, instance_id + "_cityscapes.pkl")

    config_file = "/Users/Mohamed/Documents/hsd/detectron2/projects/PanopticFCN/configs/PanopticFCN-Star-R50-3x.yaml"
    img_path = os.path.join("/Users/Mohamed/Documents/hsd/detectron2/datasets/siemens", instance_id + ".png")
    opts = []

    # Load image
    img = read_image(img_path, format="BGR")

    # Load the saved predictions from model inference
    with open(coco_pred_path, "rb") as   f:
        coco_pred = CPU_Unpickler(f).load()
    
    with open(cityscapes_pred_path, "rb") as f:
        cityscapes_pred = CPU_Unpickler(f).load()

    # Set-up config
    cfg = setup_cfg(config_file, opts)

    # Load metadata from cfg
    metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused")

    # Manually add pole and traffic sigjto COCO dataset stuff
    metadata.stuff_classes.append("pole")
    metadata.stuff_colors.append([150, 150, 150]) # grey
    metadata.stuff_dataset_id_to_contiguous_id[200] = 53

    metadata.stuff_classes.append("traffic sign")
    metadata.stuff_colors.append([220, 220, 0]) # bright yellow
    metadata.stuff_dataset_id_to_contiguous_id[201] = 54

    metadata.stuff_classes.append("traffic light")
    metadata.stuff_colors.append([250, 170, 30]) # orange
    metadata.stuff_dataset_id_to_contiguous_id[202] = 55

    # Retrieve the panoptic segmentation from Cityscapes and COCO model
    cityscapes_pan_seg, cityscapes_seg_info = cityscapes_pred["panoptic_seg"]
    coco_pan_seg, coco_seg_info = coco_pred["panoptic_seg"]

    total_coco_seg = len(coco_seg_info)

    # Iterate through each category and find the one corresponding to desired traffic features (poles, traffic lights, traffic signs)
    for info in cityscapes_seg_info:
        if info["category_id"] == 5:
            pole_id = info["id"]
            pole_area = info["area"]
        elif info["category_id"] == 6:
            light_id = info["id"]
            light_area = info["area"]
        elif info["category_id"] == 7:
            sign_id = info["id"]
            sign_area = info["area"]

    # Obtain a mask of features
    pole_mask = cityscapes_pan_seg == pole_id
    light_mask = cityscapes_pan_seg == light_id
    sign_mask = cityscapes_pan_seg == sign_id

    # Assign pole seg results from Cityscapes model to the panoptic seg of COCO model and add label to seg info
    coco_pan_seg[pole_mask] = coco_pan_seg.max() + 1
    pole_seg_info = {"id": total_coco_seg + 1, "isthing": False, "category_id": 54, "area": pole_area}
    coco_seg_info.append(pole_seg_info)

    # Repeat for traffic sign
    coco_pan_seg[sign_mask] = coco_pan_seg.max() + 1
    sign_seg_info = {"id": total_coco_seg + 2, "isthing": False, "category_id": 55, "area": sign_area}
    coco_seg_info.append(sign_seg_info)

    # Get existing traffic light labels from COCO
    idx_to_pop = []
    for i in range(total_coco_seg):
        info = coco_seg_info[i]
        if info['category_id'] == 9:
            idx_to_pop.append(i)
    
    # Reverse indices to pop to prevent errors with list getting shorter after each pop
    idx_to_pop.reverse()
    for i in idx_to_pop:
        coco_seg_info.pop(i+1)
        coco_pan_seg[coco_pan_seg == i+1] = 0

    # Repeat above for traffic lights
    coco_pan_seg[light_mask] = coco_pan_seg.max() + 1
    light_seg_info = {"id": total_coco_seg + 3, "isthing": False, "category_id": 56, "area": light_area}
    coco_seg_info.append(light_seg_info)
    
    # Visualize the segmentation and image
    visualizer = Visualizer(img, metadata)
    vis_output = visualizer.draw_panoptic_seg_predictions(coco_pan_seg, coco_seg_info)
    
    window_name = "Fused Pole Detections"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, vis_output.get_image()[:, :, ::-1])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save output
    out_path = os.path.join("/Users/Mohamed/Documents/hsd/detectron2/output/siemens_inference", instance_id, instance_id + "_fused.png")
    vis_output.save(out_path)

