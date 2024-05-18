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
    instance_id = "1651752787923372972"

    coco_pred_path = os.path.join("/Users/kelly/Documents/hsd/detectron2/output/siemens_inference", instance_id, instance_id + "_coco.pkl")
    cityscapes_pred_path = os.path.join("/Users/kelly/Documents/hsd/detectron2/output/siemens_inference", instance_id, instance_id + "_cityscapes.pkl")

    config_file = "/Users/kelly/Documents/hsd/detectron2/projects/PanopticFCN/configs/PanopticFCN-Star-R50-3x.yaml"
    img_path = os.path.join("/Users/kelly/Documents/hsd/data/od_recording_2022_05_05-12_12_11/cam_imgs", instance_id + ".png")
    opts = []

    # Load image
    img = read_image(img_path, format="BGR")

    # Load the saved predictions from model inference
    with open(coco_pred_path, "rb") as f:
        coco_pred = CPU_Unpickler(f).load()

        coco_save_path = os.path.join("/Users/kelly/Documents/hsd/detectron2/output/siemens_inference", instance_id, instance_id + "_coco_pan_seg.pkl")
        with open(coco_save_path, 'wb') as handle:
            coco_pan_seg = coco_pred['panoptic_seg'][0].detach().cpu().numpy()
            coco_seg_info = coco_pred['panoptic_seg'][1]
            pickle.dump({'pan_seg': coco_pan_seg, 'seg_info': coco_seg_info}, handle)
    
    with open(cityscapes_pred_path, "rb") as f:
        cityscapes_pred = CPU_Unpickler(f).load()

        cityscapes_save_path = os.path.join("/Users/kelly/Documents/hsd/detectron2/output/siemens_inference", instance_id, instance_id + "_cityscapes_pan_seg.pkl")
        with open(cityscapes_save_path, 'wb') as handle:
            cityscapes_pan_seg = cityscapes_pred['panoptic_seg'][0].detach().cpu().numpy()
            cityscapes_seg_info = cityscapes_pred['panoptic_seg'][1]
            pickle.dump({'pan_seg': cityscapes_pan_seg, 'seg_info': cityscapes_seg_info}, handle)

    # Set-up config
    cfg = setup_cfg(config_file, opts)

    # Load metadata from cfg
    metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused")

    # Manually add pole to COCO dataset things
    metadata.stuff_classes.append("pole")
    metadata.stuff_colors.append([150, 150, 150]) # grey
    metadata.stuff_dataset_id_to_contiguous_id[200] = 53

    # Retrieve the panoptic segmentation from Cityscapes and COCO model
    cityscapes_pan_seg, cityscapes_seg_info = cityscapes_pred["panoptic_seg"]
    coco_pan_seg, coco_seg_info = coco_pred["panoptic_seg"]

    # Iterate through each category and find the one corresponding to poles (category_id = 5)
    for info in cityscapes_seg_info:
        if info["category_id"] == 5:
            id = info["id"]
            area = info["area"]

    # Obtain a mask of the poles
    mask = cityscapes_pan_seg == id

    # Assign pole seg results from Cityscapes model to the panoptic seg of COCO model
    coco_pan_seg[mask] = coco_pan_seg.max() + 1

    # Add new pole label to 
    pole_seg_info = {"id": len(coco_seg_info) + 1, "isthing": False, "category_id": 54, "area": area}
    coco_seg_info.append(pole_seg_info)

    # Visualize the segmentation and image
    visualizer = Visualizer(img, metadata)
    vis_output = visualizer.draw_panoptic_seg_predictions(coco_pan_seg, coco_seg_info)
    
    # window_name = "Fused Pole Detections"
    # cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    # cv2.imshow(window_name, vis_output.get_image()[:, :, ::-1])
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Save output
    out_path = os.path.join("/Users/kelly/Documents/hsd/detectron2/output/siemens_inference", instance_id, instance_id + "_fused.png")
    vis_output.save(out_path)

    save_path = os.path.join("/Users/kelly/Documents/hsd/detectron2/output/siemens_inference", instance_id, instance_id + "_fused_pan_seg.pkl")
    with open(save_path, 'wb') as handle:
        pickle.dump({'pan_seg': coco_pan_seg.detach().cpu().numpy(), 'seg_info': coco_seg_info}, handle)
