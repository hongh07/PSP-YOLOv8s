from ultralytics import YOLO
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

if __name__ == '__main__':
    # Load a model
    model = YOLO(r'runs/train/exp/weights/best.pt')  # build a new model from YAML
    # Validate the model
    model.val(
        val=True,  # (bool) validate/test during training
        data=r'datasets/garbage.yaml',
        split='test',  # (str) dataset split to use for validation, i.e. 'val', 'test' or 'train'
        batch=1,  # (int) number of images per batch (-1 for AutoBatch)
        imgsz=640,  # (int) size of input images as integer or w,h
        device='',  # (int | str | list, optional) device to run on, i.e. cuda device=0 or device=0,1,2,3 or device=cpu
        workers=0,  # (int) number of worker threads for data loading (per RANK if DDP)
        save_json=True,  # (bool) save results to JSON file
        save_hybrid=False,  # (bool) save hybrid version of labels (labels + additional predictions)
        project='runs/val',  # (str, optional) project name
        name='exp',  # (str, optional) experiment name, results saved to 'project/name' directory
        max_det=300,  # (int) maximum number of detections per image
        iou=0.50,
    )