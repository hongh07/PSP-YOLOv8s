from ultralytics import YOLO
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

if __name__ == '__main__':
    # Predict
    model =YOLO(r'runs/best.pt')
    model.predict(source=r'datasets/images/test', save=True,save_txt=True,name='testresults',project='runs/predict')