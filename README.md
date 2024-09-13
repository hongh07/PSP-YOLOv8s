## I. Project Overview

Unmanned surface vehicles (USVs) are vital for water environment protection, especially in detecting and clearing floating objects. But detecting small ones is challenging due to complex scenes, small sizes, and sunlight reflection. This project presents PSP-YOLOv8s, an improved detection method based on YOLOv8, to enhance accuracy and reliability in detecting small floating objects in complex river environments and contribute to more effective water protection using USVs.

## II. Code Structure

![yolov8改进结构图-改版](C:\Users\Dong\Desktop\小论文\小论文图\成图\yolov8改进结构图-改版.jpg)

## III. Installation and Configuration

To install and configure this project, navigate to the project directory, create a virtual environment if desired, and install the required packages from the `requirements.txt` file using `pip install -r requirements.txt`.

Follow the steps below to install and configure this project:

1. Navigate to the project directory:

   ```bash
   cd your-project-directory
   ```

2. (Optional) Create a virtual environment to isolate the project dependencies:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages by running the following command:

   ```bash
   pip install -r requirements.txt
   ```

The `requirements.txt` file includes the following dependencies:

```
absl-py==2.0.0
addict==2.4.0
aliyun-python-sdk-core==2.14.0
aliyun-python-sdk-kms==2.16.2
apex 
asttokens==2.4.1
backcall==0.2.0
cachetools==5.3.2
certifi==2023.11.17
cffi==1.16.0
charset-normalizer==3.3.2
click==8.1.7
colorama==0.4.6
coloredlogs==15.0.1
contourpy==1.1.1
crcmod==1.7
cryptography==42.0.5
cycler==0.12.1
Cython==3.0.6
decorator==5.1.1
dill==0.3.7
efficientnet-pytorch==0.7.1
einops==0.7.0
executing==2.0.1
filelock==3.13.1
flatbuffers==23.5.26
fonttools==4.45.1
fsspec==2023.12.2
google-auth==2.24.0
google-auth-oauthlib==1.0.0
grad-cam==1.5.0
grpcio==1.59.3
huggingface-hub==0.20.2
humanfriendly==10.0
idna==3.6
importlib-metadata==6.9.0
importlib-resources==6.1.1
ipython==8.12.3
jedi==0.19.1
jmespath==0.10.0
joblib==1.4.0
kiwisolver==1.4.5
loguru==0.7.2
lxml==4.9.3
Markdown==3.5.1
markdown-it-py==3.0.0
MarkupSafe==2.1.3
matplotlib==3.7.4
matplotlib-inline==0.1.6
mdurl==0.1.2
mkl-fft 
mkl-random 
mkl-service==2.4.0
mmcv==2.0.0rc4
mmengine==0.10.3
model-index==0.1.11
mpmath==1.3.0
numpy 
oauthlib==3.2.2
onnx==1.15.0
onnx-simplifier==0.4.10
onnxruntime==1.16.3
opencv-python==4.8.1.78
opendatalab==0.0.10
openxlab==0.0.34
ordered-set==4.1.0
oss2==2.17.0
packaging==23.2
pandas==2.0.3
parso==0.8.3
pickleshare==0.7.5
Pillow==9.5.0
platformdirs==4.2.0
prompt-toolkit==3.0.43
protobuf==4.25.1
psutil==5.9.6
pure-eval==0.2.2
py-cpuinfo==9.0.0
pyasn1==0.5.1
pyasn1-modules==0.3.0
pycocotools==2.0.7
pycocotools-windows==2.0.0.2
pycparser==2.21
pycryptodome==3.20.0
Pygments==2.17.2
pyparsing==3.1.1
pyreadline3==3.4.1
python-dateutil==2.8.2
pytz==2023.3.post1
pywin32==306
PyYAML==6.0.1
regex==2023.12.25
requests==2.28.2
requests-oauthlib==1.3.1
rich==13.4.2
rsa==4.9
safetensors==0.4.1
scikit-learn==1.3.2
scipy==1.10.1
seaborn==0.13.0
six==1.16.0
stack-data==0.6.3
sympy==1.12
tabulate==0.9.0
tensorboard==2.14.0
tensorboard-data-server==0.7.2
termcolor==2.4.0
thop==0.1.1.post2209072238
threadpoolctl==3.4.0
timm==0.6.13
tomli==2.0.1
torch==1.7.1
torchaudio==0.7.2
torchinfo @ git+https://github.com/TylerYep/torchinfo.git@36372c2a0d593bc062d12a49aed7c873b42b027c
torchsummary==1.5.1
torchvision==0.8.2+cu110
tqdm==4.65.2
traitlets==5.14.0
ttach==0.0.3
typing_extensions 
tzdata==2023.3
urllib3==1.26.18
wcwidth==0.2.12
Werkzeug==3.0.1
win32-setctime==1.1.0
yapf==0.40.2
zipp==3.17.0

```

## IV. Usage

This project includes several scripts for different tasks. Below is an overview of how to use each script:

1. **Training the Model**:
   The model can be trained using the `trainv5.py` script. 
2. **Validating the Model**:
   To validate the model after training, use the `valv5.py` script.
3. **Making Predictions**:
   The `predictv5.py` script can be used to make predictions on new data. 

## V. Notes

Before running the project, please review the following notes to ensure smooth execution:

1. **Dependency Installation**
   Before running any scripts, ensure that all dependencies have been properly installed. You can verify the required dependencies by checking the `requirements.txt` file and using the following command to install them:

   ```bash
   pip install -r requirements.txt
   ```

   This will help avoid issues related to missing packages or incompatible versions during runtime.

2. **Dataset Preparation**
   Make sure to place the necessary dataset files in the correct location as specified by the project. For instance, the training, validation, and test datasets should be placed in the paths defined in the project’s configuration files or code. If needed, modify the paths in the configuration files to ensure the code can locate the dataset files.

3. **Environment Setup**
   Ensure that CUDA is installed and configured correctly to allow GPU-accelerated computations.

   