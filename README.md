To set up yolov8 or any version of the yolo your python version should be greater then python3.8 
or 
you can go to https://docs.ultralytics.com/ 

Make sure that your version should be same

step-1 
set up environment
python -m venv .env
.env\Scripts\activate

step 2 
if required then upgrade
python -m pip install --upgrade pip 

pip install ultralytics

# for CPU
pip install torch torchvision

# for GPU
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu113
or
pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio===0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

#to check working of yolo model
yolo predict model=yolov8n.pt source='https://ultralytics.com/images/bus.jpg'


#to detect the image
yolo task=detect mode=predict model=yolov8m.pt conf=0.5 source='img/R.jpeg'


#to check the gpu

python
import torch
torch.cuda.is_available()


if face any kind of problem feel free to contact me 
via email:rishi61802gmail.com
