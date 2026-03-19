# model training 
    1. pip install pandas numpy scikit-learn keras tensorflow
    2. python model_training.py
    3. the model training should take less than 5 seconds even without GPUs, best_model.h5 can be seen

# real-time inference
    1. [cam, laptop] in the same environment, python inference_real_time.py

# raspberry 3B+ with IMX219
    1. Enable camera module. sudo raspi-config. 
    - Go to Interface Options > Camera > Enable. 
    - Go to Finish and reboot the raspberry Pi (you may need re-ssh into Pi)
    2. 
    - (no)install python3-pip python3-opencv python3-numpy -y
    - pip3 install tensorflow pandas numpy scikit-learn keras
    - ?mediapipe
    - ?pip3 install picamera2