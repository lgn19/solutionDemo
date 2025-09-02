import numpy as np
import scipy.io as sio
import scipy.signal as signal
import matplotlib.pyplot as plt
from scipy.io.wavfile import read
from scipy.signal import resample
from gennerate_templete import generate_template
from Online_DPRTF import online_ssl_dprtf_eg
from compute_TDOA import compute_TDOA,compute_DOA
import soundfile as sf
import time
import os
import json
from scipy.signal import find_peaks

def get_degree(vector,line):
    vector = np.expand_dims(vector, axis=0)
    dot_products = np.einsum('ij,j->i', vector, line)
    length_ab = np.linalg.norm(line)
    length_vectors = np.linalg.norm(vector, axis=1)
    cos_theta = dot_products / (length_ab * length_vectors)
    angles = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # 防止浮点误差导致超出[-1, 1]
    angles_deg = np.degrees(angles)
    return angles_deg

# Sampling rate and frame length
fs = 16000
ftLen = 256  # Frame length (16 ms)

# Microphone pairs for localization (matrix MP)
MP = np.array([[1, 1, 1],
              [2, 3, 4]])
micPosition = np.array( [
    [0, 0, 0],[0.0667,0,0],[0.0667*2,0,0],[0.2,0,0]
        ])

TDOA = compute_DOA(micPosition, np.arange(181), MP)
freRan = np.arange(30, 50)
rtfTemp = generate_template(TDOA, freRan)
Hight=0.05
Distance=15

# 设置主文件夹路径
main_folder_path =  os.path.dirname((os.path.abspath(__file__)))+"/genertated"

# 获取所有子文件夹路径
subfolders = [os.path.join(main_folder_path, d) for d in os.listdir(main_folder_path) if os.path.isdir(os.path.join(main_folder_path, d))]
# 生成数组
string_array = [f"{i:05}" for i in range(0, 1)]

X=[]
Y=[]
DPRTF=[]
Class=[]
i=0
for subfolder in subfolders:
    print(subfolder,i)
    for index in string_array:
        tem_dprtf=[]
        file_name1_1=f"{subfolder}/mic00_mixed.wav"
        file_name1_2=f"{subfolder}/mic01_mixed.wav"
        file_name1_3=f"{subfolder}/mic02_mixed.wav"
        file_name1_4=f"{subfolder}/mic03_mixed.wav"
        file_name2_1=f"{subfolder}/mic04_mixed.wav"
        file_name2_2=f"{subfolder}/mic05_mixed.wav"
        file_name2_3=f"{subfolder}/mic06_mixed.wav"
        file_name2_4=f"{subfolder}/mic07_mixed.wav"
        file_name3_1=f"{subfolder}/mic08_mixed.wav"
        file_name3_2=f"{subfolder}/mic09_mixed.wav"
        file_name3_3=f"{subfolder}/mic10_mixed.wav"
        file_name3_4=f"{subfolder}/mic11_mixed.wav"
        file_name4_1=f"{subfolder}/mic12_mixed.wav"
        file_name4_2=f"{subfolder}/mic13_mixed.wav"
        file_name4_3=f"{subfolder}/mic14_mixed.wav"
        file_name4_4=f"{subfolder}/mic15_mixed.wav"
        jason_file_name=f"{subfolder}/metadata.json"
        with open(jason_file_name, 'r') as json_file:

            json_data = json.load(json_file)
            position = json_data['voice00']['position']

            pos1 = np.array([1.9, 0.6, 1])
            pos2 = np.array([2.5, 0.6, 1])
            A1 =pos1
            B1 =pos2
            AB1 = A1 - B1
            vectors1 = position - A1
            degree1 = get_degree(vectors1, AB1)

            Y.append(position)

            # 提取G1特征
            y1, rfs = sf.read(file_name1_1)
            y1 = np.array(y1)
            y2, rfs2 = sf.read(file_name1_2)
            y2 = np.array(y2)
            y3, rf3 = sf.read(file_name1_3)
            y3 = np.array(y3)
            y4, rf4 = sf.read(file_name1_4)
            y4 = np.array(y4)
            micNum = 4
            y = np.column_stack((y1, y2, y3, y4))
            x = np.zeros((48000, 4))
            for mic in range(micNum):
                x[:, mic] = resample(y[:, mic], int(len(y) * fs / rfs))
            GMMWeight, Peaks = online_ssl_dprtf_eg(x, rtfTemp, freRan, MP)
            sp=np.mean(GMMWeight[:,-250:],axis=1).reshape(-1)

            tem_dprtf.append(sp)

            p1=np.argmax(sp)



            # 提取G2特征
            y1, rfs = sf.read(file_name2_1)
            y1 = np.array(y1)
            y2, rfs2 = sf.read(file_name2_2)
            y2 = np.array(y2)
            y3, rf3 = sf.read(file_name2_3)
            y3 = np.array(y3)
            y4, rf4 = sf.read(file_name2_4)
            y4 = np.array(y4)
            micNum = 4
            y = np.column_stack((y1, y2, y3, y4))
            x = np.zeros((48000, 4))

            for mic in range(micNum):
                x[:, mic] = resample(y[:, mic], int(len(y) * fs / rfs))
            GMMWeight, Peaks = online_ssl_dprtf_eg(x, rtfTemp, freRan, MP)
            sp=np.mean(GMMWeight[:,-250:],axis=1).reshape(-1)
            tem_dprtf.append(sp)

            p2=np.argmax(sp)



            # 提取G3特征
            y1, rfs = sf.read(file_name3_1)
            y1 = np.array(y1)
            y2, rfs2 = sf.read(file_name3_2)
            y2 = np.array(y2)
            y3, rf3 = sf.read(file_name3_3)
            y3 = np.array(y3)
            y4, rf4 = sf.read(file_name3_4)
            y4 = np.array(y4)
            micNum = 4
            y = np.column_stack((y1, y2, y3, y4))
            x = np.zeros((48000, 4))
            for mic in range(micNum):
                x[:, mic] = resample(y[:, mic], int(len(y) * fs / rfs))
            GMMWeight, Peaks = online_ssl_dprtf_eg(x, rtfTemp, freRan, MP)
            sp=np.mean(GMMWeight[:,-250:],axis=1).reshape(-1)
            # sp=np.mean(Peaks[:,-250:],axis=1).reshape(-1)
            tem_dprtf.append(sp)

            p3=np.argmax(sp)

            # 提取G4特征
            y1, rfs = sf.read(file_name4_1)
            y1 = np.array(y1)
            y2, rfs2 = sf.read(file_name4_2)
            y2 = np.array(y2)
            y3, rf3 = sf.read(file_name4_3)
            y3 = np.array(y3)
            y4, rf4 = sf.read(file_name4_4)
            y4 = np.array(y4)
            micNum = 4
            y = np.column_stack((y1, y2, y3, y4))
            x = np.zeros((48000, 4))

            for mic in range(micNum):
                x[:, mic] = resample(y[:, mic], int(len(y) * fs / rfs))
            GMMWeight, Peaks = online_ssl_dprtf_eg(x, rtfTemp, freRan, MP)
            sp=np.mean(GMMWeight[:,-250:],axis=1).reshape(-1)
            tem_dprtf.append(sp)
            DPRTF.append(tem_dprtf)
            p4=np.argmax(sp)

            X.append([p1,p2,p3,p4])


DPRTF=np.array(DPRTF)
X=np.array(X)
Y=np.array(Y)
Class=np.array(Class)

np.save("sample/X.npy", X)
np.save("sample/Y.npy", Y)


