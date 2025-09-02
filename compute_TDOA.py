import numpy as np

def compute_TDOA(micPos, canPos, MP):
    # Number of microphones, candidate positions, and microphone pairs
    micNum = micPos.shape[0]
    canNum = canPos.shape[0]
    mpNum = MP.shape[1]

    # Initialize TDOA array
    TDOA = np.zeros((mpNum, canNum))

    # Compute pairwise distances between microphones and candidate positions
    dis = np.sqrt(np.sum((np.expand_dims(micPos, 1) - np.expand_dims(canPos, 0))**2, axis=2))

    # Compute Time Difference of Arrival (TDOA) for each microphone pair
    for mp in range(mpNum):
        TDOA[mp, :] = (dis[MP[1, mp]-1, :] - dis[MP[0, mp]-1, :]) / 344

    # Optional: plot the absolute value of TDOA in log scale (equivalent to imagesc in MATLAB)
    # import matplotlib.pyplot as plt
    # plt.imshow(np.log(np.abs(TDOA)), aspect='auto', cmap='jet')
    # plt.colorbar()
    # plt.show()
    print(TDOA.shape)

    return TDOA

def compute_DOA(micPos,canPos,MP):
    micNum = micPos.shape[0]
    canNum = canPos.shape[0]
    mpNum = MP.shape[1]

    # 6*360
    TDOA = np.zeros((mpNum, canNum))
    for i in range(mpNum):
        mic1_index=MP[0][i]-1
        mic2_index=MP[1][i]-1
        pos1=micPos[mic1_index]
        pos2=micPos[mic2_index]
        dis=np.linalg.norm(np.array(pos1) - np.array(pos2))
        for j in range(canNum):
            # print(canPos[j])
            # print(j)
            delay=dis*np.cos(np.radians(canPos[j]))/343
            # delay=dis*np.cos(np.radians(j))/343
            TDOA[i,j]=delay
            # print(j,delay)
    # import matplotlib.pyplot as plt
    # plt.imshow((TDOA), aspect='auto', cmap='jet')
    # plt.colorbar()
    # plt.show()
    return TDOA







