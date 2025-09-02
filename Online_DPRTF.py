import numpy as np
from scipy.signal.windows import hamming
from scipy.fft import fft
from MCMT import MCMT as mcmt
from tqdm import tqdm
import matplotlib.pyplot as plt
from loglik_complexgaussian import loglik_complexgaussian
from scipy.special import softmax

def online_ssl_dprtf_eg(soundData, rtfTemp, freRan=None, MP=None):
    """
    Online SSL using DP-RTF estimation and EG multispeaker localization.

    Parameters:
    soundData (numpy.ndarray): Audio signal, with the size of (No. of samples) x (No. of mics),
                               and with the sampling rate of 16 kHz.
    rtfTemp (numpy.ndarray): RTF template for candidate locations, with the size of
                             (No. of frequencies) x (No. of microphone pairs) x (No. of candidate locations).
    freRan (list or numpy.ndarray): Frequency range used for SSL. It is denoted in frequency bin,
                                    within the range of [2, winLen/2]. Default is [2, winLen/4].
    MP (numpy.ndarray): Microphone pairs used for localization, with the size of 2 x (No. of microphone pairs).
                        Default is all possible pairs.

    Returns:
    GMMWeight (numpy.ndarray): (No. of candidate locations) x (No. of frames).
    Peaks (numpy.ndarray): (No. of candidate locations) x (No. of frames), with 1 for detected active speakers.
    """
    d1, d2 = soundData.shape
    if d1 < d2:
        soundData = soundData.T
    micNum = soundData.shape[1]

    # Microphone pairs
    if MP is None:
        MP = []
        for m1 in range(micNum - 1):
            for m2 in range(m1 + 1, micNum):
                MP.append([m1, m2])
        MP = np.array(MP).T
    mpNum = MP.shape[1]

    # STFT parameters
    fs = 16000
    winLen = 256
    fraInc = winLen // 2
    win = hamming(winLen)

    if freRan is None:
        freRan = np.arange(2, winLen // 4 + 1)  # Frequency range up to 4 kHz
    freNum = len(freRan)

    fraNum = (soundData.shape[0] - winLen) // fraInc  # Number of frames

    # Spectral smoothing and Speech/non-speech classification parameters
    smoLen = int(np.ceil(200 * 16 / fraInc))  # Spectral smoothing history length, 200 ms
    beta = (smoLen - 1) / (smoLen + 1)  # Smoothing factor
    minSran =2  # Window length for minimum searching, in seconds
    mcmtFn = int(np.floor(minSran * fs / fraInc)) - smoLen + 1
    sTh, nTh = mcmt(smoLen, 1, mcmtFn)  # Thresholds for speech and noise frames detection

    sfreNumTh = 0.05 * freNum  # If speech frequency is less than this number, skip

    # RLS DP-RTF estimation parameters
    fiLen = 8  # IMPORTANT! CTF length, this setting is suitable for reverberation time around 0.55 s, it approximately equals T60/8.
    rctfLen = fiLen * micNum  # Length of relative CTF vector
    fraNumLs = int(np.ceil(fiLen / np.sqrt(micNum - 1)))  # Number of frames used for relative CTF estimation
    lambda_ = (fraNumLs - 1) / (fraNumLs + 1)  # Forgetting factor

    ctTh = 0.5  # Threshold for consistency test

    # EG parameters
    candNum = rtfTemp.shape[2]  # Number of candidate locations
    noInfVec = np.ones(candNum) / candNum  # Uniform vector

    eta = 0.07  # EG update rate
    eta1 = 0.065

    gamma = 0.1  # Entropy regularization weight
    spaSmooth = 0.02  # Spatial smoothing factor

    # Peak detection parameters
    psTh = 0.04  # IMPORTANT! peak detection threshold, controls the tradeoff between false alarm and miss detection
    maxPeak = 5  # Maximum number of active speakers

    # Variables in process
    ftHis = np.zeros((freNum, micNum, fiLen), dtype=np.complex128)  # STFT coefficients history
    sft = np.zeros((freNum, micNum, fiLen), dtype=np.complex128)  # Smoothed auto-/cross-spectra
    nearNoiseFra = np.zeros((freNum, micNum, fiLen), dtype=np.complex128)  # The nearest noise frame
    ftPow = np.zeros(freNum)  # Smoothed STFT power

    covInv = np.zeros((freNum, rctfLen, rctfLen), dtype=np.complex128)  # Signal covariance matrix inverse
    for fre in range(freNum):
        result_eye=1e3 * np.eye(rctfLen)
        covInv[fre, :, :] = result_eye

    gmmWeight = noInfVec  # GMM weight vector in process
    GMMWeight = np.zeros((candNum, fraNum))  # GMM weight for all frames

    Peaks = np.zeros((candNum, fraNum))  # Peaks


    for t in range(fraNum):
        # print(t)
        # Signal processing
        xt = soundData[t * fraInc:t * fraInc + winLen, :]
        xft = fft(xt * win[:, None], axis=0)
        xft = xft[freRan, :]

        ftHis[:, :, 1:fiLen] = ftHis[:, :, :fiLen - 1]
        ftHis[:, :, 0] = xft

        sft = beta * sft + (1 - beta) * ftHis * np.conj(xft[:, 0]).reshape(sft.shape[0], 1, 1)
        # print(sft[:,0,0])
        ftPow = beta * ftPow + (1 - beta) * np.sum(np.abs(xft) ** 2, axis=1)
        # print(t,np.mean(ftPow))
        if t < (smoLen-1):
            gmmWeight = (1 - eta1) * gmmWeight + eta1 * noInfVec
            GMMWeight[:, t] = gmmWeight
            continue
        elif t == (smoLen-1):
            # print(t,np.mean(ftPow),"-------------------")
            minPowHis = np.tile(ftPow[:, None], (1, mcmtFn))
            gmmWeight = (1 - eta1) * gmmWeight + eta1 * noInfVec
            GMMWeight[:, t] = gmmWeight
            continue

        minPowHis[:, :mcmtFn - 1] = minPowHis[:, 1:mcmtFn]
        minPowHis[:, mcmtFn - 1] = ftPow
        minPow = np.min(minPowHis, axis=1)


        nearNoiseFra[ftPow <= nTh * minPow, :, :] = sft[ftPow <= nTh * minPow, :, :]
        sftSS = sft - nearNoiseFra

        sInd = ftPow > sTh * minPow
        # print(sTh * minPow)
        # print(t,"--------")
        # print(sInd)

        # sInd=np.roll(sInd, 1)
        sfreNum = np.sum(sInd)
        # print(t,"-------")
        # print(sfreNum,sInd)
        if sfreNum < sfreNumTh:
            gmmWeight = (1 - eta1) * gmmWeight + eta1 * noInfVec
            GMMWeight[:, t] = gmmWeight
            continue

        sftSS = sftSS[sInd, :, :]
        # print(sftSS[:,0,0])
        # print(sftSS.shape)

        # DP-RTF estimation
        covInvt = covInv[sInd, :, :]
        # print(np.mean(covInvt))


        for m1 in range(micNum - 1):
            for m2 in range(m1 + 1, micNum):
                pos1=(m1 * fiLen)
                pos2=(m1 + 1) * fiLen
                pos3=(m2 * fiLen)
                pos4=(m2 + 1) * fiLen
                # print(pos1,pos2,pos3,pos4)

                xvec = np.zeros((sfreNum, rctfLen), dtype=np.complex128)
                xvec[:, pos1:pos2] = sftSS[:, m2, :]
                xvec[:, pos3:pos4] = -sftSS[:, m1, :]
                # print(xvec[:,0])

                gain = np.sum(covInvt * np.conj(xvec)[:, None, :], axis=2)
                gain = gain / (1 + np.sum(gain * xvec, axis=1))[:, None]

                # print(t,np.mean(gain))


                # print(gain.shape,covInvt.shape,xvec.shape)

                # 将 gain 扩展为 (8, 32, rctfLen) 的维度
                gain_expanded = np.tile(gain[:, :, np.newaxis], (1, 1, rctfLen))

                # 计算 covInvt 和 xvec 的逐元素乘积
                covInvt_times_xvec = covInvt * xvec[:, :, np.newaxis]

                # 对 covInvt_times_xvec 沿着第二个维度求和
                sum_covInvt_times_xvec = np.sum(covInvt_times_xvec, axis=1, keepdims=True)

                # 将 gain_expanded 和 sum_covInvt_times_xvec 逐元素相乘
                covInvtDelta = gain_expanded * sum_covInvt_times_xvec

                # print(t,np.mean(covInvtDelta))

                # if np.isnan(covInvt).any() or np.isinf(covInvt).any():
                #     print(t)
                #     print("Warning: NaN or Inf detected in covInvt at line 184")
                #     print("covInvt:", covInvt)
                #     raise ValueError("NaN or Inf encountered in covInvt during computation.")

                covInvt = covInvt - covInvtDelta

        covInv[sInd, :, :] = covInvt / lambda_
        dprtfRef = covInvt[:, ::fiLen, ::fiLen]
        # print(dprtfRef[:,0,0])
        dprtf = np.ones((sfreNum, mpNum), dtype=np.complex128)

        for mp in range(mpNum):
            m1 = MP[0][mp]-1
            m2 = MP[1][mp]-1
            a = dprtfRef[:, m2, m1] / (dprtfRef[:, m1, m1]+0.00000001)
            b = dprtfRef[:, m2, m2] / (dprtfRef[:, m1, m2]+0.00000001)
            simi = np.abs(1 + a * np.conj(b)) / (np.sqrt((1 + np.abs(a) ** 2) * (1 + np.abs(b) ** 2))+0.00000001)
            dprtf[:, mp] = (simi > ctTh) * (a + b) / 2

        dprtf = dprtf / (1 + np.abs(dprtf))

        if np.sum(dprtf != 0) == 0:
            gmmWeight = (1 - eta1) * gmmWeight + eta1 * noInfVec
            GMMWeight[:, t] = gmmWeight
            continue

        # EG multispeaker localization
        loglik = loglik_complexgaussian(dprtf, rtfTemp[sInd, :, :], 0.1)
        lik = np.exp(loglik)

        # print(np.mean(lik))

        # fig,(ax1,ax2,ax3,ax4)=plt.subplots(1,4)
        # ax1.plot(lik)
        # ax2.plot(np.abs(xft[:,1]))
        # ax3.plot(np.abs(xft[:,2]))
        # ax4.plot(np.abs(xft[:,3]))
        # plt.show()

        dprtf = dprtf.flatten()
        lik = lik.reshape((sfreNum * mpNum, candNum))
        lik = lik[dprtf != 0, :]

        gmmLik = lik * gmmWeight[None, :]
        DerLik = -np.mean(lik / np.sum(gmmLik, axis=1)[:, None], axis=0)

        DerEntropy = -(1 + np.log(gmmWeight))
        Der = DerLik + gamma * DerEntropy

        expDer = np.exp(-eta * Der)
        expDer[expDer > 1e2] = 1e2

        gmmWeight = gmmWeight * expDer
        gmmWeight = gmmWeight / np.sum(gmmWeight)

        gmmWeightM1 = np.roll(gmmWeight, 1)
        gmmWeightP1 = np.roll(gmmWeight, -1)
        gmmWeight = (gmmWeight + spaSmooth * (gmmWeightM1 + gmmWeightP1)) / (1 + 2 * spaSmooth)
        GMMWeight[:, t] = gmmWeight

        peaks = (gmmWeight > np.roll(gmmWeight, 1)) & (gmmWeight > np.roll(gmmWeight, -1))
        if np.sum(peaks) > maxPeak:
            sortWeight = np.sort(gmmWeight[peaks])[::-1]
            Peaks[:, t] = peaks & (gmmWeight > max(psTh, sortWeight[maxPeak]))
        else:
            Peaks[:, t] = peaks & (gmmWeight > psTh)

    return GMMWeight, Peaks

class online_ssl_dprtf_eg_c:
    # 初始化方法，实例化对象时会自动调用
    def __init__(self, segment,rtfTemp,freRan, MP):
        self.segment = segment  # 实例属性
        self.micNum=segment.shape[1]
        self.rtfTemp=rtfTemp
        self.freRan=freRan
        self.MP=MP
        self.mpNum=MP.shape[1]
        self.fs = 16000
        self.winLen = 256
        self.fraInc = self.winLen // 2
        self.win = hamming(self.winLen)
        self.freNum=len(freRan)
        self.fraNum = (self.segment.shape[0] - self.winLen) // self.fraInc #需要修改
        self.smoLen = int(np.ceil(200 * 8 / self.fraInc))
        self.beta = (self.smoLen - 1) / (self.smoLen + 1)
        self.minSran = 3
        self.mcmtFn = int(np.floor(self.minSran * self.fs / self.fraInc)) -self.smoLen + 1
        self.sTh, self.nTh = mcmt(self.smoLen, 1,self.mcmtFn)
        self.sfreNumTh = 0.05 * self.freNum
        self.fiLen = 12 # IMPORTANT! CTF length, this setting is suitable for reverberation time around 0.55 s, it approximately equals T60/8.
        self.rctfLen = self.fiLen * self.micNum
        self.fraNumLs = int(np.ceil(self.fiLen / np.sqrt(self.micNum - 1)))
        self.lambda_ = (self.fraNumLs - 1) / (self.fraNumLs + 1)
        self.ctTh = 0.6

        # EG parameters
        self.candNum = rtfTemp.shape[2]  # Number of candidate locations
        self.noInfVec = np.ones(self.candNum) / self.candNum  # Uniform vector

        self.eta = 0.07  # EG update rate
        self.eta1 = 0.065

        self.gamma = 0.1 * 2  # Entropy regularization weight
        self.spaSmooth = 0.01  # Spatial smoothing factor
        self.psTh = 0.09  # IMPORTANT! peak detection threshold, controls the tradeoff between false alarm and miss detection
        self.maxPeak = 2  # Maximum number of active speakers

        # Variables in process处理过程中需要修改的变量?
        self.ftHis = np.zeros((self.freNum, self.micNum, self.fiLen), dtype=np.complex128)
        self.sft = np.zeros((self.freNum, self.micNum, self.fiLen), dtype=np.complex128)
        self.nearNoiseFra = np.zeros((self.freNum, self.micNum, self.fiLen), dtype=np.complex128)
        self.ftPow = np.zeros(self.freNum)  # Smoothed STFT power

        self.covInv = np.zeros((self.freNum, self.rctfLen, self.rctfLen), dtype=np.complex128)  # Signal covariance matrix inverse
        for fre in range(self.freNum):
            result_eye = 1e3 * np.eye(self.rctfLen)
            self.covInv[fre, :, :] = result_eye

        self.gmmWeight = self.noInfVec  # GMM weight vector in process
        self.GMMWeight = np.zeros((self.candNum, self.fraNum))  # GMM weight for all frames
        self.Peaks = np.zeros((self.candNum, self.fraNum))  # Peaks


        for t in range(self.fraNum):
            xt = self.segment[t * self.fraInc:t * self.fraInc + self.winLen, :]
            xft = fft(xt * self.win[:, None], axis=0)
            xft = xft[freRan, :]

            self.ftHis[:, :, 1:self.fiLen] = self.ftHis[:, :, :self.fiLen - 1]
            self.ftHis[:, :, 0] = xft

            self.sft = self.beta * self.sft + (1 - self.beta) * self.ftHis * np.conj(xft[:, 0]).reshape(self.sft.shape[0], 1, 1)
            self.ftPow = self.beta * self.ftPow + (1 - self.beta) * np.sum(np.abs(xft) ** 2, axis=1)
            # print(t,np.mean(ftPow))
            if t < (self.smoLen - 1):
                self.gmmWeight = (1 - self.eta1) * self.gmmWeight + self.eta1 * self.noInfVec
                self.GMMWeight[:, t] = self.gmmWeight
                continue
            elif t == (self.smoLen - 1):
                self.minPowHis = np.tile(self.ftPow[:, None], (1, self.mcmtFn))
                self.gmmWeight = (1 - self.eta1) * self.gmmWeight + self.eta1 * self.noInfVec
                self.GMMWeight[:, t] = self.gmmWeight
                continue

            self.minPowHis[:, :self.mcmtFn - 1] = self.minPowHis[:, 1:self.mcmtFn]
            self.minPowHis[:, self.mcmtFn - 1] = self.ftPow
            self.minPow = np.min(self.minPowHis, axis=1)

            self.nearNoiseFra[self.ftPow <= self.nTh * self.minPow, :, :] = self.sft[self.ftPow <= self.nTh * self.minPow, :, :]
            self.sftSS = self.sft - self.nearNoiseFra

            self.sInd = self.ftPow > self.sTh * self.minPow
            self.sfreNum = np.sum(self.sInd)

            if self.sfreNum < self.sfreNumTh:
                self.gmmWeight = (1 - self.eta1) * self.gmmWeight + self.eta1 * self.noInfVec
                self.GMMWeight[:, t] = self.gmmWeight
                continue

            self.sftSS = self.sftSS[self.sInd, :, :]

            # DP-RTF estimation
            self.covInvt = self.covInv[self.sInd, :, :]

            for m1 in range(self.micNum - 1):
                for m2 in range(m1 + 1, self.micNum):
                    pos1 = (m1 * self.fiLen)
                    pos2 = (m1 + 1) * self.fiLen
                    pos3 = (m2 * self.fiLen)
                    pos4 = (m2 + 1) * self.fiLen

                    xvec = np.zeros((self.sfreNum, self.rctfLen), dtype=np.complex128)
                    xvec[:, pos1:pos2] = self.sftSS[:, m2, :]
                    xvec[:, pos3:pos4] = -self.sftSS[:, m1, :]

                    gain = np.sum(self.covInvt * np.conj(xvec)[:, None, :], axis=2)
                    gain = gain / (1 + np.sum(gain * xvec, axis=1))[:, None]

                    # 将 gain 扩展为 (8, 32, rctfLen) 的维度
                    gain_expanded = np.tile(gain[:, :, np.newaxis], (1, 1, self.rctfLen))

                    # 计算 covInvt 和 xvec 的逐元素乘积
                    covInvt_times_xvec = self.covInvt * xvec[:, :, np.newaxis]

                    # 对 covInvt_times_xvec 沿着第二个维度求和
                    sum_covInvt_times_xvec = np.sum(covInvt_times_xvec, axis=1, keepdims=True)

                    # 将 gain_expanded 和 sum_covInvt_times_xvec 逐元素相乘
                    covInvtDelta = gain_expanded * sum_covInvt_times_xvec


                    self.covInvt = self.covInvt - covInvtDelta

            self.covInv[self.sInd, :, :] = self.covInvt / self.lambda_
            dprtfRef = self.covInvt[:, ::self.fiLen, ::self.fiLen]
            dprtf = np.ones((self.sfreNum, self.mpNum), dtype=np.complex128)

            for mp in range(self.mpNum):
                m1 = self.MP[0][mp] - 1
                m2 = self.MP[1][mp] - 1
                a = dprtfRef[:, m2, m1] / (dprtfRef[:, m1, m1] + 0.00000001)
                b = dprtfRef[:, m2, m2] / (dprtfRef[:, m1, m2] + 0.00000001)
                simi = np.abs(1 + a * np.conj(b)) / (np.sqrt((1 + np.abs(a) ** 2) * (1 + np.abs(b) ** 2)) + 0.00000001)
                dprtf[:, mp] = (simi > self.ctTh) * (a + b) / 2

            dprtf = dprtf / (1 + np.abs(dprtf))

            if np.sum(dprtf != 0) == 0:
                self.gmmWeight = (1 - self.eta1) * self.gmmWeight + self.eta1 * self.noInfVec
                self.GMMWeight[:, t] = self.gmmWeight
                continue

            # EG multispeaker localization
            loglik = loglik_complexgaussian(dprtf, self.rtfTemp[self.sInd, :, :], 0.1)
            lik = np.exp(loglik)

            dprtf = dprtf.flatten()
            lik = lik.reshape((self.sfreNum * self.mpNum, self.candNum))
            lik = lik[dprtf != 0, :]

            gmmLik = lik * self.gmmWeight[None, :]
            DerLik = -np.mean(lik / np.sum(gmmLik, axis=1)[:, None], axis=0)

            DerEntropy = -(1 + np.log(self.gmmWeight))
            Der = DerLik + self.gamma * DerEntropy

            expDer = np.exp(-self.eta * Der)
            expDer[expDer > 1e2] = 1e2

            self.gmmWeight = self.gmmWeight * expDer
            self.gmmWeight = self.gmmWeight / np.sum(self.gmmWeight)

            gmmWeightM1 = np.roll(self.gmmWeight, 1)
            gmmWeightP1 = np.roll(self.gmmWeight, -1)
            self.gmmWeight = (self.gmmWeight + self.spaSmooth * (gmmWeightM1 + gmmWeightP1)) / (1 + 2 * self.spaSmooth)
            self.GMMWeight[:, t] = self.gmmWeight

            peaks = (self.gmmWeight > np.roll(self.gmmWeight, 1)) & (self.gmmWeight > np.roll(self.gmmWeight, -1))
            if np.sum(peaks) > self.maxPeak:
                sortWeight = np.sort(self.gmmWeight[peaks])[::-1]
                self.Peaks[:, t] = peaks & (self.gmmWeight > max(self.psTh, sortWeight[self.maxPeak]))
            else:
                self.Peaks[:, t] = peaks & (self.gmmWeight > self.psTh)

    def process(self,new_segment):
        self.segment=new_segment

        for t in range(self.fraNum):
            xt = self.segment[t * self.fraInc:t * self.fraInc + self.winLen, :]
            xft = fft(xt * self.win[:, None], axis=0)
            xft = xft[self.freRan, :]

            self.ftHis[:, :, 1:self.fiLen] = self.ftHis[:, :, :self.fiLen - 1]
            self.ftHis[:, :, 0] = xft

            self.sft = self.beta * self.sft + (1 - self.beta) * self.ftHis * np.conj(xft[:, 0]).reshape(self.sft.shape[0], 1, 1)
            self.ftPow = self.beta * self.ftPow + (1 - self.beta) * np.sum(np.abs(xft) ** 2, axis=1)

            self.minPowHis[:, :self.mcmtFn - 1] = self.minPowHis[:, 1:self.mcmtFn]
            self.minPowHis[:, self.mcmtFn - 1] = self.ftPow
            self.minPow = np.min(self.minPowHis, axis=1)

            self.nearNoiseFra[self.ftPow <= self.nTh * self.minPow, :, :] = self.sft[self.ftPow <= self.nTh * self.minPow, :, :]
            self.sftSS = self.sft - self.nearNoiseFra

            self.sInd = self.ftPow > self.sTh * self.minPow
            self.sfreNum = np.sum(self.sInd)

            if self.sfreNum < self.sfreNumTh:
                self.gmmWeight = (1 - self.eta1) * self.gmmWeight + self.eta1 * self.noInfVec
                self.GMMWeight[:, :-1]=self.GMMWeight[:, 1:]
                self.GMMWeight[:, -1] = self.gmmWeight
                continue

            self.sftSS = self.sftSS[self.sInd, :, :]

            # DP-RTF estimation
            self.covInvt = self.covInv[self.sInd, :, :]

            for m1 in range(self.micNum - 1):
                for m2 in range(m1 + 1, self.micNum):
                    pos1 = (m1 * self.fiLen)
                    pos2 = (m1 + 1) * self.fiLen
                    pos3 = (m2 * self.fiLen)
                    pos4 = (m2 + 1) * self.fiLen

                    xvec = np.zeros((self.sfreNum, self.rctfLen), dtype=np.complex128)
                    xvec[:, pos1:pos2] = self.sftSS[:, m2, :]
                    xvec[:, pos3:pos4] = -self.sftSS[:, m1, :]

                    gain = np.sum(self.covInvt * np.conj(xvec)[:, None, :], axis=2)
                    gain = gain / (1 + np.sum(gain * xvec, axis=1))[:, None]

                    # 将 gain 扩展为 (8, 32, rctfLen) 的维度
                    gain_expanded = np.tile(gain[:, :, np.newaxis], (1, 1, self.rctfLen))

                    # 计算 covInvt 和 xvec 的逐元素乘积
                    covInvt_times_xvec = self.covInvt * xvec[:, :, np.newaxis]

                    # 对 covInvt_times_xvec 沿着第二个维度求和
                    sum_covInvt_times_xvec = np.sum(covInvt_times_xvec, axis=1, keepdims=True)

                    # 将 gain_expanded 和 sum_covInvt_times_xvec 逐元素相乘
                    covInvtDelta = gain_expanded * sum_covInvt_times_xvec


                    self.covInvt = self.covInvt - covInvtDelta

            self.covInv[self.sInd, :, :] = self.covInvt / self.lambda_
            dprtfRef = self.covInvt[:, ::self.fiLen, ::self.fiLen]
            dprtf = np.ones((self.sfreNum, self.mpNum), dtype=np.complex128)

            for mp in range(self.mpNum):
                m1 = self.MP[0][mp] - 1
                m2 = self.MP[1][mp] - 1
                a = dprtfRef[:, m2, m1] / (dprtfRef[:, m1, m1] + 0.00000001)
                b = dprtfRef[:, m2, m2] / (dprtfRef[:, m1, m2] + 0.00000001)
                simi = np.abs(1 + a * np.conj(b)) / (np.sqrt((1 + np.abs(a) ** 2) * (1 + np.abs(b) ** 2)) + 0.00000001)
                dprtf[:, mp] = (simi > self.ctTh) * (a + b) / 2

            dprtf = dprtf / (1 + np.abs(dprtf))

            if np.sum(dprtf != 0) == 0:
                self.gmmWeight = (1 - self.eta1) * self.gmmWeight + self.eta1 * self.noInfVec
                self.GMMWeight[:, :-1]=self.GMMWeight[:, 1:]
                self.GMMWeight[:, -1] = self.gmmWeight
                continue

            # EG multispeaker localization
            loglik = loglik_complexgaussian(dprtf, self.rtfTemp[self.sInd, :, :], 0.1)
            lik = np.exp(loglik)

            dprtf = dprtf.flatten()
            lik = lik.reshape((self.sfreNum * self.mpNum, self.candNum))
            lik = lik[dprtf != 0, :]

            gmmLik = lik * self.gmmWeight[None, :]
            DerLik = -np.mean(lik / np.sum(gmmLik, axis=1)[:, None], axis=0)

            DerEntropy = -(1 + np.log(self.gmmWeight))
            Der = DerLik + self.gamma * DerEntropy

            expDer = np.exp(-self.eta * Der)
            expDer[expDer > 1e2] = 1e2

            self.gmmWeight = self.gmmWeight * expDer
            self.gmmWeight = self.gmmWeight / np.sum(self.gmmWeight)

            gmmWeightM1 = np.roll(self.gmmWeight, 1)
            gmmWeightP1 = np.roll(self.gmmWeight, -1)
            self.gmmWeight = (self.gmmWeight + self.spaSmooth * (gmmWeightM1 + gmmWeightP1)) / (1 + 2 * self.spaSmooth)
            self.GMMWeight[:, :-1] = self.GMMWeight[:, 1:]
            self.GMMWeight[:, -1] = self.gmmWeight

            peaks = (self.gmmWeight > np.roll(self.gmmWeight, 1)) & (self.gmmWeight > np.roll(self.gmmWeight, -1))
            if np.sum(peaks) > self.maxPeak:
                sortWeight = np.sort(self.gmmWeight[peaks])[::-1]
                self.Peaks[:, t] = peaks & (self.gmmWeight > max(self.psTh, sortWeight[self.maxPeak]))
            else:
                self.Peaks[:, t] = peaks & (self.gmmWeight > self.psTh)
        return self.GMMWeight,self.Peaks


