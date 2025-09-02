import numpy as np

def generate_template(dpp, freRan=None, MP=None):
    # 输入参数
    # dpp: 直接传播路径，可能是 i) 直接路径脉冲响应，例如虚拟头部情况的HRIR，大小为 (样本数) x (麦克风数) x (候选位置数)，
    # ii) 麦克风间的TDOA，大小为 (麦克风对数) x (候选位置数)。采样率为16 kHz。
    # MP: 用于定位的麦克风对，大小为 2 x (麦克风对数)
    # freRan: 用于SSL的频率范围。用频率bin表示，在 [2 ftLen/2] 范围内。

    # 输出
    # rtfTemp: 大小为 (频率数) x (麦克风对数) x (候选位置数)

    fs = 16000  # 采样率 16 kHz
    ftLen = 256  # 傅里叶变换的长度

    if freRan is None:
        freRan = np.arange(2, ftLen // 4 + 1)  # SSL使用的频率范围，最多到4 kHz
    freNum = len(freRan)

    if dpp.ndim == 3:  # 如果dpp的维度为3
        _, micNum, candNum = dpp.shape

        if MP is None:
            MP = []
            for m1 in range(micNum - 1):
                for m2 in range(m1 + 1, micNum):
                    MP.append([m1, m2])
            MP = np.array(MP).T  # 将麦克风对按列存储

        mpNum = MP.shape[1]  # 麦克风对的数量

        # 对dpp进行傅里叶变换
        TF = np.fft.fft(dpp, ftLen, axis=0)
        TF = TF[freRan, :, :]  # 选择频率范围内的数据

        rtfTemp = np.zeros((freNum, mpNum, candNum), dtype=complex)  # 初始化输出

        for mp in range(mpNum):
            m1, m2 = MP[:, mp]-1
            rtfTemp[:, mp, :] = TF[:, m2, :] / TF[:, m1, :]  # 计算相对传递函数

        # 归一化到[0, 1]
        rtfTemp = rtfTemp / (np.abs(rtfTemp) + 1)
    else:
        mpNum, candNum = dpp.shape  # 如果dpp维度为2

        # 使用指数计算
        rtfTemp = 0.5 * np.exp(-1j * np.multiply(
            2 * np.pi * fs * freRan[:, None, None] / ftLen,
            dpp[None, :, :]
        ))

    return rtfTemp
