import numpy as np

def loglik_complexgaussian(x, mu, sigma):
    """
    计算复高斯分布的对数似然值

    参数：
        x (numpy.ndarray): dprtf 估计值，形状为 (频率数, 麦克风对数)

        sigma (float): 方差，为常量（设置为0.1）

    返回：
        numpy.ndarray: 对数似然值，形状为 (频率数, 麦克风对数, 候选位置数)
    """
    D = mu.shape[2]
    x = np.repeat(x[:, :, np.newaxis], D, axis=2)

    # 找到幅度大于 0.5 的元素位置
    upindx = np.abs(x) > 0.5

    # 对 x 和 mu 进行处理，限制幅度
    x = np.where(upindx, (x * (1 - np.abs(x)) / (np.abs(x) + np.finfo(float).eps)), x * (1 - upindx))
    mu = np.where(upindx, (mu * (1 - np.abs(mu)) / (np.abs(mu) + np.finfo(float).eps)), mu * (1 - upindx))

    # 计算对数似然
    loglik = -np.log(np.pi * sigma) - np.abs(x - mu)**2 / sigma
    return loglik
