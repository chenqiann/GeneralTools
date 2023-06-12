import numpy as np
import cv2
import torch

UNKNOWN_FLOW_THRESH = 1e7
# SMALLFLOW = 0.0
# LARGEFLOW = 1e8
def readFlow(fn):
    """ Read .flo file in Middlebury format"""
    # Code adapted from:
    # http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy

    # WARNING: this will work on little-endian architectures (eg Intel x86) only!
    # print 'fn = %s'%(fn)
    with open(fn, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
            return None
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            # print 'Reading %d x %d flo file\n' % (w, h)
            data = np.fromfile(f, np.float32, count=2*int(w)*int(h))
            # Reshape data into 3D array (columns, rows, bands)
            # The reshape here is for visualization, the original code is (w,h,2)
            return np.resize(data, (int(h), int(w), 2))
            # from https://github.com/gengshan-y/VCN

def make_color_wheel():
    """
    Generate color wheel according Middlebury color code
    :return: Color wheel
    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros([ncols, 3])

    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255 * np.arange(0, RY) / RY))
    col += RY

    # YG
    colorwheel[col:col + YG, 0] = 255 - np.transpose(np.floor(255 * np.arange(0, YG) / YG))
    colorwheel[col:col + YG, 1] = 255
    col += YG

    # GC
    colorwheel[col:col + GC, 1] = 255
    colorwheel[col:col + GC, 2] = np.transpose(np.floor(255 * np.arange(0, GC) / GC))
    col += GC

    # CB
    colorwheel[col:col + CB, 1] = 255 - np.transpose(np.floor(255 * np.arange(0, CB) / CB))
    colorwheel[col:col + CB, 2] = 255
    col += CB

    # BM
    colorwheel[col:col + BM, 2] = 255
    colorwheel[col:col + BM, 0] = np.transpose(np.floor(255 * np.arange(0, BM) / BM))
    col += + BM

    # MR
    colorwheel[col:col + MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col + MR, 0] = 255

    return colorwheel


def compute_color(u, v):
    """
    compute optical flow color map
    :param u: optical flow horizontal map
    :param v: optical flow vertical map
    :return: optical flow in color code
    """
    [h, w] = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0

    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)

    rad = np.sqrt(u ** 2 + v ** 2)

    a = np.arctan2(-v, -u) / np.pi

    fk = (a + 1) / 2 * (ncols - 1) + 1

    k0 = np.floor(fk).astype(int)

    k1 = k0 + 1
    k1[k1 == ncols + 1] = 1
    f = fk - k0

    for i in range(0, np.size(colorwheel, 1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0 - 1] / 255
        col1 = tmp[k1 - 1] / 255
        col = (1 - f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        notidx = np.logical_not(idx)

        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col * (1 - nanIdx)))

    return img


def flow_to_image(flow):
    """
    Convert flow into middlebury color code image
    :param flow: optical flow map
    :return: optical flow image in middlebury color
    """
    u = flow[:, :, 0]
    v = flow[:, :, 1]

    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.

    idxUnknow = (abs(u) > UNKNOWN_FLOW_THRESH) | (abs(v) > UNKNOWN_FLOW_THRESH)
    u[idxUnknow] = 0
    v[idxUnknow] = 0

    maxu = max(maxu, np.max(u))
    minu = min(minu, np.min(u))

    maxv = max(maxv, np.max(v))
    minv = min(minv, np.min(v))

    rad = np.sqrt(u ** 2 + v ** 2)
    maxrad = max(-1, np.max(rad))

    u = u / (maxrad + np.finfo(float).eps)
    v = v / (maxrad + np.finfo(float).eps)

    img = compute_color(u, v)

    idx = np.repeat(idxUnknow[:, :, np.newaxis], 3, axis=2)
    img[idx] = 0

    return np.uint8(img)


def forward_warp(img0, flow, output_path):
    img1 = np.zeros_like(img0)
    H = img0.shape[0]
    W = img0.shape[1]
    round_flow = np.round(flow)
    # 遍历source image中的每个点p_source
    for h in range(H):
        for w in range(W):
            # 获取投影到destination image中的对应坐标点p_destination，采用四舍五入取整
            x = w + int(round_flow[h, w, 0])
            y = h + int(round_flow[h, w, 1])
            # 判断映射位置是否在有效范围内，若在则赋值，否则保留zero
            if x >= 0 and x < W and y >= 0 and y < H:
                img1[y, x, :] = img0[h, w, :]
            else:
                print("beyond figure: (", x, y, ")")
    cv2.imwrite(output_path, img1)
    return img1


if __name__ == "__main__":

    # file_name = "F:/Prj/Research/datasets/AnimeRun_v2/train/Flow"
    # file_name += "/cami_09_03_A"
    # file_name += "/forward/0288.flo"
    # file_name += "/backward/0289.flo"
    # flow = readFlow(file_name).astype(np.float32)
    # flow_image = flow_to_image(flow)
    # cv2.imwrite("0289_bacflow.png", flow_image)
    #
    # file_name = "F:/Prj/Research/datasets/AnimeRun_v2/train/UnmatchedForward/agent_basement2_weapon_approach/Image0186.npy"
    # data = np.load(file_name)
    # data = data * 255
    # data = data.astype('uint8')
    # cv2.imshow("frame", data)
    # cv2.waitKey(0)

    from PIL import Image
    # file_name = "F:/Prj/Research/datasets/AnimeRun_v2/test/LineArea/agent_basement1_descent/Image0078.npy"
    # data = np.load(file_name)
    # img = data * 255
    # img = img.astype('uint8')
    # cv2.imshow("contour", img)
    # cv2.waitKey(0)

    file_name = "F:/Prj/Research/datasets/AnimeRun_v2/test/Flow"
    scene = "/sprite_080_0010_A"
    file_name += scene
    file_name += "/forward/0184.flo"
    # file_name += "/backward/0289.flo"
    flow = readFlow(file_name).astype(np.float32) * 0.5

    frame_name = "F:/Prj/Research/datasets/AnimeRun_v2/test/Frame_Anime"
    frame_name = frame_name + scene + "/original"
    frame0_file = frame_name + "/0184.png"
    frame0 = cv2.imread(frame0_file)
    h, w, c = frame0.shape
    one = np.ones((h, w, 1)) * 255
    forward_warp(frame0, flow, "0184_forward.png")


