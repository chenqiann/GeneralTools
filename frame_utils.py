import json
import cv2
import numpy as np
import os
from glob import glob
import imageio.v2 as imageio



def draw_rectangle(frame_file, x, y, width, height, output_path):

    img = cv2.imread(frame_file)
    img = cv2.rectangle(img, (x, y), (x + width, y + height), (0, 0, 255), 3)
    cv2.imwrite(output_path, img)

def overlay_images(img1_file, img2_file, output_path):
    img1 = cv2.imread(img1_file)
    img2 = cv2.imread(img2_file)
    img = cv2.addWeighted(img1, 0.5, img2, 0.5, 0)
    cv2.imwrite(output_path, img)

def cut_image(frame_file, x, y, width, height, output_path):
    img = cv2.imread(frame_file)
    img = img[y:y+height, x:x+width]
    cv2.imwrite(output_path, img)


def generate_gif(k, output_path):
    img_paths = []
    for f in range(0, k):
        img_paths.append(('%02d.png') % f)
    gif_images = []
    for path in img_paths:
        gif_images.append(imageio.imread(output_path + "/" + path))
    imageio.mimsave(output_path + "/test.gif", gif_images, 'GIF', fps=7)



if __name__ == "__main__":
    # sample_file = "Disney_v4_8_002959_s2/"
    # label_file = "F:/Prj/Research/datasets/atd_12k/datasets/test_2k_annotations"
    # label_file += sample_file
    # label_file += "Disney_v4_8_002959_s2.json"
    # input_file = "F:/Prj/Research/datasets/atd_12k/datasets/test_2k_540p/"
    # frame_file += sample_file
    # frame_file += "frame2.jpg"
    # with open(label_file, 'r') as f:
    #     jsonObj = json.load(f)
    #     motion_RoI = jsonObj["motion_RoI"]
    #     print(motion_RoI)
    #
    # x = motion_RoI['x']
    # y = motion_RoI['y']
    # width = motion_RoI['width']
    # height = motion_RoI['height']
    # draw_rectangle(frame_file, x, y, width, height, "frame.jpg")

    frame_file = "F:/Prj/Research/multi_flows/Japan_v2_0_161238_s3/multiflows/"
    files = glob(os.path.join(frame_file, "*.png"))
    for f in files:
        x = 190
        y = 120
        width = 125
        height = 70
        name = os.path.splitext(f)[0]
        print(name)
        draw_rectangle(f, x, y, width, height, name + "_.png")
        cut_image(f, x, y, width, height, name + "_cut.png")



    # frame_file = "F:/Prj/Research/datasets/AnimeRun_v2/test/Frame_Anime/"
    # frame_file += "sprite_080_0010_A/original/"
    # frame1 = frame_file + "0184.png"
    # frame2 = frame_file + "0185.png"
    # overlay_images(frame1, frame2, "0184over.png")

    # frame_file = "C:/Users/Chen/Desktop/Research/thesis/images/multitest/mid"
    # scene = "Japan_v2_3_103334_s2"
    # roi_x = 124
    # roi_y = 70
    # roi_w = 135
    # roi_h = int(roi_w * 270.0 / 480.0)
    # output_root = os.path.join(os.getcwd(), scene)
    # if not os.path.exists(output_root):
    #     os.mkdir(output_root)
    # frames = sorted(glob(os.path.join(frame_file, scene, "*.png")))
    # for f in frames:
    #     file = os.path.basename(f).split('.')[0]
    #     draw_rectangle(f, roi_x, roi_y, roi_w, roi_h, os.path.join(output_root, file + "_.png"))
    #     cut_image(f, roi_x, roi_y, roi_w, roi_h, os.path.join(output_root, file + "_cut.png"))

    # for scene in os.listdir(frame_file):
    #     output_root = os.path.join(os.getcwd(), scene)
    #     if not os.path.exists(output_root):
    #         os.mkdir(output_root)
    #     frame1 = os.path.join(input_file, scene, "frame1.png")
    #     frame2 = os.path.join(input_file, scene, "frame3.png")
    #     img1 = cv2.imread(frame1)
    #     img1 = cv2.resize(img1, (480, 270), interpolation=cv2.INTER_LINEAR)
    #     img2 = cv2.imread(frame2)
    #     img2 = cv2.resize(img2, (480, 270), interpolation=cv2.INTER_LINEAR)
    #     cv2.imwrite(os.path.join(output_root, "frame1.png"), img1)
    #     cv2.imwrite(os.path.join(output_root, "frame3.png"), img2)

    ############# generate gif #################
    # k = 17
    # base_path = "F:/Prj/Research/gif"
    # for scene in os.listdir(base_path):
    #     generate_gif(k, os.path.join(base_path, scene, "abme"))




