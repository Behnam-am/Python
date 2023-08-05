import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # # Read Image
    img = cv.imread(cv.samples.findFile("golden.jpg"), cv.IMREAD_COLOR)
    # # Gray-Scale Image
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_x, img_y = img_gray.shape
    x = img_gray.reshape(-1, 1)
    y = (x & (pow(2, np.arange(8))) != 0).astype(int)
    bit_planes = np.zeros(shape=(8, img_x, img_y)).astype(int)
    for i in range(8):
        bit_planes[i] = y[:, i].reshape(img_x, img_y)
    fig, ax = plt.subplots(3, 3)
    ax[0, 0].imshow(img_gray, cmap=plt.get_cmap('gray'))
    ax[0, 1].imshow(bit_planes[0], cmap=plt.get_cmap('gray'))
    ax[0, 2].imshow(bit_planes[1], cmap="binary")
    ax[1, 0].imshow(bit_planes[2], cmap="binary")
    ax[1, 1].imshow(bit_planes[3], cmap="binary")
    ax[1, 2].imshow(bit_planes[4], cmap="binary")
    ax[2, 0].imshow(bit_planes[5], cmap="binary")
    ax[2, 1].imshow(bit_planes[6], cmap="binary")
    ax[2, 2].imshow(bit_planes[7], cmap="binary")
    ax[0, 0].set_title("Original")
    ax[0, 1].set_title("bit plane 0")
    ax[0, 2].set_title("bit plane 1")
    ax[1, 0].set_title("bit plane 2")
    ax[1, 1].set_title("bit plane 3")
    ax[1, 2].set_title("bit plane 4")
    ax[2, 0].set_title("bit plane 5")
    ax[2, 1].set_title("bit plane 6")
    ax[2, 2].set_title("bit plane 7")
    fig.tight_layout()
    plt.savefig("bit planes.jpg")
