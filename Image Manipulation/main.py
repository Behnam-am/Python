import cv2 as cv
import numpy as np
import random


def salt_pepper(img):
    image = np.copy(img)
    row, col = image.shape
    random.seed(0)
    # add white points
    number_of_pixels = random.randint(300, 10000)
    for i in range(number_of_pixels):
        y_coord = random.randint(0, row - 1)
        x_coord = random.randint(0, col - 1)
        image[y_coord][x_coord] = 255

    # add black points
    number_of_pixels = random.randint(300, 10000)
    for i in range(number_of_pixels):
        y_coord = random.randint(0, row - 1)
        x_coord = random.randint(0, col - 1)
        image[y_coord][x_coord] = 0

    return image


if __name__ == '__main__':
    # # Read Image
    img = cv.imread(cv.samples.findFile("golden.jpg"), cv.IMREAD_COLOR)

    # # A: Gray-Scale Image
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # # B: Laplace Image
    dst = cv.Laplacian(img_gray, cv.CV_16S, ksize=3)
    img_lap = cv.convertScaleAbs(dst)

    # # C: Add Gaussian Noise And Laplace
    mean = 0
    var = 0.1
    stddev = np.sqrt(var)
    gaus_noise = np.zeros(img_gray.shape, dtype=np.uint8)
    cv.randn(gaus_noise, mean, stddev)
    # add noise
    img_gaus_noise = cv.add(img_gray, gaus_noise)
    # laplace for noisy image
    dst = cv.Laplacian(img_gaus_noise, cv.CV_16S, ksize=3)
    img_gaus_noise_lap = cv.convertScaleAbs(dst)

    # # D: Filters For Noisy Image
    # median filter
    img_gaus_noise_median = cv.medianBlur(img_gaus_noise, 3)
    # laplace for noisy image with median filter
    dst = cv.Laplacian(img_gaus_noise_median, cv.CV_16S, ksize=3)
    img_gaus_noise_median_lap = cv.convertScaleAbs(dst)
    difference1 = (img_gaus_noise_median - img_gaus_noise_median_lap)
    d_result1 = np.hstack((img_gaus_noise_median, img_gaus_noise_median_lap, difference1))

    # average filter
    img_gaus_noise_avg = cv.blur(img_gaus_noise, (3, 3))
    # laplace for noisy image with average filter
    dst = cv.Laplacian(img_gaus_noise_avg, cv.CV_16S, ksize=3)
    img_gaus_noise_avg_lap = cv.convertScaleAbs(dst)
    difference2 = (img_gaus_noise_avg - img_gaus_noise_avg_lap)
    d_result2 = np.hstack((img_gaus_noise_avg, img_gaus_noise_avg_lap, difference2))

    # # E: Add Salt and Pepper Noise
    img_sltppr_noise = salt_pepper(img_gray)
    dst = cv.Laplacian(img_sltppr_noise, cv.CV_16S, ksize=3)
    img_sltppr_noise_lap = cv.convertScaleAbs(dst)

    # mesian filter
    img_sltppr_noise_median = cv.medianBlur(img_sltppr_noise, 3)
    # laplace for noisy image with median filter
    dst = cv.Laplacian(img_sltppr_noise_median, cv.CV_16S, ksize=3)
    img_sltppr_noise_median_lap = cv.convertScaleAbs(dst)
    difference1 = (img_sltppr_noise_median - img_sltppr_noise_median_lap)
    e_result1 = np.hstack((img_sltppr_noise_median, img_sltppr_noise_median_lap, difference1))

    # average filter
    img_sltppr_noise_avg = cv.blur(img_sltppr_noise, (3, 3))
    # laplace for noisy image with average filter
    dst = cv.Laplacian(img_sltppr_noise_avg, cv.CV_16S, ksize=3)
    img_sltppr_noise_avg_lap = cv.convertScaleAbs(dst)
    difference2 = (img_sltppr_noise_avg - img_sltppr_noise_avg_lap)
    e_result2 = np.hstack((img_sltppr_noise_avg, img_sltppr_noise_avg_lap, difference2))

    # # Show Images
    # cv.imshow("original image", img)
    cv.imwrite("gray-scale image.jpg", img_gray)  # A
    cv.imwrite("laplace image.jpg", img_lap)  # B

    # # cv.imshow("gaussian noisy image", img_gaus_noise)
    cv.imwrite("laplace on gaussian noisy image.jpg", img_gaus_noise_lap)  # C
    # # cv.imshow("median on gaussian noisy image", img_gaus_noise_median)
    # # cv.imshow("laplace on median gaussian noisy image", img_gaus_noise_median_lap)
    cv.imwrite("gaussian noisy image with median filter, it's laplace, difference.jpg", d_result1)  # D
    cv.imwrite("gaussian noisy image with average filter, it's laplace, difference.jpg", d_result2)  # D

    # # cv.imshow("salt and pepper noisy image", img_sltppr_noise)
    cv.imwrite("laplace on salt and pepper noisy image.jpg", img_sltppr_noise_lap)  # E
    # # cv.imshow("median on salt and pepper noisy image", img_sltppr_noise_median)
    # # cv.imshow("laplace on median salt and pepper noisy image", img_sltppr_noise_median_lap)
    cv.imwrite("salt and pepper noisy image with median filter, it's laplace, difference.jpg", e_result1)  # E
    cv.imwrite("salt and pepper noisy image with average filter, it's laplace, difference.jpg", e_result2)  # E

    cv.waitKey(0)
