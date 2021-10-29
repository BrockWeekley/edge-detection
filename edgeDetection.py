import cv2 as cv
import numpy as np
import sys
import matplotlib.pyplot as plt

# Apply convolution twice
double = True


def convolution(image, kernel):
    color = False
    if len(image.shape) > 2:
        color = True

    kernel_height, kernel_width = 3, 3
    pad_height, pad_width = 1, 1

    if color:
        height, width, _ = image.shape
        (B, G, R) = cv.split(image)
        outputB = np.zeros(B.shape)
        outputG = np.zeros(G.shape)
        outputR = np.zeros(R.shape)
    else:
        height, width = image.shape
        output = np.zeros(image.shape)

    if color:
        padded_B = np.zeros((B.shape[0] + (2 * pad_height), B.shape[1] + (2 * pad_width)))
        padded_B[pad_height:padded_B.shape[0] - pad_height, pad_width:padded_B.shape[1] - pad_width] = B

        padded_G = np.zeros((G.shape[0] + (2 * pad_height), G.shape[1] + (2 * pad_width)))
        padded_G[pad_height:padded_G.shape[0] - pad_height, pad_width:padded_G.shape[1] - pad_width] = G

        padded_R = np.zeros((R.shape[0] + (2 * pad_height), R.shape[1] + (2 * pad_width)))
        padded_R[pad_height:padded_R.shape[0] - pad_height, pad_width:padded_R.shape[1] - pad_width] = R
    else:
        padded_image = np.zeros((height + (2 * pad_height), width + (2 * pad_width)))
        padded_image[pad_height:padded_image.shape[0] - pad_height, pad_width:padded_image.shape[1] - pad_width] = image

    for r in range(height):
        for c in range(width):
            if color:
                outputB[r, c] = np.sum(kernel * padded_B[r:r + kernel_height, c:c + kernel_width])
                outputG[r, c] = np.sum(kernel * padded_G[r:r + kernel_height, c:c + kernel_width])
                outputR[r, c] = np.sum(kernel * padded_R[r:r + kernel_height, c:c + kernel_width])
            else:
                output[r, c] = np.sum(kernel * padded_image[r:r + kernel_height, c:c + kernel_width])

    if color:
        output = cv.merge([outputB, outputG, outputR])

    output = (output + 255) / 510 * 255

    return output


def edge_detection(image, input_filter):
    edge_x = convolution(image, input_filter)
    edge_y = convolution(image, np.flip(input_filter.T, axis=0))

    if double:
        double_x = convolution(edge_x, input_filter)
        double_y = convolution(edge_y, np.flip(input_filter.T, axis=0))
        double_x = double_x.astype(np.uint8)
        double_y = double_y.astype(np.uint8)
        cv.namedWindow('Edge X', cv.WINDOW_NORMAL)
        cv.imshow("Edge X", double_x)
        cv.namedWindow('Edge Y', cv.WINDOW_NORMAL)
        cv.imshow("Edge Y", double_y)
        cv.waitKey(0)

    color = False
    if len(image.shape) > 2:
        color = True

    if color:
        (Bx, Gx, Rx) = cv.split(edge_x)
        (By, Gy, Ry) = cv.split(edge_y)
        Bmagnitude = np.sqrt(np.square(Bx) + np.square(By))
        Gmagnitude = np.sqrt(np.square(Gx) + np.square(Gy))
        Rmagnitude = np.sqrt(np.square(Rx) + np.square(Ry))
        magnitude = cv.merge([Bmagnitude, Gmagnitude, Rmagnitude])
        (fig, ax) = plt.subplots(nrows=1, ncols=1, figsize=(8, 4))
    else:
        magnitude = np.sqrt(np.square(edge_x) + np.square(edge_y))
        orientation = np.arctan2(edge_y, edge_x)
        orientation = np.degrees(orientation)
        orientation = np.where(orientation < 0, 360 + orientation, orientation)
        (fig, axs) = plt.subplots(nrows=1, ncols=3, figsize=(8, 4))
        # Average 10 pixels
        orientation_by_ten = orientation[0::10, ::10]
        X, Y = np.meshgrid(np.arange(orientation_by_ten.shape[1]), np.arange(orientation_by_ten.shape[0]))

    magnitude = (magnitude) / (255 * np.sqrt(2)) * 255
    magnitude = magnitude.astype(np.uint8)

    if color:
        ax.imshow(magnitude, cmap="gray")
        ax.set_title("Gradient Magnitude")
    else:
        axs[0].imshow(magnitude, cmap="gray")
        axs[1].imshow(orientation, cmap="jet")

        axs[2].quiver(X, Y, np.cos(orientation_by_ten.reshape(-1)), np.sin(orientation_by_ten.reshape(-1)))

        axs[0].set_title("Gradient Magnitude")
        axs[1].set_title("Gradient Orientation [0, 360]")
        axs[2].set_title("Orientation Quiver")

    plt.show()
    edge_x = edge_x.astype(np.uint8)
    edge_y = edge_y.astype(np.uint8)
    cv.namedWindow('Edge X', cv.WINDOW_NORMAL)
    cv.imshow("Edge X", edge_x)
    cv.namedWindow('Edge Y', cv.WINDOW_NORMAL)
    cv.imshow("Edge Y", edge_y)


if __name__ == "__main__":
    file_input = sys.argv[1]
    if sys.argv[len(sys.argv) - 1] == "--gray":
        input_img = cv.imread(cv.samples.findFile(file_input), 0)
    else:
        input_img = cv.imread(cv.samples.findFile(file_input))
    cv.namedWindow('Original', cv.WINDOW_NORMAL)
    cv.imshow("Original", input_img)

    sobel_filter = np.array([[.25, 0, -.25],
                            [.50, 0, -.50],
                            [.25, 0, -.25]])
    edge_detection(input_img, sobel_filter)
    cv.waitKey(0)
