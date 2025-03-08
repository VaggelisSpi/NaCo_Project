from PIL import Image
import numpy as np


def color_distance(col1, col2):
    # for now I implemented Euclidean distance but if we want we can optimize this function to
    # be more perceptually accurate: https://en.wikipedia.org/wiki/Color_difference
    return np.sqrt((col1[0] - col2[0]) ** 2 +
                   (col1[1] - col2[1]) ** 2 +
                   (col1[2] - col2[2]) ** 2)


def color_quantization(image, colors):
    image = np.array(image, dtype=np.uint8)
    pixels = image.reshape(-1, 3)

    # change every pixel to the nearest color
    for i, pixel in enumerate(pixels):
        best_color = None
        best_distance = np.inf

        # check which color is the nearest color
        for color in colors:
            distance = color_distance(pixel, color)
            if distance < best_distance:
                best_distance = distance
                best_color = color

        # assign the nearest color to the pixel
        pixels[i] = best_color

    return Image.fromarray(pixels.reshape(image.shape))


def PSO():
    pass


def k_means():
    pass


if __name__ == "__main__":
    # import image
    path = "./assignment3/si-exercises/exercise_pso/"
    img = Image.open(path + 'image.png')

    # define color palette from assignment (RGB)
    light_green = [175, 198, 114]
    dark_red = [83, 20, 14]
    light_red = [185, 49, 40]
    dark_green = [124, 148, 71]
    colors = np.array([light_green, dark_red, light_red, dark_green])

    # color quantization
    quantized_img = color_quantization(img, colors)
    quantized_img.show()

    # export image
    quantized_img.save(path + 'quantized_imgage.png')
