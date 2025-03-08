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

    # make an image again
    new_image = Image.fromarray(pixels.reshape(image.shape))

    # export image
    path = "./assignment3/si-exercises/exercise_pso/"
    new_image.save(path + 'quantized_imgage.png')


def PSO(image, n, evaluation, omega, alpha1, alpha2):
    # get n random pixel positions
    height, width = image.size
    x_coords = np.randint(0, width, n)
    y_coords = np.randint(0, height, n)

    x = np.array(zip(x_coords, y_coords))

    # initialize velocities, local best and global best
    velocity = np.zeros(n)
    local_best = np.zeros(n)
    global_best = 0
    happy = False

    # write initial state to file
    # TODO

    # execute PSO algorithm
    while not happy:
        for i in range(n):
            r1 = 0  # TODO
            r2 = 0  # TODO
            velocity = omega * velocity + alpha1 * r1 * (local_best - x) + alpha2 * r2 * (global_best - x)
        for i in range(n):
            x = x + velocity
            if evaluation(x[i]) < evaluation(x[i]):  # TODO
                local_best[i] = x[i]
            if evaluation(x[i]) < evaluation(global_best):
                global_best = x[i]
        # write state to file
        # TODO

        # At several iterations of your PSO algorithm, show both the best color palette found so far, and the image quantized with this palette.
        # TODO


# thoughts (not sure if they are right and my most recent thoughts)
# points: colored pixels
# clusters: colors that are used most
# agents: pixels
# fitness: how often a color is used, or, distance
# particle: the best color to choose from


def k_means():
    pass


if __name__ == "__main__":
    # QUANTIZATION
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
    color_quantization(img, colors)

    # PSO
    PSO(img, 4, color_distance, 0.73, 1.5, 1.5)  # parameter omega, alpha1, alpha2 from lecture

    # K-MEANS
