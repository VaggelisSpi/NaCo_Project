from PIL import Image
import numpy as np
from sklearn.cluster import KMeans


def nearest_color(pixel, colors):
    best_color = None
    best_distance = np.inf

    for color in colors:
        distance = color_distance(pixel, color)
        if distance < best_distance:
            best_distance = distance
            best_color = color

    return best_color, best_distance


def color_distance(col1, col2):
    # for now I implemented Euclidean distance but if we want we can optimize this function to
    # be more perceptually accurate: https://en.wikipedia.org/wiki/Color_difference
    return np.sqrt((col1[0] - col2[0]) ** 2 +
                   (col1[1] - col2[1]) ** 2 +
                   (col1[2] - col2[2]) ** 2)


def color_quantization(image, colors, name):
    image = np.array(image, dtype=np.uint8)
    pixels = image.reshape(-1, 3)

    # change every pixel to the nearest color
    for i, pixel in enumerate(pixels):
        # assign the nearest color to the pixel
        pixels[i], _ = nearest_color(pixel, colors)

    # make an image again
    new_image = Image.fromarray(pixels.reshape(image.shape))

    # export image
    path = "./assignment3/si-exercises/exercise_pso/"
    new_image.save(path + name)


def PSO(image, k, n_particles, evaluation=nearest_color, max_iter=50, omega=0.73, alpha1=1.5, alpha2=1.5):
    image = np.array(image, dtype=np.uint8)
    pixels = image.reshape(-1, 3)

    # get k random colors for each particle
    x = np.random.randint(0, 256, (n_particles, k, 3))

    # initialize velocities, local best and global best
    velocity = np.zeros_like(x)
    local_best = np.copy(x)
    local_best_score = np.full(n_particles, np.inf)
    global_best = np.copy(x)
    global_best_score = np.inf
    iter = 0

    # write initial state to file
    path = "./assignment3/si-exercises/exercise_pso/"
    f = open(path + "PSO_results.txt", "a")
    f.write("Iteration: " + str(iter) + "\n")
    f.write("x: " + str(x) + "\n")
    f.write("velocity: " + str(velocity) + "\n")
    f.write("local best: " + str(local_best) + "\n")
    f.write("global_best: " + str(global_best) + "\n\n")
    f.close()

    # execute PSO algorithm
    while iter < max_iter:
        iter += 1
        for i in range(n_particles):
            r1 = np.random.rand(3)
            r2 = np.random.rand(3)
            velocity = omega * velocity + alpha1 * r1 * (local_best - x) + alpha2 * r2 * (global_best - x)
        x = np.array(x + velocity, dtype=int)
        for i in range(n_particles):
            score = 0
            for pixel in pixels:
                score += evaluation(pixel, x[i])[1]
            if score < local_best_score[i]:
                local_best[i] = x[i]
                local_best_score[i] = score
            if score < global_best_score:
                global_best = x[i]
                global_best_score = score

        # write state to file
        f = open(path + "PSO_results.txt", "a")
        f.write("Iteration: " + str(iter) + "\n")
        f.write("x: " + str(x) + "\n")
        f.write("velocity: " + str(velocity) + "\n")
        f.write("local best: " + str(local_best) + "\n")
        f.write("global_best: " + str(global_best) + "\n\n")
        f.close()

        # show color palette so far + image every 5 iterations
        if iter % 5 == 0:
            print(global_best)
            name = "PSO" + str(iter) + ".png"
            color_quantization(image, global_best, name)


def k_means(image, k):
    image = np.array(image, dtype=np.uint8)
    pixels = image.reshape(-1, 3)

    kmeans = KMeans(k).fit(pixels)
    colors = kmeans.cluster_centers_

    print("k-means")
    print(colors)
    name = name = "K_means.png"
    color_quantization(image, colors, name)


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
    color_quantization(img, colors, 'quantized_image.png')

    # PSO
    PSO(img, 4, 4, nearest_color)  # parameter omega, alpha1, alpha2 from lecture

    # K-MEANS
    k_means(img, 4)
