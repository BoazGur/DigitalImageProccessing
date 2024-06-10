from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import numpy as np

def bilinear_interpolation(img):
    n, m = img.shape
    interpolated_pic = np.zeros((2 * n, 2 * m))
    
    for i in range(n):
        for j in range(m):
            interpolated_pic[2 * i, 2 * j] = img[i,j]

    for i in range(0, 2 * n, 2):
        for j in range(1, 2 * m, 2):
            if j == 2 * m - 1:
                interpolated_pic[i, j] = interpolated_pic[i, j - 1]
            else:
                interpolated_pic[i, j] = np.mean([interpolated_pic[i, j - 1], interpolated_pic[i, j + 1]])

    for j in range(0, 2 * m):  
        for i in range(1, 2 * n, 2):
            if i == 2 * n - 1:
                interpolated_pic[i, j] = interpolated_pic[i - 1, j]
            else:
                interpolated_pic[i, j] = np.mean([interpolated_pic[i - 1, j], interpolated_pic[i + 1, j]])

    return interpolated_pic

def save_hist(img, name):
    flatten_img = img.flatten()
    plt.hist(flatten_img, bins=255)
    plt.title('Leafs histogram')
    plt.xlabel('Intensity')
    plt.ylabel('Frequency')
    plt.savefig(name + '.png')
    plt.clf()

def strech_hist(img):
    min_freq = np.min(img)
    max_freq = np.max(img)

    return 255 * ((img - min_freq) / (max_freq - min_freq))

def equalize_hist(img):
    n, m = img.shape
    hist = np.histogram(img, bins=256)[0]
    probability = hist / (n * m)
    
    cdf = np.cumsum(probability)
    
    return np.round(255 * cdf)

def equalize_img(img):
    equalized_hist = equalize_hist(img)
    equalized_img_lambda = lambda x: equalized_hist[x.astype(int)]

    return equalized_img_lambda(img)

def main():
    ######################### PART 1 - 1 ##################################
    # img = Image.open('./hw12024_input_img/hw12024/peppers.jpg')
    # img = ImageOps.grayscale(img)

    # interpolated_peppers = bilinear_interpolation(np.asarray(img))
    
    # image = Image.fromarray(interpolated_peppers).convert('L')
    # image.save('InterpolatedPeppersX2.png')

    # interpolated_peppers = bilinear_interpolation(interpolated_peppers)
    # interpolated_peppers = bilinear_interpolation(interpolated_peppers)

    # image = Image.fromarray(interpolated_peppers).convert('L')
    # image.save('InterpolatedPeppersX8.png')

    ######################### PART 1 - 2 ##################################
    img = Image.open('./hw12024_input_img/hw12024/leafs.jpg')
    img = ImageOps.grayscale(img)

    # save_hist(np.asarray(img), 'before_flattening')

    streched_img = strech_hist(np.asarray(img))
    
    # save_hist(streched_img, 'after_streching')
    
    # streched_img = Image.fromarray(streched_img).convert('L')
    # streched_img.save('streched_histogram_leafs.png')

    equalized_img = equalize_img(streched_img)

    save_hist(equalized_img, 'after_equalizing')
    
    equalized_img = Image.fromarray(equalized_img).convert('L')
    equalized_img.save('equalized_histogram_leafs.png')
    

if __name__ == '__main__':
    main()

    