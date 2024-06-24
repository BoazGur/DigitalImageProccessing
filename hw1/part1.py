from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import numpy as np

def bilinear_interpolation(img):
    # initialize parameters
    n, m = img.shape
    interpolated_pic = np.zeros((2 * n, 2 * m))

    # fill original image into inetrpolated image at even places
    for i in range(n):
        for j in range(m):
            interpolated_pic[2 * i, 2 * j] = img[i,j]

    # fill all the odd places in even rows
    for i in range(0, 2 * n, 2):
        for j in range(1, 2 * m, 2):
            if j == 2 * m - 1:
                interpolated_pic[i, j] = interpolated_pic[i, j - 1]
            else:
                interpolated_pic[i, j] = np.mean([interpolated_pic[i, j - 1], interpolated_pic[i, j + 1]])

    # fill all the odd rows
    for j in range(0, 2 * m):  
        for i in range(1, 2 * n, 2):
            if i == 2 * n - 1:
                interpolated_pic[i, j] = interpolated_pic[i - 1, j]
            else:
                interpolated_pic[i, j] = np.mean([interpolated_pic[i - 1, j], interpolated_pic[i + 1, j]])

    return interpolated_pic

def save_hist(img, name):
    # flatten img for use of plt.hist
    flatten_img = img.flatten()

    # plot the histogram
    plt.hist(flatten_img, bins=255)
    plt.title(f'{name} histogram')
    plt.xlabel('Intensity')
    plt.ylabel('Frequency')
    plt.savefig('answers/' + name + '.png')
    
    # clear figure for later use
    plt.clf()

def strech_hist(img):
    min_freq = np.min(img)
    max_freq = np.max(img)

    # strech the image's histogram
    return 255 * ((img - min_freq) / (max_freq - min_freq))

def equalize_hist(img):
    # get histogram
    n, m = img.shape
    hist = np.histogram(img, bins=256)[0]
    
    # get the cdf of the histogram
    probability = hist / (n * m)
    cdf = np.cumsum(probability)
    
    # equalize histogram
    return np.round(255 * cdf)

def equalize_img(img):
    # equalize histogram
    equalized_hist = equalize_hist(img)
    
    # create a lambda function that uses the equalized histogram to equalize the image
    equalized_img_lambda = lambda x: equalized_hist[x.astype(int)]

    # equalize the image
    return equalized_img_lambda(img)

def main():
    ######################### PART 1 - 1 ##################################
    
    # open peppers photo into greyscale image
    img = Image.open('hw12024_input_img/hw12024/peppers.jpg')
    img = ImageOps.grayscale(img)

    # interpolate as asked
    interpolated_peppers = bilinear_interpolation(np.asarray(img))
    
    # save X2 super resolution image
    image = Image.fromarray(interpolated_peppers).convert('L')
    image.save('answers/InterpolatedPeppersX2.jpg')

    # interpolate to get X8 resolution image
    interpolated_peppers = bilinear_interpolation(interpolated_peppers)
    interpolated_peppers = bilinear_interpolation(interpolated_peppers)

    # save
    image = Image.fromarray(interpolated_peppers).convert('L')
    image.save('answers/InterpolatedPeppersX8.jpg')

    ######################### PART 1 - 2 ##################################
    
    # open leafs image
    img = Image.open('hw12024_input_img/hw12024/leafs.jpg')
    img = ImageOps.grayscale(img)

    # plot and save histogram
    save_hist(np.asarray(img), 'before_flattening')

    # strech histogram
    streched_img = strech_hist(np.asarray(img))
    
    # plot and save streched histogram
    save_hist(streched_img, 'after_streching')
    
    # save image after streching histogram
    pil_streched_img = Image.fromarray(streched_img).convert('L')
    pil_streched_img.save('answers/streched_histogram_leafs.png')

    # equalize image after streching
    equalized_img = equalize_img(streched_img)

    # plot the histogram of the image
    save_hist(equalized_img, 'after_equalizing')
    
    # save the equalized image
    equalized_img = Image.fromarray(equalized_img).convert('L')
    equalized_img.save('answers/equalized_histogram_leafs.png')
    

if __name__ == '__main__':
    main()

    