from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import numpy as np
import cv2

def conv2d(image, kernel):
        
    # Padding to get original size
    k = kernel.shape[0]
    padding_width = k // 2
    
    padded_image = pad(image, padding_width)
    conved_image = np.zeros(image.shape)

    kernel_placements = [(i, i + image.shape[0], j, j + image.shape[1]) for i in range(k) for j in range(k)]
    for i_start, i_stop, j_start, j_stop in kernel_placements:
        conved_image += padded_image[i_start:i_stop, j_start:j_stop] * kernel[i_start, j_start]

    return conved_image

def pad(image, padding_width):
    padded = np.zeros((image.shape[0] + padding_width * 2, image.shape[1] + padding_width * 2))
    padded[padding_width:-padding_width, padding_width:-padding_width] = image

    return padded

def main():

    ######################### PART 2 - 2a ##################################

    # derivation_kernel = np.array([[1, 0, -1], [0, 0, 0], [-1, 0, 1]])
   
    # image1 = Image.open('./hw12024_input_img/hw12024/I.jpg')
    # image1 = ImageOps.grayscale(image1)
    
    # image2 = Image.open('./hw12024_input_img/hw12024/I_n.jpg')
    # image2 = ImageOps.grayscale(image2)

    # conved_img1 = conv2d(np.asarray(image1), derivation_kernel)
    # conved_img2 = conv2d(np.asarray(image2), derivation_kernel)

    # conved_img1 = Image.fromarray(conved_img1).convert('L')
    # conved_img1.save('derived_I.jpeg')

    # conved_img2 = Image.fromarray(conved_img2).convert('L')
    # conved_img2.save('derived_I_n.jpeg')

    ######################### PART 2 - 2b ##################################
 
    # I_n = Image.open('./hw12024_input_img/hw12024/I_n.jpg')
    # I_n = ImageOps.grayscale(I_n)

    # I_dn = cv2.GaussianBlur(np.asarray(I_n), (3, 3), 0)

    # I_dn = Image.fromarray(I_dn).convert('L')
    # I_dn.save('I_dn.jpg')

 ######################### PART 2 - 2c ##################################

    # I_n = Image.open('./hw12024_input_img/hw12024/I_n.jpg')
    # I_n = ImageOps.grayscale(I_n)

    # I_dn2_x = cv2.Sobel(np.asarray(I_n), cv2.CV_64F, 1, 0)
    # I_dn2_y = cv2.Sobel(np.asarray(I_n), cv2.CV_64F, 0, 1)

    # I_dn2_sqrt = np.sqrt(I_dn2_x ** 2 + I_dn2_y ** 2)
    # I_dn2 = I_dn2_sqrt*255/I_dn2_sqrt.max()

    # I_dn2 = Image.fromarray(I_dn2).convert('L')
    # I_dn2.save('I_dn2.jpg')

 ######################### PART 2 - 3a ##################################

    I_n = Image.open('./hw12024_input_img/hw12024/I_n.jpg')
    I_n = ImageOps.grayscale(I_n)

    I_n_fft = np.fft.fft2(np.asarray(I_n))
    
    amplitude = np.abs(I_n_fft)

    amplitude = (255 * (amplitude - np.min(amplitude)) / (np.max(amplitude) - np.min(amplitude))).astype(np.uint32)

    print(amplitude)
    
    phase = np.angle(I_n_fft)

    I_n_amp = Image.fromarray(amplitude).convert('L')
    I_n_amp.save('I_n_amp.jpg')

    # 
 


if __name__ == '__main__':
    main()