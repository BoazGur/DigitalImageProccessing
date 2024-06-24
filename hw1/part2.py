from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import numpy as np
import cv2

def conv2d(image, kernel):
        
    # padding size and kernel size
    k = kernel.shape[0]
    padding_width = k // 2
    
    # initialization and padding
    padded_image = pad(image, padding_width)
    conved_image = np.zeros(image.shape)

    # the convolution
    kernel_placements = [(i, i + image.shape[0], j, j + image.shape[1]) for i in range(k) for j in range(k)]
    for i_start, i_stop, j_start, j_stop in kernel_placements:
        conved_image += padded_image[i_start:i_stop, j_start:j_stop] * kernel[i_start, j_start]

    return conved_image

def pad(image, padding_width):
    # pad with necessary padding for proper convolution
    padded = np.zeros((image.shape[0] + padding_width * 2, image.shape[1] + padding_width * 2))
    padded[padding_width:-padding_width, padding_width:-padding_width] = image

    return padded

def calculate_amplitude_phase(path, factor=None, resize=False):
    # import image
    img = Image.open(path).convert('L')
    img = np.array(img)

    # crop the image if asked (useful for question 3d) 
    if resize:
        img = img[:factor[0],:factor[1]]
    
    # fft and shift
    fft = np.fft.fft2(img)
    fft_shifted = np.fft.fftshift(fft)

    # retirive amplitude and phase
    amplitude = np.abs(fft_shifted)
    phase = np.angle(fft_shifted)

    return amplitude, phase

def main():

    ######################### PART 2 - 2a ##################################

    # kernel of gradient
    derivation_kernel = np.array([[1, 0, -1], [0, 0, 0], [-1, 0, 1]])

    # load both images
    image1 = Image.open('hw12024_input_img/hw12024/I.jpg')
    image1 = ImageOps.grayscale(image1)
    
    image2 = Image.open('hw12024_input_img/hw12024/I_n.jpg')
    image2 = ImageOps.grayscale(image2)

    # convolve with the gradiant
    conved_img1 = conv2d(np.asarray(image1), derivation_kernel)
    conved_img2 = conv2d(np.asarray(image2), derivation_kernel)

    # export images
    conved_img1 = Image.fromarray(conved_img1).convert('L')
    conved_img1.save('answers/derived_I.jpg')

    conved_img2 = Image.fromarray(conved_img2).convert('L')
    conved_img2.save('answers/derived_I_n.jpg')

    ######################### PART 2 - 2b ##################################
    
    # import image
    I_n = Image.open('hw12024_input_img/hw12024/I_n.jpg')
    I_n = ImageOps.grayscale(I_n)

    # apply gaussian blur using 3x3 kernel
    I_dn = cv2.GaussianBlur(np.asarray(I_n), (3, 3), 0)

    # export image
    I_dn = Image.fromarray(I_dn).convert('L')
    I_dn.save('answers/I_dn.jpg')

    ######################### PART 2 - 2c ##################################

    # import image
    I_n = Image.open('hw12024_input_img/hw12024/I_n.jpg')
    I_n = ImageOps.grayscale(I_n)

    # apply sobel inx and y directions
    I_dn2_x = cv2.Sobel(np.asarray(I_n), cv2.CV_64F, 1, 0)
    I_dn2_y = cv2.Sobel(np.asarray(I_n), cv2.CV_64F, 0, 1)

    # normalize
    I_dn2_sqrt = np.sqrt(I_dn2_x ** 2 + I_dn2_y ** 2)
    I_dn2 = I_dn2_sqrt * 255 / I_dn2_sqrt.max()

    # export image
    I_dn2 = Image.fromarray(I_dn2).convert('L')
    I_dn2.save('answers/I_dn2.jpg')

    ######################### PART 2 - 3a ##################################

    # retrieve amplitude and phase of I.jpg
    amplitude_I, phase_I = calculate_amplitude_phase('hw12024_input_img/hw12024/I.jpg')

    # normalize
    amplitude_db_I = 20 * np.log10(amplitude_I + 1)
    phase_normalized_I = 255 * (phase_I + np.pi) / (2 * np.pi)

    # export images
    I_amp = Image.fromarray(amplitude_db_I).convert('L')
    I_amp.save('answers/I_amp.jpg')

    I_phase = Image.fromarray(phase_normalized_I).convert('L')
    I_phase.save('answers/I_phase.jpg')

    # retrieve amplitude and phase of I_n.jpg
    amplitude_In, phaseIn = calculate_amplitude_phase('hw12024_input_img/hw12024/I_n.jpg')

    # normalize
    amplitude_db_In = 20 * np.log10(amplitude_In + 1)
    phase_normalized_In = 255 * (phaseIn + np.pi) / (2 * np.pi)

    # export images
    I_n_amp = Image.fromarray(amplitude_db_In).convert('L')
    I_n_amp.save('answers/I_n_amp.jpg')

    I_n_phase = Image.fromarray(phase_normalized_In).convert('L')
    I_n_phase.save('answers/I_n_phase.jpg')
    
    ######################### PART 2 - 3b ##################################

    # calculate the subtraction of both amplitudes
    sub_amplitude = np.abs(amplitude_I - amplitude_In)
    sub_amplitude_db = 20 * np.log10(sub_amplitude + 1)
    
    # export image
    sub_amplitude_img = Image.fromarray(sub_amplitude_db).convert('L')
    sub_amplitude_img.save('answers/sub_amplitude.jpg')

    ######################### PART 2 - 3c ##################################

    # retrieve amplitude of chita and phase of zebra
    amplitude_chita, _ = calculate_amplitude_phase('hw12024_input_img/hw12024/chita.jpeg')
    _, phase_zebra = calculate_amplitude_phase('hw12024_input_img/hw12024/zebra.jpeg')

    # normalize
    amplitude_db_chita = 20 * np.log10(amplitude_chita + 1)
    phase_normalized_zebra = 255 * (phase_zebra + np.pi) / (2 * np.pi)

    # export chita amplitude and zebra phase
    chita_amp = Image.fromarray(amplitude_db_chita).convert('L')
    chita_amp.save('answers/chita_amp.jpg')

    zebra_phase = Image.fromarray(phase_normalized_zebra).convert('L')
    zebra_phase.save('answers/zebra_phase.jpg')

    ######################### PART 2 - 3d ##################################

    # retrieve amplitude of chita and phase of zebra and crop them to be the same size
    amplitude_chita, _ = calculate_amplitude_phase('hw12024_input_img/hw12024/chita.jpeg', (176, 225), True)
    _, phase_zebra = calculate_amplitude_phase('hw12024_input_img/hw12024/zebra.jpeg', (176, 225), True)

    # recover the image using inverse fft
    recovered_fft = amplitude_chita * np.exp(1j * phase_zebra)
    unshifted_fft = np.fft.ifftshift(recovered_fft)
    recovered_img = np.real(np.fft.ifft2(unshifted_fft)).astype(np.uint8)

    # export image
    recovered_img = Image.fromarray(recovered_img).convert('L')
    recovered_img.save('answers/recovered_zebra_chita.jpg')

    pass

if __name__ == '__main__':
    main()