from PIL import Image
import numpy as np

def bilinear_interpolation(img):
    n,m = np.shape(img)
    interpolated_pic = np.zeros(2*n, 2*m)
    
    for i in range(n):
        for j in range(m):
            interpolated_pic[2*i, 2*j] = img[i,j]

    for i in range(0, 2*n, 2):
        for j in range(1, 2*m, 2):
            if j == 2*m - 1:
                interpolated_pic[i, j] = interpolated_pic[i, j - 1]
            else:
                interpolated_pic[i, j] = np.mean([img[i, j - 1], img[i, j + 1]])

    for j in range(0, 2*m):  
        for i in range(1, 2*n, 2):
            if i == 2*n - 1:
                interpolated_pic[i, j] = interpolated_pic[i - 1, j]
            else:
                interpolated_pic[i, j] = np.mean([img[i - 1, j], img[i + 1, j]])

    return interpolated_pic


def main():
    img = Image.open('./hw12024_input_img/hw12024/peppers.jpg')
    interpolated_peppers = bilinear_interpolation(np.asarray(img))
    Image.show(interpolated_peppers)

if '__name__' == '__main__':
    main()

    