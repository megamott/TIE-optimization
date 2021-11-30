import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from mpl_toolkits.axes_grid1 import make_axes_locatable

matrix_height = 500 + 1
matrix_width = 500 + 1
px_size = 5 * 10 ** -3  # mm
nu_min = 0
nu_max = 100  # mm
nu_descr = 1 / px_size

nu_h_1d_grid = np.arange(nu_min, nu_max + 1, nu_max / 5) / nu_descr  # нормированная частотная сетка
H = 1 - np.sin(np.pi * nu_h_1d_grid)  # передаточная функция цифрового фильтра

# Определение коэффициентов фильтра
filter_matrix = np.array([[2 * np.cos(i * 2 * np.pi * nu_0) for i in range(1, 5 + 1, 1)] for nu_0 in nu_h_1d_grid])
filter_matrix = np.c_[np.ones(6), filter_matrix]
h = np.matmul(H, np.linalg.inv(filter_matrix))

h_1d = np.concatenate((h, h[4::-1]))  # сим метричный одномерный фильтр

h0, h1, h2, h3, h4, h5 = h
h_2d = np.array([
    [h5, h5, h5, h5, h5, h5, h5, h5, h5, h5, h5],
    [h5, h4, h4, h4, h4, h4, h4, h4, h4, h4, h5],
    [h5, h4, h3, h3, h3, h3, h3, h3, h3, h4, h5],
    [h5, h4, h3, h2, h2, h2, h2, h2, h3, h4, h5],
    [h5, h4, h3, h2, h1, h1, h1, h2, h3, h4, h5],
    [h5, h4, h3, h2, h1, h0, h1, h2, h3, h4, h5],
    [h5, h4, h3, h2, h1, h1, h1, h2, h3, h4, h5],
    [h5, h4, h3, h2, h2, h2, h2, h2, h3, h4, h5],
    [h5, h4, h3, h3, h3, h3, h3, h3, h3, h4, h5],
    [h5, h4, h4, h4, h4, h4, h4, h4, h4, h4, h5],
    [h5, h5, h5, h5, h5, h5, h5, h5, h5, h5, h5]
])  # симметричный двумерный фильтр


# Задам тестовый сигнал
def triangle(z: np.ndarray, a: int = 0, b: int = 2, c: int = 4):
    z[z == a] = 0
    z[z == c] = 0
    first_half = np.logical_and(a < z, z <= b)
    z[first_half] = (z[first_half] - a) / (b - a)
    second_half = np.logical_and(b < z, z < c)
    z[second_half] = (c - z[second_half]) / (c - b)


triangle_slice = np.arange(0, matrix_width, dtype=np.float64)
triangle(triangle_slice, 69, 115, 161)
triangle(triangle_slice, 207, 253, 299)
triangle(triangle_slice, 345, 391, 437)
triangle_slice[0:69] = 0
triangle_slice[161:207] = 0
triangle_slice[299:345] = 0
triangle_slice[437:matrix_width] = 0
normalized_triangle_image = (
        (triangle_slice - triangle_slice.min()) * (1 / (triangle_slice.max() - triangle_slice.min()) * 255)
).astype('uint8')  # нормировка
normalized_triangle_image = np.tile(normalized_triangle_image, (matrix_height, 1))  # двумерный тестовый сигнал

final_image = convolve2d(normalized_triangle_image, h_2d)
final_image = ((final_image - final_image.min()) * (1 / (final_image.max() - final_image.min()) * 255)).astype('uint8')

# визуализация
fig = plt.figure()
ax1, ax2 = fig.add_subplot(1, 2, 1), fig.add_subplot(1, 2, 2)

ax1.title.set_text('исходное изображение')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
im1 = ax1.imshow(normalized_triangle_image, cmap='gray')

divider1 = make_axes_locatable(ax1)

ax2.title.set_text('обработанное изображение')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
im2 = ax2.imshow(final_image, cmap='gray')

divider2 = make_axes_locatable(ax2)
cax2 = divider2.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im2, cax=cax2, orientation='vertical')
fig.show()
