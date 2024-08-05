# MIT License

# Copyright (c) 2024 Benjamin Steenkamer

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from multiprocessing import Pool
import itertools
import matplotlib.pyplot
import numpy as np

def cuda_generate(image_size, x_range, y_range):
    import cupy
    # FIXME
    kernel = cupy.ElementwiseKernel(
        "complex64 c, uint32 MAX_ITERATIONS",
        "float64 result",
        """
            thrust::complex<float> z = c;

            unsigned int iter;
            for (iter = 0; iter < MAX_ITERATIONS; iter++){
                if (!isfinite(z)){
                    break;
                }
                z = z * z + c;  // FIXME: This operation is not producing nan or infinite when it should
            }
            result = (double) iter / (double) MAX_ITERATIONS;
        """
        )

    image_array = np.zeros((image_size, image_size, 4))

    x_iter = cupy.asarray(cupy.arange(*x_range, 2 / image_size))
    y_iter = cupy.asarray(cupy.arange(*y_range, 2 / image_size))
    c = x_iter + 1j * y_iter
    c = c.astype(cupy.complex64)

    results = cupy.zeros(shape=image_array.shape[:-1], dtype=cupy.float64)

    kernel(c, MAX_ITERATIONS, results)  # Run kernel

    print(results)

    image_array[:, :] = COLOR_MAP(cupy.asnumpy(results[:, :]))

    return image_array

def cpu_worker(y_iter, x_iter):
    iy, y = y_iter
    ix, x = x_iter

    c = complex(x, y)
    z = c
    try:
        for i in range(MAX_ITERATIONS):
            if not np.isfinite(z):  # TODO timeit: Is this better than np.isnan()?
                break
            z = z**2 + c
    except OverflowError:
        pass

    return iy, ix, COLOR_MAP(i / MAX_ITERATIONS)


def cpu_generate(image_size, x_range, y_range):
    y_iter = enumerate(np.arange(*y_range, 2 / image_size))
    x_iter = enumerate(np.arange(*x_range, 2 / image_size))

    image_array = np.zeros((image_size, image_size, 4))

    with Pool() as p:
        result = p.starmap(cpu_worker, itertools.product(y_iter, x_iter))
        for r in result:
            image_array[r[0], r[1]] = r[2]

    return image_array

if __name__ == "__main__":
    COLOR_MAP = matplotlib.pyplot.get_cmap("cividis")   # Color scheme of image

    x_range = (-1.5, 0.5)
    y_range = (-1 , 1)

    IMAGE_SIZE = 800       # Overall X,Y image resolution
    MAX_ITERATIONS = 150   # How "crisp" the edges of the fractal will be

    image_array = cpu_generate(IMAGE_SIZE, x_range, y_range)
    # image_array = cuda_generate(IMAGE_SIZE, x_range, y_range)

    # Save image to file
    matplotlib.pyplot.imsave(f"{IMAGE_SIZE}x{IMAGE_SIZE}_{MAX_ITERATIONS}@{x_range},{y_range}.png", image_array)

    # Show image in new window
    # matplotlib.pyplot.imshow(image_array)
    # matplotlib.pyplot.show()
