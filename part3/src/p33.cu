#include "complex.cuh"
#include "input_image.cuh"
#include "math.h"
//#include <chrono>

#define NUM_THREADS 128
const float PI = 3.14159265358979f;

// Device functions that compute the dft for row and column respectively
__global__ void row_dft(Complex* data, Complex* dft_data, int w, float PI);
__global__ void col_dft(Complex* data, Complex* dft_data, int h, float PI);


int main(int argc, char *argv[]) {
    //auto start = std::chrono::system_clock::now();

    // file names, foward/reverse
    char *in_file, *out_file;
    in_file = argv[2];
    out_file = argv[3];
    // If the command line asks for anything other than a foward dft, reject it
    if (argc != 4) {
        printf("Incorrect number of arguments\n");
        return -1;
    }

    // Get a image and then get the data/w/h from it
    InputImage img(in_file);
    int w = img.get_width();
    int h = img.get_height();
    Complex* data = img.get_image_data();

    // Size and number of blocks
    int size = w*h*sizeof(Complex);
    int NUM_BLOCKS = w/NUM_THREADS;

    // Allocate some device mem
    Complex* d_row_data; 
    Complex* d_col_data;
    cudaMalloc((void**)&d_row_data, size);
    cudaMalloc((void**)&d_col_data, size);

    // Upload host data to device
    cudaMemcpy(d_row_data, data, size, cudaMemcpyHostToDevice);
    // Call device function (updates grid for this timestep)
    row_dft<<<NUM_BLOCKS, NUM_THREADS>>>(d_row_data, d_col_data, w, PI);
    // Call device function (updates grid for this timestep)
    col_dft<<<NUM_BLOCKS, NUM_THREADS>>>(d_row_data, d_col_data, h, PI);
    // Copy device grid to host
    cudaMemcpy(data, d_row_data, size, cudaMemcpyDeviceToHost);

    // save the file using given function
    img.save_image_data(out_file, data, w, h);

    // Free up that memory
    free(data);
    cudaFree(d_row_data);
    cudaFree(d_col_data);
	
    //auto end = std::chrono::system_clock::now();
    //std::chrono::duration<double> elapsed_seconds = end-start;
    //std::cout << elapsed_seconds.count() << std::endl;


    return 0;
}


/**************************************************************************
    * How to compute 1D fouier transform
        - H[n] = h[k] * W^*(n*k) from k = 0 to k = N-1
        - W = cos(2pi/N) - jsin(2pi/N)
    
    * How to compute 2D fourier transform
    - Compute 1D transform for all rows
    - Compute 1D transfrom for all columns (using inputs from first step)
    - Enjoy the glorius dft that you have performed
***************************************************************************/

__global__ void row_dft(Complex* row_data, Complex* col_data, int w, float PI) {
    // Get the current index in the device array
    int row = threadIdx.x + blockIdx.x * blockDim.x;   
    
    Complex H_n;
    // Loop over all elements in row
    for (int n = 0; n < w; ++n) {
        // reset current
        H_n.real = 0;
        H_n.imag = 0;
        for (int k = 0; k < w; ++k)
            H_n = H_n + Complex(cos((2*PI*n*k)/w), sin((2*PI*n*k)/w))*row_data[row*w + k];

        col_data[row*w + n] = H_n;
    }

}
__global__ void col_dft(Complex* row_data, Complex* col_data, int h, float PI) {
    // Get the current index in the device array
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    
    // Complex W used for calculations
    Complex H_n;

    // Loop over this row and get H[n]
    for (int n = 0; n < h; ++n) {
        // reset current
        H_n.real = 0;
        H_n.imag = 0;
        // Get the summation for this H[n]
        for (int k = 0; k < h; ++k)          
            H_n = H_n + Complex(cos((2*PI*n*k)/h), sin((2*PI*n*k)/h))*col_data[col + k*h];

        // Store the computed value in dft_data
        row_data[col + n*h] = H_n;
    }
}