#include <thread>
#include <cmath>
#include "stdio.h"
#include "complex.h"
#include "input_image.h"
//#include <chrono>


// Play with this to get best peformance (should be a power of 2)
#define NUM_THREADS 128
const float PI = 3.14159265358979f;


// Thread function declarations (s_ prefix means starting)
void row_dft(Complex* data, Complex* dft_data, int w, int s_h, int rows_per_thread);
void col_dft(Complex* data, Complex* dft_data, int s_w, int h, int cols_per_thread);


int main(int argc, char **argv) {
    //auto start = std::chrono::system_clock::now();
	
    // file names, foward/reverse
    char *in_file, *out_file;
    in_file = argv[2];
    out_file = argv[3];
    // If the command line asks for anything other than a foward dft, reject that junk
    if (argc != 4) {
        printf("Incorrect number of arguments\n");
        return -1;
    }

    // Get a image and then get the data/w/h from it
    InputImage img(in_file);
    int w = img.get_width();
    int h = img.get_height();
    Complex* data = img.get_image_data();
    Complex* dft_data = new Complex[w*h];


    // Array of threads
    std::thread thread_arr[NUM_THREADS];

    // Get number of rows/cols per thread (same thing if its square)
    int rows_per_thread = w/NUM_THREADS;
    int cols_per_thread = h/NUM_THREADS;

    // Launch threads for row
    for (int i = 0; i < NUM_THREADS; ++i) {
        thread_arr[i] = std::thread(row_dft, data, dft_data, w, i*rows_per_thread, rows_per_thread);
    }
    // Join threads for row_dft
    for (int i = 0; i < NUM_THREADS; ++i) {
        thread_arr[i].join();
    }

    // Launch threads for col_dft (swap dft_data and data so that the final data is stored in data)
    for (int i = 0; i < NUM_THREADS; ++i) {
        thread_arr[i] = std::thread(col_dft, dft_data, data, i*cols_per_thread, h, cols_per_thread );
    }
    // Join threads for col_dft
    for (int i = 0; i < NUM_THREADS; ++i) {
        thread_arr[i].join();
    }


    // save the file using given function
    img.save_image_data(out_file, data, w, h);
    

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

// Thread function definitions
void row_dft(Complex* data, Complex* dft_data, int w, int s_h, int rows_per_thread) {
    // Complex W used for calculations
    Complex H_n;

    // Outer row loop
    for (int i = s_h; i < s_h+rows_per_thread; ++i) {
        // Loop over this row and get H[n]
        for (int n = 0; n < w; ++n) {            
            // reset current
            H_n.real = 0;
            H_n.imag = 0;
            for (int k = 0; k < w; ++k)
                H_n = H_n + Complex(cos((2*PI*n*k)/w), sin((2*PI*n*k)/w))*data[i*w + k];

            dft_data[i*w + n] = H_n;
        }
    }
}

void col_dft(Complex* data, Complex* dft_data, int s_w, int h, int cols_per_thread) {
    // Complex W used for calculations
    Complex H_n;
    // Outer row loop
    for (int i = s_w; i < s_w+cols_per_thread; ++i) {
        // Loop over this row and get H[n]
        for (int n = 0; n < h; ++n) {
            // reset current
            H_n.real = 0;
            H_n.imag = 0;
            // Get the summation for this H[n]
            for (int k = 0; k < h; ++k)          
                H_n = H_n + Complex(cos((2*PI*n*k)/h), sin((2*PI*n*k)/h))*data[i + k*h];

            // Store the computed value in dft_data
            dft_data[i + n*h] = H_n;
        }
    }
}