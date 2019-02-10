#include <iostream>
#include <cmath>
#include "fft.h"

void fft(Complex *input, int length)
{
    int rev, logLength;
    logLength = static_cast<int>(std::log2(length));

    // bit reverse every element and swap
    // needed to avoid recursion
    for (unsigned int i = 0; i < length; ++i)
    {
        rev = reverse_bit(i, logLength);
        if (rev > i)
            std::swap(input[rev], input[i]);
    }

    // start of fft calculation
    // goes through every fft level from bottom up
    // loops through every butterfly starting location and calculates the butterfly
    // s: fft tree level, m: butterfly size,
    // k: ind of butterfly, j: each element of butterfly, ind: speed up calc of ind
    int s, m, k, j, ind;
    Complex i(0.0f, 1.0f);

    // w: 1, wm: roots of unity
    for (s = 1; s <= logLength; ++s)
    {
        m = 1 << s;
        Complex wm(std::cos(2 * PI / m), -1 * std::sin(2 * PI / m));
        for (k = 0; k < length; k += m)
        {
            Complex w(1.0f, 0.0f);
            for (j = 0; j < (m >> 1); ++j)
            {
                ind = k + j;
                Complex t = w * input[ind + (m >> 1)];
                Complex u = input[ind];
                input[ind] = u + t;
                input[ind + (m >> 1)] = u - t;
                w = w * wm;
            }
        }
    }
}

unsigned int reverse_bit(unsigned int in, int logLength)
{
    unsigned int rev = 0;
    for (int i = 0; i < logLength; ++i)
    {
        rev <<= 1;
        rev |= in & 1;
        in >>= 1;
    }
    return rev;
}