#pragma once
#include "complex.h"

const float PI = 3.14159265358979f;

void fft(Complex*, int);
// gives the reversed bit index
unsigned int reverse_bit(unsigned int, int);