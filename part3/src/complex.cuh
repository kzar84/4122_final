//
// Created by brian on 11/20/18.
//

#pragma once

#include <iostream>
#include <cuda.h>

class Complex {
public:
    __device__ __host__ Complex();
    __device__ __host__ Complex(float r, float i);
    __device__ __host__ Complex(float r);
    __device__ __host__ Complex operator+(const Complex& b) const;
    __device__ __host__ Complex operator-(const Complex& b) const;
    __device__ __host__ Complex operator*(const Complex& b) const;

    __device__ __host__ float mag() const;
    __device__ __host__ float angle() const;
    __device__ __host__ Complex conj() const;

    float real;
    float imag;
};

std::ostream& operator<<(std::ostream& os, const Complex& rhs);