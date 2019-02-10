//
// Created by brian on 11/20/18.
//

#include "complex.cuh"
#include <cmath>


__device__ __host__ Complex::Complex() : real(0.0f), imag(0.0f) {}

__device__ __host__ Complex::Complex(float r) : real(r), imag(0.0f) {}

__device__ __host__ Complex::Complex(float r, float i) : real(r), imag(i) {}

__device__ __host__ Complex Complex::operator+(const Complex &b) const {
	Complex ans;
    ans.real = real + b.real;
    ans.imag = imag + b.imag;
    return ans;
}

__device__ __host__ Complex Complex::operator-(const Complex &b) const {
    Complex ans;
    ans.real = real - b.real;
    ans.imag = imag - b.imag;
    return ans;
}

__device__ __host__ Complex Complex::operator*(const Complex &b) const {
    Complex ans;
    ans.real = (real * b.real) - (imag * b.imag);
    ans.imag = (real * b.imag) + (imag * b.real);
    return ans;
}

__device__ __host__ float Complex::mag() const {
    return sqrt(real * real + imag * imag);
}

__device__ __host__ float Complex::angle() const {
    return atan(imag / real);
}

__device__ __host__ Complex Complex::conj() const {
	Complex conj;
	conj.real = real;
	conj.imag = - imag;
	return conj;
}

std::ostream& operator<< (std::ostream& os, const Complex& rhs) {
    Complex c(rhs);
    if(fabsf(rhs.imag) < 1e-10) c.imag = 0.0f;
    if(fabsf(rhs.real) < 1e-10) c.real = 0.0f;

    if(c.imag == 0) {
        os << c.real;
    }
    else {
        os << "(" << c.real << "," << c.imag << ")";
    }
    return os;
}