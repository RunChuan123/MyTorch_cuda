#pragma once
#ifndef RAND_H
#define RAND_H

#include <math.h>

#define MERSENNE_STATE_M 397u
#define MERSENNE_STATE_N 624u

#define LMASK 0x7ffffffful
#define UMASK 0x80000000ul

// Copyright(c) Makoto Matsumoto and Takuji Nishimura

// This implementation follows PyTorch so that we are numerically identical when running verification tests.

typedef struct {
    unsigned long long seed_;
    int left_;
    unsigned int next_;
    unsigned int state_[MERSENNE_STATE_N];
    unsigned int MATRIX_A[2];
} mt19937_state;

void manual_seed(mt19937_state* state, unsigned int seed);

void next_state(mt19937_state* state);

unsigned int randint32(mt19937_state* state) ;

inline unsigned long long randint64(mt19937_state* state) {
    return (((unsigned long long)(randint32(state)) << 32) | randint32(state));
}

inline float randfloat32(mt19937_state* state) {
    return (randint32(state) & ((1ull << 24) - 1)) * (1.0f / (1ull << 24));
}

inline double randfloat64(mt19937_state* state) {
    return (randint64(state) & ((1ull << 53) - 1)) * (1.0 / (1ull << 53));
}

void uniform_(float* data, unsigned int numel, float from, float to, mt19937_state* state) ;

// Box-Muller transform: maps uniform random numbers to Gaussian distributed numbers
// https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
void normal_fill_16(float* data, float mean, float std) ;

void normal_fill(float* data, unsigned int numel, float mean, float std, mt19937_state* state);

void normal_(float* data, unsigned int numel, float mean, float std, mt19937_state* state) ;

void init_identity_permutation(int *data, int numel) ;

void random_permutation(int* data, int numel, mt19937_state* state);
#endif