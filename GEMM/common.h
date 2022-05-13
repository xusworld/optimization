#pragma once

#include <iostream>
#include <vector>
#include <random>
#include <functional>

using namespace std;


#define A(i,j) a[ (j)*sa + (i) ]
#define B(i,j) b[ (j)*sb + (i) ]
#define C(i,j) c[ (j)*sc + (i) ]

int sa;
int sb;
int sc;
