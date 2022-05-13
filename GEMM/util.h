#include <iostream>
#include <random>
#include <cstdlib>
#include <sys/time.h>
#include <time.h>

double* RandomMatrix(int m, int n) {
  srand(42);
  double* matrix = (double*)malloc(sizeof(double) * m * n);

  for(auto i = 0; i < m; ++i) {
    for(auto j = 0; j < n; ++j) {
      matrix[i*n + j] = (double)(rand());
    }
  }
  return matrix;
}

double time_diff_ms(struct timeval *start, struct timeval *end) {
    auto s = (end->tv_sec - start->tv_sec) + 1e-6*(end->tv_usec - start->tv_usec);
    return s * 1000;
}

double benchmark(std::function<void(void)> func) {
  auto warmup_times = 20;
  auto run_times = 20;

  // warmup
  for (auto i = 0; i < warmup_times; ++i) func();

  struct timeval tv_start, tv_end;

  gettimeofday(&tv_start,NULL);
  for (auto i = 0; i < run_times; ++i) func();
  gettimeofday(&tv_end,NULL);

  return time_diff_ms(&tv_start, &tv_end) / run_times;
}

