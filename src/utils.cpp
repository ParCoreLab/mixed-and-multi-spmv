#include "utils.hpp"

double relative_error(double x) { return fabs(((double)((float)(x)) - x)) / fabs(x); }

bool file_exists(char* filename) {
  struct stat buffer;
  return (stat(filename, &buffer) == 0);
}
