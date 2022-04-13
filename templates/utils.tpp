#include "utils.hpp"

/**
 * @brief Generate a random number in range [min, max].
 *
 * @tparam T type of the number
 * @param max
 * @param min
 * @return randomly generated number
 */
template <typename T>
T random_value(T max, T min) {
  return (T)((((double)rand() / RAND_MAX) * (double)(max - min)) + (double)(min));
}

/**
 * @brief Write a value to the array.
 *
 * @tparam T type of the values in the array
 * @param vec array
 * @param N size of the array
 * @param val value to set
 */
template <typename T>
void write_vector(T *vec, int N, T val) {
  for (int i = 0; i < N; i++) {
    vec[i] = val;
  }
}

/**
 * @brief Write uniformly distributed random values to the array.
 *
 * @tparam T type of the values in the array
 * @param vec array
 * @param N size of the array
 * @param min minimum value in distribution
 * @param max maximum value in distribution
 */
template <typename T>
void write_vector_random(T *vec, int N, T min, T max) {
  assert(min <= max);
  for (int i = 0; i < N; i++) {
    vec[i] = random_value<T>(min, max);
  }
}

/**
 * @brief Copy a vector, with type-casting.
 *
 * @tparam TSRC source type
 * @tparam TDEST destination type
 * @param src source array
 * @param dest destination array
 * @param N size of the array
 */
template <typename TSRC, typename TDEST>
void transfer_vector(TSRC *src, TDEST *dest, int N) {
  for (int i = 0; i < N; i++) {
    dest[i] = src[i];
  }
}