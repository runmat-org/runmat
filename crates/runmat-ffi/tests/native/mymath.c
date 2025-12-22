/**
 * Test native library for runmat-ffi testing.
 *
 * Build:
 *   Windows (MSYS2): gcc -shared -o mymath.dll mymath.c
 *   Linux: gcc -shared -fPIC -o libmymath.so mymath.c
 *   macOS: clang -shared -o libmymath.dylib mymath.c
 */

#include <stddef.h>

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

/* Scalar functions */

EXPORT double add(double x, double y) {
    return x + y;
}

EXPORT double subtract(double x, double y) {
    return x - y;
}

EXPORT double multiply(double x, double y) {
    return x * y;
}

EXPORT double divide(double x, double y) {
    if (y == 0.0) return 0.0;
    return x / y;
}

EXPORT double square(double x) {
    return x * x;
}

EXPORT double cube(double x) {
    return x * x * x;
}

EXPORT double negate(double x) {
    return -x;
}

EXPORT double get_pi(void) {
    return 3.14159265358979323846;
}

EXPORT double sum3(double a, double b, double c) {
    return a + b + c;
}

EXPORT double sum4(double a, double b, double c, double d) {
    return a + b + c + d;
}

EXPORT double sum5(double a, double b, double c, double d, double e) {
    return a + b + c + d + e;
}

/* Array function example */

EXPORT int scale_array(
    const double* input, size_t rows, size_t cols,
    double* output, size_t* out_rows, size_t* out_cols,
    double factor
) {
    size_t n = rows * cols;
    for (size_t i = 0; i < n; i++) {
        output[i] = input[i] * factor;
    }
    *out_rows = rows;
    *out_cols = cols;
    return 0;  /* success */
}
