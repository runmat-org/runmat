#ifndef RUNMAT_RUNTIME_NATIVE_FFI_INTERFACE_H
#define RUNMAT_RUNTIME_NATIVE_FFI_INTERFACE_H

#include <stdint.h>

typedef struct fixture_record {
    int32_t left;
    int32_t right;
} fixture_record;

int32_t fixture_add(int32_t left, int32_t right);
void fixture_increment(int32_t *value);
int32_t fixture_record_sum(const fixture_record *value);
int32_t *fixture_borrowed_value(int32_t present);

#endif
