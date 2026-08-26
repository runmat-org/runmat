#ifndef RUNMAT_NATIVE_FFI_TEST_INTERFACE_H
#define RUNMAT_NATIVE_FFI_TEST_INTERFACE_H

#include <stdint.h>

typedef struct fixture_record {
    double value;
    uint32_t tag;
} fixture_record;

typedef enum fixture_mode {
    FIXTURE_MODE_DIRECT = 0,
    FIXTURE_MODE_SCALED = 4
} fixture_mode;

double fixture_scale(fixture_record *record, double factor);
uint32_t fixture_tag(const fixture_record *record);
int32_t fixture_sum(const int32_t *values, uint32_t length);
void fixture_increment(int32_t *value);
fixture_record fixture_make(double value, uint32_t tag);
int32_t fixture_apply(int32_t value, int32_t (*callback)(int32_t));
int32_t *fixture_borrowed_value(int32_t present);

#endif
