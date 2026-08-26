#include "interface.h"

double fixture_scale(fixture_record *record, double factor) {
    record->value *= factor;
    return record->value;
}

int32_t fixture_sum(const int32_t *values, uint32_t length) {
    int32_t total = 0;
    for (uint32_t index = 0; index < length; ++index) {
        total += values[index];
    }
    return total;
}

void fixture_increment(int32_t *value) {
    *value += 1;
}

fixture_record fixture_make(double value, uint32_t tag) {
    fixture_record result = {value, tag};
    return result;
}

int32_t fixture_apply(int32_t value, int32_t (*callback)(int32_t)) {
    return callback(value) + callback(value + 1);
}
