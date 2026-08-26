#include "interface.h"

int32_t fixture_add(int32_t left, int32_t right) {
    return left + right;
}

void fixture_increment(int32_t *value) {
    *value += 1;
}

int32_t fixture_record_sum(const fixture_record *value) {
    return value->left + value->right;
}

int32_t *fixture_borrowed_value(int32_t present) {
    static int32_t value = 17;
    return present ? &value : 0;
}
