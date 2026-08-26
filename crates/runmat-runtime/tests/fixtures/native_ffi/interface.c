#include "interface.h"

int32_t fixture_add(int32_t left, int32_t right) {
    return left + right;
}

void fixture_increment(int32_t *value) {
    *value += 1;
}
