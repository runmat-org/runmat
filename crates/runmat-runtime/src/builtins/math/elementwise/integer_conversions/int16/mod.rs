use runmat_types::IntegerClass;

use super::contract::define_integer_conversion_builtin;

define_integer_conversion_builtin!(
    int16_builtin,
    "int16",
    IntegerClass::Int16,
    "crate::builtins::math::elementwise::integer_conversions::int16",
    runmat_builtins::INT16_ERROR_INVALID_ARGUMENT,
    runmat_builtins::INT16_ERROR_INVALID_INPUT,
    runmat_builtins::INT16_ERROR_INTERNAL
);
