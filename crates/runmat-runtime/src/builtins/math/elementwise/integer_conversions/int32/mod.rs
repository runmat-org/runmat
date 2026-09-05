use runmat_types::IntegerClass;

use super::contract::define_integer_conversion_builtin;

define_integer_conversion_builtin!(
    int32_builtin,
    "int32",
    IntegerClass::Int32,
    "crate::builtins::math::elementwise::integer_conversions::int32",
    runmat_builtins::INT32_ERROR_INVALID_ARGUMENT,
    runmat_builtins::INT32_ERROR_INVALID_INPUT,
    runmat_builtins::INT32_ERROR_INTERNAL
);
