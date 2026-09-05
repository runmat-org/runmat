use runmat_types::IntegerClass;

use super::contract::define_integer_conversion_builtin;

define_integer_conversion_builtin!(
    int64_builtin,
    "int64",
    IntegerClass::Int64,
    "crate::builtins::math::elementwise::integer_conversions::int64",
    runmat_builtins::INT64_ERROR_INVALID_ARGUMENT,
    runmat_builtins::INT64_ERROR_INVALID_INPUT,
    runmat_builtins::INT64_ERROR_INTERNAL
);
