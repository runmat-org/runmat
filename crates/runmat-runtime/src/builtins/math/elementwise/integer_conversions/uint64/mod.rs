use runmat_types::IntegerClass;

use super::contract::define_integer_conversion_builtin;

define_integer_conversion_builtin!(
    uint64_builtin,
    "uint64",
    IntegerClass::UInt64,
    "crate::builtins::math::elementwise::integer_conversions::uint64",
    runmat_builtins::UINT64_ERROR_INVALID_ARGUMENT,
    runmat_builtins::UINT64_ERROR_INVALID_INPUT,
    runmat_builtins::UINT64_ERROR_INTERNAL
);
