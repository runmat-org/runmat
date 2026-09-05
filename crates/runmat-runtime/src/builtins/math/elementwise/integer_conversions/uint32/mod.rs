use runmat_types::IntegerClass;

use super::contract::define_integer_conversion_builtin;

define_integer_conversion_builtin!(
    uint32_builtin,
    "uint32",
    IntegerClass::UInt32,
    "crate::builtins::math::elementwise::integer_conversions::uint32",
    runmat_builtins::UINT32_ERROR_INVALID_ARGUMENT,
    runmat_builtins::UINT32_ERROR_INVALID_INPUT,
    runmat_builtins::UINT32_ERROR_INTERNAL
);
