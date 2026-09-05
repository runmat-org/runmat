use runmat_types::IntegerClass;

use super::contract::define_integer_conversion_builtin;

define_integer_conversion_builtin!(
    uint8_builtin,
    "uint8",
    IntegerClass::UInt8,
    "crate::builtins::math::elementwise::integer_conversions::uint8",
    runmat_builtins::UINT8_ERROR_INVALID_ARGUMENT,
    runmat_builtins::UINT8_ERROR_INVALID_INPUT,
    runmat_builtins::UINT8_ERROR_INTERNAL
);
