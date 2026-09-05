use runmat_types::IntegerClass;

use super::contract::define_integer_conversion_builtin;

define_integer_conversion_builtin!(
    uint16_builtin,
    "uint16",
    IntegerClass::UInt16,
    "crate::builtins::math::elementwise::integer_conversions::uint16",
    runmat_builtins::UINT16_ERROR_INVALID_ARGUMENT,
    runmat_builtins::UINT16_ERROR_INVALID_INPUT,
    runmat_builtins::UINT16_ERROR_INTERNAL
);
