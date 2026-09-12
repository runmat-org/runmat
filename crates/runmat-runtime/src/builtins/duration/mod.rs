mod builtins;
mod display;
mod metadata;
mod model;
mod prelude;

pub use display::{
    duration_char_array, duration_display_text, duration_string_array, duration_summary,
};
pub(crate) use metadata::DEFAULT_DURATION_FORMAT;
use metadata::*;
pub use metadata::{
    DURATION_BINARY_DESCRIPTOR, DURATION_DESCRIPTOR, DURATION_INTEGER_CAPABILITIES,
    DURATION_SUBSASGN_DESCRIPTOR, DURATION_SUBSREF_DESCRIPTOR,
};
pub use model::is_duration_object;
use model::*;
pub(crate) use model::{
    duration_format_from_value, duration_object_from_days_tensor,
    duration_tensor_from_duration_value,
};
use prelude::*;

#[cfg(test)]
mod tests;

#[cfg(target_arch = "wasm32")]
pub(crate) use builtins::{
    __runmat_wasm_register_builtin_days_builtin, __runmat_wasm_register_builtin_duration_builtin,
    __runmat_wasm_register_builtin_duration_eq, __runmat_wasm_register_builtin_duration_ge,
    __runmat_wasm_register_builtin_duration_gt, __runmat_wasm_register_builtin_duration_le,
    __runmat_wasm_register_builtin_duration_lt, __runmat_wasm_register_builtin_duration_minus,
    __runmat_wasm_register_builtin_duration_ne, __runmat_wasm_register_builtin_duration_plus,
    __runmat_wasm_register_builtin_duration_subsasgn,
    __runmat_wasm_register_builtin_duration_subsref, __runmat_wasm_register_builtin_hours_builtin,
    __runmat_wasm_register_builtin_isduration_builtin,
    __runmat_wasm_register_builtin_milliseconds_builtin,
    __runmat_wasm_register_builtin_minutes_builtin, __runmat_wasm_register_builtin_seconds_builtin,
    __runmat_wasm_register_builtin_years_builtin,
};
