use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    ResolveContext, Type,
};
use runmat_macros::runtime_builtin;
use runmat_value::{
    CellArray, CharArray, IntValue, LogicalArray, NumericDType, ObjectInstance, StringArray,
    StructValue, Tensor, Value,
};

use crate::builtins::common::{gpu_helpers, tensor as tensor_utils};
use crate::builtins::math::reduction::{mean, median, min, std as std_reduction, sum, var};
use crate::builtins::table::{
    is_tabular_object, select_rows, selected_row_names, table_from_columns_like, table_height,
    table_variable_names_from_object, table_variables, table_width,
};
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

use self::metadata::core::MISSING_ERRORS;

macro_rules! descriptor {
    ($name:ident, $signatures:ident, $mode:expr) => {
        pub const $name: BuiltinDescriptor = BuiltinDescriptor {
            signatures: &$signatures,
            output_mode: $mode,
            completion_policy: BuiltinCompletionPolicy::Public,
            errors: &MISSING_ERRORS,
        };
    };
}

mod detection;
mod entrypoints;
mod filling;
mod metadata;
mod moving;
mod numeric;
mod removal;
mod standardize;

pub use metadata::cleanup::{
    FILLMISSING_DESCRIPTOR, FILLMISSING_EXTENSIONS, FILLMISSING_INTEGER_CAPABILITIES,
    RMMISSING_DESCRIPTOR, RMMISSING_EXTENSIONS, RMMISSING_INTEGER_CAPABILITIES,
    STANDARDIZE_MISSING_DESCRIPTOR,
};
pub use metadata::core::{
    ANYMISSING_DESCRIPTOR, ANYMISSING_INTEGER_CAPABILITIES, ISMISSING_DESCRIPTOR,
    ISMISSING_INTEGER_CAPABILITIES, MISSING_DESCRIPTOR, MISSING_EXTENSIONS,
    MISSING_INTEGER_CAPABILITIES, STANDARDIZE_MISSING_EXTENSIONS,
    STANDARDIZE_MISSING_INTEGER_CAPABILITIES,
};
pub use metadata::nan::{
    MOVMAD_EXTENSIONS, MOVMAD_INTEGER_CAPABILITIES, NANMEAN_EXTENSIONS,
    NANMEAN_INTEGER_CAPABILITIES, NANMEDIAN_EXTENSIONS, NANMEDIAN_INTEGER_CAPABILITIES,
    NANMIN_EXTENSIONS, NANMIN_INTEGER_CAPABILITIES, NANSTD_EXTENSIONS, NANSTD_INTEGER_CAPABILITIES,
    NANSUM_EXTENSIONS, NANSUM_INTEGER_CAPABILITIES, NANVAR_EXTENSIONS, NANVAR_INTEGER_CAPABILITIES,
    NAN_AWARE_DESCRIPTOR,
};
pub(super) use metadata::ISMISSING_EXTENSIONS;
use metadata::{
    any_type, internal_error, invalid_argument, logical_type, unsupported_type,
    FILLMISSING_AGGREGATE_INTEGER_DATA_EXTENSION, FILLMISSING_INTEGER_DATA_EXTENSION,
    ISMISSING_RESIDENT_INPUT_EXTENSION, MISSING_SHAPED_ARRAY_EXTENSION, MISSING_TEXT,
    MOVMAD_GPU_LARGE_WINDOW_EXTENSION, NANMEAN_INTEGER_EXTENSION, NANMEDIAN_INTEGER_EXTENSION,
    NANMIN_INTEGER_EXTENSION, NANSTD_INTEGER_CONTROL_EXTENSION, NANSUM_INTEGER_EXTENSION,
    NANVAR_INTEGER_CONTROL_EXTENSION, RMMISSING_INTEGER_DIM_EXTENSION,
    STANDARDIZE_MISSING_EXPLICIT_GPU_INDICATOR_EXTENSION,
    STANDARDIZE_MISSING_INTEGER_DATA_EXTENSION,
};

#[cfg(test)]
mod tests;

#[cfg(target_arch = "wasm32")]
pub(crate) use entrypoints::{
    __runmat_wasm_register_builtin_anymissing_builtin,
    __runmat_wasm_register_builtin_fillmissing_builtin,
    __runmat_wasm_register_builtin_ismissing_builtin,
    __runmat_wasm_register_builtin_missing_builtin, __runmat_wasm_register_builtin_movmad_builtin,
    __runmat_wasm_register_builtin_nanmean_builtin,
    __runmat_wasm_register_builtin_nanmedian_builtin,
    __runmat_wasm_register_builtin_nanmin_builtin, __runmat_wasm_register_builtin_nanstd_builtin,
    __runmat_wasm_register_builtin_nansum_builtin, __runmat_wasm_register_builtin_nanvar_builtin,
    __runmat_wasm_register_builtin_rmmissing_builtin,
    __runmat_wasm_register_builtin_standardize_missing_builtin,
};
