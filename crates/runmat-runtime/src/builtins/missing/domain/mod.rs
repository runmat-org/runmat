use runmat_builtins::{BuiltinErrorDescriptor, BuiltinExtensionDescriptor};
use runmat_macros::runtime_builtin;
use runmat_value::{
    CellArray, CharArray, IntValue, LogicalArray, NumericDType, ObjectInstance, StringArray,
    Tensor, Value,
};

use crate::builtins::common::{gpu_helpers, tensor as tensor_utils};
use crate::builtins::math::reduction::{mean, median, min, std as std_reduction, sum, var};
use crate::builtins::table::{
    is_tabular_object, select_rows, selected_row_names, table_from_columns_like, table_height,
    table_variable_names_from_object, table_variables, table_width,
};
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

mod detection;
mod entrypoints;
mod errors;
mod filling;
mod metadata;
mod moving;
mod numeric;
mod removal;
mod standardize;
mod text_output;

use errors::{gather_error, internal_error, invalid_argument, unsupported_type};
#[cfg(test)]
use metadata::MISSING_TEXT;
use runmat_builtins::{
    FILLMISSING_AGGREGATE_INTEGER_DATA_EXTENSION, FILLMISSING_INTEGER_DATA_EXTENSION,
    MOVMAD_GPU_LARGE_WINDOW_EXTENSION, NANMEAN_INTEGER_EXTENSION, NANMEDIAN_INTEGER_EXTENSION,
    NANMIN_INTEGER_EXTENSION, NANSTD_INTEGER_EXTENSION, NANSUM_INTEGER_EXTENSION,
    NANVAR_INTEGER_EXTENSION, RMMISSING_INTEGER_DIM_EXTENSION,
};
pub use runmat_builtins::{
    FILLMISSING_DESCRIPTOR, FILLMISSING_EXTENSIONS, FILLMISSING_INTEGER_CAPABILITIES,
    MOVMAD_EXTENSIONS, MOVMAD_INTEGER_CAPABILITIES, NANMEAN_EXTENSIONS,
    NANMEAN_INTEGER_CAPABILITIES, NANMEDIAN_EXTENSIONS, NANMEDIAN_INTEGER_CAPABILITIES,
    NANMIN_EXTENSIONS, NANMIN_INTEGER_CAPABILITIES, NANSTD_EXTENSIONS, NANSTD_INTEGER_CAPABILITIES,
    NANSUM_EXTENSIONS, NANSUM_INTEGER_CAPABILITIES, NANVAR_EXTENSIONS, NANVAR_INTEGER_CAPABILITIES,
    RMMISSING_DESCRIPTOR, RMMISSING_EXTENSIONS, RMMISSING_INTEGER_CAPABILITIES,
};
use runmat_builtins::{
    ISMISSING_RESIDENT_INPUT_EXTENSION, MISSING_SHAPED_ARRAY_EXTENSION,
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
