//! MATLAB-compatible missing-value construction, predicates, cleanup, and NaN-aware aliases.

mod domain;

use domain::ISMISSING_EXTENSIONS;

pub use domain::{
    ANYMISSING_DESCRIPTOR, ANYMISSING_INTEGER_CAPABILITIES, FILLMISSING_DESCRIPTOR,
    FILLMISSING_EXTENSIONS, FILLMISSING_INTEGER_CAPABILITIES, ISMISSING_DESCRIPTOR,
    ISMISSING_INTEGER_CAPABILITIES, MISSING_DESCRIPTOR, MISSING_EXTENSIONS,
    MISSING_INTEGER_CAPABILITIES, MOVMAD_EXTENSIONS, MOVMAD_INTEGER_CAPABILITIES,
    NANMEAN_EXTENSIONS, NANMEAN_INTEGER_CAPABILITIES, NANMEDIAN_EXTENSIONS,
    NANMEDIAN_INTEGER_CAPABILITIES, NANMIN_EXTENSIONS, NANMIN_INTEGER_CAPABILITIES,
    NANSTD_EXTENSIONS, NANSTD_INTEGER_CAPABILITIES, NANSUM_EXTENSIONS, NANSUM_INTEGER_CAPABILITIES,
    NANVAR_EXTENSIONS, NANVAR_INTEGER_CAPABILITIES, NAN_AWARE_DESCRIPTOR, RMMISSING_DESCRIPTOR,
    RMMISSING_EXTENSIONS, RMMISSING_INTEGER_CAPABILITIES, STANDARDIZE_MISSING_DESCRIPTOR,
    STANDARDIZE_MISSING_EXTENSIONS, STANDARDIZE_MISSING_INTEGER_CAPABILITIES,
};

#[cfg(target_arch = "wasm32")]
pub(crate) use domain::{
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
