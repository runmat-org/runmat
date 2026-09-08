pub(in crate::catalog) mod arrayfun;
mod documentation;
mod gpu_array;
mod transfer;

pub use arrayfun::{
    arrayfun_gpu_callback, ArrayfunGpuBinary, ArrayfunGpuCallback, ArrayfunGpuUnary,
    ARRAYFUN_CATALOG_ENTRY, ARRAYFUN_DESCRIPTOR, ARRAYFUN_ERROR_CALLBACK_FAILED,
    ARRAYFUN_ERROR_INTERNAL, ARRAYFUN_ERROR_INVALID_INPUT, ARRAYFUN_ERROR_UNDEFINED_FUNCTION,
    ARRAYFUN_ERROR_UNIFORM_OUTPUT_OPTION, ARRAYFUN_ERROR_UNIFORM_OUTPUT_TYPE, ARRAYFUN_EXTENSIONS,
    ARRAYFUN_GPU_OPTIONS_EXTENSION, ARRAYFUN_HOST_SCALAR_EXPANSION_EXTENSION,
    ARRAYFUN_INTEGER_CAPABILITIES, ARRAYFUN_TEXT_CALLABLE_EXTENSION,
};
pub use gpu_array::*;
pub use transfer::*;

pub(super) fn extend_entries(entries: &mut Vec<&'static crate::BuiltinCatalogEntry>) {
    super::extend_groups(
        entries,
        &[arrayfun::ENTRIES, gpu_array::ENTRIES, transfer::ENTRIES],
    );
}
