mod cputime;
mod pause;
mod stopwatch;

pub use cputime::{CPUTIME_DESCRIPTOR, CPUTIME_ERROR_TOO_MANY_INPUTS, CPUTIME_ERROR_UNAVAILABLE};
pub use pause::{
    PAUSE_ERROR_CONTEXT_UNAVAILABLE, PAUSE_ERROR_GATHER_FAILED, PAUSE_ERROR_INVALID_ARG,
    PAUSE_ERROR_TOO_MANY_INPUTS, PAUSE_GPU_INPUT_EXTENSION,
};
pub use stopwatch::{
    TIC_ERROR_CONTEXT_UNAVAILABLE, TIC_ERROR_TOKEN_EXHAUSTED, TOC_ERROR_INVALID_HANDLE,
    TOC_ERROR_NO_MATCHING_TIC, TOC_ERROR_TOO_MANY_INPUTS,
};

pub(crate) fn extend_entries(values: &mut Vec<&'static crate::BuiltinCatalogEntry>) {
    values.extend_from_slice(cputime::ENTRIES);
    values.extend_from_slice(pause::ENTRIES);
    values.extend_from_slice(stopwatch::ENTRIES);
}
