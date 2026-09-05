//! Provider-owned result validation, host fallback, and exact-owner restoration.

mod execution;
mod precision;
mod upload;

pub(crate) use execution::{
    gather_compute_restore, gather_value_compute_restore, validate_real_unary_provider_output,
};
pub(crate) use precision::align_floating_value_precision;
pub(crate) use upload::{upload_value_like, upload_value_like_protected, upload_value_protected};

#[cfg(test)]
mod tests;
