//! Provider-owned result validation, host fallback, and exact-owner restoration.

mod execution;
mod precision;
mod upload;

pub(crate) use execution::{
    gather_compute_restore, gather_compute_restore_protecting, gather_value_compute_restore,
    gather_value_compute_restore_protecting, validate_real_unary_provider_output,
    validate_real_unary_provider_output_protecting,
};
pub(crate) use precision::align_floating_value_precision;
pub(crate) use upload::{upload_value_like, upload_value_like_protected};

#[cfg(test)]
mod tests;
