//! Reusable provider mechanics for binary arithmetic.

mod broadcast;
mod dispatch;
mod expanded;
mod operation;
mod residency;
mod scalar;
mod validation;

pub(super) use broadcast::broadcast_repetitions;
pub(super) use dispatch::{try_host_left, try_host_right, try_pair};
pub(super) use expanded::ExpandedPair;
pub(super) use operation::ArithmeticProviderOperation;
pub(super) use residency::resident_output_from_sources;
pub(super) use scalar::{device_real_scalar, host_real_scalar};
pub(super) use validation::valid_real_binary_output;
