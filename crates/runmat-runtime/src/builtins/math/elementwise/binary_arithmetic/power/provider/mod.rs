//! Provider-resident dispatch for element-wise power.

mod conversion;
pub(super) mod admission;
mod pair;
mod resident;

pub(super) use pair::{power_gpu_pair, publish_pair, typed_provider_error};
pub(super) use resident::{power_gpu_host_left, power_gpu_host_right};
