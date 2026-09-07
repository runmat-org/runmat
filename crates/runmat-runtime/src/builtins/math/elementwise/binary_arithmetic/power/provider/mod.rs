//! Provider-resident dispatch for element-wise power.

mod conversion;
mod pair;
mod resident;

pub(super) use pair::power_gpu_pair;
pub(super) use resident::{power_gpu_host_left, power_gpu_host_right};
