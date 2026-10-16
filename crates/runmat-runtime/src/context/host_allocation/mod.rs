//! Managed physical host allocations for ordinary numeric calls.
//!
//! The installed parallel service remains the only finite-capacity authority.
//! This module supplies checked workload bounds and an allocation receipt; it
//! neither estimates process RSS nor certifies host I/O effects or placement.

mod plan;
mod receipt;
mod temporary;

pub use plan::HostAllocationPlan;
pub use receipt::{HostAllocationError, HostAllocationReceipt, HostRetainedViewPermit};
pub use temporary::HostNumericWorkspace;

#[cfg(test)]
mod tests;
