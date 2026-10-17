//! Managed physical host allocations for ordinary numeric calls.
//!
//! The installed parallel service remains the only finite-capacity authority.
//! This module supplies checked workload bounds and an allocation receipt; it
//! neither estimates process RSS nor certifies host I/O effects or placement.

mod plan;
mod provider_registration;
mod receipt;
mod session;
mod temporary;

pub use plan::HostAllocationPlan;
pub use provider_registration::{provider_registration_host_quote, ProviderRegistrationHostQuote};
pub use receipt::{HostAllocationError, HostAllocationReceipt, HostRetainedViewPermit};
pub(crate) use session::HostRetainedSessionAllocation;
pub use temporary::{HostComplexWorkspace, HostNumericWorkspace};

#[cfg(test)]
mod tests;
