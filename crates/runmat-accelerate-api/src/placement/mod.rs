//! Side-effect-free provider discovery and feasibility contracts.
//!
//! These types describe what a provider can execute; they do not select where
//! an operation should execute. Selection and profitability remain placement
//! concerns in `runmat-accelerate`, while command scheduling remains owned by
//! each provider implementation.

mod capability;
mod cost;
mod feasibility;
mod representation;

pub use capability::{
    ProviderCapabilityOperation, ProviderCapabilitySnapshot, ProviderConcurrencyCapabilities,
    ProviderOperationFamily, PROVIDER_CAPABILITY_SCHEMA_VERSION,
};
pub use cost::{ProviderCostEstimate, ProviderCostQuery};
pub use feasibility::{
    ProviderFeasibility, ProviderFeasibilityQuery, ProviderOperationIdentity, ProviderRejection,
    ProviderRejectionCode, ProviderResourceEstimate, ProviderWorkload,
};
pub use representation::{
    ProviderElementType, ProviderLayout, ProviderRepresentation, ProviderResidency, ProviderStorage,
};
