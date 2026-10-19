//! Physical reduction ABI, not language semantics or scheduling authority.
//!
//! A checked packet establishes internal geometry and binding consistency. It
//! does not authenticate shader behavior, grant provider access, or replace
//! backend checks against actual buffers, device limits and compiled pipelines.
//! Runtime controls, canonical binding evidence and graph IDs remain upstream.
mod input_preparation;
mod recipe;
mod submission;
mod types;
mod validation;
mod work;

pub use input_preparation::{
    ProviderReductionInputInventory, ProviderReductionInputPreparation, ProviderReductionInputStage,
};
pub use recipe::{ProviderReductionRecipe, ReductionRecipeDescriptor, ReductionShaderSources};
pub use submission::{
    ReductionSelection, ReductionStrategy, ReductionSubmission, SelectedReductionWork,
};
pub use types::*;

#[cfg(test)]
mod tests;
