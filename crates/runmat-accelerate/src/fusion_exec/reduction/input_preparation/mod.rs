//! Request-local typed reduction input inventory and selected preparation.
//! Planning is effect-free; execution consumes the selected plan once.
use crate::fusion_exec::ResolvedReductionProgram;
use runmat_accelerate_api::reduction::{
    ProviderReductionInputInventory, ProviderReductionInputStage,
};
use runmat_accelerate_api::{
    AccelProvider, GpuTensorHandle, ProviderExecutionError, ProviderPrecision,
    ProviderRealElementwiseRecipe, ProviderRepresentation,
};
use runmat_value::Value;

mod host;
mod inputs;
mod preflight;
mod prepare;

pub(super) use inputs::plan;
pub(super) use preflight::preflight_sources;

/// Query-free owner/class/liveness proof for the exact ordered data leaves.
/// It borrows the canonical resolved program, so a later selected recipe
/// cannot silently substitute a different set of source carriers.
pub(super) struct PreflightedInputs<'p, 'a> {
    program: &'p ResolvedReductionProgram<'a>,
    owner: &'static dyn AccelProvider,
    sources: Vec<ProviderRepresentation>,
}

pub(super) enum InputAction<'a> {
    Borrow(&'a GpuTensorHandle),
    Upload {
        source: &'a Value,
        target: ProviderPrecision,
    },
    Cast {
        source: &'a GpuTensorHandle,
        recipe_index: usize,
    },
}

pub(super) struct TypedInputPlan<'a> {
    pub(super) inventory: ProviderReductionInputInventory,
    owner: &'static dyn AccelProvider,
    binding_cap: u64,
    pub(super) actions: Vec<InputAction<'a>>,
    pub(super) casts: Vec<ProviderRealElementwiseRecipe>,
}

impl TypedInputPlan<'_> {
    /// On successful preparation, this distinguishes an all-borrow request
    /// from one that has crossed an upload/cast effect boundary. The caller
    /// must preserve it when classifying later submit/output failures.
    pub(super) fn has_physical_effects(&self) -> bool {
        self.inventory.entries().iter().any(|entry| {
            !matches!(
                &entry.stage,
                ProviderReductionInputStage::BorrowResident { .. }
            )
        })
    }
}

fn invalid(message: &'static str) -> ProviderExecutionError {
    ProviderExecutionError::from(anyhow::anyhow!(message))
}

#[cfg(test)]
mod tests;
