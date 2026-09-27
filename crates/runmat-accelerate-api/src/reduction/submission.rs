use super::*;
use crate::{
    GpuTensorHandle, GpuTensorStorage, NumericElementType, ProviderExecutionError,
    ProviderExecutionResult, ProviderPrecision,
};
use anyhow::{ensure, Result};
use runmat_execution::Digest;
use serde::Serialize;

/// The existing provider tuning owner chooses this once. The packet does not
/// choose a strategy or alter chunk sizes; costing and execution share it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum ReductionStrategy {
    SinglePass,
    TwoPass,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct SelectedReductionWork {
    pub output_requested_bytes: u64,
    pub partial_requested_bytes: u64,
    pub dispatches: u32,
    pub workgroup_bytes: u64,
    pub uniform_bytes: u64,
    pub producer_scalar_work_units: u64,
}

/// Borrowing the immutable recipe prevents a different recipe from being
/// substituted between selection, resource admission and submission.
pub struct ReductionSelection<'a> {
    recipe: &'a ProviderReductionRecipe,
    strategy: ReductionStrategy,
    identity: Digest,
    work: SelectedReductionWork,
}
impl<'a> ReductionSelection<'a> {
    pub fn new(recipe: &'a ProviderReductionRecipe, strategy: ReductionStrategy) -> Result<Self> {
        let w = &recipe.descriptor().work;
        let (partial_requested_bytes, dispatches) = match strategy {
            ReductionStrategy::SinglePass => (0, w.single_dispatches),
            ReductionStrategy::TwoPass => (w.partial_requested_bytes, w.two_pass_dispatches),
        };
        let work = SelectedReductionWork {
            output_requested_bytes: w.output_requested_bytes,
            partial_requested_bytes,
            dispatches,
            workgroup_bytes: w.workgroup_bytes,
            uniform_bytes: w.uniform_bytes,
            producer_scalar_work_units: w.producer_scalar_work_units,
        };
        let identity = Digest::sha256(serde_json::to_vec(&(
            "typed-reduction-selection-v1",
            recipe.identity(),
            strategy,
            work,
        ))?);
        Ok(Self {
            recipe,
            strategy,
            identity,
            work,
        })
    }
    pub fn recipe(&self) -> &ProviderReductionRecipe {
        self.recipe
    }
    pub fn strategy(&self) -> ReductionStrategy {
        self.strategy
    }
    pub fn identity(&self) -> Digest {
        self.identity
    }
    pub fn work(&self) -> SelectedReductionWork {
        self.work
    }
}

/// Borrowed owning inputs, not a second allocation lifetime manager. A future
/// provider hook returns ProviderExecutionResult<GpuTensorHandle>; partials
/// remain private scratch and rejected outputs retain their genuine owners.
/// Device IDs only screen routing here. The executing backend must authenticate
/// each physical lease against its own allocation state before accessing bytes;
/// a matching numeric device ID is not proof of provider ownership. Current
/// views have no arbitrary offset/stride field: the full immutable view layout
/// and shape must match the recipe, whose index bounds were checked separately.
pub struct ReductionSubmission<'a> {
    selection: ReductionSelection<'a>,
    inputs: &'a [GpuTensorHandle],
    device_id: u32,
}
impl<'a> ReductionSubmission<'a> {
    pub fn new(
        selection: ReductionSelection<'a>,
        inputs: &'a [GpuTensorHandle],
        device_id: u32,
    ) -> ProviderExecutionResult<Self> {
        let descriptor = selection.recipe().descriptor();
        validate_inputs(descriptor, inputs, device_id).map_err(ProviderExecutionError::from)?;
        for handle in inputs {
            handle
                .allocation_lease()
                .validate_live()
                .map_err(ProviderExecutionError::from_source)?;
        }
        Ok(Self {
            selection,
            inputs,
            device_id,
        })
    }
    pub fn selection(&self) -> &ReductionSelection<'a> {
        &self.selection
    }
    pub fn inputs(&self) -> &[GpuTensorHandle] {
        self.inputs
    }
    pub fn device_id(&self) -> u32 {
        self.device_id
    }
}

fn validate_inputs(
    descriptor: &ReductionRecipeDescriptor,
    inputs: &[GpuTensorHandle],
    device_id: u32,
) -> Result<()> {
    ensure!(
        inputs.len() == descriptor.inputs.len(),
        "reduction prepared input count mismatch"
    );
    for (handle, expected) in inputs.iter().zip(&descriptor.inputs) {
        let element = match expected.precision {
            ProviderPrecision::F32 => NumericElementType::F32,
            ProviderPrecision::F64 => NumericElementType::F64,
        };
        ensure!(
            handle.device_id() == device_id,
            "reduction prepared input device mismatch"
        );
        ensure!(
            handle.shape() == expected.shape
                && handle.view().layout() == expected.layout
                && !handle.view().is_logical(),
            "reduction prepared input view mismatch"
        );
        ensure!(
            handle.physical().element_type() == element
                && handle.physical().storage() == GpuTensorStorage::Real,
            "reduction prepared input native class mismatch"
        );
        ensure!(
            handle.physical().payload_bytes() >= expected.payload_bytes
                && handle.physical().requested_bytes()
                    <= descriptor.config.max_storage_binding_bytes,
            "reduction prepared input allocation binding limit"
        );
    }
    Ok(())
}
