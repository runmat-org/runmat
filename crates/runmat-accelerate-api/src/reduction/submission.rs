use super::*;
use crate::{
    GpuTensorHandle, GpuTensorStorage, NumericElementType, ProviderExecutionError,
    ProviderExecutionResult, ProviderPrecision,
};
use anyhow::{ensure, Result};
use runmat_execution::Digest;
use serde::Serialize;

/// The provider enumerates executable strategies; canonical placement chooses
/// one. Costing and execution retain that same immutable selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, serde::Deserialize)]
pub enum ReductionStrategy {
    SinglePass,
    TwoPass,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, serde::Deserialize)]
pub struct SelectedReductionWork {
    pub output_requested_bytes: u64,
    pub partial_requested_bytes: u64,
    pub dispatches: u32,
    pub workgroup_bytes: u64,
    pub uniform_bytes: u64,
    pub producer_scalar_work_units: u64,
    /// Checked terminal tree/forwarding prior, separate from the scalar producer.
    pub terminal_scalar_work_units: u64,
}

impl SelectedReductionWork {
    pub fn checked_total_scalar_work_units(self) -> Option<u64> {
        self.producer_scalar_work_units
            .checked_add(self.terminal_scalar_work_units)
    }
}

/// Borrowing the immutable recipe prevents a different recipe from being
/// substituted between selection, resource admission and submission.
pub struct ReductionSelection<'a> {
    recipe: &'a ProviderReductionRecipe,
    strategy: ReductionStrategy,
    identity: Digest,
    work: SelectedReductionWork,
}
impl std::fmt::Debug for ReductionSelection<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ReductionSelection")
            .field("recipe", &self.recipe.identity())
            .field("strategy", &self.strategy)
            .field("identity", &self.identity)
            .field("work", &self.work)
            .finish()
    }
}
impl<'a> ReductionSelection<'a> {
    pub fn new(recipe: &'a ProviderReductionRecipe, strategy: ReductionStrategy) -> Result<Self> {
        let w = &recipe.descriptor().work;
        let (partial_requested_bytes, dispatches) = match strategy {
            ReductionStrategy::SinglePass => (0, w.single_dispatches),
            ReductionStrategy::TwoPass => (w.partial_requested_bytes, w.two_pass_dispatches),
        };
        if strategy == ReductionStrategy::TwoPass {
            ensure!(
                partial_requested_bytes <= recipe.descriptor().config.max_storage_binding_bytes,
                "selected reduction partial storage binding limit"
            );
            ensure!(
                recipe
                    .descriptor()
                    .geometry
                    .pass1
                    .workgroups
                    .iter()
                    .all(|&dimension| {
                        dimension <= recipe.descriptor().config.max_dispatch_dimension
                    }),
                "selected reduction pass1 dispatch dimension limit"
            );
        }
        let work = SelectedReductionWork {
            output_requested_bytes: w.output_requested_bytes,
            partial_requested_bytes,
            dispatches,
            workgroup_bytes: w.workgroup_bytes,
            uniform_bytes: w.uniform_bytes,
            producer_scalar_work_units: w.producer_scalar_work_units,
            terminal_scalar_work_units: recipe
                .terminal_work_prior(strategy)?
                .checked_units()
                .ok_or_else(|| anyhow::anyhow!("terminal reduction work overflow"))?,
        };
        ensure!(
            work.checked_total_scalar_work_units().is_some(),
            "complete reduction work overflow"
        );
        let identity = Digest::sha256_serialized(&(
            "typed-reduction-selection-v2",
            recipe.identity(),
            strategy,
            work,
        ))?;
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
    host_metadata: Option<&'a dyn runmat_execution::resource::ProviderHostMetadataAdmission>,
    output_registration_metadata:
        Option<&'a dyn runmat_execution::resource::ProviderHostMetadataAdmission>,
}
impl<'a> ReductionSubmission<'a> {
    /// Explicit first-party control admission from the selected original
    /// Runtime. Unbounded direct callers explicitly pass None; managed
    /// execution supplies its captured port, never ambient state.
    pub fn new(
        selection: ReductionSelection<'a>,
        inputs: &'a [GpuTensorHandle],
        device_id: u32,
        host_metadata: Option<&'a dyn runmat_execution::resource::ProviderHostMetadataAdmission>,
        output_registration_metadata: Option<
            &'a dyn runmat_execution::resource::ProviderHostMetadataAdmission,
        >,
    ) -> ProviderExecutionResult<Self> {
        if host_metadata.is_some() && output_registration_metadata.is_none() {
            return Err(ProviderExecutionError::from(anyhow::anyhow!(
                "managed reduction output has no registration metadata authority"
            )));
        }
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
            host_metadata,
            output_registration_metadata,
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
    pub fn host_metadata(
        &self,
    ) -> Option<&dyn runmat_execution::resource::ProviderHostMetadataAdmission> {
        self.host_metadata
    }
    /// Separate tensor record/lease/view rights. A control-only allocation
    /// port cannot substitute for this exact provider's prepared registration.
    pub fn output_registration_metadata(
        &self,
    ) -> Option<&dyn runmat_execution::resource::ProviderHostMetadataAdmission> {
        self.output_registration_metadata
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
