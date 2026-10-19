//! One exact global floating extrema tree. This is physical admission only;
//! its result is not numerical-domain evidence until the owning caller reads it.
//! The recipe contract includes NaNs: a represented NaN must propagate to the
//! scalar result. Omit-NaN extrema are not a substitute for this probe tree.
use super::ProviderExtremaKind;
mod checked;
use crate::{
    AccelProvider, GpuTensorHandle, GpuTensorStorage, ProviderElementType,
    ProviderExecutionCompletionError, ProviderExecutionError, ProviderExecutionFuture,
    ProviderExecutionResult, ProviderFeasibilityQuery, ProviderLayout, ProviderRepresentation,
    ProviderResidency, ProviderStorage,
};
pub use checked::{ProviderCheckedGlobalExtrema, ProviderGlobalExtremaScalar};

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct ProviderGlobalExtremaRequest {
    pub kind: ProviderExtremaKind,
    pub source: ProviderRepresentation,
    pub source_buffer_bytes: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub enum ProviderGlobalExtremaOutputRoute {
    FreshAllocation,
    SourceScalarAlias,
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct ProviderGlobalExtremaPhysicalProposal {
    pub output_route: ProviderGlobalExtremaOutputRoute,
    pub output_buffer_bytes: u64,
    /// All intermediate allocations, excluding the retained scalar result.
    /// Summing allocations remains a bound even while submissions retain them.
    pub intermediate_buffer_bytes: u64,
    pub uniform_allocations: u32,
    pub uniform_buffer_bytes: u64,
    pub dispatches: u32,
    pub queue_submissions: u32,
    pub workgroup_size: u32,
    pub chunk_capacity: u64,
    pub pipeline_key: u64,
}

impl ProviderGlobalExtremaPhysicalProposal {
    pub fn requested_incremental_peak_bytes(&self) -> Option<u64> {
        let output = match self.output_route {
            ProviderGlobalExtremaOutputRoute::FreshAllocation => self.output_buffer_bytes,
            ProviderGlobalExtremaOutputRoute::SourceScalarAlias => 0,
        };
        output
            .checked_add(self.intermediate_buffer_bytes)?
            .checked_add(self.uniform_buffer_bytes)
    }

    pub fn expected_observations(&self) -> Option<ProviderGlobalExtremaObserved> {
        let output = match self.output_route {
            ProviderGlobalExtremaOutputRoute::FreshAllocation => self.output_buffer_bytes,
            ProviderGlobalExtremaOutputRoute::SourceScalarAlias => 0,
        };
        Some(ProviderGlobalExtremaObserved {
            allocated_buffer_bytes: output.checked_add(self.intermediate_buffer_bytes)?,
            uniform_allocations: self.uniform_allocations,
            uniform_buffer_bytes: self.uniform_buffer_bytes,
            dispatches: self.dispatches,
            queue_submissions: self.queue_submissions,
        })
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ProviderGlobalExtremaObserved {
    pub allocated_buffer_bytes: u64,
    pub uniform_allocations: u32,
    pub uniform_buffer_bytes: u64,
    pub dispatches: u32,
    pub queue_submissions: u32,
}

pub struct ProviderExecutedGlobalExtrema {
    pub output: GpuTensorHandle,
    pub observed: ProviderGlobalExtremaObserved,
}

/// Retains the original live view/lease and exact provider instance, not an
/// ambient provider or a deserialized physical proposal.
#[derive(Clone)]
pub struct ProviderGlobalExtremaRecipe {
    owner: &'static dyn AccelProvider,
    source: GpuTensorHandle,
    request: ProviderGlobalExtremaRequest,
    physical: ProviderGlobalExtremaPhysicalProposal,
}

impl ProviderGlobalExtremaRecipe {
    pub fn plan(
        owner: &'static dyn AccelProvider,
        source: &GpuTensorHandle,
        kind: ProviderExtremaKind,
    ) -> ProviderExecutionResult<Option<Self>> {
        source
            .allocation_lease()
            .validate_live()
            .map_err(ProviderExecutionError::from_source)?;
        source
            .view()
            .validate(source.physical())
            .map_err(ProviderExecutionError::from_source)?;
        if crate::provider_for_handle(source)
            .is_none_or(|actual| !crate::same_provider_instance(actual, owner))
        {
            return Err(ProviderExecutionError::from(anyhow::anyhow!(
                "global extrema source has another owner"
            )));
        }
        let Some(query) = ProviderFeasibilityQuery::exact_numeric_download(source) else {
            return Ok(None);
        };
        let Some(input) = query.inputs.first() else {
            return Ok(None);
        };
        if !matches!(
            input.element_type,
            ProviderElementType::F32 | ProviderElementType::F64
        ) || input.storage != ProviderStorage::DenseReal
            || input
                .checked_element_count()
                .is_none_or(|n| n == 0 || n != source.physical().element_capacity())
        {
            return Ok(None);
        }
        let request = ProviderGlobalExtremaRequest {
            kind,
            source: input.clone(),
            source_buffer_bytes: source.physical().requested_bytes(),
        };
        let Some(physical) = owner.propose_global_extrema(&request)? else {
            return Ok(None);
        };
        let alias = request.source.checked_element_count() == Some(1);
        if physical.workgroup_size == 0
            || physical.chunk_capacity == 0
            || physical.requested_incremental_peak_bytes().is_none()
            || physical.expected_observations().is_none()
            || physical.output_buffer_bytes < request.source.element_type.byte_width()
            || physical.queue_submissions < physical.dispatches
            || (alias
                && (physical.output_route != ProviderGlobalExtremaOutputRoute::SourceScalarAlias
                    || physical.output_buffer_bytes != request.source_buffer_bytes
                    || physical.expected_observations()
                        != Some(ProviderGlobalExtremaObserved::default())))
            || (!alias
                && physical.output_route != ProviderGlobalExtremaOutputRoute::FreshAllocation)
        {
            return Ok(None);
        }
        Ok(Some(Self {
            owner,
            source: source.clone(),
            request,
            physical,
        }))
    }

    pub fn request(&self) -> &ProviderGlobalExtremaRequest {
        &self.request
    }
    pub fn physical(&self) -> &ProviderGlobalExtremaPhysicalProposal {
        &self.physical
    }
    pub fn source(&self) -> &GpuTensorHandle {
        &self.source
    }
    pub fn is_owned_by(&self, owner: &dyn AccelProvider) -> bool {
        crate::same_provider_instance(self.owner, owner)
    }
    pub fn output(&self) -> ProviderRepresentation {
        ProviderRepresentation {
            element_type: self.request.source.element_type,
            storage: ProviderStorage::DenseReal,
            layout: ProviderLayout::ColumnMajorContiguous,
            shape: vec![1, 1],
            residency: ProviderResidency::Device,
        }
    }
    pub fn identity(&self) -> anyhow::Result<runmat_execution::Digest> {
        Ok(runmat_execution::Digest::sha256(serde_json::to_vec(&(
            "global-extrema-v1",
            self.request(),
            self.physical(),
        ))?))
    }
    pub fn execute(&self) -> ProviderExecutionFuture<'_, ProviderExecutedGlobalExtrema> {
        Box::pin(async move {
            self.source
                .allocation_lease()
                .validate_live()
                .map_err(ProviderExecutionError::from_source)?;
            if self.owner.propose_global_extrema(&self.request)?.as_ref() != Some(&self.physical) {
                return Err(ProviderExecutionError::from(anyhow::anyhow!(
                    "global extrema physical recipe changed"
                )));
            }
            let metadata = crate::ResidencyRestoreGuard::new(&self.source);
            let result = self
                .owner
                .execute_global_extrema_recipe(&self.source, self)
                .await;
            metadata.restore();
            drop(metadata);
            let result = result?;
            let aliases = result.output.shares_allocation_with(&self.source);
            let valid_alias = match self.physical.output_route {
                ProviderGlobalExtremaOutputRoute::SourceScalarAlias => {
                    aliases && result.output.physical() == self.source.physical()
                }
                ProviderGlobalExtremaOutputRoute::FreshAllocation => !aliases,
            };
            let valid = valid_alias
                && result.output.shape() == [1, 1]
                && result.output.physical().storage() == GpuTensorStorage::Real
                && result.output.physical().element_type() == self.source.physical().element_type()
                && !result.output.view().is_logical()
                && result.output.physical().requested_bytes() == self.physical.output_buffer_bytes
                && result.output.allocation_lease().validate_live().is_ok()
                && result
                    .output
                    .view()
                    .validate(result.output.physical())
                    .is_ok()
                && self.source.allocation_lease().validate_live().is_ok()
                && crate::provider_for_handle(&result.output)
                    .is_some_and(|actual| crate::same_provider_instance(actual, self.owner))
                && Some(&result.observed) == self.physical.expected_observations().as_ref();
            if valid {
                return Ok(result);
            }
            let error = ProviderExecutionError::from(anyhow::anyhow!(
                "global extrema execution differs from its recipe"
            ))
            .after_effect();
            if aliases {
                drop(result.output);
                return Err(error);
            }
            let cleanup = result
                .output
                .release()
                .err()
                .into_iter()
                .collect::<Vec<_>>();
            if cleanup.is_empty() {
                Err(error)
            } else {
                Err(
                    ProviderExecutionError::from_source(ProviderExecutionCompletionError::new(
                        Some(error),
                        cleanup,
                    ))
                    .after_effect(),
                )
            }
        })
    }

    /// Keep a nonconstructible actual-execution witness across the separate
    /// scalar readback await. Callers can check cancellation between stages.
    pub async fn execute_checked(&self) -> ProviderExecutionResult<ProviderCheckedGlobalExtrema> {
        let executed = self.execute().await?;
        Ok(ProviderCheckedGlobalExtrema::new(self, executed))
    }
}
