//! Native paired POWER evaluation after an actual authenticated domain probe.
//! Both alternatives are physically planned before the probe is consumed.
use super::{PowerProbeDomain, PowerProbeRecipe};
use crate::{
    AccelProvider, GpuTensorHandle, GpuTensorStorage, PowerResultDomain, ProviderElementType,
    ProviderExecutedRealElementwise, ProviderExecutionCompletionError, ProviderExecutionError,
    ProviderExecutionResult, ProviderLayout, ProviderRealElementwiseOutputRoute,
    ProviderRealElementwisePhysicalProposal, ProviderRepresentation, ProviderResidency,
    ProviderStorage,
};

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct PowerEvaluationPhysicalProposal {
    pub output: ProviderRepresentation,
    pub work: ProviderRealElementwisePhysicalProposal,
    pub shader_identity: runmat_execution::Digest,
}

#[derive(Clone)]
pub struct PowerEvaluationRecipe {
    probe: PowerProbeRecipe,
    domain: PowerResultDomain,
    physical: PowerEvaluationPhysicalProposal,
}

impl PowerEvaluationRecipe {
    pub fn plan(
        owner: &'static dyn AccelProvider,
        probe: &PowerProbeRecipe,
        domain: PowerResultDomain,
    ) -> ProviderExecutionResult<Option<Self>> {
        if !probe.is_owned_by(owner) {
            return Err(changed());
        }
        let [base, exponent] = probe.sources();
        let Some(physical) = owner.propose_power_evaluation(base, exponent, domain)? else {
            return Ok(None);
        };
        let Some(query) = crate::ProviderFeasibilityQuery::exact_numeric_download(base) else {
            return Ok(None);
        };
        let Some(input) = query.inputs.first() else {
            return Ok(None);
        };
        let output_type = match (input.element_type, domain) {
            (ProviderElementType::F32, PowerResultDomain::Real) => ProviderElementType::F32,
            (ProviderElementType::F64, PowerResultDomain::Real) => ProviderElementType::F64,
            (ProviderElementType::F32, PowerResultDomain::Complex) => {
                ProviderElementType::ComplexF32
            }
            (ProviderElementType::F64, PowerResultDomain::Complex) => {
                ProviderElementType::ComplexF64
            }
            _ => return Ok(None),
        };
        let output_storage = match domain {
            PowerResultDomain::Real => ProviderStorage::DenseReal,
            PowerResultDomain::Complex => ProviderStorage::DenseComplexInterleaved,
        };
        let shape = probe
            .physical()
            .output_shape
            .iter()
            .copied()
            .map(u64::try_from)
            .collect::<Result<Vec<_>, _>>()
            .map_err(ProviderExecutionError::from_source)?;
        let expected = ProviderRepresentation {
            element_type: output_type,
            storage: output_storage,
            layout: ProviderLayout::ColumnMajorContiguous,
            shape,
            residency: ProviderResidency::Device,
        };
        if physical.output != expected
            || physical.work.output_route != ProviderRealElementwiseOutputRoute::FreshAllocation
            || physical.work.workgroup_size == 0
            || physical.work.chunk_capacity == 0
            || expected
                .checked_byte_len()
                .is_none_or(|bytes| bytes > physical.work.output_buffer_bytes)
            || physical.work.requested_incremental_peak_bytes().is_none()
        {
            return Err(changed());
        }
        Ok(Some(Self {
            probe: probe.clone(),
            domain,
            physical,
        }))
    }

    pub fn probe(&self) -> &PowerProbeRecipe {
        &self.probe
    }
    pub fn domain(&self) -> PowerResultDomain {
        self.domain
    }
    pub fn physical(&self) -> &PowerEvaluationPhysicalProposal {
        &self.physical
    }
    pub fn identity(&self) -> anyhow::Result<runmat_execution::Digest> {
        Ok(runmat_execution::Digest::sha256_serialized(&(
            "power-native-evaluation-v1",
            self.probe.identity()?,
            self.domain,
            &self.physical,
        ))?)
    }

    pub async fn execute(
        &self,
        evidence: &PowerProbeDomain,
    ) -> ProviderExecutionResult<ProviderExecutedRealElementwise> {
        self.execute_admitted(evidence, None).await
    }

    pub async fn execute_admitted(
        &self,
        evidence: &PowerProbeDomain,
        admission: Option<&dyn runmat_execution::resource::ProviderHostMetadataAdmission>,
    ) -> ProviderExecutionResult<ProviderExecutedRealElementwise> {
        // Every failure belongs to an already-consumed probe, including a
        // child Unsupported. There is no permission to replay this compound.
        self.execute_inner(evidence, admission)
            .await
            .map_err(ProviderExecutionError::after_effect)
    }

    async fn execute_inner(
        &self,
        evidence: &PowerProbeDomain,
        admission: Option<&dyn runmat_execution::resource::ProviderHostMetadataAdmission>,
    ) -> ProviderExecutionResult<ProviderExecutedRealElementwise> {
        if !self.probe.authenticates(evidence) || evidence.domain() != self.domain {
            return Err(changed());
        }
        let owner = self.probe.owner();
        let [base, exponent] = self.probe.sources();
        if owner
            .propose_power_evaluation(base, exponent, self.domain)?
            .as_ref()
            != Some(&self.physical)
        {
            return Err(changed());
        }
        let guards = [
            crate::ResidencyRestoreGuard::new(base),
            crate::ResidencyRestoreGuard::new(exponent),
        ];
        let result = owner
            .execute_power_evaluation_recipe_admitted(self, admission)
            .await;
        for guard in &guards {
            guard.restore();
        }
        drop(guards);
        let result = result?;
        let work = &self.physical.work;
        let observed = &result.observed;
        let aliases = self
            .probe
            .sources()
            .iter()
            .any(|source| result.output.shares_allocation_with(source));
        if !self.probe.authenticates(evidence)
            || aliases
            || !self.output_matches(&result.output)
            || observed.output_buffer_bytes != Some(work.output_buffer_bytes)
            || observed.uniform_allocations != work.uniform_allocations
            || observed.uniform_buffer_bytes != work.uniform_buffer_bytes
            || observed.dispatches != work.dispatches
            || observed.queue_submissions != work.queue_submissions
        {
            let primary = changed().after_effect();
            if aliases {
                drop(result.output);
                return Err(primary);
            }
            let cleanup = result
                .output
                .release()
                .err()
                .into_iter()
                .collect::<Vec<_>>();
            return if cleanup.is_empty() {
                Err(primary)
            } else {
                Err(
                    ProviderExecutionError::from_source(ProviderExecutionCompletionError::new(
                        Some(primary),
                        cleanup,
                    ))
                    .after_effect(),
                )
            };
        }
        Ok(result)
    }

    fn output_matches(&self, output: &GpuTensorHandle) -> bool {
        let expected = &self.physical.output;
        let (element, storage) = match expected.element_type {
            ProviderElementType::F32 => (crate::NumericElementType::F32, GpuTensorStorage::Real),
            ProviderElementType::F64 => (crate::NumericElementType::F64, GpuTensorStorage::Real),
            ProviderElementType::ComplexF32 => (
                crate::NumericElementType::F32,
                GpuTensorStorage::ComplexInterleaved,
            ),
            ProviderElementType::ComplexF64 => (
                crate::NumericElementType::F64,
                GpuTensorStorage::ComplexInterleaved,
            ),
            _ => return false,
        };
        output.allocation_lease().validate_live().is_ok()
            && output.view().validate(output.physical()).is_ok()
            && crate::provider_for_handle(output).is_some_and(|owner| self.probe.is_owned_by(owner))
            && output.physical().element_type() == element
            && output.physical().storage() == storage
            && !output.view().is_logical()
            && output.view().layout() == crate::GpuTensorLayout::Contiguous
            && output.physical().requested_bytes() == self.physical.work.output_buffer_bytes
            && output.shape() == self.probe.physical().output_shape
    }
}

fn changed() -> ProviderExecutionError {
    ProviderExecutionError::from(anyhow::anyhow!(
        "POWER evaluation differs from its consumed probe or physical recipe"
    ))
}
