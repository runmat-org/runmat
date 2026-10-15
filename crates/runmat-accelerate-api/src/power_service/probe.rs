//! Exact paired POWER domain-probe recipe. This physical owner supplies no
//! placement decision, numerical promise, host fallback or completion right.
use crate::{
    AccelProvider, GpuTensorHandle, PowerResultDomain, ProviderElementType, ProviderExecutionError,
    ProviderExecutionResult, ProviderFeasibilityQuery, ProviderStorage,
};
use runmat_execution::Digest;

/// Projection of the provider's actual frozen flag/reduction/readback tree.
/// Data phase ceilings include all preceding requested backing, not just the
/// last stage. Optional observation fence/query backing is added by its actual
/// owner exactly once. Host workspace is separate from provider staging.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct PowerProbePhysicalProposal {
    pub output_shape: Vec<usize>,
    pub snapshot_copy: bool,
    pub shader_identity: Digest,
    pub flag_pipeline_key: u64,
    pub phase_data_limits: [u64; 4],
    pub dispatches: u32,
    pub queue_submissions: u32,
    pub host_workspace_bytes: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PowerProbeObserved {
    pub domain: PowerResultDomain,
    pub dispatches: u32,
    pub queue_submissions: u32,
}

#[derive(Clone)]
pub struct PowerProbeRecipe {
    owner: &'static dyn AccelProvider,
    sources: [GpuTensorHandle; 2],
    physical: PowerProbePhysicalProposal,
}

/// Only actual consumption of this exact retained probe mints this witness.
/// An inference domain, classification hook or output class cannot create it.
pub struct PowerProbeDomain {
    source_views: [crate::allocation::TensorViewIdentity; 2],
    recipe: Digest,
    domain: PowerResultDomain,
}

impl PowerProbeDomain {
    pub fn domain(&self) -> PowerResultDomain {
        self.domain
    }
}

impl PowerProbeRecipe {
    pub(super) fn owner(&self) -> &'static dyn AccelProvider {
        self.owner
    }
    pub fn plan(
        owner: &'static dyn AccelProvider,
        base: &GpuTensorHandle,
        exponent: &GpuTensorHandle,
    ) -> ProviderExecutionResult<Option<Self>> {
        Self::plan_with_snapshot(owner, base, exponent, None)
    }

    pub(super) fn plan_with_snapshot(
        owner: &'static dyn AccelProvider,
        base: &GpuTensorHandle,
        exponent: &GpuTensorHandle,
        frozen: Option<&PowerProbePhysicalProposal>,
    ) -> ProviderExecutionResult<Option<Self>> {
        for source in [base, exponent] {
            validate_source(owner, source)?;
        }
        let [Some(base_query), Some(exponent_query)] =
            [base, exponent].map(ProviderFeasibilityQuery::exact_numeric_download)
        else {
            return Ok(None);
        };
        let (Some(base_rep), Some(exponent_rep)) =
            (base_query.inputs.first(), exponent_query.inputs.first())
        else {
            return Ok(None);
        };
        if base_rep.element_type != exponent_rep.element_type
            || !matches!(
                base_rep.element_type,
                ProviderElementType::F32 | ProviderElementType::F64
            )
            || base_rep.storage != ProviderStorage::DenseReal
            || exponent_rep.storage != ProviderStorage::DenseReal
        {
            return Ok(None);
        }
        let shape =
            runmat_types::geometry::broadcast_concrete_shape(base.shape(), exponent.shape())
                .map_err(ProviderExecutionError::from_source)?;
        let Some(physical) = owner.propose_power_probe(base, exponent, frozen)? else {
            return Ok(None);
        };
        if physical.output_shape != shape
            || physical.phase_data_limits.contains(&0)
            || physical.dispatches == 0
            || physical.queue_submissions < physical.dispatches
        {
            return Err(changed());
        }
        Ok(Some(Self {
            owner,
            sources: [base.clone(), exponent.clone()],
            physical,
        }))
    }

    pub fn sources(&self) -> &[GpuTensorHandle; 2] {
        &self.sources
    }
    pub fn physical(&self) -> &PowerProbePhysicalProposal {
        &self.physical
    }
    pub fn is_owned_by(&self, owner: &dyn AccelProvider) -> bool {
        crate::same_provider_instance(self.owner, owner)
    }
    pub fn identity(&self) -> anyhow::Result<Digest> {
        Ok(Digest::sha256_serialized(&(
            "power-domain-probe-v1",
            self.sources.each_ref().map(|source| {
                let identity = source.view_identity();
                (identity.allocation(), identity.view())
            }),
            &self.physical,
        ))?)
    }
    pub fn authenticates(&self, evidence: &PowerProbeDomain) -> bool {
        evidence.source_views == self.sources.each_ref().map(GpuTensorHandle::view_identity)
            && self.identity().ok() == Some(evidence.recipe)
            && self
                .sources
                .iter()
                .all(|source| validate_source(self.owner, source).is_ok())
    }

    pub async fn execute(&self) -> ProviderExecutionResult<PowerProbeDomain> {
        for source in &self.sources {
            validate_source(self.owner, source)?;
        }
        let [base, exponent] = &self.sources;
        if self
            .owner
            .propose_power_probe(base, exponent, Some(&self.physical))?
            .as_ref()
            != Some(&self.physical)
        {
            return Err(changed());
        }
        let guards = [
            crate::ResidencyRestoreGuard::new(base),
            crate::ResidencyRestoreGuard::new(exponent),
        ];
        let result = self.owner.execute_power_probe_recipe(self).await;
        for guard in &guards {
            guard.restore();
        }
        drop(guards);
        let observed = result?;
        if observed.dispatches != self.physical.dispatches
            || observed.queue_submissions != self.physical.queue_submissions
        {
            return Err(changed().after_effect());
        }
        for source in &self.sources {
            validate_source(self.owner, source).map_err(ProviderExecutionError::after_effect)?;
        }
        Ok(PowerProbeDomain {
            source_views: self.sources.each_ref().map(GpuTensorHandle::view_identity),
            recipe: self
                .identity()
                .map_err(ProviderExecutionError::from)
                .map_err(ProviderExecutionError::after_effect)?,
            domain: observed.domain,
        })
    }
}

fn validate_source(
    owner: &dyn AccelProvider,
    source: &GpuTensorHandle,
) -> ProviderExecutionResult<()> {
    source
        .allocation_lease()
        .validate_live()
        .map_err(ProviderExecutionError::from_source)?;
    source
        .view()
        .validate(source.physical())
        .map_err(ProviderExecutionError::from_source)?;
    if crate::provider_for_handle(source)
        .is_none_or(|actual| !crate::same_provider_instance(owner, actual))
    {
        return Err(changed());
    }
    Ok(())
}

fn changed() -> ProviderExecutionError {
    ProviderExecutionError::from(anyhow::anyhow!(
        "POWER probe source or frozen physical recipe changed"
    ))
}
