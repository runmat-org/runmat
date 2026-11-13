//! Handle-free paired POWER physical geometry. Retained live views or future
//! allocator extents are descriptive until bound to independently owned leases.
use super::{PowerEvaluationPhysicalProposal, PowerPairCompound, PowerProbePhysicalProposal};
use crate::{
    AccelProvider, GpuTensorHandle, PowerInputDescription, ProviderExecutionError,
    ProviderExecutionResult, ProviderRepresentation,
};

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct PowerPairPhysicalProjection {
    pub probe: PowerProbePhysicalProposal,
    pub real: PowerEvaluationPhysicalProposal,
    pub complex: PowerEvaluationPhysicalProposal,
}

/// Owner-bound planning contract; neither this value nor its digest attests a
/// domain, an allocation, a submission or a placement decision.
#[derive(Clone)]
pub struct PowerPairProjection {
    owner: &'static dyn AccelProvider,
    inputs: [ProviderRepresentation; 2],
    descriptions: [PowerInputDescription; 2],
    physical: PowerPairPhysicalProjection,
}

impl PowerPairProjection {
    pub fn plan(
        owner: &'static dyn AccelProvider,
        inputs: [ProviderRepresentation; 2],
        descriptions: [PowerInputDescription; 2],
    ) -> ProviderExecutionResult<Option<Self>> {
        let Some(physical) = owner.propose_power_pair_geometry(&inputs, &descriptions, None)?
        else {
            return Ok(None);
        };
        Ok(Some(Self {
            owner,
            inputs,
            descriptions,
            physical,
        }))
    }
    pub fn physical(&self) -> &PowerPairPhysicalProjection {
        &self.physical
    }
    pub fn inputs(&self) -> &[ProviderRepresentation; 2] {
        &self.inputs
    }
    pub fn descriptions(&self) -> &[PowerInputDescription; 2] {
        &self.descriptions
    }
    pub fn is_owned_by(&self, owner: &dyn AccelProvider) -> bool {
        crate::same_provider_instance(self.owner, owner)
    }
    pub fn identity(&self) -> anyhow::Result<runmat_execution::Digest> {
        Ok(runmat_execution::Digest::sha256_serialized(&(
            "power-pair-projection-v1",
            &self.inputs,
            &self.descriptions,
            &self.physical,
        ))?)
    }
    /// No child has run when this is called. Both authentic handles must match
    /// the original descriptions and the exact frozen physical alternatives.
    pub fn bind(
        &self,
        base: &GpuTensorHandle,
        exponent: &GpuTensorHandle,
    ) -> ProviderExecutionResult<PowerPairCompound> {
        for ((source, expected), description) in [base, exponent]
            .into_iter()
            .zip(&self.inputs)
            .zip(&self.descriptions)
        {
            source
                .allocation_lease()
                .validate_live()
                .map_err(ProviderExecutionError::from_source)?;
            source
                .view()
                .validate(source.physical())
                .map_err(ProviderExecutionError::from_source)?;
            if crate::provider_for_handle(source)
                .is_none_or(|owner| !crate::same_provider_instance(owner, self.owner))
            {
                return Err(changed());
            }
            let actual = crate::ProviderFeasibilityQuery::exact_numeric_download(source)
                .and_then(|query| query.inputs.into_iter().next());
            if actual.as_ref() != Some(expected) || !description.matches(source) {
                return Err(changed());
            }
        }
        if self
            .owner
            .propose_power_pair_geometry(
                &self.inputs,
                &self.descriptions,
                Some(&self.physical.probe),
            )?
            .as_ref()
            != Some(&self.physical)
        {
            return Err(changed());
        }
        let probe = super::PowerProbeRecipe::plan_with_snapshot(
            self.owner,
            base,
            exponent,
            Some(&self.physical.probe),
        )?
        .ok_or_else(changed)?;
        let actual = PowerPairCompound::from_probe(self.owner, probe)?.ok_or_else(changed)?;
        if actual.probe().physical() != &self.physical.probe
            || actual.real().physical() != &self.physical.real
            || actual.complex().physical() != &self.physical.complex
        {
            return Err(changed());
        }
        Ok(actual)
    }
}

fn changed() -> ProviderExecutionError {
    ProviderExecutionError::from(anyhow::anyhow!(
        "planned POWER input or physical alternatives changed"
    ))
}
