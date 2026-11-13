use super::{ProviderStochasticEvolutionPhysicalProposal, ProviderStochasticEvolutionPlanRequest};
use crate::{
    provider_for_handle, same_provider_instance, AccelProvider, GpuTensorHandle, GpuTensorStorage,
    NumericElementType, ProviderExecutionError, ProviderExecutionResult,
};
use runmat_execution::Digest;
use runmat_types::NumericClass;

fn numeric_element(class: NumericClass) -> Option<NumericElementType> {
    match class {
        NumericClass::Single => Some(NumericElementType::F32),
        NumericClass::Double => Some(NumericElementType::F64),
        _ => None,
    }
}

#[derive(Clone)]
pub struct ProviderStochasticEvolutionPending {
    output: GpuTensorHandle,
    plan_identity: Digest,
    step: u32,
    next_offset: u64,
    source_view: crate::allocation::TensorViewIdentity,
}

impl ProviderStochasticEvolutionPending {
    pub fn output(&self) -> &GpuTensorHandle {
        &self.output
    }

    pub fn plan_identity(&self) -> Digest {
        self.plan_identity
    }

    pub fn step(&self) -> u32 {
        self.step
    }

    pub fn next_offset(&self) -> u64 {
        self.next_offset
    }

    pub fn source_view(&self) -> crate::allocation::TensorViewIdentity {
        self.source_view
    }
}

pub struct ProviderStochasticEvolutionProgress {
    pending: ProviderStochasticEvolutionPending,
    complete: bool,
}

impl ProviderStochasticEvolutionProgress {
    /// Records only the exact tile described by `request`. The request's
    /// prior token is the lineage authority; callers cannot name an arbitrary
    /// aligned offset or replace a continuation's provisional output.
    pub fn from_completed_tile(
        owner: &dyn AccelProvider,
        request: ProviderStochasticEvolutionTileRequest<'_>,
        output: GpuTensorHandle,
    ) -> ProviderExecutionResult<Self> {
        let (count, total) = request.validate_for(owner)?;
        let arithmetic = request
            .plan
            .arithmetic_for_step(request.step)
            .ok_or_else(|| {
                ProviderExecutionError::from(anyhow::anyhow!(
                    "stochastic evolution progress has an invalid step"
                ))
            })?;
        let next_offset = request.offset.checked_add(count).ok_or_else(|| {
            ProviderExecutionError::from(anyhow::anyhow!(
                "stochastic evolution progress offset overflows"
            ))
        })?;
        output
            .allocation_lease()
            .validate_live()
            .map_err(ProviderExecutionError::from_source)?;
        output
            .view()
            .validate(output.physical())
            .map_err(ProviderExecutionError::from_source)?;
        let output_shape_matches = output
            .shape()
            .iter()
            .map(|extent| u64::try_from(*extent).ok())
            .eq(request.plan.output.shape.iter().copied().map(Some));
        let exact_prior_output = request
            .pending
            .is_none_or(|prior| prior.output().view_identity() == output.view_identity());
        let expected_output = numeric_element(arithmetic.result_class()).ok_or_else(|| {
            ProviderExecutionError::from(anyhow::anyhow!(
                "stochastic evolution progress has an invalid output class"
            ))
        })?;
        if provider_for_handle(&output).is_none_or(|actual| !same_provider_instance(actual, owner))
            || output.physical().element_type() != expected_output
            || output.physical().storage() != GpuTensorStorage::Real
            || !output_shape_matches
            || !exact_prior_output
            || output.allocation_identity() == request.source.allocation_identity()
            || output.allocation_identity() == request.noise.allocation_identity()
        {
            return Err(ProviderExecutionError::from(anyhow::anyhow!(
                "stochastic evolution progress is not the selected completed prefix"
            )));
        }
        Ok(Self {
            pending: ProviderStochasticEvolutionPending {
                output,
                plan_identity: request.proposal.plan_identity,
                step: request.step,
                next_offset,
                source_view: request.source.view_identity(),
            },
            complete: next_offset == total,
        })
    }

    pub fn pending(&self) -> &ProviderStochasticEvolutionPending {
        &self.pending
    }

    pub fn is_complete(&self) -> bool {
        self.complete
    }

    pub fn into_completed_output(
        self,
        owner: &dyn AccelProvider,
        plan: &ProviderStochasticEvolutionPlanRequest,
        proposal: &ProviderStochasticEvolutionPhysicalProposal,
        step: u32,
    ) -> ProviderExecutionResult<GpuTensorHandle> {
        let total = plan.validate().ok_or_else(|| {
            ProviderExecutionError::from(anyhow::anyhow!(
                "stochastic evolution completion has an invalid plan"
            ))
        })?;
        self.pending
            .output
            .allocation_lease()
            .validate_live()
            .map_err(ProviderExecutionError::from_source)?;
        self.pending
            .output
            .view()
            .validate(self.pending.output.physical())
            .map_err(ProviderExecutionError::from_source)?;
        if !self.complete
            || !proposal.validate_for(plan)
            || self.pending.plan_identity != proposal.plan_identity
            || self.pending.step != step
            || self.pending.next_offset != total
            || provider_for_handle(&self.pending.output)
                .is_none_or(|actual| !same_provider_instance(actual, owner))
        {
            return Err(ProviderExecutionError::from(anyhow::anyhow!(
                "stochastic evolution output is not a completed selected step"
            )));
        }
        Ok(self.pending.output)
    }
}

#[derive(Clone, Copy)]
pub struct ProviderStochasticEvolutionTileRequest<'a> {
    pub plan: &'a ProviderStochasticEvolutionPlanRequest,
    pub proposal: &'a ProviderStochasticEvolutionPhysicalProposal,
    pub source: &'a GpuTensorHandle,
    pub noise: &'a GpuTensorHandle,
    pub step: u32,
    pub offset: u64,
    pub pending: Option<&'a ProviderStochasticEvolutionPending>,
}

impl ProviderStochasticEvolutionTileRequest<'_> {
    pub fn validate_for(self, owner: &dyn AccelProvider) -> ProviderExecutionResult<(u64, u64)> {
        if !self.proposal.validate_for(self.plan) || self.step >= self.plan.steps {
            return Err(ProviderExecutionError::from(anyhow::anyhow!(
                "stochastic evolution tile does not match its selected physical proposal"
            )));
        }
        for handle in [self.source, self.noise] {
            handle
                .allocation_lease()
                .validate_live()
                .map_err(ProviderExecutionError::from_source)?;
            handle
                .view()
                .validate(handle.physical())
                .map_err(ProviderExecutionError::from_source)?;
            if provider_for_handle(handle)
                .is_none_or(|actual| !same_provider_instance(actual, owner))
            {
                return Err(ProviderExecutionError::from(anyhow::anyhow!(
                    "stochastic evolution tile has a foreign provider owner"
                )));
            }
        }
        let total = self.plan.validate().ok_or_else(|| {
            ProviderExecutionError::from(anyhow::anyhow!(
                "stochastic evolution plan is no longer valid"
            ))
        })?;
        let count = self
            .noise
            .shape()
            .iter()
            .try_fold(1_u64, |product, extent| {
                product.checked_mul(u64::try_from(*extent).ok()?)
            });
        let count = count.ok_or_else(|| {
            ProviderExecutionError::from(anyhow::anyhow!(
                "stochastic evolution noise tile shape overflows"
            ))
        })?;
        let arithmetic = self.plan.arithmetic_for_step(self.step).ok_or_else(|| {
            ProviderExecutionError::from(anyhow::anyhow!(
                "stochastic evolution step has no arithmetic recipe"
            ))
        })?;
        let expected_noise = numeric_element(arithmetic.noise_class()).ok_or_else(|| {
            ProviderExecutionError::from(anyhow::anyhow!(
                "stochastic evolution noise class is not representable"
            ))
        })?;
        let expected_state = numeric_element(arithmetic.state_class()).ok_or_else(|| {
            ProviderExecutionError::from(anyhow::anyhow!(
                "stochastic evolution state class is not representable"
            ))
        })?;
        if count == 0
            || count > self.plan.noise_tile_elements
            || self.offset.checked_add(count).is_none_or(|end| end > total)
            || self.noise.physical().element_type() != expected_noise
            || self.source.physical().element_type() != expected_state
            || self.noise.physical().storage() != GpuTensorStorage::Real
            || self.source.physical().storage() != GpuTensorStorage::Real
            || self
                .source
                .shape()
                .iter()
                .map(|extent| u64::try_from(*extent).ok())
                .ne(self.plan.state.shape.iter().copied().map(Some))
            || self.noise.shape() != [usize::try_from(count).unwrap_or(usize::MAX), 1]
            || self.source.allocation_identity() == self.noise.allocation_identity()
        {
            return Err(ProviderExecutionError::from(anyhow::anyhow!(
                "stochastic evolution tile representation is invalid"
            )));
        }
        match self.pending {
            None if self.offset == 0 => {}
            Some(pending)
                if pending.plan_identity() == self.proposal.plan_identity
                    && pending.step() == self.step
                    && pending.next_offset() == self.offset
                    && pending.source_view() == self.source.view_identity()
                    && pending.output().allocation_identity()
                        != self.source.allocation_identity()
                    && pending.output().allocation_identity()
                        != self.noise.allocation_identity() =>
            {
                pending
                    .output()
                    .allocation_lease()
                    .validate_live()
                    .map_err(ProviderExecutionError::from_source)?;
                if provider_for_handle(pending.output())
                    .is_none_or(|actual| !same_provider_instance(actual, owner))
                {
                    return Err(ProviderExecutionError::from(anyhow::anyhow!(
                        "stochastic evolution pending output has a foreign owner"
                    )));
                }
            }
            _ => {
                return Err(ProviderExecutionError::from(anyhow::anyhow!(
                    "stochastic evolution tile continuation is not canonical"
                )))
            }
        }
        Ok((count, total))
    }
}
