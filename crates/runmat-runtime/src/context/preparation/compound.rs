//! Exact admitted body delivery; one typed compound, not competing optional
//! proofs or a string dispatcher. Domains retain numerical/publication owners.
use super::{
    AdmittedRealPowerCompound, AdmittedRealPowerScalarCompound, AdmittedRealUnaryCompound,
};
use runmat_accelerate_api::{
    observation::{
        ProviderObservationCapability, ProviderObservationOverhead, ProviderObservationSupport,
        ProviderOperationReceipt,
    },
    AccelProvider, ProviderRepresentation,
};
use runmat_execution::{CandidateOutputResidency, CandidateResourceDemand, Digest};
use runmat_value::Value;

pub enum AdmittedBuiltinCompound {
    RealUnary(Box<AdmittedRealUnaryCompound>),
    RealPower(Box<AdmittedRealPowerCompound>),
    RealPowerScalar(Box<AdmittedRealPowerScalarCompound>),
}

impl AdmittedBuiltinCompound {
    pub fn preserves_host_scalar_operand(&self, offset: usize) -> bool {
        match self {
            Self::RealPowerScalar(plan) => plan.host_control_offset() == offset,
            Self::RealUnary(_) | Self::RealPower(_) => false,
        }
    }
    pub fn identity(&self) -> Digest {
        match self {
            Self::RealUnary(plan) => plan.identity(),
            Self::RealPower(plan) => plan.identity(),
            Self::RealPowerScalar(plan) => plan.identity(),
        }
    }
    pub fn resources(&self) -> CandidateResourceDemand {
        match self {
            Self::RealUnary(plan) => plan.resources(),
            Self::RealPower(plan) => plan.resources(),
            Self::RealPowerScalar(plan) => plan.resources(),
        }
    }
    pub fn resources_with_observation(
        &self,
        overhead: Option<ProviderObservationOverhead>,
    ) -> crate::BuiltinResult<CandidateResourceDemand> {
        match self {
            Self::RealUnary(plan) => plan.resources_with_observation(overhead),
            Self::RealPower(plan) => plan.resources_with_observation(overhead),
            Self::RealPowerScalar(plan) => plan.resources_with_observation(overhead),
        }
    }
    pub fn output_residency(&self) -> CandidateOutputResidency {
        match self {
            Self::RealUnary(plan) => plan.output_residency(),
            Self::RealPower(plan) => plan.output_residency(),
            Self::RealPowerScalar(plan) => plan.output_residency(),
        }
    }
    pub fn matches(&self, owner: &dyn AccelProvider, values: &[&Value]) -> bool {
        match self {
            Self::RealUnary(plan) => plan.matches(owner, values),
            Self::RealPower(plan) => plan.matches(owner, values),
            Self::RealPowerScalar(plan) => plan.matches(owner, values),
        }
    }
    pub fn source_inputs(&self) -> Vec<ProviderRepresentation> {
        match self {
            Self::RealUnary(plan) => vec![plan.physical().real().request().source.clone()],
            Self::RealPower(plan) => plan.inputs().to_vec(),
            Self::RealPowerScalar(plan) => plan.inputs().to_vec(),
        }
    }
    /// Representative real alternative for original query geometry only.
    /// This does not replace Unknown workload/domain evidence: resources and
    /// actual execution use the enclosing authenticated alternative envelope.
    pub fn query_output(&self) -> ProviderRepresentation {
        match self {
            Self::RealUnary(plan) => plan.physical().real().output().clone(),
            Self::RealPower(plan) => plan.physical().real().physical().output.clone(),
            Self::RealPowerScalar(plan) => plan.output().clone(),
        }
    }
    pub fn preparation_program(&self) -> runmat_accelerate_api::ProviderPreparationProgram<'_> {
        match self {
            Self::RealUnary(plan) => runmat_accelerate_api::ProviderPreparationProgram::RealUnary {
                recipe: plan.physical(),
            },
            Self::RealPower(plan) => runmat_accelerate_api::ProviderPreparationProgram::PowerPair {
                recipe: plan.physical(),
            },
            Self::RealPowerScalar(plan) => {
                runmat_accelerate_api::ProviderPreparationProgram::PowerScalar {
                    projection: plan.projection(),
                }
            }
        }
    }
    pub fn observation_capability(
        &self,
        owner: &dyn AccelProvider,
    ) -> ProviderObservationCapability {
        match self {
            Self::RealUnary(plan) => {
                owner.real_unary_observation_capability(plan.physical(), plan.continuations())
            }
            Self::RealPower(plan) => owner.power_pair_observation_capability(plan.physical()),
            Self::RealPowerScalar(plan) => {
                owner.power_scalar_observation_capability(plan.projection())
            }
        }
    }
    pub fn begin_observation(&self, owner: &dyn AccelProvider) -> ProviderObservationSupport {
        match self {
            Self::RealUnary(plan) => {
                owner.begin_real_unary_observation(plan.physical(), plan.continuations())
            }
            Self::RealPower(plan) => owner.begin_power_pair_observation(plan.physical()),
            Self::RealPowerScalar(plan) => owner.begin_power_scalar_observation(plan.projection()),
        }
    }
    pub fn require_execution(
        &self,
        receipt: &mut ProviderOperationReceipt,
        operation: runmat_accelerate_api::ProviderOperationIdentity,
    ) {
        match self {
            Self::RealUnary(plan) => receipt.require_real_unary_execution(
                operation,
                plan.physical(),
                plan.continuations(),
            ),
            Self::RealPower(plan) => {
                receipt.require_power_pair_execution(operation, plan.physical())
            }
            Self::RealPowerScalar(plan) => {
                receipt.require_power_scalar_execution(operation, plan.projection(), plan.source())
            }
        }
    }
}
