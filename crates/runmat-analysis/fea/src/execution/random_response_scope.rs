use runmat_canonical_codec::{decode_contract, encode_contract, CanonicalLimits};
use runmat_execution::Digest;
use serde::{Deserialize, Serialize};

use crate::{
    solve::dynamics::{random_excitation_spectrum_digest, RandomResponsePlanDescription},
    ModalBasisIdentity,
};

use super::{
    AnalysisExecutionError, AnalysisExecutionResult, StateIdentity,
    StructuralRandomResponseScopeIdentity, StructuralRandomResponseWorkload,
    StructuralRandomResponseWorkloadIdentity,
};

const SCOPE_PREFIX: &[u8] = b"runmat-structural-random-response-scope\0";
const SCOPE_LIMITS: CanonicalLimits =
    crate::contracts::canonical_digest::wire_bounded(CanonicalLimits::new(0, 0, 65_536, 32));

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, try_from = "StructuralRandomResponseScopeWire")]
pub struct StructuralRandomResponseScope {
    schema_version: u16,
    identity: StructuralRandomResponseScopeIdentity,
    workload: StructuralRandomResponseWorkloadIdentity,
    topology_digest: Digest,
    source_state: StateIdentity,
    operator_digest: Digest,
    plan: RandomResponsePlanDescription,
    maximum_intervals_per_task: usize,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct StructuralRandomResponseScopeWire {
    schema_version: u16,
    identity: StructuralRandomResponseScopeIdentity,
    workload: StructuralRandomResponseWorkloadIdentity,
    topology_digest: Digest,
    source_state: StateIdentity,
    operator_digest: Digest,
    plan: RandomResponsePlanDescription,
    maximum_intervals_per_task: usize,
}

pub struct StructuralRandomResponseScopeInput {
    pub topology_digest: Digest,
    pub source_state: StateIdentity,
    pub operator_digest: Digest,
    pub plan: RandomResponsePlanDescription,
}

impl TryFrom<StructuralRandomResponseScopeWire> for StructuralRandomResponseScope {
    type Error = AnalysisExecutionError;

    fn try_from(wire: StructuralRandomResponseScopeWire) -> Result<Self, Self::Error> {
        let scope = Self {
            schema_version: wire.schema_version,
            identity: wire.identity,
            workload: wire.workload,
            topology_digest: wire.topology_digest,
            source_state: wire.source_state,
            operator_digest: wire.operator_digest,
            plan: wire.plan,
            maximum_intervals_per_task: wire.maximum_intervals_per_task,
        };
        scope.validate()?;
        Ok(scope)
    }
}

impl StructuralRandomResponseScope {
    pub fn new(
        workload: &StructuralRandomResponseWorkload,
        input: StructuralRandomResponseScopeInput,
    ) -> AnalysisExecutionResult<Self> {
        workload.validate()?;
        let mut scope = Self {
            schema_version: 1,
            identity: StructuralRandomResponseScopeIdentity::derive(
                "structural-random-response-scope-placeholder",
                &1_u8,
            )?,
            workload: workload.identity(),
            topology_digest: input.topology_digest,
            source_state: input.source_state,
            operator_digest: input.operator_digest,
            plan: input.plan,
            maximum_intervals_per_task: workload.maximum_intervals_per_task(),
        };
        scope.identity = scope.derive_identity()?;
        scope.validate_for(workload)?;
        Ok(scope)
    }

    pub const fn identity(&self) -> StructuralRandomResponseScopeIdentity {
        self.identity
    }

    pub const fn workload(&self) -> StructuralRandomResponseWorkloadIdentity {
        self.workload
    }

    pub const fn topology_digest(&self) -> Digest {
        self.topology_digest
    }

    pub const fn source_state(&self) -> StateIdentity {
        self.source_state
    }

    pub const fn operator_digest(&self) -> Digest {
        self.operator_digest
    }

    pub const fn basis(&self) -> ModalBasisIdentity {
        self.plan.basis()
    }

    pub const fn plan(&self) -> &RandomResponsePlanDescription {
        &self.plan
    }

    pub const fn maximum_intervals_per_task(&self) -> usize {
        self.maximum_intervals_per_task
    }

    pub fn task_ranges(&self) -> Vec<[usize; 2]> {
        (0..self.plan.interval_count())
            .step_by(self.maximum_intervals_per_task)
            .map(|start| {
                [
                    start,
                    (start + self.maximum_intervals_per_task).min(self.plan.interval_count()),
                ]
            })
            .collect()
    }

    pub fn validate_for(
        &self,
        workload: &StructuralRandomResponseWorkload,
    ) -> AnalysisExecutionResult<()> {
        self.validate()?;
        workload.validate()?;
        let excitation = workload.excitation()?;
        if self.workload != workload.identity()
            || self.source_state != workload.modal_basis_transition().source_state
            || self.topology_digest != workload.modal_basis_transition().target_topology_digest
            || self.maximum_intervals_per_task != workload.maximum_intervals_per_task()
            || self.plan.excitation() != &excitation.excitation_id
            || self.plan.excitation_digest()
                != random_excitation_spectrum_digest(excitation)
                    .map_err(|error| invalid(error.to_string()))?
            || self.plan.policy() != workload.integration_policy()
        {
            return Err(AnalysisExecutionError::Identity(
                "random-response scope differs from its workload or modal-basis transition",
            ));
        }
        Ok(())
    }

    pub fn canonical_bytes(&self) -> AnalysisExecutionResult<Vec<u8>> {
        self.validate()?;
        Ok(encode_contract(
            SCOPE_PREFIX,
            "structural-random-response-scope",
            self,
            SCOPE_LIMITS,
        )?)
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> AnalysisExecutionResult<Self> {
        Ok(decode_contract(
            SCOPE_PREFIX,
            "structural-random-response-scope",
            bytes,
            SCOPE_LIMITS,
        )?)
    }

    pub fn validate(&self) -> AnalysisExecutionResult<()> {
        self.plan
            .validate()
            .map_err(|error| invalid(error.to_string()))?;
        if self.schema_version != 1
            || !(1..=4096).contains(&self.maximum_intervals_per_task)
            || self.task_ranges().is_empty()
            || self.identity != self.derive_identity()?
        {
            return Err(invalid("random-response scope is not canonical"));
        }
        Ok(())
    }

    fn derive_identity(&self) -> AnalysisExecutionResult<StructuralRandomResponseScopeIdentity> {
        StructuralRandomResponseScopeIdentity::derive(
            "structural-random-response-scope",
            &(
                self.schema_version,
                self.workload,
                self.topology_digest,
                self.source_state,
                self.operator_digest,
                &self.plan,
                self.maximum_intervals_per_task,
            ),
        )
    }
}

fn invalid(message: impl Into<String>) -> AnalysisExecutionError {
    AnalysisExecutionError::Invalid(message.into())
}
