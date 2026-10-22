use runmat_canonical_codec::{decode_contract, encode_contract, CanonicalLimits};
use serde::{Deserialize, Serialize};

use super::FieldTopologyIdentity;

use super::{
    AnalysisExecutionError, AnalysisExecutionResult, FieldRecoveryPlan,
    FieldRecoveryRequestIdentity, SolverResultIdentity, StepIdentity,
    StructuralFrequencyEntryResultIdentity,
};

const FIELD_RECOVERY_PREFIX: &[u8] = b"runmat-analysis-field-recovery/v1\0";
const FIELD_RECOVERY_LIMITS: CanonicalLimits = CanonicalLimits::new(1024 * 1024, 65_536, 4096, 32);

/// Exact numerical output whose physical fields are being recovered.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FieldRecoverySource {
    SolverResult(SolverResultIdentity),
    StructuralFrequencyEntry(StructuralFrequencyEntryResultIdentity),
}

impl FieldRecoverySource {
    pub const fn solver_result(self) -> Option<SolverResultIdentity> {
        match self {
            Self::SolverResult(result) => Some(result),
            Self::StructuralFrequencyEntry(_) => None,
        }
    }
}

/// Immutable request for family-owned recovery from one exact numerical result.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, try_from = "FieldRecoveryRequestWire")]
pub struct FieldRecoveryRequest {
    pub schema_version: u16,
    identity: FieldRecoveryRequestIdentity,
    pub producer: StepIdentity,
    pub source: FieldRecoverySource,
    pub topology: FieldTopologyIdentity,
    pub plan: FieldRecoveryPlan,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct FieldRecoveryRequestWire {
    schema_version: u16,
    identity: FieldRecoveryRequestIdentity,
    producer: StepIdentity,
    source: FieldRecoverySource,
    topology: FieldTopologyIdentity,
    plan: FieldRecoveryPlan,
}

impl TryFrom<FieldRecoveryRequestWire> for FieldRecoveryRequest {
    type Error = AnalysisExecutionError;

    fn try_from(wire: FieldRecoveryRequestWire) -> Result<Self, Self::Error> {
        let request = Self {
            schema_version: wire.schema_version,
            identity: wire.identity,
            producer: wire.producer,
            source: wire.source,
            topology: wire.topology,
            plan: wire.plan,
        };
        request.validate()?;
        Ok(request)
    }
}

impl FieldRecoveryRequest {
    pub fn new(
        producer: StepIdentity,
        source: FieldRecoverySource,
        topology: FieldTopologyIdentity,
        plan: FieldRecoveryPlan,
    ) -> AnalysisExecutionResult<Self> {
        let request = Self {
            schema_version: 1,
            identity: FieldRecoveryRequestIdentity::derive(
                "field-recovery-request",
                &(producer, source, topology, &plan),
            )?,
            producer,
            source,
            topology,
            plan,
        };
        request.validate()?;
        Ok(request)
    }

    pub const fn identity(&self) -> FieldRecoveryRequestIdentity {
        self.identity
    }

    pub fn validate(&self) -> AnalysisExecutionResult<()> {
        self.plan.validate()?;
        let expected = FieldRecoveryRequestIdentity::derive(
            "field-recovery-request",
            &(self.producer, self.source, self.topology, &self.plan),
        )?;
        if self.schema_version != 1
            || !self.topology.has_valid_identity()
            || self.identity != expected
        {
            return Err(AnalysisExecutionError::Identity(
                "field recovery request identity does not converge",
            ));
        }
        Ok(())
    }

    pub fn canonical_bytes(&self) -> AnalysisExecutionResult<Vec<u8>> {
        self.validate()?;
        Ok(encode_contract(
            FIELD_RECOVERY_PREFIX,
            "field-recovery-request",
            self,
            FIELD_RECOVERY_LIMITS,
        )?)
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> AnalysisExecutionResult<Self> {
        let request = decode_contract(
            FIELD_RECOVERY_PREFIX,
            "field-recovery-request",
            bytes,
            FIELD_RECOVERY_LIMITS,
        )?;
        Ok(request)
    }
}

#[cfg(test)]
mod tests {
    use runmat_analysis_core::{
        FieldLocation, ObservationQuantity, ObservationValueShape, RecoveryPolicy,
    };
    use runmat_execution::Digest;

    use crate::execution::{FieldRecoveryCost, FieldRecoveryRequirement};
    use crate::physics::structural::topology::CanonicalStructuralTopologyIdentity;
    use crate::results::field::RecoveredFieldEntityDomain;

    use super::*;

    #[test]
    fn recovery_source_kind_is_canonical_and_identity_significant() {
        let sequence = super::super::SequenceIdentity::new(Digest::sha256("sequence")).unwrap();
        let step = StepIdentity::new(sequence, Digest::sha256("step")).unwrap();
        let solver =
            SolverResultIdentity::from_step_result(step, Digest::sha256("result")).unwrap();
        let entry = StructuralFrequencyEntryResultIdentity::derive(
            "test-frequency-entry",
            &Digest::sha256("result"),
        )
        .unwrap();
        let topology = CanonicalStructuralTopologyIdentity::Explicit {
            structural_topology_digest: Digest::sha256("topology"),
        };
        let solver_request = FieldRecoveryRequest::new(
            step,
            FieldRecoverySource::SolverResult(solver),
            crate::execution::FieldTopologyIdentity::Structural { topology },
            FieldRecoveryPlan::default(),
        )
        .unwrap();
        let frequency_request = FieldRecoveryRequest::new(
            step,
            FieldRecoverySource::StructuralFrequencyEntry(entry),
            crate::execution::FieldTopologyIdentity::Structural { topology },
            FieldRecoveryPlan::default(),
        )
        .unwrap();

        assert_ne!(solver_request.identity(), frequency_request.identity());
        assert_eq!(
            FieldRecoveryRequest::from_canonical_bytes(
                &frequency_request.canonical_bytes().unwrap()
            )
            .unwrap(),
            frequency_request
        );
        assert_eq!(solver_request.source.solver_result(), Some(solver));
        assert_eq!(frequency_request.source.solver_result(), None);
    }

    #[test]
    fn recovery_method_and_topology_cost_are_identity_significant() {
        let sequence = super::super::SequenceIdentity::new(Digest::sha256("sequence")).unwrap();
        let step = StepIdentity::new(sequence, Digest::sha256("step")).unwrap();
        let solver =
            SolverResultIdentity::from_step_result(step, Digest::sha256("result")).unwrap();
        let topology = CanonicalStructuralTopologyIdentity::Explicit {
            structural_topology_digest: Digest::sha256("topology"),
        };
        let requirement = FieldRecoveryRequirement {
            support: Default::default(),
            quantity: ObservationQuantity::VonMisesStress,
            source_location: FieldLocation::Element,
            output_location: FieldLocation::Node,
            recovery: RecoveryPolicy::VolumeWeightedAverage,
            value_shape: ObservationValueShape::Scalar,
            entity_domain: RecoveredFieldEntityDomain::Nodes,
            cost: FieldRecoveryCost {
                source_entity_count: 2,
                output_entity_count: 7,
                topology_incidence_count: 8,
                arithmetic_operation_count: 23,
            },
        };
        let planned = FieldRecoveryRequest::new(
            step,
            FieldRecoverySource::SolverResult(solver),
            crate::execution::FieldTopologyIdentity::Structural { topology },
            FieldRecoveryPlan {
                requirements: vec![requirement],
            },
        )
        .unwrap();
        let unplanned = FieldRecoveryRequest::new(
            step,
            FieldRecoverySource::SolverResult(solver),
            crate::execution::FieldTopologyIdentity::Structural { topology },
            FieldRecoveryPlan::default(),
        )
        .unwrap();

        assert_ne!(planned.identity(), unplanned.identity());
        assert_eq!(
            FieldRecoveryRequest::from_canonical_bytes(&planned.canonical_bytes().unwrap())
                .unwrap(),
            planned
        );
    }
}
