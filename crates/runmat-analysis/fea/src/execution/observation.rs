use runmat_canonical_codec::{decode_contract, encode_contract, CanonicalLimits};
use serde::{Deserialize, Serialize};

use crate::results::{
    field::RecoveredFieldSample,
    observation::{
        resolved_observation_accepts_sample, ObservationEventAttribution,
        ObservationFieldSelection, ResolvedObservation, ResolvedObservationPlan,
    },
};

use super::{
    AnalysisExecutionError, AnalysisExecutionResult, ObservationEvaluationRequestIdentity,
    StepIdentity,
};

const OBSERVATION_EVALUATION_PREFIX: &[u8] = b"runmat-analysis-observation-evaluation/v2\0";
/// The request grows with the observed regions, so no fixed size caps it: the object inventory
/// admits the stored request against the run's budget (`ObjectInventoryLimits::admitted`), and
/// the codec keeps only the format's string and nesting bounds.
const OBSERVATION_EVALUATION_LIMITS: CanonicalLimits =
    crate::contracts::canonical_digest::wire_bounded(CanonicalLimits::new(0, 0, 65_536, 64));

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, try_from = "ObservationEvaluationRequestWire")]
pub struct ObservationEvaluationRequest {
    pub schema_version: u16,
    identity: ObservationEvaluationRequestIdentity,
    pub producer: StepIdentity,
    pub selection: ObservationFieldSelection,
    pub plan: ResolvedObservationPlan,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ObservationEvaluationRequestWire {
    schema_version: u16,
    identity: ObservationEvaluationRequestIdentity,
    producer: StepIdentity,
    selection: ObservationFieldSelection,
    plan: ResolvedObservationPlan,
}

impl TryFrom<ObservationEvaluationRequestWire> for ObservationEvaluationRequest {
    type Error = AnalysisExecutionError;

    fn try_from(wire: ObservationEvaluationRequestWire) -> Result<Self, Self::Error> {
        let request = Self {
            schema_version: wire.schema_version,
            identity: wire.identity,
            producer: wire.producer,
            selection: wire.selection,
            plan: wire.plan,
        };
        request.validate()?;
        Ok(request)
    }
}

impl ObservationEvaluationRequest {
    pub fn new(
        selection: ObservationFieldSelection,
        plan: ResolvedObservationPlan,
    ) -> AnalysisExecutionResult<Self> {
        selection.validate()?;
        let producer = selection.producer();
        let request = Self {
            schema_version: 2,
            identity: ObservationEvaluationRequestIdentity::derive(
                "observation-evaluation-request",
                &(
                    producer,
                    selection.identity(),
                    plan.identity,
                    plan.topology_digest,
                ),
            )?,
            producer,
            selection,
            plan,
        };
        request.validate()?;
        Ok(request)
    }

    pub const fn identity(&self) -> ObservationEvaluationRequestIdentity {
        self.identity
    }

    /// Exact authored-order coverage at a coordinate, bound to the selected source fields.
    /// Event sampling requires the caller's actual attribution; metadata callers pass None.
    pub(crate) fn scheduled_observations(
        &self,
        sample: RecoveredFieldSample,
        event_attribution: Option<ObservationEventAttribution>,
    ) -> AnalysisExecutionResult<Vec<&ResolvedObservation>> {
        self.validate()?;
        let mut scheduled = Vec::new();
        for observation in &self.plan.observations {
            if !resolved_observation_accepts_sample(
                &self.plan,
                observation,
                sample,
                event_attribution,
            )
            .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?
            {
                continue;
            }
            if !self.selection.fields().iter().any(|field| {
                field.definition.identity() == observation.source_field_definition_identity
                    && field.sample == sample
            }) {
                return Err(AnalysisExecutionError::Identity(
                    "scheduled observation coordinate or definition absent from selected source",
                ));
            }
            scheduled.push(observation);
        }
        Ok(scheduled)
    }

    pub fn validate(&self) -> AnalysisExecutionResult<()> {
        self.selection.validate()?;
        if self.schema_version != 2 || self.plan.observations.is_empty() {
            return Err(AnalysisExecutionError::Invalid(
                "observation evaluation requires a supported schema and nonempty plan".into(),
            ));
        }
        let expected = ObservationEvaluationRequestIdentity::derive(
            "observation-evaluation-request",
            &(
                self.producer,
                self.selection.identity(),
                self.plan.identity,
                self.plan.topology_digest,
            ),
        )?;
        if self.producer != self.selection.producer()
            || self.plan.topology_digest != self.selection.source().observation_topology_digest()
            || self.identity != expected
        {
            return Err(AnalysisExecutionError::Identity(
                "observation evaluation request identity does not converge",
            ));
        }
        if self.plan.observations.iter().any(|observation| {
            !self.selection.fields().iter().any(|entry| {
                entry.definition.identity() == observation.source_field_definition_identity
            })
        }) {
            return Err(AnalysisExecutionError::Identity(
                "observation plan field is absent from its source selection",
            ));
        }
        Ok(())
    }

    pub fn canonical_bytes(&self) -> AnalysisExecutionResult<Vec<u8>> {
        self.validate()?;
        Ok(encode_contract(
            OBSERVATION_EVALUATION_PREFIX,
            "observation-evaluation-request",
            self,
            OBSERVATION_EVALUATION_LIMITS,
        )?)
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> AnalysisExecutionResult<Self> {
        let request = decode_contract(
            OBSERVATION_EVALUATION_PREFIX,
            "observation-evaluation-request",
            bytes,
            OBSERVATION_EVALUATION_LIMITS,
        )?;
        Ok(request)
    }
}

#[cfg(test)]
#[test]
fn observation_requests_are_bounded_by_the_admitted_inventory_not_a_fixed_size() {
    let wire = crate::contracts::canonical_digest::WIRE_BOUND;
    assert_eq!(OBSERVATION_EVALUATION_LIMITS.maximum_encoded_bytes, wire);
    assert_eq!(OBSERVATION_EVALUATION_LIMITS.maximum_collection_items, wire);
}
