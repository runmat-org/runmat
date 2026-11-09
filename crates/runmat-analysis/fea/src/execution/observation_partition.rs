use runmat_canonical_codec::{decode_contract, encode_contract, CanonicalLimits};
use serde::{Deserialize, Serialize};

use crate::results::{
    field::{RecoveredFieldPartitionSet, RecoveredFieldPartitionSetIdentity},
    observation::{
        ObservationEventAttribution, ObservationPartition, ObservationPartitionContribution,
        ObservationPartitionContributionIdentity, ObservationPartitionIdentity,
        ObservationPartitionPlan, ObservationPartitionPlanIdentity,
    },
};

use super::{
    AnalysisExecutionError, AnalysisExecutionResult, ObservationEvaluationRequest,
    ObservationEvaluationRequestIdentity, ObservationPartitionEvaluationRequestIdentity,
    ObservationPartitionReductionRequestIdentity, StepIdentity,
};

mod attribution;
mod identity;
pub(crate) use identity::validate_published_observation_source;
use identity::{evaluation_identity, reduction_identity};

use attribution::{validate_contribution_attribution, validate_event_attribution};

const PARTITION_EVALUATION_PREFIX: &[u8] = b"runmat-analysis-observation-partition-evaluation/v1\0";
const PARTITION_REDUCTION_PREFIX: &[u8] = b"runmat-analysis-observation-partition-reduction/v1\0";
const PARTITION_REQUEST_LIMITS: CanonicalLimits =
    crate::contracts::canonical_digest::wire_bounded(CanonicalLimits::new(0, 0, 65_536, 64));

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(
    deny_unknown_fields,
    try_from = "ObservationPartitionEvaluationRequestWire"
)]
pub struct ObservationPartitionEvaluationRequest {
    pub schema_version: u16,
    identity: ObservationPartitionEvaluationRequestIdentity,
    pub producer: StepIdentity,
    pub source_request: ObservationEvaluationRequestIdentity,
    pub partition_plan: ObservationPartitionPlanIdentity,
    pub partition: ObservationPartitionIdentity,
    pub source_field_partitions: RecoveredFieldPartitionSetIdentity,
    pub event_attribution: Option<ObservationEventAttribution>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ObservationPartitionEvaluationRequestWire {
    schema_version: u16,
    identity: ObservationPartitionEvaluationRequestIdentity,
    producer: StepIdentity,
    source_request: ObservationEvaluationRequestIdentity,
    partition_plan: ObservationPartitionPlanIdentity,
    partition: ObservationPartitionIdentity,
    source_field_partitions: RecoveredFieldPartitionSetIdentity,
    event_attribution: Option<ObservationEventAttribution>,
}

impl TryFrom<ObservationPartitionEvaluationRequestWire> for ObservationPartitionEvaluationRequest {
    type Error = AnalysisExecutionError;

    fn try_from(wire: ObservationPartitionEvaluationRequestWire) -> Result<Self, Self::Error> {
        let request = Self {
            schema_version: wire.schema_version,
            identity: wire.identity,
            producer: wire.producer,
            source_request: wire.source_request,
            partition_plan: wire.partition_plan,
            partition: wire.partition,
            source_field_partitions: wire.source_field_partitions,
            event_attribution: wire.event_attribution,
        };
        request.validate_identity()?;
        Ok(request)
    }
}

impl ObservationPartitionEvaluationRequest {
    pub fn new(
        source: &ObservationEvaluationRequest,
        partition_plan: &ObservationPartitionPlan,
        partition: &ObservationPartition,
        source_field: &RecoveredFieldPartitionSet,
        event_attribution: Option<ObservationEventAttribution>,
    ) -> AnalysisExecutionResult<Self> {
        source.validate()?;
        partition_plan
            .validate_for(&source.plan)
            .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
        source_field
            .validate()
            .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
        source.selection.validate_partitions(source_field)?;
        validate_event_attribution(source, partition_plan, source_field, event_attribution)?;
        if partition_plan
            .partitions
            .get(partition.ordinal as usize)
            .is_none_or(|expected| expected != partition)
        {
            return Err(AnalysisExecutionError::Identity(
                "observation partition evaluation names a noncanonical partition",
            ));
        }
        let request = Self {
            schema_version: 1,
            // `source.validate()` and `partition_plan.validate_for` succeeded above.
            identity: Self::published_identity_validated(
                source,
                partition_plan,
                partition,
                source_field.identity,
                event_attribution,
            )?,
            producer: source.producer,
            source_request: source.identity(),
            partition_plan: partition_plan.identity,
            partition: partition.identity,
            source_field_partitions: source_field.identity,
            event_attribution,
        };
        request.validate_identity()?;
        Ok(request)
    }

    pub const fn identity(&self) -> ObservationPartitionEvaluationRequestIdentity {
        self.identity
    }

    pub fn validate_for(
        &self,
        source: &ObservationEvaluationRequest,
        partition_plan: &ObservationPartitionPlan,
        partition: &ObservationPartition,
        source_field: &RecoveredFieldPartitionSet,
    ) -> AnalysisExecutionResult<()> {
        self.validate_identity()?;
        source.validate()?;
        partition_plan
            .validate_for(&source.plan)
            .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
        source_field
            .validate()
            .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
        source.selection.validate_partitions(source_field)?;
        validate_event_attribution(source, partition_plan, source_field, self.event_attribution)?;
        if self.producer != source.producer
            || self.source_request != source.identity()
            || self.partition_plan != partition_plan.identity
            || self.partition != partition.identity
            || self.source_field_partitions != source_field.identity
            || partition_plan
                .partitions
                .get(partition.ordinal as usize)
                .is_none_or(|expected| expected != partition)
        {
            return Err(AnalysisExecutionError::Identity(
                "observation partition evaluation inputs differ from its request",
            ));
        }
        Ok(())
    }

    pub fn canonical_bytes(&self) -> AnalysisExecutionResult<Vec<u8>> {
        self.validate_identity()?;
        Ok(encode_contract(
            PARTITION_EVALUATION_PREFIX,
            "observation-partition-evaluation-request",
            self,
            PARTITION_REQUEST_LIMITS,
        )?)
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> AnalysisExecutionResult<Self> {
        Ok(decode_contract(
            PARTITION_EVALUATION_PREFIX,
            "observation-partition-evaluation-request",
            bytes,
            PARTITION_REQUEST_LIMITS,
        )?)
    }

    fn validate_identity(&self) -> AnalysisExecutionResult<()> {
        let expected = evaluation_identity(
            self.producer,
            self.source_request,
            self.partition_plan,
            self.partition,
            self.source_field_partitions,
            self.event_attribution,
        )?;
        if let Some(attribution) = self.event_attribution {
            attribution
                .validate()
                .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
        }
        if self.schema_version != 1 || self.identity != expected {
            return Err(AnalysisExecutionError::Identity(
                "observation partition evaluation request identity does not converge",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(
    deny_unknown_fields,
    try_from = "ObservationPartitionReductionRequestWire"
)]
pub struct ObservationPartitionReductionRequest {
    pub schema_version: u16,
    identity: ObservationPartitionReductionRequestIdentity,
    pub producer: StepIdentity,
    pub source_request: ObservationEvaluationRequestIdentity,
    pub partition_plan: ObservationPartitionPlanIdentity,
    pub contributions: Vec<ObservationPartitionContributionIdentity>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ObservationPartitionReductionRequestWire {
    schema_version: u16,
    identity: ObservationPartitionReductionRequestIdentity,
    producer: StepIdentity,
    source_request: ObservationEvaluationRequestIdentity,
    partition_plan: ObservationPartitionPlanIdentity,
    contributions: Vec<ObservationPartitionContributionIdentity>,
}

impl TryFrom<ObservationPartitionReductionRequestWire> for ObservationPartitionReductionRequest {
    type Error = AnalysisExecutionError;

    fn try_from(wire: ObservationPartitionReductionRequestWire) -> Result<Self, Self::Error> {
        let request = Self {
            schema_version: wire.schema_version,
            identity: wire.identity,
            producer: wire.producer,
            source_request: wire.source_request,
            partition_plan: wire.partition_plan,
            contributions: wire.contributions,
        };
        request.validate_identity()?;
        Ok(request)
    }
}

impl ObservationPartitionReductionRequest {
    pub fn new(
        source: &ObservationEvaluationRequest,
        partition_plan: &ObservationPartitionPlan,
        contributions: &[ObservationPartitionContribution],
    ) -> AnalysisExecutionResult<Self> {
        source.validate()?;
        partition_plan
            .validate_for(&source.plan)
            .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
        validate_contributions(partition_plan, contributions)?;
        validate_contribution_attribution(source, partition_plan, contributions)?;
        let contribution_identities = contributions
            .iter()
            .map(|contribution| contribution.identity)
            .collect::<Vec<_>>();
        let request = Self {
            schema_version: 1,
            // `source.validate()` and `partition_plan.validate_for` succeeded above.
            identity: Self::published_identity_validated(
                source,
                partition_plan,
                &contribution_identities,
            )?,
            producer: source.producer,
            source_request: source.identity(),
            partition_plan: partition_plan.identity,
            contributions: contribution_identities,
        };
        request.validate_identity()?;
        Ok(request)
    }

    pub const fn identity(&self) -> ObservationPartitionReductionRequestIdentity {
        self.identity
    }

    pub fn validate_for(
        &self,
        source: &ObservationEvaluationRequest,
        partition_plan: &ObservationPartitionPlan,
        contributions: &[ObservationPartitionContribution],
    ) -> AnalysisExecutionResult<()> {
        self.validate_identity()?;
        source.validate()?;
        partition_plan
            .validate_for(&source.plan)
            .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
        validate_contributions(partition_plan, contributions)?;
        validate_contribution_attribution(source, partition_plan, contributions)?;
        if self.producer != source.producer
            || self.source_request != source.identity()
            || self.partition_plan != partition_plan.identity
            || self.contributions
                != contributions
                    .iter()
                    .map(|contribution| contribution.identity)
                    .collect::<Vec<_>>()
        {
            return Err(AnalysisExecutionError::Identity(
                "observation partition reduction inputs differ from its request",
            ));
        }
        Ok(())
    }

    pub fn canonical_bytes(&self) -> AnalysisExecutionResult<Vec<u8>> {
        self.validate_identity()?;
        Ok(encode_contract(
            PARTITION_REDUCTION_PREFIX,
            "observation-partition-reduction-request",
            self,
            PARTITION_REQUEST_LIMITS,
        )?)
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> AnalysisExecutionResult<Self> {
        Ok(decode_contract(
            PARTITION_REDUCTION_PREFIX,
            "observation-partition-reduction-request",
            bytes,
            PARTITION_REQUEST_LIMITS,
        )?)
    }

    fn validate_identity(&self) -> AnalysisExecutionResult<()> {
        let expected = reduction_identity(
            self.producer,
            self.source_request,
            self.partition_plan,
            &self.contributions,
        )?;
        if self.schema_version != 1 || self.contributions.is_empty() || self.identity != expected {
            return Err(AnalysisExecutionError::Identity(
                "observation partition reduction request identity does not converge",
            ));
        }
        Ok(())
    }
}

fn validate_contributions(
    partition_plan: &ObservationPartitionPlan,
    contributions: &[ObservationPartitionContribution],
) -> AnalysisExecutionResult<()> {
    if contributions.len() != partition_plan.partitions.len() {
        return Err(AnalysisExecutionError::Identity(
            "observation reduction does not contain every planned partition",
        ));
    }
    for (partition, contribution) in partition_plan.partitions.iter().zip(contributions) {
        contribution
            .validate()
            .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
        if contribution.partition_plan != partition_plan.identity
            || contribution.partition != partition.identity
            || contribution.partition_ordinal != partition.ordinal
        {
            return Err(AnalysisExecutionError::Identity(
                "observation contribution order or ownership differs from its plan",
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
