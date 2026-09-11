use runmat_canonical_codec::{decode_contract, encode_contract, CanonicalLimits};
use serde::{Deserialize, Serialize};

use crate::planning::resource::{FeaResourcePlan, FeaResourcePlanIdentity};
use crate::results::{
    field::{RecoveredFieldPartitionSet, RecoveredFieldPartitionSetIdentity},
    observation::{
        ObservationPartition, ObservationPartitionContribution,
        ObservationPartitionContributionIdentity, ObservationPartitionIdentity,
        ObservationPartitionPlan, ObservationPartitionPlanIdentity,
    },
};

use super::{
    AnalysisExecutionError, AnalysisExecutionResult, ObservationEvaluationRequest,
    ObservationEvaluationRequestIdentity, ObservationPartitionEvaluationRequestIdentity,
    ObservationPartitionReductionRequestIdentity, StepIdentity,
};

const PARTITION_EVALUATION_PREFIX: &[u8] = b"runmat-analysis-observation-partition-evaluation/v1\0";
const PARTITION_REDUCTION_PREFIX: &[u8] = b"runmat-analysis-observation-partition-reduction/v1\0";
const PARTITION_REQUEST_LIMITS: CanonicalLimits =
    CanonicalLimits::new(64 * 1024 * 1024, 1_000_000, 65_536, 64);

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
    pub resource_plan: FeaResourcePlanIdentity,
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
    resource_plan: FeaResourcePlanIdentity,
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
            resource_plan: wire.resource_plan,
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
        resource_plan: &FeaResourcePlan,
    ) -> AnalysisExecutionResult<Self> {
        source.validate()?;
        partition_plan
            .validate_for(&source.plan)
            .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
        source_field
            .validate()
            .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
        resource_plan
            .validate_identity()
            .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
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
            identity: ObservationPartitionEvaluationRequestIdentity::derive(
                "observation-partition-evaluation-request",
                &(
                    source.producer,
                    source.identity(),
                    partition_plan.identity,
                    partition.identity,
                    source_field.identity,
                    resource_plan.identity,
                ),
            )?,
            producer: source.producer,
            source_request: source.identity(),
            partition_plan: partition_plan.identity,
            partition: partition.identity,
            source_field_partitions: source_field.identity,
            resource_plan: resource_plan.identity,
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
        resource_plan: &FeaResourcePlan,
    ) -> AnalysisExecutionResult<()> {
        self.validate_identity()?;
        source.validate()?;
        partition_plan
            .validate_for(&source.plan)
            .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
        source_field
            .validate()
            .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
        resource_plan
            .validate_identity()
            .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
        if self.producer != source.producer
            || self.source_request != source.identity()
            || self.partition_plan != partition_plan.identity
            || self.partition != partition.identity
            || self.source_field_partitions != source_field.identity
            || self.resource_plan != resource_plan.identity
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
        let expected = ObservationPartitionEvaluationRequestIdentity::derive(
            "observation-partition-evaluation-request",
            &(
                self.producer,
                self.source_request,
                self.partition_plan,
                self.partition,
                self.source_field_partitions,
                self.resource_plan,
            ),
        )?;
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
    pub resource_plan: FeaResourcePlanIdentity,
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
    resource_plan: FeaResourcePlanIdentity,
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
            resource_plan: wire.resource_plan,
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
        resource_plan: &FeaResourcePlan,
    ) -> AnalysisExecutionResult<Self> {
        source.validate()?;
        partition_plan
            .validate_for(&source.plan)
            .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
        validate_contributions(partition_plan, contributions)?;
        resource_plan
            .validate_identity()
            .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
        let contribution_identities = contributions
            .iter()
            .map(|contribution| contribution.identity)
            .collect::<Vec<_>>();
        let request = Self {
            schema_version: 1,
            identity: ObservationPartitionReductionRequestIdentity::derive(
                "observation-partition-reduction-request",
                &(
                    source.producer,
                    source.identity(),
                    partition_plan.identity,
                    &contribution_identities,
                    resource_plan.identity,
                ),
            )?,
            producer: source.producer,
            source_request: source.identity(),
            partition_plan: partition_plan.identity,
            contributions: contribution_identities,
            resource_plan: resource_plan.identity,
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
        resource_plan: &FeaResourcePlan,
    ) -> AnalysisExecutionResult<()> {
        self.validate_identity()?;
        source.validate()?;
        partition_plan
            .validate_for(&source.plan)
            .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
        validate_contributions(partition_plan, contributions)?;
        resource_plan
            .validate_identity()
            .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
        if self.producer != source.producer
            || self.source_request != source.identity()
            || self.partition_plan != partition_plan.identity
            || self.resource_plan != resource_plan.identity
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
        let expected = ObservationPartitionReductionRequestIdentity::derive(
            "observation-partition-reduction-request",
            &(
                self.producer,
                self.source_request,
                self.partition_plan,
                &self.contributions,
                self.resource_plan,
            ),
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
