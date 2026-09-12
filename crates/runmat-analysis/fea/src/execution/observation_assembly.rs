use runmat_canonical_codec::{decode_contract, encode_contract, CanonicalLimits};
use serde::{Deserialize, Serialize};

use crate::planning::resource::{FeaResourcePlan, FeaResourcePlanIdentity};
use crate::results::observation::{
    ObservationPartitionEvidenceIdentity, ObservationResult, ObservationResultIdentity,
};

use super::{
    AnalysisExecutionError, AnalysisExecutionResult, ObservationAssemblyRequestIdentity,
    ObservationEvaluationRequest, ObservationEvaluationRequestIdentity, StepIdentity,
};

const ASSEMBLY_PREFIX: &[u8] = b"runmat-analysis-observation-assembly/v1\0";
const ASSEMBLY_LIMITS: CanonicalLimits =
    CanonicalLimits::new(16 * 1024 * 1024, 1_000_000, 65_536, 32);

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, try_from = "ObservationAssemblyRequestWire")]
pub struct ObservationAssemblyRequest {
    pub schema_version: u16,
    identity: ObservationAssemblyRequestIdentity,
    pub producer: StepIdentity,
    pub source_request: ObservationEvaluationRequestIdentity,
    pub results: Vec<ObservationResultIdentity>,
    pub partition_evidence: Vec<ObservationPartitionEvidenceIdentity>,
    pub resource_plan: FeaResourcePlanIdentity,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ObservationAssemblyRequestWire {
    schema_version: u16,
    identity: ObservationAssemblyRequestIdentity,
    producer: StepIdentity,
    source_request: ObservationEvaluationRequestIdentity,
    results: Vec<ObservationResultIdentity>,
    partition_evidence: Vec<ObservationPartitionEvidenceIdentity>,
    resource_plan: FeaResourcePlanIdentity,
}

impl TryFrom<ObservationAssemblyRequestWire> for ObservationAssemblyRequest {
    type Error = AnalysisExecutionError;

    fn try_from(wire: ObservationAssemblyRequestWire) -> Result<Self, Self::Error> {
        let request = Self {
            schema_version: wire.schema_version,
            identity: wire.identity,
            producer: wire.producer,
            source_request: wire.source_request,
            results: wire.results,
            partition_evidence: wire.partition_evidence,
            resource_plan: wire.resource_plan,
        };
        request.validate_identity()?;
        Ok(request)
    }
}

impl ObservationAssemblyRequest {
    pub fn new(
        source: &ObservationEvaluationRequest,
        results: &[ObservationResult],
        resource_plan: &FeaResourcePlan,
    ) -> AnalysisExecutionResult<Self> {
        validate_results(source, results)?;
        validate_acceleration_allocations(resource_plan, results)?;
        resource_plan
            .validate_identity()
            .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
        let result_identities = results
            .iter()
            .map(|result| result.logical_identity)
            .collect::<Vec<_>>();
        let evidence_identities = results
            .iter()
            .map(|result| result.partition_evidence.identity)
            .collect::<Vec<_>>();
        let request = Self {
            schema_version: 1,
            identity: ObservationAssemblyRequestIdentity::derive(
                "observation-assembly-request",
                &(
                    source.producer,
                    source.identity(),
                    &result_identities,
                    &evidence_identities,
                    resource_plan.identity,
                ),
            )?,
            producer: source.producer,
            source_request: source.identity(),
            results: result_identities,
            partition_evidence: evidence_identities,
            resource_plan: resource_plan.identity,
        };
        request.validate_identity()?;
        Ok(request)
    }

    pub const fn identity(&self) -> ObservationAssemblyRequestIdentity {
        self.identity
    }

    pub fn validate_for(
        &self,
        source: &ObservationEvaluationRequest,
        results: &[ObservationResult],
        resource_plan: &FeaResourcePlan,
    ) -> AnalysisExecutionResult<()> {
        self.validate_identity()?;
        validate_results(source, results)?;
        validate_acceleration_allocations(resource_plan, results)?;
        resource_plan
            .validate_identity()
            .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
        if self.producer != source.producer
            || self.source_request != source.identity()
            || self.resource_plan != resource_plan.identity
            || self.results
                != results
                    .iter()
                    .map(|result| result.logical_identity)
                    .collect::<Vec<_>>()
            || self.partition_evidence
                != results
                    .iter()
                    .map(|result| result.partition_evidence.identity)
                    .collect::<Vec<_>>()
        {
            return Err(AnalysisExecutionError::Identity(
                "observation assembly inputs differ from its request",
            ));
        }
        Ok(())
    }

    pub fn canonical_bytes(&self) -> AnalysisExecutionResult<Vec<u8>> {
        self.validate_identity()?;
        Ok(encode_contract(
            ASSEMBLY_PREFIX,
            "observation-assembly-request",
            self,
            ASSEMBLY_LIMITS,
        )?)
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> AnalysisExecutionResult<Self> {
        Ok(decode_contract(
            ASSEMBLY_PREFIX,
            "observation-assembly-request",
            bytes,
            ASSEMBLY_LIMITS,
        )?)
    }

    fn validate_identity(&self) -> AnalysisExecutionResult<()> {
        let expected = ObservationAssemblyRequestIdentity::derive(
            "observation-assembly-request",
            &(
                self.producer,
                self.source_request,
                &self.results,
                &self.partition_evidence,
                self.resource_plan,
            ),
        )?;
        if self.schema_version != 1
            || self.results.is_empty()
            || self.results.len() != self.partition_evidence.len()
            || self.identity != expected
        {
            return Err(AnalysisExecutionError::Identity(
                "observation assembly request identity does not converge",
            ));
        }
        Ok(())
    }
}

fn validate_results(
    source: &ObservationEvaluationRequest,
    results: &[ObservationResult],
) -> AnalysisExecutionResult<()> {
    source.validate()?;
    if results.len() != source.plan.observations.len() {
        return Err(AnalysisExecutionError::Identity(
            "observation assembly does not contain every resolved observation",
        ));
    }
    for (observation, result) in source.plan.observations.iter().zip(results) {
        result
            .validate()
            .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
        if result.plan_identity != source.plan.identity
            || result.topology_digest != source.plan.topology_digest
            || result.observation_id != observation.spec.observation_id
        {
            return Err(AnalysisExecutionError::Identity(
                "observation assembly result order or source differs from its resolved plan",
            ));
        }
    }
    Ok(())
}

pub(crate) fn validate_acceleration_allocations(
    resource_plan: &FeaResourcePlan,
    results: &[ObservationResult],
) -> AnalysisExecutionResult<()> {
    let evidence = results
        .iter()
        .flat_map(|result| &result.partition_evidence.accelerations)
        .collect::<Vec<_>>();
    if evidence.is_empty() {
        return Ok(());
    }
    let [allocation] = resource_plan.request.accelerators.as_slice() else {
        return Err(AnalysisExecutionError::Identity(
            "published observation acceleration has no unique resource-plan allocation",
        ));
    };
    for operation in evidence {
        operation
            .validate_for_allocation(allocation)
            .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
    }
    Ok(())
}
