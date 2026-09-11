use runmat_canonical_codec::{decode_contract, encode_contract, CanonicalLimits};
use serde::{Deserialize, Serialize};

use crate::planning::resource::{FeaResourcePlan, FeaResourcePlanIdentity};
use crate::results::{
    field::{RecoveredFieldPartitionSet, RecoveredFieldPartitionSetIdentity},
    observation::{
        ObservationEventAttribution, ObservationPartition, ObservationPartitionContribution,
        ObservationPartitionContributionIdentity, ObservationPartitionIdentity,
        ObservationPartitionPlan, ObservationPartitionPlanIdentity, SolverEventSource,
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
    pub event_attribution: Option<ObservationEventAttribution>,
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
    event_attribution: Option<ObservationEventAttribution>,
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
            event_attribution: wire.event_attribution,
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
        event_attribution: Option<ObservationEventAttribution>,
        resource_plan: &FeaResourcePlan,
    ) -> AnalysisExecutionResult<Self> {
        source.validate()?;
        partition_plan
            .validate_for(&source.plan)
            .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
        source_field
            .validate()
            .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
        validate_event_attribution(source, partition_plan, source_field, event_attribution)?;
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
                    event_attribution,
                    resource_plan.identity,
                ),
            )?,
            producer: source.producer,
            source_request: source.identity(),
            partition_plan: partition_plan.identity,
            partition: partition.identity,
            source_field_partitions: source_field.identity,
            event_attribution,
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
        event_attribution: Option<ObservationEventAttribution>,
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
        validate_event_attribution(source, partition_plan, source_field, event_attribution)?;
        resource_plan
            .validate_identity()
            .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
        if self.producer != source.producer
            || self.source_request != source.identity()
            || self.partition_plan != partition_plan.identity
            || self.partition != partition.identity
            || self.source_field_partitions != source_field.identity
            || self.event_attribution != event_attribution
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
                self.event_attribution,
                self.resource_plan,
            ),
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

fn validate_event_attribution(
    source: &ObservationEvaluationRequest,
    partition_plan: &ObservationPartitionPlan,
    source_field: &RecoveredFieldPartitionSet,
    attribution: Option<ObservationEventAttribution>,
) -> AnalysisExecutionResult<()> {
    let observation = source
        .plan
        .observations
        .iter()
        .find(|observation| observation.spec.observation_id == partition_plan.observation_id)
        .ok_or(AnalysisExecutionError::Identity(
            "observation partition event is absent from its source plan",
        ))?;
    match (&observation.spec.sampling, attribution) {
        (runmat_analysis_core::SamplingPolicy::Event(expected), Some(attribution)) => {
            attribution
                .validate()
                .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
            if attribution.event() != *expected || attribution.sample() != source_field.sample {
                return Err(AnalysisExecutionError::Identity(
                    "observation partition event differs from its plan or source sample",
                ));
            }
            if let ObservationEventAttribution::Solver(
                crate::results::observation::SolverEventAttribution {
                    source: SolverEventSource::CompletedResult { result },
                    ..
                },
            ) = attribution
            {
                if result != source.source_result {
                    return Err(AnalysisExecutionError::Identity(
                        "completed solver event names a different source result",
                    ));
                }
            }
            Ok(())
        }
        (runmat_analysis_core::SamplingPolicy::Event(_), None) => {
            Err(AnalysisExecutionError::Invalid(
                "event observation partition requires source attribution".into(),
            ))
        }
        (_, Some(_)) => Err(AnalysisExecutionError::Invalid(
            "non-event observation partition cannot carry event attribution".into(),
        )),
        (_, None) => Ok(()),
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
        validate_contribution_attribution(source, partition_plan, contributions)?;
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
        validate_contribution_attribution(source, partition_plan, contributions)?;
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

fn validate_contribution_attribution(
    source: &ObservationEvaluationRequest,
    partition_plan: &ObservationPartitionPlan,
    contributions: &[ObservationPartitionContribution],
) -> AnalysisExecutionResult<()> {
    let expected = contributions
        .first()
        .and_then(|contribution| contribution.event_attribution);
    if contributions
        .iter()
        .any(|contribution| contribution.event_attribution != expected)
    {
        return Err(AnalysisExecutionError::Identity(
            "observation reduction contributions name different event occurrences",
        ));
    }
    let observation = source
        .plan
        .observations
        .iter()
        .find(|observation| observation.spec.observation_id == partition_plan.observation_id)
        .ok_or(AnalysisExecutionError::Identity(
            "observation reduction is absent from its source plan",
        ))?;
    match (&observation.spec.sampling, expected) {
        (runmat_analysis_core::SamplingPolicy::Event(event), Some(attribution))
            if attribution.event() == *event =>
        {
            if let ObservationEventAttribution::Solver(
                crate::results::observation::SolverEventAttribution {
                    source: SolverEventSource::CompletedResult { result },
                    ..
                },
            ) = attribution
            {
                if result != source.source_result {
                    return Err(AnalysisExecutionError::Identity(
                        "observation reduction event names a different source result",
                    ));
                }
            }
            Ok(())
        }
        (runmat_analysis_core::SamplingPolicy::Event(_), _) => {
            Err(AnalysisExecutionError::Identity(
                "observation reduction lacks the planned event attribution",
            ))
        }
        (_, Some(_)) => Err(AnalysisExecutionError::Identity(
            "non-event observation reduction carries event attribution",
        )),
        (_, None) => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use runmat_analysis_core::{
        AnalysisField, ComplexRepresentation, CompressionPolicy, CoordinateFrame, FieldLocation,
        IndependentDomain, ObservationComponent, ObservationCostEstimate, ObservationEvent,
        ObservationId, ObservationPurpose, ObservationQuantity, ObservationRequest,
        ObservationSpec, ObservationStoragePolicy, ObservationTarget, ObservationValueShape,
        PartialAccessPolicy, ProcedureStep, RecoveryPolicy, RetentionPolicy, SamplingPolicy,
        SignConvention, SignalPrecision, SolverObservationEvent,
    };
    use runmat_execution::Digest;
    use runmat_meshing_core::{fixtures::canonical_tetrahedron_solver_mesh, ElementOrder};

    use crate::{
        execution::{
            FieldRecoveryRequest, ObservationEvaluationRequest, SequenceIdentity,
            SolverResultIdentity, StepIdentity,
        },
        physics::structural::topology::CanonicalStructuralTopologyIdentity,
        results::{
            field::{
                RecoveredField, RecoveredFieldArtifact, RecoveredFieldDefinition,
                RecoveredFieldEntityCounts, RecoveredFieldEntityDomain,
                RecoveredFieldNumericRepresentation, RecoveredFieldSample, RecoveredFieldSelection,
            },
            observation::{
                prepare_solid_observation_field_partition, resolve_observation_plan,
                ObservationEventAttribution, ObservationExecutionPolicy, ObservationLimits,
                ObservationPartitionPlan, SolverEventAttribution, SolverEventSource,
            },
        },
        topology::AnalysisProjection,
    };

    use super::validate_event_attribution;

    #[test]
    fn portable_partition_event_requires_the_exact_completed_result() {
        let projection =
            AnalysisProjection::derive(&canonical_tetrahedron_solver_mesh(ElementOrder::Tet4))
                .unwrap();
        let definition = RecoveredFieldDefinition::new(
            "temperature.event",
            ObservationQuantity::Temperature,
            FieldLocation::Node,
            RecoveredFieldEntityDomain::StructuralNodes,
            ObservationValueShape::Scalar,
            RecoveredFieldNumericRepresentation::Real,
        )
        .unwrap();
        let sample = RecoveredFieldSample::TimeSeconds(0.5);
        let field = RecoveredField::new(
            definition.clone(),
            sample,
            AnalysisField::host_f64(
                "temperature.event",
                vec![projection.nodes.len()],
                vec![300.0; projection.nodes.len()],
            ),
        )
        .unwrap();
        let mut step = ProcedureStep::thermal_transient("portable-event");
        step.independent_domain = IndependentDomain::TimeSeconds {
            origin: 0.0,
            start: 0.0,
            end: 1.0,
        };
        step.accuracy.resources.maximum_samples = 4;
        step.observations = vec![ObservationRequest::History(ObservationSpec {
            observation_id: ObservationId("first_yield".into()),
            quantity: ObservationQuantity::Temperature,
            value_shape: ObservationValueShape::Scalar,
            target: ObservationTarget::WholeModel,
            source_location: FieldLocation::Node,
            output_location: FieldLocation::Node,
            recovery: RecoveryPolicy::Native,
            component: ObservationComponent::All,
            frame: CoordinateFrame::Global,
            sign_convention: SignConvention::Native,
            complex_representation: ComplexRepresentation::Real,
            sampling: SamplingPolicy::Event(ObservationEvent::Solver(
                SolverObservationEvent::FirstYield,
            )),
            reduction: None,
            storage: ObservationStoragePolicy {
                precision: SignalPrecision::Real64,
                compression: CompressionPolicy::Lossless,
                retention: RetentionPolicy::Canonical,
                partial_access: PartialAccessPolicy::ManifestAndChunks,
            },
            purpose: ObservationPurpose::Publication,
            cost: ObservationCostEstimate {
                operation_count: 1_000_000,
                peak_resident_bytes: 10_000_000,
                stored_bytes: 10_000_000,
                transfer_bytes: 10_000_000,
                collective_count: 0,
            },
        })];
        let plan = resolve_observation_plan(
            &step,
            &projection,
            std::slice::from_ref(&definition),
            ObservationLimits::default(),
        )
        .unwrap();
        let sequence = SequenceIdentity::new(Digest::sha256("portable-event-sequence")).unwrap();
        let producer = StepIdentity::new(sequence, Digest::sha256("portable-event-step")).unwrap();
        let source_result = SolverResultIdentity::from_step_result(
            producer,
            Digest::sha256("portable-event-result"),
        )
        .unwrap();
        let topology = CanonicalStructuralTopologyIdentity::Explicit {
            structural_topology_digest: projection.projection_digest,
        };
        let recovery = FieldRecoveryRequest::new(producer, source_result, topology).unwrap();
        let artifact = RecoveredFieldArtifact::new(
            recovery.identity(),
            source_result,
            projection.projection_digest,
            RecoveredFieldEntityCounts {
                structural_nodes: projection.nodes.len(),
                solid_elements: projection.elements.len(),
                beam_elements: 0,
                shell_elements: 0,
            },
            vec![field.clone()],
        )
        .unwrap();
        let selection =
            RecoveredFieldSelection::new(artifact.manifest().unwrap(), vec![field.identity()])
                .unwrap();
        let request =
            ObservationEvaluationRequest::new(producer, &selection, topology, plan).unwrap();
        let partition_plan = ObservationPartitionPlan::derive(
            &request.plan,
            &ObservationId("first_yield".into()),
            ObservationExecutionPolicy::serial(),
        )
        .unwrap();
        let partition = &partition_plan.partitions[0];
        let source_fields = prepare_solid_observation_field_partition(
            &request.plan,
            &partition_plan,
            partition,
            &projection,
            &field,
        )
        .unwrap();
        let attribution = ObservationEventAttribution::Solver(SolverEventAttribution {
            event: SolverObservationEvent::FirstYield,
            sample,
            occurrence_index: 0,
            source: SolverEventSource::CompletedResult {
                result: source_result,
            },
        });
        validate_event_attribution(&request, &partition_plan, &source_fields, Some(attribution))
            .unwrap();

        let substituted = ObservationEventAttribution::Solver(SolverEventAttribution {
            event: SolverObservationEvent::FirstYield,
            sample,
            occurrence_index: 0,
            source: SolverEventSource::CompletedResult {
                result: SolverResultIdentity::from_step_result(
                    producer,
                    Digest::sha256("substituted-portable-event-result"),
                )
                .unwrap(),
            },
        });
        assert!(validate_event_attribution(
            &request,
            &partition_plan,
            &source_fields,
            Some(substituted),
        )
        .is_err());
        assert!(
            validate_event_attribution(&request, &partition_plan, &source_fields, None,).is_err()
        );
    }
}
