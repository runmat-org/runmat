use super::*;
use crate::execution::{
    FieldRecoveryPlan, FieldRecoveryRequest, FieldRecoverySource, SequenceIdentity,
    SolverResultIdentity,
};
use crate::physics::structural::topology::CanonicalStructuralTopologyIdentity;
use crate::results::field::*;
use crate::topology::AnalysisProjection;
use runmat_analysis_core::*;
use runmat_meshing_core::{fixtures::canonical_tetrahedron_solver_mesh, ElementOrder};

fn spec(id: &str) -> ObservationSpec {
    ObservationSpec {
        observation_id: ObservationId(id.into()),
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
        sampling: SamplingPolicy::Final,
        reduction: None,
        storage: ObservationStoragePolicy {
            precision: SignalPrecision::Real64,
            compression: CompressionPolicy::Lossless,
            retention: RetentionPolicy::Canonical,
            partial_access: PartialAccessPolicy::ManifestAndChunks,
        },
        purpose: ObservationPurpose::Publication,
        cost: ObservationCostPolicy::UpperBound(ObservationCostEstimate {
            operation_count: 100_000,
            peak_resident_bytes: 1_000_000,
            stored_bytes: 1_000_000,
            transfer_bytes: 1_000_000,
            collective_count: 0,
        }),
    }
}

fn fixture(
    selected_time: f64,
    mixed: bool,
) -> (ObservationEvaluationRequest, Vec<ObservationResult>) {
    fixture_with_policy(selected_time, mixed, ObservationExecutionPolicy::serial())
}

fn fixture_with_policy(
    selected_time: f64,
    mixed: bool,
    policy: ObservationExecutionPolicy,
) -> (ObservationEvaluationRequest, Vec<ObservationResult>) {
    let mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    let projection = AnalysisProjection::derive(&mesh).unwrap();
    let mut step = ProcedureStep::thermal_transient("heat");
    step.independent_domain = IndependentDomain::TimeSeconds {
        origin: 0.0,
        start: 0.0,
        end: 1.0,
        interval_count: 1,
    };
    step.observations = vec![ObservationRequest::Field(spec("field"))];
    if mixed {
        let mut history = spec("history");
        history.sampling = SamplingPolicy::TimeSeconds(vec![0.0, 1.0]);
        step.observations.push(ObservationRequest::History(history));
    }
    let definition = RecoveredFieldDefinition::new(
        "temperature",
        ObservationQuantity::Temperature,
        FieldLocation::Node,
        RecoveredFieldEntityDomain::Nodes,
        ObservationValueShape::Scalar,
        RecoveredFieldNumericRepresentation::Real,
    )
    .unwrap();
    let plan = resolve_observation_plan(
        &step,
        &projection,
        std::slice::from_ref(&definition),
        ObservationLimits::default(),
    )
    .unwrap();
    let field_at = |time| {
        RecoveredField::new(
            definition.clone(),
            RecoveredFieldSample::TimeSeconds(time),
            AnalysisField::host_f64(
                "temperature", definition.layout(),
                vec![projection.nodes.len()],
                vec![293.15; projection.nodes.len()],
            ),
        )
        .unwrap()
    };
    let terminal_field = field_at(1.0);
    let fields = RecoveredFieldPartitionSet::new(
        RecoveredFieldView::from_field(&terminal_field)
            .unwrap()
            .canonical_partitions()
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap(),
    )
    .unwrap();
    let results = plan
        .observations
        .iter()
        .map(|observation| {
            let partition_plan =
                ObservationPartitionPlan::derive(&plan, &observation.spec.observation_id, policy)
                    .unwrap();
            let contributions = partition_plan
                .partitions
                .iter()
                .map(|partition| {
                    evaluate_solid_observation_partition(
                        &plan,
                        &partition_plan,
                        partition,
                        &projection,
                        &fields,
                        None,
                    )
                    .unwrap()
                })
                .collect();
            reduce_observation_partitions(&plan, &partition_plan, contributions).unwrap()
        })
        .collect();
    let sequence = SequenceIdentity::new(Digest::sha256("sample publication tests")).unwrap();
    let producer = StepIdentity::from_solver_contract(sequence, &step).unwrap();
    let numerical_source = FieldRecoverySource::SolverResult(
        SolverResultIdentity::from_step_result(producer, Digest::sha256("result")).unwrap(),
    );
    let recovery = FieldRecoveryRequest::new(
        producer,
        numerical_source,
        crate::execution::FieldTopologyIdentity::Structural {
            topology: CanonicalStructuralTopologyIdentity::Explicit {
                structural_topology_digest: plan.topology_digest,
            },
        },
        FieldRecoveryPlan::default(),
    )
    .unwrap();
    let selected = field_at(selected_time);
    let manifest = RecoveredFieldArtifact::new(
        recovery.identity(),
        numerical_source,
        plan.topology_digest,
        RecoveredFieldEntityCounts {
            oriented_faces: 0,
            nodes: projection.nodes.len(),
            volume_elements: projection.elements.len(),
            volume_integration_points: 0,
            oriented_edges: 0,
            beam_elements: 0,
            shell_elements: 0,
            connectors: 0,
            constraint_groups: 0,
        },
        vec![selected.clone()],
    )
    .unwrap()
    .manifest()
    .unwrap();
    let selection =
        ObservationFieldSelection::structural(&recovery, &manifest, vec![selected.identity()])
            .unwrap();
    selection.validate_structural(&recovery, &manifest).unwrap();
    (
        ObservationEvaluationRequest::new(selection, plan).unwrap(),
        results,
    )
}

fn artifact(
    request: &ObservationEvaluationRequest,
    results: Vec<ObservationResult>,
) -> ObservationArtifact {
    // Build fresh identities rather than mutating stale hashes: the malformed relationship,
    // not invalid artifact integrity, must be what admission rejects.
    ObservationArtifact::new(ObservationArtifactInput {
        evaluation_request: request.identity(),
        producer: request.producer,
        source_selection: request.selection.identity(),
        source: request.selection.source(),
        topology_digest: request.plan.topology_digest,
        plan_identity: request.plan.identity,
        results,
    })
    .unwrap()
}

#[test]
fn sample_publication_rejects_valid_source_selection_at_wrong_coordinate() {
    let (request, results) = fixture(0.0, false);
    let artifact = artifact(&request, results);
    artifact.validate().unwrap();
    let publication = ObservationSamplePublication {
        request: request.clone(),
        manifest: artifact.manifest().unwrap(),
        partition_evidence: artifact
            .results()
            .iter()
            .map(|result| result.partition_evidence.clone())
            .collect(),
    };
    assert!(publication.validate().is_err());
    assert!(ObservationAssemblyRequest::new(&request, artifact.results()).is_err());
    assert!(ObservationSamplePublication::new(request, &artifact).is_err());
}

#[test]
fn sample_publication_requires_scheduled_history_alongside_field() {
    let (request, results) = fixture(1.0, true);
    let complete = artifact(&request, results.clone());
    ObservationAssemblyRequest::new(&request, complete.results()).unwrap();
    ObservationSamplePublication::new(request.clone(), &complete).unwrap();
    let incomplete = artifact(&request, vec![results[0].clone()]);
    incomplete.validate().unwrap();
    let publication = ObservationSamplePublication {
        request: request.clone(),
        manifest: incomplete.manifest().unwrap(),
        partition_evidence: incomplete
            .results()
            .iter()
            .map(|result| result.partition_evidence.clone())
            .collect(),
    };
    assert!(publication.validate().is_err());
    assert!(ObservationAssemblyRequest::new(&request, incomplete.results()).is_err());
}

#[test]
fn sample_publication_retains_exact_execution_evidence_separately_from_values() {
    let (request, results) = fixture(1.0, true);
    let serial = artifact(&request, results);
    let publication = ObservationSamplePublication::new(request.clone(), &serial).unwrap();
    publication.validate_artifact(&serial).unwrap();
    let (split_request, split_results) = fixture_with_policy(
        1.0,
        true,
        ObservationExecutionPolicy::canonical(std::num::NonZeroU32::new(2).unwrap()),
    );
    assert_eq!(request, split_request);
    let split = artifact(&split_request, split_results);
    for (left, right) in serial.results().iter().zip(split.results()) {
        assert_eq!(left.logical_identity, right.logical_identity);
        assert_ne!(
            left.partition_evidence.identity,
            right.partition_evidence.identity
        );
    }
    let split_publication = ObservationSamplePublication::new(split_request, &split).unwrap();
    assert!(publication.validate_artifact(&split).is_err());
    let mut substituted = publication.clone();
    substituted.partition_evidence = split_publication.partition_evidence;
    assert!(substituted.validate().is_err());
    let mut omitted = publication.clone();
    omitted.partition_evidence.pop();
    assert!(omitted.validate().is_err());
    let mut reordered = publication.clone();
    reordered.partition_evidence.swap(0, 1);
    assert!(reordered.validate().is_err());
    let mut extra = publication;
    extra
        .partition_evidence
        .push(extra.partition_evidence[0].clone());
    assert!(extra.validate().is_err());
}
