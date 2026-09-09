use runmat_execution::Digest;

use super::*;

fn digest(label: &str) -> Digest {
    Digest::sha256(label)
}

fn step_identity() -> StepIdentity {
    let sequence = SequenceIdentity::new(digest("sequence")).unwrap();
    StepIdentity::new(sequence, digest("step")).unwrap()
}

fn identity_graph() -> (
    StudyIdentity,
    ExecutionPlanIdentity,
    SignalRealizationIdentity,
) {
    let study =
        StudyIdentity::from_authored_digest(digest("study"), Some(digest("project"))).unwrap();
    let step = step_identity();
    let constraints =
        ConstraintPlanIdentity::new(step, digest("mesh"), digest("constraints")).unwrap();
    let generator =
        GeneratorIdentity::new(digest("callable"), digest("captures"), digest("generator"))
            .unwrap();
    let materialization =
        SignalMaterializationIdentity::new(digest("signal"), Some(generator), digest("sampling"))
            .unwrap();
    let signal = SignalRealizationIdentity::new(materialization, digest("samples")).unwrap();
    let excitation =
        SolverExcitationIdentity::new(step, signal, digest("target"), digest("projection"))
            .unwrap();
    let problem =
        ProblemIdentity::new(step, digest("model"), None, constraints, vec![excitation]).unwrap();
    let discretization =
        DiscretizationIdentity::new(problem, digest("mesh"), digest("formulation")).unwrap();
    let operator = OperatorIdentity::new(discretization, digest("operator")).unwrap();
    let numerical = NumericalPlanIdentity::new(operator, digest("solver")).unwrap();
    let execution = ExecutionPlanIdentity::new(numerical, digest("portable-execution")).unwrap();
    (study, execution, signal)
}

#[test]
fn layered_identities_are_deterministic_and_preserve_argument_order() {
    let first = identity_graph();
    let second = identity_graph();
    assert_eq!(first, second);

    let workload_a = AnalysisWorkloadIdentity::new(
        first.0,
        step_identity(),
        first.1,
        AnalysisWorkKind::Solve,
        digest("solve-contract"),
        vec![digest("mesh"), digest("signal")],
    )
    .unwrap();
    let workload_b = AnalysisWorkloadIdentity::new(
        first.0,
        step_identity(),
        first.1,
        AnalysisWorkKind::Solve,
        digest("solve-contract"),
        vec![digest("signal"), digest("mesh")],
    )
    .unwrap();
    assert_ne!(workload_a, workload_b);

    let attempt_a =
        AnalysisAttemptIdentity::new(workload_task(workload_a), 1, digest("worker-a")).unwrap();
    let attempt_b =
        AnalysisAttemptIdentity::new(workload_task(workload_a), 1, digest("worker-b")).unwrap();
    assert_ne!(attempt_a, attempt_b);
    assert_ne!(workload_a, workload_b);
}

#[test]
fn generator_capture_and_realized_content_invalidate_the_realization() {
    let (_, _, baseline) = identity_graph();
    let generator = GeneratorIdentity::new(
        digest("callable"),
        digest("changed-captures"),
        digest("generator"),
    )
    .unwrap();
    let changed_materialization =
        SignalMaterializationIdentity::new(digest("signal"), Some(generator), digest("sampling"))
            .unwrap();
    let changed_capture =
        SignalRealizationIdentity::new(changed_materialization, digest("samples")).unwrap();
    let baseline_materialization = SignalMaterializationIdentity::new(
        digest("signal"),
        Some(
            GeneratorIdentity::new(digest("callable"), digest("captures"), digest("generator"))
                .unwrap(),
        ),
        digest("sampling"),
    )
    .unwrap();
    let changed_content =
        SignalRealizationIdentity::new(baseline_materialization, digest("changed-samples"))
            .unwrap();
    let changed_input_generator = GeneratorIdentity::new(
        digest("callable"),
        digest("captures"),
        digest("changed-generator-inputs"),
    )
    .unwrap();
    let changed_input_materialization = SignalMaterializationIdentity::new(
        digest("signal"),
        Some(changed_input_generator),
        digest("sampling"),
    )
    .unwrap();
    let changed_input =
        SignalRealizationIdentity::new(changed_input_materialization, digest("samples")).unwrap();
    assert_ne!(baseline, changed_capture);
    assert_ne!(baseline, changed_content);
    assert_ne!(baseline, changed_input);
}

#[test]
fn sweep_identity_preserves_authored_order_and_rejects_duplicate_studies() {
    let first = StudyIdentity::from_authored_digest(digest("study-a"), None).unwrap();
    let second = StudyIdentity::from_authored_digest(digest("study-b"), None).unwrap();
    let forward =
        SweepIdentity::new(digest("sweep"), vec![first, second], digest("composition")).unwrap();
    let reverse =
        SweepIdentity::new(digest("sweep"), vec![second, first], digest("composition")).unwrap();

    assert_ne!(forward, reverse);
    assert!(
        SweepIdentity::new(digest("sweep"), vec![first, first], digest("composition")).is_err()
    );
}

#[test]
fn sweep_results_compose_in_authored_order_not_completion_order() {
    let first = StudyIdentity::from_authored_digest(digest("sweep-study-a"), None).unwrap();
    let second = StudyIdentity::from_authored_digest(digest("sweep-study-b"), None).unwrap();
    let (_, execution, _) = identity_graph();
    let result = |study, label| {
        let workload = AnalysisWorkloadIdentity::new(
            study,
            step_identity(),
            execution,
            AnalysisWorkKind::Solve,
            digest(label),
            vec![digest("input")],
        )
        .unwrap();
        AnalysisResultIdentity::new(workload, vec![digest(label)]).unwrap()
    };
    let expected_studies = [first, second];
    let sweep = SweepIdentity::new(
        digest("ordered-sweep"),
        expected_studies.to_vec(),
        digest("composition"),
    )
    .unwrap();
    let first_entry = SweepEntryOutcome {
        ordinal: 0,
        study: first,
        disposition: SweepEntryDisposition::Succeeded {
            result: result(first, "result-a"),
        },
    };
    let second_entry = SweepEntryOutcome {
        ordinal: 1,
        study: second,
        disposition: SweepEntryDisposition::Succeeded {
            result: result(second, "result-b"),
        },
    };

    let ordered =
        ComposedSweepResult::new(sweep, &expected_studies, vec![first_entry, second_entry])
            .unwrap();
    let reversed =
        ComposedSweepResult::new(sweep, &expected_studies, vec![second_entry, first_entry])
            .unwrap();
    assert_eq!(ordered, reversed);

    let mut wrong = reversed.entries;
    wrong[1].study = first;
    assert!(ComposedSweepResult::new(sweep, &expected_studies, wrong).is_err());
}

#[test]
fn numerical_objects_are_deterministically_chunked_and_self_validating() {
    let payload = (0_u8..=31).collect::<Vec<_>>();
    let prepared = chunk_numerical_payload(
        AnalysisNumericalObjectKind::Field,
        digest("field"),
        &payload,
        7,
    )
    .unwrap();
    assert_eq!(prepared.manifest.chunks.len(), 5);
    assert_eq!(prepared.chunks.len(), 5);
    assert_eq!(prepared.manifest.payload_digest, Digest::sha256(&payload));
    prepared.revalidate(7).unwrap();

    let encoded = prepared.manifest.canonical_bytes().unwrap();
    let decoded = AnalysisObjectManifest::from_canonical_bytes(&encoded).unwrap();
    assert_eq!(decoded, prepared.manifest);

    let mut tampered = decoded;
    tampered.encoded_length += 1;
    assert!(tampered.validate().is_err());
}

#[test]
fn semantic_payloads_have_independent_chunk_intervals_and_identity() {
    let header = b"field catalog".as_slice();
    let displacement = (0_u8..=31).collect::<Vec<_>>();
    let stress = (64_u8..=95).collect::<Vec<_>>();
    let prepared = chunk_partitioned_numerical_payload(
        AnalysisNumericalObjectKind::Field,
        digest("field-artifact"),
        &[
            AnalysisPayloadPartition {
                logical_identity: digest("catalog"),
                payload: header,
            },
            AnalysisPayloadPartition {
                logical_identity: digest("displacement"),
                payload: &displacement,
            },
            AnalysisPayloadPartition {
                logical_identity: digest("stress"),
                payload: &stress,
            },
        ],
        13,
    )
    .unwrap();

    assert_eq!(prepared.manifest.partitions.len(), 3);
    assert_eq!(prepared.manifest.partitions[0].chunk_count, 1);
    assert_eq!(prepared.manifest.partitions[1].first_chunk_ordinal, 1);
    assert_eq!(prepared.manifest.partitions[1].chunk_count, 3);
    assert_eq!(prepared.manifest.partitions[2].first_chunk_ordinal, 4);
    assert_eq!(prepared.manifest.partitions[2].chunk_count, 3);
    assert_eq!(
        prepared.manifest.partitions[1].payload_digest,
        Digest::sha256(&displacement)
    );
    prepared.revalidate(13).unwrap();

    let mut duplicated = [
        AnalysisPayloadPartition {
            logical_identity: digest("same"),
            payload: header,
        },
        AnalysisPayloadPartition {
            logical_identity: digest("same"),
            payload: &stress,
        },
    ];
    assert!(chunk_partitioned_numerical_payload(
        AnalysisNumericalObjectKind::Field,
        digest("invalid-field-artifact"),
        &duplicated,
        13,
    )
    .is_err());
    duplicated[1].logical_identity = digest("other");
    let mut malformed = chunk_partitioned_numerical_payload(
        AnalysisNumericalObjectKind::Field,
        digest("malformed-field-artifact"),
        &duplicated,
        13,
    )
    .unwrap()
    .manifest;
    malformed.partitions[1].first_chunk_ordinal += 1;
    assert!(malformed.validate().is_err());
}

#[test]
fn semantic_state_and_reuse_transition_bind_exact_content_and_topology() {
    use runmat_analysis_core::{
        AnalysisField, ProcedureStepId, ReferenceStatePolicy, StateCompleteness, StateId,
        StateKind, StateOutput, StatePersistence, StateTransition, StateTransitionMethod,
        TopologyTransitionPolicy, TransitionId,
    };

    let study = StudyIdentity::from_authored_digest(digest("state-study"), None).unwrap();
    let sequence = SequenceIdentity::new(digest("state-sequence")).unwrap();
    let producer = StepIdentity::new(sequence, digest("producer")).unwrap();
    let consumer = StepIdentity::new(sequence, digest("consumer")).unwrap();
    let (_, execution_plan, _) = identity_graph();
    let workload = AnalysisWorkloadIdentity::new(
        study,
        producer,
        execution_plan,
        AnalysisWorkKind::Solve,
        digest("state-solve"),
        vec![digest("state-input")],
    )
    .unwrap();
    let source_result = SolverResultIdentity::new(workload, digest("solver-fields")).unwrap();
    let topology = digest("shared-topology");
    let state = CanonicalSemanticState::new(CanonicalSemanticStateInput {
        producer,
        declaration: StateOutput {
            state_id: StateId("equilibrium".into()),
            kind: StateKind::StructuralPrimary,
            completeness: StateCompleteness::Complete,
            persistence: StatePersistence::Canonical,
        },
        topology_digest: topology,
        source_result,
        fields: vec![AnalysisField::host_f64(
            "structural.displacement",
            vec![2, 3],
            vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        )],
    })
    .unwrap();
    assert_eq!(
        CanonicalSemanticState::from_canonical_bytes(&state.canonical_bytes().unwrap()).unwrap(),
        state
    );

    let authored = StateTransition {
        transition_id: TransitionId("reuse-equilibrium".into()),
        producer_step: ProcedureStepId("preload".into()),
        consumer_step: ProcedureStepId("branch".into()),
        state_id: state.state_id.clone(),
        method: StateTransitionMethod::Reuse,
        topology: TopologyTransitionPolicy::RequireIdentical,
        reference: ReferenceStatePolicy::Preserve,
    };
    let transition = CanonicalStateTransition::new(CanonicalStateTransitionInput {
        sequence,
        producer,
        consumer,
        authored: &authored,
        source: &state,
        target_topology_digest: topology,
    })
    .unwrap();
    assert_eq!(
        CanonicalStateTransition::from_canonical_bytes(&transition.canonical_bytes().unwrap())
            .unwrap(),
        transition
    );

    let mut substituted_state = state.clone();
    substituted_state.fields[0] = AnalysisField::host_f64(
        "structural.displacement",
        vec![2, 3],
        vec![0.0, 0.0, 0.0, 2.0, 0.0, 0.0],
    );
    assert!(substituted_state.validate().is_err());

    let mut substituted_transition = transition.clone();
    substituted_transition.target_topology_digest = digest("other-topology");
    assert!(substituted_transition.validate().is_err());

    let wrong_producer = StepIdentity::new(sequence, digest("wrong-producer")).unwrap();
    assert!(
        CanonicalStateTransition::new(CanonicalStateTransitionInput {
            sequence,
            producer: wrong_producer,
            consumer,
            authored: &authored,
            source: &state,
            target_topology_digest: topology,
        })
        .is_err()
    );

    let mut wrong_state = authored.clone();
    wrong_state.state_id = StateId("other-state".into());
    assert!(
        CanonicalStateTransition::new(CanonicalStateTransitionInput {
            sequence,
            producer,
            consumer,
            authored: &wrong_state,
            source: &state,
            target_topology_digest: topology,
        })
        .is_err()
    );
}

#[test]
fn semantic_state_rejects_unimplemented_projection_transitions() {
    use runmat_analysis_core::{
        AnalysisField, ProcedureStepId, ReferenceStatePolicy, StateCompleteness, StateId,
        StateKind, StateOutput, StatePersistence, StateTransition, StateTransitionMethod,
        TopologyTransitionPolicy, TransitionId,
    };

    let study = StudyIdentity::from_authored_digest(digest("projection-study"), None).unwrap();
    let sequence = SequenceIdentity::new(digest("projection-sequence")).unwrap();
    let producer = StepIdentity::new(sequence, digest("producer")).unwrap();
    let consumer = StepIdentity::new(sequence, digest("consumer")).unwrap();
    let (_, execution_plan, _) = identity_graph();
    let workload = AnalysisWorkloadIdentity::new(
        study,
        producer,
        execution_plan,
        AnalysisWorkKind::Solve,
        digest("projection-solve"),
        vec![digest("projection-input")],
    )
    .unwrap();
    let state = CanonicalSemanticState::new(CanonicalSemanticStateInput {
        producer,
        declaration: StateOutput {
            state_id: StateId("equilibrium".into()),
            kind: StateKind::StructuralPrimary,
            completeness: StateCompleteness::Complete,
            persistence: StatePersistence::Canonical,
        },
        topology_digest: digest("source-topology"),
        source_result: SolverResultIdentity::new(workload, digest("solver-fields")).unwrap(),
        fields: vec![AnalysisField::host_f64(
            "structural.displacement",
            vec![1, 3],
            vec![0.0, 0.0, 0.0],
        )],
    })
    .unwrap();
    let authored = StateTransition {
        transition_id: TransitionId("project-equilibrium".into()),
        producer_step: ProcedureStepId("source".into()),
        consumer_step: ProcedureStepId("target".into()),
        state_id: state.state_id.clone(),
        method: StateTransitionMethod::ConservativeProjection,
        topology: TopologyTransitionPolicy::ProjectWithEvidence,
        reference: ReferenceStatePolicy::Preserve,
    };
    assert!(
        CanonicalStateTransition::new(CanonicalStateTransitionInput {
            sequence,
            producer,
            consumer,
            authored: &authored,
            source: &state,
            target_topology_digest: digest("target-topology"),
        })
        .is_err()
    );
}

#[test]
fn structural_linear_workload_round_trips_without_a_mesh_path() {
    let model = crate::fixtures::fixture_model(crate::fixtures::FixtureId::CantileverLinearStatic);
    let step_id = model.sequence.steps[0].step_id.clone();
    let workload = StructuralLinearWorkload {
        schema_version: 1,
        model,
        step_id,
        backend: crate::ComputeBackend::Cpu,
        formulation_requirement: runmat_analysis_core::FormulationRequirement::Experimental,
        preconditioner: crate::solve::preconditioner::SpdPreconditionerKind::Jacobi,
        algebra_backend: crate::solve::backend::kind::LinearAlgebraBackendKind::CpuReference,
        require_solver_mesh_for_solid: true,
        thermo_mechanical_context: None,
        electro_thermal_context: None,
        initial_guess_transition: None,
    };
    let bytes = workload.canonical_bytes().unwrap();
    let decoded = StructuralLinearWorkload::from_canonical_bytes(&bytes).unwrap();
    assert_eq!(decoded, workload);
    assert!(!String::from_utf8_lossy(&bytes).contains("artifact_path"));
}

fn signal_realization(values: Vec<f64>) -> CanonicalSignalRealization {
    signal_realization_with_precision(values, runmat_analysis_core::SignalPrecision::Real64)
}

fn signal_realization_with_precision(
    values: Vec<f64>,
    precision: runmat_analysis_core::SignalPrecision,
) -> CanonicalSignalRealization {
    use runmat_analysis_core::{
        GeneratorDomain, GeneratorProvenance, PhysicalQuantityKind, SignalApproximationEvidence,
        SignalExtrapolation, SignalId, SignalInterpolation, SignalShape,
    };

    let generator = GeneratorIdentity::new(
        digest("callable"),
        digest("captures"),
        digest("generator-contract"),
    )
    .unwrap();
    let materialization = SignalMaterializationIdentity::new(
        digest("authored-signal"),
        Some(generator),
        digest("sampling"),
    )
    .unwrap();
    CanonicalSignalRealization::new(CanonicalSignalRealizationInput {
        materialization,
        signal_id: SignalId("launch.force".into()),
        quantity: PhysicalQuantityKind::Force,
        shape: SignalShape::Scalar,
        topology_entity_count: None,
        domain: GeneratorDomain::TimeSeconds,
        knots: [0.0, 0.5, 1.0]
            .into_iter()
            .map(CanonicalSignalKnot::continuous)
            .collect(),
        precision,
        values: CanonicalSignalValues::real(values),
        interpolation: SignalInterpolation::Linear,
        extrapolation: SignalExtrapolation::Reject,
        evidence: SignalApproximationEvidence {
            max_absolute_error: Some(0.0),
            max_relative_error: Some(0.0),
            derivative_max_absolute_error: None,
            sample_count: 3,
        },
        provenance: GeneratorProvenance {
            project_revision: digest("project").to_string(),
            callable_id: "loads.launch_force".into(),
            frozen_input_digest: digest("frozen-input").to_string(),
        },
        input_admissions: Vec::new(),
    })
    .unwrap()
}

#[test]
fn canonical_signal_realization_round_trips_and_layers_content_identity() {
    let first = signal_realization(vec![0.0, 4.0, 0.0]);
    let decoded =
        CanonicalSignalRealization::from_canonical_bytes(&first.canonical_bytes().unwrap())
            .unwrap();
    assert_eq!(decoded, first);

    let changed = signal_realization(vec![0.0, 5.0, 0.0]);
    assert_ne!(first.content_digest, changed.content_digest);
    assert_ne!(first.identity, changed.identity);

    let mut tampered = first;
    tampered.values = CanonicalSignalValues::real([0.0, f64::NAN, 0.0]);
    assert!(tampered.validate().is_err());
}

#[test]
fn direct_serde_rejects_a_nonconvergent_signal_realization() {
    let mut malformed = signal_realization(vec![0.0, 4.0, 0.0]);
    malformed.content_digest = digest("wrong-content");
    let bytes = serde_json::to_vec(&malformed).unwrap();
    assert!(serde_json::from_slice::<CanonicalSignalRealization>(&bytes).is_err());
}

#[test]
fn canonical_signal_knots_normalize_and_reject_signed_zero() {
    assert_eq!(
        CanonicalSignalKnot::continuous(-0.0),
        CanonicalSignalKnot::continuous(0.0)
    );
    let base = signal_realization(vec![0.0, 4.0, 0.0]);
    let mut knots = base.knots.clone();
    knots[0] = CanonicalSignalKnot {
        abscissa: (-0.0_f64).to_bits(),
        continuity: None,
    };
    assert!(
        CanonicalSignalRealization::new(CanonicalSignalRealizationInput {
            materialization: base.materialization,
            signal_id: base.signal_id,
            quantity: base.quantity,
            shape: base.shape,
            topology_entity_count: base.topology_entity_count,
            domain: base.domain,
            knots,
            precision: base.precision,
            values: base.values,
            interpolation: base.interpolation,
            extrapolation: base.extrapolation,
            evidence: base.evidence,
            provenance: base.provenance,
            input_admissions: base.input_admissions,
        })
        .is_err()
    );
}

#[test]
fn canonical_signal_identity_preserves_jump_limits_and_point_continuity() {
    let base = signal_realization(vec![0.0, 1.0, 4.0]);
    let build = |continuity| {
        CanonicalSignalRealization::new(CanonicalSignalRealizationInput {
            materialization: base.materialization,
            signal_id: base.signal_id.clone(),
            quantity: base.quantity,
            shape: base.shape.clone(),
            topology_entity_count: base.topology_entity_count,
            domain: base.domain.clone(),
            knots: vec![
                CanonicalSignalKnot::continuous(0.0),
                CanonicalSignalKnot::jump(0.5, continuity),
                CanonicalSignalKnot::continuous(1.0),
            ],
            precision: base.precision,
            values: CanonicalSignalValues::real([0.0, 1.0, 3.0, 4.0]),
            interpolation: base.interpolation,
            extrapolation: base.extrapolation,
            evidence: base.evidence.clone(),
            provenance: base.provenance.clone(),
            input_admissions: base.input_admissions.clone(),
        })
        .unwrap()
    };
    let left = build(runmat_analysis_core::SignalPointContinuity::Left);
    let decoded =
        CanonicalSignalRealization::from_canonical_bytes(&left.canonical_bytes().unwrap()).unwrap();
    assert_eq!(decoded, left);
    let right = build(runmat_analysis_core::SignalPointContinuity::Right);
    assert_ne!(left.content_digest, right.content_digest);
}

#[test]
fn canonical_signal_realization_normalizes_signed_zero() {
    let positive = signal_realization(vec![0.0, 4.0, 0.0]);
    let negative = signal_realization(vec![-0.0, 4.0, -0.0]);
    assert_eq!(negative.values, positive.values);
    assert_eq!(negative.content_digest, positive.content_digest);
    assert_eq!(negative.identity, positive.identity);
}

#[test]
fn canonical_signal_realization_enforces_declared_precision() {
    let noncanonical = 1.0 / 3.0;
    let canonical = f64::from(noncanonical as f32);
    let mut realization = signal_realization_with_precision(
        vec![0.0, canonical, 0.0],
        runmat_analysis_core::SignalPrecision::Real32,
    );
    realization.values = CanonicalSignalValues::real([0.0, noncanonical, 0.0]);
    assert!(realization.validate().is_err());
}

#[test]
fn topology_signal_realization_requires_exact_entity_shape() {
    use runmat_analysis_core::{
        GeneratorDomain, GeneratorProvenance, PhysicalQuantityKind, SignalApproximationEvidence,
        SignalExtrapolation, SignalId, SignalInterpolation, SignalPrecision, SignalShape,
    };

    let materialization =
        SignalMaterializationIdentity::new(digest("field-signal"), None, digest("field-sampling"))
            .unwrap();
    let input = CanonicalSignalRealizationInput {
        materialization,
        signal_id: SignalId("thermal.field".into()),
        quantity: PhysicalQuantityKind::Temperature,
        shape: SignalShape::TopologyField {
            topology_id: digest("mesh").to_string(),
            component_count: 1,
        },
        topology_entity_count: Some(4),
        domain: GeneratorDomain::TimeSeconds,
        knots: [0.0, 1.0]
            .into_iter()
            .map(CanonicalSignalKnot::continuous)
            .collect(),
        precision: SignalPrecision::Real64,
        values: CanonicalSignalValues::real([300.0; 7]),
        interpolation: SignalInterpolation::Linear,
        extrapolation: SignalExtrapolation::Reject,
        evidence: SignalApproximationEvidence {
            max_absolute_error: None,
            max_relative_error: None,
            derivative_max_absolute_error: None,
            sample_count: 2,
        },
        provenance: GeneratorProvenance {
            project_revision: digest("project").to_string(),
            callable_id: "thermal.field".into(),
            frozen_input_digest: digest("inputs").to_string(),
        },
        input_admissions: Vec::new(),
    };
    assert!(CanonicalSignalRealization::new(input).is_err());
}

fn workload_task(workload: AnalysisWorkloadIdentity) -> AnalysisTaskIdentity {
    AnalysisTaskIdentity::new(workload, Vec::new()).unwrap()
}
