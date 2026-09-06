use runmat_execution::Digest;

use super::*;

fn digest(label: &str) -> Digest {
    Digest::sha256(label)
}

fn step_identity(study: StudyIdentity) -> StepIdentity {
    let sequence = SequenceIdentity::new(study, digest("sequence")).unwrap();
    StepIdentity::new(sequence, digest("step")).unwrap()
}

fn identity_graph() -> (
    StudyIdentity,
    ExecutionPlanIdentity,
    SignalRealizationIdentity,
) {
    let study = StudyIdentity::new(digest("study"), Some(digest("project"))).unwrap();
    let step = step_identity(study);
    let constraints =
        ConstraintPlanIdentity::new(step, digest("mesh"), digest("constraints")).unwrap();
    let generator =
        GeneratorIdentity::new(digest("callable"), digest("captures"), digest("generator"))
            .unwrap();
    let materialization = SignalMaterializationIdentity::new(
        digest("signal"),
        Some(generator),
        digest("sampling"),
        vec![digest("frozen-input")],
    )
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
    let accuracy = AccuracyIdentity::new(step, digest("accuracy")).unwrap();
    let numerical = NumericalPlanIdentity::new(operator, digest("solver"), accuracy).unwrap();
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
        step_identity(first.0),
        first.1,
        AnalysisWorkKind::Solve,
        digest("solve-contract"),
        vec![digest("mesh"), digest("signal")],
    )
    .unwrap();
    let workload_b = AnalysisWorkloadIdentity::new(
        first.0,
        step_identity(first.0),
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
    let changed_materialization = SignalMaterializationIdentity::new(
        digest("signal"),
        Some(generator),
        digest("sampling"),
        vec![digest("frozen-input")],
    )
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
        vec![digest("frozen-input")],
    )
    .unwrap();
    let changed_content =
        SignalRealizationIdentity::new(baseline_materialization, digest("changed-samples"))
            .unwrap();
    let changed_input_materialization = SignalMaterializationIdentity::new(
        digest("signal"),
        Some(
            GeneratorIdentity::new(digest("callable"), digest("captures"), digest("generator"))
                .unwrap(),
        ),
        digest("sampling"),
        vec![digest("changed-frozen-input")],
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
    let first = StudyIdentity::new(digest("study-a"), None).unwrap();
    let second = StudyIdentity::new(digest("study-b"), None).unwrap();
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
    let first = StudyIdentity::new(digest("sweep-study-a"), None).unwrap();
    let second = StudyIdentity::new(digest("sweep-study-b"), None).unwrap();
    let (_, execution, _) = identity_graph();
    let result = |study, label| {
        let workload = AnalysisWorkloadIdentity::new(
            study,
            step_identity(study),
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
fn structural_linear_workload_round_trips_without_a_mesh_path() {
    let workload = StructuralLinearWorkload {
        schema_version: 1,
        model: crate::fixtures::fixture_model(crate::fixtures::FixtureId::CantileverLinearStatic),
        backend: crate::ComputeBackend::Cpu,
        formulation_requirement: runmat_analysis_core::FormulationRequirement::Experimental,
        preconditioner: crate::solve::preconditioner::SpdPreconditionerKind::Jacobi,
        algebra_backend: crate::solve::backend::kind::LinearAlgebraBackendKind::CpuReference,
        require_solver_mesh_for_solid: true,
        thermo_mechanical_context: None,
        electro_thermal_context: None,
    };
    let bytes = workload.canonical_bytes().unwrap();
    let decoded = StructuralLinearWorkload::from_canonical_bytes(&bytes).unwrap();
    assert_eq!(decoded, workload);
    assert!(!String::from_utf8_lossy(&bytes).contains("artifact_path"));
}

fn signal_realization(values: Vec<f64>) -> CanonicalSignalRealization {
    use runmat_analysis_core::{
        GeneratorDomain, GeneratorProvenance, PhysicalQuantityKind, SignalApproximationEvidence,
        SignalExtrapolation, SignalId, SignalInterpolation, SignalPrecision, SignalShape,
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
        vec![digest("frozen-input")],
    )
    .unwrap();
    CanonicalSignalRealization::new(CanonicalSignalRealizationInput {
        materialization,
        signal_id: SignalId("launch.force".into()),
        quantity: PhysicalQuantityKind::Force,
        shape: SignalShape::Scalar,
        topology_entity_count: None,
        domain: GeneratorDomain::TimeSeconds,
        sample_points: vec![0.0, 0.5, 1.0],
        precision: SignalPrecision::Real64,
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
fn topology_signal_realization_requires_exact_entity_shape() {
    use runmat_analysis_core::{
        GeneratorDomain, GeneratorProvenance, PhysicalQuantityKind, SignalApproximationEvidence,
        SignalExtrapolation, SignalId, SignalInterpolation, SignalPrecision, SignalShape,
    };

    let materialization = SignalMaterializationIdentity::new(
        digest("field-signal"),
        None,
        digest("field-sampling"),
        Vec::new(),
    )
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
        sample_points: vec![0.0, 1.0],
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
    };
    assert!(CanonicalSignalRealization::new(input).is_err());
}

fn workload_task(workload: AnalysisWorkloadIdentity) -> AnalysisTaskIdentity {
    AnalysisTaskIdentity::new(workload, Vec::new()).unwrap()
}
