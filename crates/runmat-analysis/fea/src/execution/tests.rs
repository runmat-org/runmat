use runmat_execution::Digest;

use super::*;

fn digest(label: &str) -> Digest {
    Digest::sha256(label)
}

fn identity_graph() -> (
    StudyIdentity,
    ExecutionPlanIdentity,
    SignalRealizationIdentity,
) {
    let study = StudyIdentity::new(digest("study"), Some(digest("project"))).unwrap();
    let sequence = SequenceIdentity::new(study, digest("sequence")).unwrap();
    let step = StepIdentity::new(sequence, digest("step")).unwrap();
    let constraints =
        ConstraintPlanIdentity::new(step, digest("mesh"), digest("constraints")).unwrap();
    let generator =
        GeneratorIdentity::new(digest("callable"), digest("captures"), digest("generator"))
            .unwrap();
    let signal = SignalRealizationIdentity::new(
        digest("signal"),
        Some(generator),
        digest("sampling"),
        digest("samples"),
    )
    .unwrap();
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
        first.1,
        AnalysisWorkKind::Solve,
        digest("solve-contract"),
        vec![digest("mesh"), digest("signal")],
    )
    .unwrap();
    let workload_b = AnalysisWorkloadIdentity::new(
        first.0,
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
    let changed_capture = SignalRealizationIdentity::new(
        digest("signal"),
        Some(generator),
        digest("sampling"),
        digest("samples"),
    )
    .unwrap();
    let changed_content = SignalRealizationIdentity::new(
        digest("signal"),
        None,
        digest("sampling"),
        digest("changed-samples"),
    )
    .unwrap();
    assert_ne!(baseline, changed_capture);
    assert_ne!(baseline, changed_content);
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

fn workload_task(workload: AnalysisWorkloadIdentity) -> AnalysisTaskIdentity {
    AnalysisTaskIdentity::new(workload, Vec::new()).unwrap()
}
