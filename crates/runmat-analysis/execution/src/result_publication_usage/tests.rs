use super::*;
use runmat_analysis_core::{AccuracyPlan, FormulationRequirement, ProcedureStep};
use runmat_analysis_fea::execution::{
    AccuracyEvaluationRequest, AccuracySourceKind, ResultPublicationRequest, SequenceIdentity,
    SolverResultIdentity, StepIdentity,
};
use runmat_analysis_fea::planning::resource::*;
use runmat_analysis_fea::results::accuracy::{
    accuracy_evaluation_resource_profile, evaluate_accuracy_artifact,
};
use runmat_analysis_fea::results::publication::published_result_encoding_profile;
use runmat_analysis_fea::results::publication::PublishedAnalysisResult;
use runmat_execution::{
    identity::{AttemptId, WorkerId},
    resource::ResourceAccounting,
    Digest,
};

fn facts(bytes: u64) -> PreparedPhaseResourceFacts {
    PreparedPhaseResourceFacts {
        inputs: PreparedInputResourceFacts {
            encoded_payload_bytes: ByteCount(1_000_000),
            transport_closure_bytes: ByteCount(2_000_000),
            decoded_resident_bytes_admission_budget: ByteCount(64_000_000),
        },
        output: projected_canonical_output_resource_facts(bytes).unwrap(),
        output_set: PreparedOutputSetResourceFacts {
            encoded_bytes_upper_bound: ByteCount(4096),
        },
    }
}

fn fixture() -> (
    PublishedAnalysisResult,
    ResourceUsageEvidence,
    ResourceUsageEvidence,
) {
    let producer = StepIdentity::from_solver_contract(
        SequenceIdentity::new(Digest::sha256(b"sequence")).unwrap(),
        &ProcedureStep::structural_static("producer"),
    )
    .unwrap();
    let result =
        SolverResultIdentity::from_step_result(producer, Digest::sha256(b"result")).unwrap();
    let request = AccuracyEvaluationRequest::new(
        producer,
        AccuracySourceKind::StructuralLinear,
        result,
        None,
        FormulationRequirement::Experimental,
        AccuracyPlan::experimental_baseline("accuracy"),
    )
    .unwrap();
    let profile = accuracy_evaluation_resource_profile(&request, &[]).unwrap();
    let primary = build_accuracy_resource_plan(
        &request,
        &[],
        facts(profile.output_encoded_bytes_upper_bound),
    )
    .unwrap();
    let accuracy = evaluate_accuracy_artifact(&request, vec![]).unwrap();
    let source = ResourceUsageAttempt {
        attempt_id: AttemptId::derive(&[b"source"]),
        worker_id: WorkerId::derive(&[b"worker"]),
        plan: primary.identity,
        resources: ResourceAccounting::for_reserved_interval(primary.request.clone(), 1, 0, 0, 0)
            .unwrap(),
    };
    let baseline = record_resource_usage(
        &primary,
        Digest::sha256(b"workload"),
        result.digest(),
        vec![source.clone()],
        1,
    )
    .unwrap();
    let request = ResultPublicationRequest::new(&accuracy, None, &baseline).unwrap();
    let receipt = PublishedAnalysisResult::new(&request, &accuracy, &baseline).unwrap();
    let profile = published_result_encoding_profile(&request, &accuracy, &baseline).unwrap();
    let plan = build_result_publication_resource_plan(
        &request,
        &accuracy,
        &baseline,
        None,
        facts(profile.encoded_bytes_upper_bound),
    )
    .unwrap();
    let publication = ResourceUsageAttempt {
        attempt_id: AttemptId::derive(&[b"publication"]),
        worker_id: source.worker_id,
        plan: plan.identity,
        resources: ResourceAccounting::for_reserved_interval(plan.request.clone(), 2, 100, 0, 0)
            .unwrap(),
    };
    let terminal = record_resource_graph_usage(
        &primary,
        vec![plan],
        baseline.workload_digest,
        baseline.result_digest,
        vec![source, publication],
        3,
    )
    .unwrap();
    (receipt, baseline, terminal)
}

fn rebuild(
    base: &ResourceUsageEvidence,
    attempts: Vec<ResourceUsageAttempt>,
    wall: u64,
) -> ResourceUsageEvidence {
    record_resource_graph_usage(
        base.primary_plan().unwrap(),
        base.plans
            .iter()
            .filter(|p| p.identity != base.primary_plan)
            .cloned()
            .collect(),
        base.workload_digest,
        base.result_digest,
        attempts,
        wall,
    )
    .unwrap()
}

#[test]
fn terminal_is_exact_canonical_extension() {
    let (receipt, baseline, terminal) = fixture();
    validate_result_publication_resource_usage(&receipt, &baseline, &terminal).unwrap();
    let decoded =
        ResourceUsageEvidence::from_canonical_bytes(&terminal.canonical_bytes().unwrap()).unwrap();
    assert_eq!(terminal, decoded);
    assert!(validate_result_publication_resource_usage(&receipt, &baseline, &baseline).is_err());
    assert!(validate_result_publication_resource_usage(&receipt, &terminal, &terminal).is_err());
}

#[test]
fn rejects_rewritten_baseline_extra_attempt_and_invented_wall() {
    let (receipt, baseline, terminal) = fixture();
    let mut attempts = terminal.attempts.clone();
    attempts
        .iter_mut()
        .find(|a| a.plan == baseline.primary_plan)
        .unwrap()
        .worker_id = WorkerId::derive(&[b"foreign"]);
    assert!(validate_result_publication_resource_usage(
        &receipt,
        &baseline,
        &rebuild(&terminal, attempts, 3)
    )
    .is_err());
    let mut attempts = terminal.attempts.clone();
    let mut extra = attempts[0].clone();
    extra.attempt_id = AttemptId::derive(&[b"extra"]);
    attempts.push(extra);
    assert!(validate_result_publication_resource_usage(
        &receipt,
        &baseline,
        &rebuild(&terminal, attempts, 4)
    )
    .is_err());
    assert!(validate_result_publication_resource_usage(
        &receipt,
        &baseline,
        &rebuild(&terminal, terminal.attempts.clone(), 4)
    )
    .is_err());
}

#[test]
fn rejects_publication_overrun() {
    let (receipt, baseline, terminal) = fixture();
    let mut attempts = terminal.attempts.clone();
    let added = attempts
        .iter_mut()
        .find(|a| a.plan != baseline.primary_plan)
        .unwrap();
    added.resources = ResourceAccounting::for_reserved_interval(
        added.resources.allocation.clone(),
        added.resources.usage.wall_millis,
        added.resources.allocation.max_artifact_bytes + 1,
        0,
        0,
    )
    .unwrap();
    let overrun = rebuild(&terminal, attempts, 3);
    assert!(!overrun.overruns.is_empty());
    assert!(validate_result_publication_resource_usage(&receipt, &baseline, &overrun).is_err());
}

#[test]
fn rejects_canonical_foreign_workload_and_missing_baseline_attempt() {
    let (receipt, baseline, terminal) = fixture();
    let additional = terminal
        .plans
        .iter()
        .filter(|p| p.identity != terminal.primary_plan)
        .cloned()
        .collect::<Vec<_>>();
    let foreign = record_resource_graph_usage(
        terminal.primary_plan().unwrap(),
        additional.clone(),
        Digest::sha256(b"foreign-workload"),
        terminal.result_digest,
        terminal.attempts.clone(),
        3,
    )
    .unwrap();
    assert!(validate_result_publication_resource_usage(&receipt, &baseline, &foreign).is_err());
    // Keep a primary-plan attempt, but substitute its identity. The evidence is internally
    // canonical and balanced; only exact preservation of the baseline detects this omission.
    let mut attempts = terminal.attempts.clone();
    attempts
        .iter_mut()
        .find(|a| a.plan == baseline.primary_plan)
        .unwrap()
        .attempt_id = AttemptId::derive(&[b"replacement"]);
    let substituted = rebuild(&terminal, attempts, 3);
    assert!(validate_result_publication_resource_usage(&receipt, &baseline, &substituted).is_err());
}

#[test]
fn accepts_platform_capability_augmentation_but_rejects_changed_numeric_allocation() {
    use runmat_execution::resource::Capability;
    let (receipt, baseline, terminal) = fixture();
    let mut attempts = terminal.attempts.clone();
    let added = attempts
        .iter_mut()
        .find(|a| a.plan != baseline.primary_plan)
        .unwrap();
    added
        .resources
        .allocation
        .required_capabilities
        .insert(Capability::ProcessIsolation);
    added
        .resources
        .allocation
        .required_capabilities
        .insert(Capability::Custom(format!(
            "runmat.analysis.host:{}",
            crate::ANALYSIS_HOST_ABI
        )));
    let augmented = rebuild(&terminal, attempts.clone(), 3);
    validate_result_publication_resource_usage(&receipt, &baseline, &augmented).unwrap();

    let added = attempts
        .iter_mut()
        .find(|a| a.plan != baseline.primary_plan)
        .unwrap();
    let mut allocation = added.resources.allocation.clone();
    allocation.memory_bytes += 1;
    added.resources = ResourceAccounting::for_reserved_interval(allocation, 2, 100, 0, 0).unwrap();
    assert!(record_resource_graph_usage(
        terminal.primary_plan().unwrap(),
        terminal
            .plans
            .iter()
            .filter(|p| p.identity != terminal.primary_plan)
            .cloned()
            .collect(),
        terminal.workload_digest,
        terminal.result_digest,
        attempts,
        3,
    )
    .is_err());
}
