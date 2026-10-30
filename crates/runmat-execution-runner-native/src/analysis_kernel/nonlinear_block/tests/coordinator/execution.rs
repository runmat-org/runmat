//! Original native lifecycle test runner shared by independent material references.
use super::*;

fn execute(
    root: &std::path::Path,
    program: &PreparedAnalysisProgram,
    host: &runmat_analysis_execution::AnalysisHostWorkload,
    plan: &FeaResourcePlan,
    policy: AnalysisOutputPolicy,
) -> (
    ProgramExecutionResponse,
    runmat_analysis_execution::AnalysisPhaseExecutionEvidence,
) {
    let response = super::process::run_terminated(root, program, policy);
    // Physical execution is a real terminated producer. Admission/history below
    // is portable-driver fixture evidence, not measured native worker placement.
    static GENERATION: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
    let fence = GENERATION.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let (submission, _) = crate::analysis_task::prepare_analysis_task(
        runmat_execution::ExecutionScopeId::derive(&[b"native-j2-fixture", &fence.to_be_bytes()]),
        runmat_execution::PoolId::derive(&[b"native-j2-fixture", &fence.to_be_bytes()]),
        host,
        program,
        vec![],
        Default::default(),
        crate::analysis_task::AnalysisTaskPolicy {
            resources: plan.request.clone(),
            resource_plan: Some(plan),
            prepared_closure: None,
            retry: runmat_execution::task::RetryPolicy::Never,
            deadline_unix_millis: None,
            priority: 0,
            permitted_trust_tiers: crate::program_host::local_trust(),
        },
    )
    .unwrap();
    let accounting =
        crate::analysis_task_pool::phase_fixture::successful_phase(submission, &response, fence);
    (response, accounting)
}

pub(in super::super) fn execute_preparation(
    p: NonlinearSolidStudyPreparation,
) -> (
    runmat_analysis_execution::CompletedNonlinearSolidStudy,
    tempfile::TempDir,
) {
    let policy = p.output_policy;
    let mut study = NonlinearSolidStudyCoordinator::new(p).unwrap();
    assert!(study.prepare_accuracy().is_err());
    assert!(study.prepare_next_state().is_err());
    let directory = tempfile::tempdir().unwrap();
    let open = || {
        FilesystemObjectStore::open(
            directory.path().join("store"),
            policy.inventory_limits().max_object_bytes,
        )
        .unwrap()
    };
    loop {
        match study.phase() {
            NonlinearSolidStudyPhase::ReadyForBlock { ordinal } => {
                let task = study.prepare_block().unwrap();
                assert_eq!(task.request.block_ordinal, ordinal);
                let (done, accounting) = execute(
                    directory.path(),
                    &task.program,
                    &task.host,
                    &task.resource_plan,
                    policy,
                );
                let altered_response = ProgramExecutionResponse::ExternalizedSuccess {
                    outputs: vec![],
                    result_objects: vec![],
                };
                assert!(study
                    .complete_block(&open(), altered_response, accounting.clone())
                    .is_err());
                assert_eq!(
                    study.phase(),
                    NonlinearSolidStudyPhase::AwaitingBlock { ordinal }
                );
                study.complete_block(&open(), done, accounting).unwrap();
                if ordinal == 0 {
                    assert_eq!(
                        study.phase(),
                        NonlinearSolidStudyPhase::ReadyForBlock { ordinal: 1 }
                    );
                    assert!(study.terminal_block().is_none());
                    assert!(study.prepare_accuracy().is_err());
                }
            }
            NonlinearSolidStudyPhase::ReadyForFieldRecovery => {
                let task = study.prepare_field_recovery().unwrap();
                super::observations::reject_recovery_understatements(task);
                let (done, accounting) = execute(
                    directory.path(),
                    &task.program,
                    &task.host,
                    &task.resource_plan,
                    policy,
                );
                let altered_response = ProgramExecutionResponse::ExternalizedSuccess {
                    outputs: vec![],
                    result_objects: vec![],
                };
                assert!(study
                    .complete_field_recovery(&open(), altered_response, accounting.clone())
                    .is_err());
                assert_eq!(
                    study.phase(),
                    NonlinearSolidStudyPhase::AwaitingFieldRecovery
                );
                study
                    .complete_field_recovery(&open(), done, accounting)
                    .unwrap();
            }
            NonlinearSolidStudyPhase::ReadyForObservationFanout => {
                let count = study
                    .prepare_observation_fanout(&open())
                    .unwrap()
                    .partitions
                    .len();
                // Reverse completion order proves task ordinals survive cold replay.
                for ordinal in (0..count).rev() {
                    let task = study.observation_partition(ordinal).unwrap();
                    let (done, accounting) = execute(
                        directory.path(),
                        &task.program,
                        &task.host,
                        &task.resource_plan,
                        policy,
                    );
                    study
                        .complete_observation_partition(
                            &open(),
                            ordinal,
                            done.clone(),
                            accounting.clone(),
                        )
                        .unwrap();
                    assert!(study
                        .complete_observation_partition(&open(), ordinal, done, accounting)
                        .is_err());
                }
            }
            NonlinearSolidStudyPhase::ReadyForObservationReduction { .. } => {
                let task = study.prepare_observation_reduction().unwrap();
                let (done, accounting) = execute(
                    directory.path(),
                    &task.program,
                    &task.host,
                    &task.resource_plan,
                    policy,
                );
                study
                    .complete_observation_reduction(&open(), done, accounting)
                    .unwrap();
            }
            NonlinearSolidStudyPhase::ReadyForObservationAssembly => {
                let task = study.prepare_observation_assembly().unwrap();
                let (done, accounting) = execute(
                    directory.path(),
                    &task.program,
                    &task.host,
                    &task.resource_plan,
                    policy,
                );
                study
                    .complete_observation_assembly(&open(), done, accounting)
                    .unwrap();
            }
            NonlinearSolidStudyPhase::ReadyForAccuracy => {
                let task = study.prepare_accuracy().unwrap();
                let original_plan = task.request.plan.clone();
                let (done, accounting) = execute(
                    directory.path(),
                    &task.program,
                    &task.host,
                    &task.resource_plan,
                    policy,
                );
                study.complete_accuracy(&open(), done, accounting).unwrap();
                let status = study
                    .accuracy()
                    .unwrap()
                    .artifact
                    .evaluation()
                    .report()
                    .status;
                // Receipts answer Newton closure and the achieved tangent-solve residual
                // (fae5bc356b): a plan with only those criteria is satisfied; a fixture whose plan
                // also requires evidence the receipt cannot supply stays indeterminate.
                assert!(matches!(
                    status,
                    AccuracyStatus::Satisfied | AccuracyStatus::Indeterminate
                ));
                assert_eq!(
                    runmat_analysis_fea::results::accuracy::publication_gate(
                        &original_plan,
                        study.accuracy().unwrap().artifact.evaluation()
                    )
                    .unwrap()
                    .decision,
                    if status == AccuracyStatus::Satisfied
                        || original_plan.unavailable_evidence
                            == runmat_analysis_core::UnavailableEvidencePolicy::MarkIndeterminate
                    {
                        runmat_analysis_fea::results::accuracy::PublicationDecision::Allow
                    } else {
                        runmat_analysis_fea::results::accuracy::PublicationDecision::Block
                    }
                );
            }
            NonlinearSolidStudyPhase::ReadyForState { ordinal } => {
                let task = study.prepare_next_state().unwrap();
                let (done, accounting) = execute(
                    directory.path(),
                    &task.program,
                    &task.host,
                    &task.resource_plan,
                    policy,
                );
                study
                    .complete_next_state(&open(), done, accounting)
                    .unwrap();
                assert_eq!(study.states().len(), ordinal + 1);
            }
            NonlinearSolidStudyPhase::ReadyForBaseline => {
                study.freeze_baseline().unwrap();
            }
            NonlinearSolidStudyPhase::ReadyForPublication => {
                let task = study.prepare_publication().unwrap();
                let (done, accounting) = execute(
                    directory.path(),
                    &task.program,
                    &task.host,
                    &task.resource_plan,
                    policy,
                );
                study
                    .complete_publication(&open(), done, accounting)
                    .unwrap();
            }
            NonlinearSolidStudyPhase::Complete => break,
            phase => panic!("unexpected awaited phase {phase:?}"),
        }
    }
    (study.finish().unwrap(), directory)
}
