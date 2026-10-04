//! Original native lifecycle test runner shared by independent material references.
use super::*;

fn execute(
    root: &std::path::Path,
    program: &PreparedAnalysisProgram,
    host: &runmat_analysis_execution::AnalysisHostWorkload,
    plan: &FeaResourcePlan,
    policy: AnalysisOutputPolicy,
) -> (ProgramExecutionResponse, AttemptResourceAccounting) {
    let start = std::time::Instant::now();
    let response = super::process::run_terminated(root, program, policy);
    let ProgramExecutionResponse::ExternalizedSuccess { result_objects, .. } = &response else {
        panic!("original generic native program failed: {response:?}");
    };
    let bytes = result_objects.iter().map(|r| r.encoded_length).sum();
    let mut allocation = plan.request.clone();
    allocation
        .required_capabilities
        .extend(host.workload.generic_capabilities().unwrap());
    let accounting = AttemptResourceAccounting {
        attempt_id: AttemptId::derive(&[plan.identity.digest().bytes()]),
        worker_id: WorkerId::derive(&[b"terminated-native-j2-coordinator"]),
        resources: ResourceAccounting::for_reserved_interval(
            allocation,
            u64::try_from(start.elapsed().as_millis()).unwrap().max(1),
            bytes,
            0,
            0,
        )
        .unwrap(),
    };
    // Reserved allocations with observed elapsed time/bytes; no RSS attestation claimed.
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
                let mut altered = accounting.clone();
                altered.resources.artifact_bytes += 1;
                assert!(study
                    .complete_block(&open(), done.clone(), altered)
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
                let mut altered = accounting.clone();
                altered.resources.artifact_bytes += 1;
                assert!(study
                    .complete_field_recovery(&open(), done.clone(), altered)
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
                assert_eq!(
                    study
                        .accuracy()
                        .unwrap()
                        .artifact
                        .evaluation()
                        .report()
                        .status,
                    AccuracyStatus::Indeterminate
                );
                assert_eq!(
                    runmat_analysis_fea::results::accuracy::publication_gate(
                        &original_plan,
                        FormulationRequirement::Production,
                        study.accuracy().unwrap().artifact.evaluation()
                    )
                    .unwrap()
                    .decision,
                    runmat_analysis_fea::results::accuracy::PublicationDecision::Block
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
