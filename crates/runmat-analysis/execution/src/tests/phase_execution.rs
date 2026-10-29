//! Bookkeeping fixtures use real generic admission/history, not invented winning IDs.
//! Their reported lease interval is test data, not native utilization or RSS evidence.
use crate::{AnalysisHostWorkload, AnalysisPhaseExecutionEvidence, PreparedAnalysisProgram};
use runmat_execution::{
    identity::{ArtifactId, WorkerId},
    resource::{ResourceInventory, ResourceRequest},
    state::PoolState,
    task::RetryPolicy,
    value::ValuePayload,
    ExecutionScopeId, PoolId,
};
use runmat_execution_artifact::ProgramExecutionResponse;
use runmat_execution_runner::{
    port::BackendReport, AttemptReport, AttemptSuccess, Driver, DriverAction, DriverCommand,
    DriverConfig, PoolSpec, WorkerSpec,
};
use std::{
    collections::BTreeSet,
    sync::atomic::{AtomicU64, Ordering},
};

pub(crate) fn phase_execution(
    host: &AnalysisHostWorkload,
    program: &PreparedAnalysisProgram,
    allocation: &ResourceRequest,
    response: &ProgramExecutionResponse,
) -> AnalysisPhaseExecutionEvidence {
    phase_execution_with_retry(host, program, allocation, response, None, 1).unwrap()
}

/// Some(true) reports a failed attempt's known usage; Some(false) reports unknown loss.
fn phase_execution_with_retry(
    host: &AnalysisHostWorkload,
    program: &PreparedAnalysisProgram,
    allocation: &ResourceRequest,
    response: &ProgramExecutionResponse,
    first: Option<bool>,
    graph_wall_millis: u64,
) -> crate::AnalysisBridgeResult<AnalysisPhaseExecutionEvidence> {
    static GENERATION: AtomicU64 = AtomicU64::new(1);
    let fence = GENERATION.fetch_add(1, Ordering::Relaxed);
    let scope = ExecutionScopeId::derive(&[b"phase-fixture", &fence.to_be_bytes()]);
    let pool = PoolId::derive(&[scope.bytes()]);
    let worker = WorkerId::derive(&[scope.bytes()]);
    let mut resources = allocation.clone();
    resources
        .required_capabilities
        .extend(host.workload.generic_capabilities().unwrap());
    let roots = program
        .request
        .arguments
        .iter()
        .map(|argument| match argument {
            ValuePayload::Object(root) => root.as_ref().clone(),
            _ => panic!("phase fixture requires original object references"),
        })
        .collect();
    let submission = crate::build_analysis_task_submission(
        host,
        roots,
        Vec::new(),
        BTreeSet::new(),
        crate::AnalysisExecutionContext {
            host: program
                .request
                .artifact
                .execution_host_requirement(
                    &program.request.recipe,
                    BTreeSet::from([
                        runmat_execution::security::ExecutionTrustTier::CustomerTrusted,
                    ]),
                )
                .unwrap(),
            scope_id: scope,
            pool_id: pool,
            program_artifact_id: ArtifactId::derive(&[program.request.artifact.id.0.bytes()]),
            resources: resources.clone(),
            retry: RetryPolicy::ExplicitlyIdempotent { max_attempts: 2 },
            deadline_unix_millis: None,
            priority: 0,
        },
    )
    .unwrap();
    let inventory = ResourceInventory {
        cpu_millicores: resources.cpu_millicores,
        memory_bytes: resources.memory_bytes,
        scratch_bytes: resources.scratch_bytes.max(1),
        accelerators: crate::tests::contract_devices(&resources.accelerators),
        capabilities: resources.required_capabilities,
    };
    let mut driver = Driver::new(DriverConfig::default(), fence).unwrap();
    driver
        .handle(DriverCommand::RegisterScope {
            scope_id: scope,
            parent: None,
        })
        .unwrap();
    driver
        .handle(DriverCommand::CreatePool(PoolSpec {
            id: pool,
            min_workers: 1,
            max_workers: 1,
            max_in_flight: 1,
            resource_limit: inventory.clone(),
        }))
        .unwrap();
    driver
        .handle(DriverCommand::SetPoolState {
            pool_id: pool,
            state: PoolState::Ready,
        })
        .unwrap();
    driver
        .handle(DriverCommand::RegisterWorker(WorkerSpec {
            host: crate::tests::contract_worker(&submission.request.host),
            id: worker,
            pool_id: pool,
            resources: inventory,
        }))
        .unwrap();
    let mut request = driver
        .handle(DriverCommand::Submit(Box::new(submission)))
        .unwrap()
        .into_iter()
        .find_map(|action| match action {
            DriverAction::Launch(request) => Some(request),
            _ => None,
        })
        .unwrap();
    if let Some(known) = first {
        let mut report = BackendReport::for_request(
            &request,
            if known {
                AttemptReport::Failed {
                    kind: runmat_execution_runner::AttemptFailureKind::Infrastructure,
                    message: "original reported failure".into(),
                }
            } else {
                AttemptReport::Lost {
                    message: "original unknown loss".into(),
                }
            },
        );
        if known {
            report.unsuccessful_resources = Some(Box::new(
                runmat_execution_runner::AttemptResourceAccounting::for_attempt(
                    &request, 2, 7, 11, 13,
                )
                .unwrap(),
            ));
        }
        let actions = driver.handle(DriverCommand::BackendReport(report)).unwrap();
        if !known {
            assert!(!actions
                .iter()
                .any(|action| matches!(action, DriverAction::Launch(_))));
            let evidence = driver
                .snapshot()
                .task_execution_evidence(request.task_id)
                .unwrap();
            assert_eq!(
                evidence.task.state,
                runmat_execution::state::TaskState::Indeterminate
            );
            assert!(evidence.task.committed.is_none());
            return AnalysisPhaseExecutionEvidence::new(evidence, graph_wall_millis);
        }
        request = actions
            .into_iter()
            .find_map(|action| match action {
                DriverAction::Launch(request) => Some(request),
                _ => None,
            })
            .unwrap();
    }
    let ProgramExecutionResponse::ExternalizedSuccess {
        outputs,
        result_objects,
    } = response
    else {
        panic!("phase fixture requires an original successful externalized response")
    };
    let success =
        AttemptSuccess::new(&request, outputs.clone(), result_objects.clone(), 1, 0, 0).unwrap();
    driver
        .handle(DriverCommand::BackendReport(BackendReport::for_request(
            &request,
            AttemptReport::Succeeded { result: success },
        )))
        .unwrap();
    AnalysisPhaseExecutionEvidence::new(
        driver
            .snapshot()
            .task_execution_evidence(request.task_id)
            .unwrap(),
        graph_wall_millis,
    )
}

mod validation;
