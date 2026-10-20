//! Inline physics tests use the portable driver's actual admission and commit history.
//! Their reported interval is fixture data, not native scheduler or measured RSS evidence.
use runmat_analysis_execution::AnalysisPhaseExecutionEvidence;
use runmat_execution::{identity::WorkerId, resource::ResourceInventory, state::PoolState};
use runmat_execution_artifact::ProgramExecutionResponse;
use runmat_execution_runner::{
    port::BackendReport, AttemptReport, AttemptSuccess, Driver, DriverAction, DriverCommand,
    DriverConfig, PoolSpec, TaskSubmission, WorkerSpec,
};

pub(crate) fn successful_phase(
    submission: TaskSubmission,
    response: &ProgramExecutionResponse,
    fence: u64,
) -> AnalysisPhaseExecutionEvidence {
    let scope = submission.request.scope_id;
    let pool = submission.request.pool_id;
    let worker = WorkerId::derive(&[scope.bytes(), &fence.to_be_bytes()]);
    let resources = &submission.request.resources;
    let inventory = ResourceInventory {
        cpu_millicores: resources.cpu_millicores,
        memory_bytes: resources.memory_bytes,
        scratch_bytes: resources.scratch_bytes.max(1),
        accelerators: runmat_accelerate_api::registered_providers()
            .into_iter()
            .map(|provider| provider.execution_accelerator_device(1))
            .filter_map(Result::transpose)
            .collect::<Result<Vec<_>, _>>()
            .expect("actual registered phase fixture devices"),
        capabilities: resources.required_capabilities.clone(),
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
            host: crate::test_support::host_inventory(),
            id: worker,
            pool_id: pool,
            resources: inventory,
        }))
        .unwrap();
    let request = driver
        .handle(DriverCommand::Submit(Box::new(submission)))
        .unwrap()
        .into_iter()
        .find_map(|action| match action {
            DriverAction::Launch(request) => Some(request),
            _ => None,
        })
        .unwrap();
    let ProgramExecutionResponse::ExternalizedSuccess {
        outputs,
        result_objects,
    } = response
    else {
        panic!("inline native phase failed: {response:?}")
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
        1,
    )
    .unwrap()
}
