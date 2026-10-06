//! Bookkeeping witnesses use the actual portable driver, not invented retry IDs.
use super::*;
use runmat_execution::{
    identity::{ArtifactId, WorkerId},
    resource::{Capability, ResourceInventory},
    state::PoolState,
    task::{Callable, RetryPolicy, TaskRequest},
    Digest, ExecutionScopeId, OutputContract, PoolId, TaskId,
};
use runmat_execution_runner::{
    port::BackendReport, AttemptReport, AttemptSuccess, Driver, DriverAction, DriverCommand,
    DriverConfig, PoolSpec, TaskSubmission, WorkerSpec,
};

fn original(plan: &FeaResourcePlan, lose_first: bool) -> TaskExecutionEvidence {
    let scope = ExecutionScopeId::derive(&[b"usage-history"]);
    let pool = PoolId::derive(&[b"usage-history"]);
    let worker = WorkerId::derive(&[b"usage-history"]);
    let mut resources = plan.request.clone();
    resources
        .required_capabilities
        .insert(Capability::ProcessIsolation);
    let inventory = ResourceInventory {
        cpu_millicores: plan.request.cpu_millicores,
        memory_bytes: plan.request.memory_bytes,
        scratch_bytes: plan.request.scratch_bytes.max(1),
        accelerators: plan.request.accelerators.clone(),
        capabilities: resources.required_capabilities.clone(),
    };
    let mut driver = Driver::new(DriverConfig::default(), 1).unwrap();
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
            id: worker,
            pool_id: pool,
            resources: inventory,
        }))
        .unwrap();
    let submission = TaskSubmission {
        request: TaskRequest {
            id: TaskId::derive(&[b"usage-history"]),
            scope_id: scope,
            pool_id: pool,
            program_artifact_id: ArtifactId::derive(&[b"usage-history"]),
            callable: Callable {
                owner_identity: "bookkeeping-test".into(),
                qualified_name: "original-history".into(),
                entrypoint_digest: Digest::sha256(b"usage-history"),
            },
            inputs: Vec::new(),
            outputs: OutputContract {
                requested_outputs: 0,
            },
            resources,
            retry: RetryPolicy::ExplicitlyIdempotent { max_attempts: 2 },
            deadline_unix_millis: None,
        },
        dependencies: Default::default(),
        priority: 0,
    };
    let launch = |actions: Vec<DriverAction>| {
        actions
            .into_iter()
            .find_map(|action| match action {
                DriverAction::Launch(request) => Some(request),
                _ => None,
            })
            .unwrap()
    };
    let mut request = launch(
        driver
            .handle(DriverCommand::Submit(Box::new(submission)))
            .unwrap(),
    );
    if lose_first {
        request = launch(
            driver
                .handle(DriverCommand::BackendReport(BackendReport::for_request(
                    &request,
                    AttemptReport::Lost {
                        message: "original unknown consumption".into(),
                    },
                )))
                .unwrap(),
        );
    }
    driver
        .handle(DriverCommand::Tick { now_millis: 1 })
        .unwrap();
    let success = AttemptSuccess::new(&request, Vec::new(), Vec::new(), 1, 0, 0).unwrap();
    driver
        .handle(DriverCommand::BackendReport(BackendReport::for_request(
            &request,
            AttemptReport::Succeeded { result: success },
        )))
        .unwrap();
    driver
        .snapshot()
        .task_execution_evidence(request.task_id)
        .unwrap()
}

#[test]
fn complete_original_usage_rejects_unknown_retry_and_plan_substitution() {
    let mut coordinator =
        crate::FluidStudyCoordinator::new(crate::fluid_study::tests::preparation()).unwrap();
    let plan = &coordinator.prepare_solve().unwrap().resource_plan;
    let success = original(plan, false);
    let graph = PlannedResourceUsageGraph::from_execution(plan, &success, 1).unwrap();
    assert_eq!(graph.attempts.len(), 1);
    assert_eq!(
        graph.attempts[0].accounting,
        *success
            .task
            .committed
            .as_ref()
            .unwrap()
            .resources
            .as_deref()
            .unwrap()
    );
    let lost = original(plan, true);
    assert_eq!(lost.attempts.len(), 2);
    assert!(PlannedResourceUsageGraph::from_execution(plan, &lost, 1).is_err());
    let mut foreign = success;
    foreign.task.submission.request.resources.memory_bytes += 1;
    assert!(PlannedResourceUsageGraph::from_execution(plan, &foreign, 1).is_err());
}
