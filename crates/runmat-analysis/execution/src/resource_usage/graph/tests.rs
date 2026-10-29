//! Bookkeeping witnesses use the actual portable driver, not invented retry IDs.
use super::*;
use runmat_execution::{
    identity::{ArtifactId, WorkerId},
    resource::{Capability, ResourceInventory},
    state::PoolState,
    task::{Callable, RetryPolicy, TaskRequest},
    ExecutionScopeId, OutputContract, PoolId, TaskId,
};
use runmat_execution_runner::{
    port::BackendReport, AttemptReport, AttemptSuccess, Driver, DriverAction, DriverCommand,
    DriverConfig, PoolSpec, TaskSubmission, WorkerSpec,
};

enum FirstAttempt {
    Succeeded,
    UnknownLoss,
    AccountedFailure,
}

fn original(plan: &FeaResourcePlan, first: FirstAttempt) -> TaskExecutionEvidence {
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
        accelerators: crate::tests::contract_devices(&plan.request.accelerators),
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
            host: crate::tests::contract_worker(&crate::tests::contract_default_host()),
            id: worker,
            pool_id: pool,
            resources: inventory,
        }))
        .unwrap();
    let submission = TaskSubmission {
        request: TaskRequest {
            console: None,
            environment: runmat_execution::EnvironmentSnapshot::default(),
            id: TaskId::derive(&[b"usage-history"]),
            scope_id: scope,
            pool_id: pool,
            schema_version: runmat_execution::task::TASK_REQUEST_SCHEMA_VERSION,
            executable: runmat_execution::TaskExecutableIdentity::ProgramArtifact(
                ArtifactId::derive(&[b"usage-history"]),
            ),
            host: crate::tests::contract_default_host(),
            invocation_context: runmat_execution::ProgramInvocationContext::Direct,
            callable: Callable::for_program(
                "bookkeeping-test",
                &runmat_execution::ProgramCallable::semantic(
                    runmat_execution::ProgramFunctionId(0),
                    Some("original-history".into()),
                ),
            ),
            inputs: Vec::new(),
            outputs: OutputContract {
                owned_work: None,
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
    if !matches!(first, FirstAttempt::Succeeded) {
        let mut report = BackendReport::for_request(
            &request,
            match first {
                FirstAttempt::AccountedFailure => AttemptReport::Failed {
                    kind: runmat_execution_runner::AttemptFailureKind::Infrastructure,
                    message: "original accounted failure".into(),
                },
                _ => AttemptReport::Lost {
                    message: "original unknown consumption".into(),
                },
            },
        );
        if matches!(first, FirstAttempt::AccountedFailure) {
            // Explicit test data, not a provider measurement claim.
            report.unsuccessful_resources = Some(Box::new(
                AttemptResourceAccounting::for_attempt(&request, 2, 17, 19, 23).unwrap(),
            ));
        }
        let actions = driver.handle(DriverCommand::BackendReport(report)).unwrap();
        if matches!(first, FirstAttempt::UnknownLoss) {
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
            return evidence;
        }
        request = launch(actions);
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
    let success = original(plan, FirstAttempt::Succeeded);
    let graph = PlannedResourceUsageGraph::from_execution(plan, &success, 1).unwrap();
    assert_eq!(graph.attempts.len(), 1);
    assert_eq!(
        graph.attempts[0].accounting,
        *success
            .task
            .committed
            .as_ref()
            .unwrap()
            .result
            .resources()
            .unwrap()
    );
    let lost = original(plan, FirstAttempt::UnknownLoss);
    assert_eq!(lost.attempts.len(), 1);
    assert!(PlannedResourceUsageGraph::from_execution(plan, &lost, 1).is_err());
    let mut foreign = success;
    foreign.task.submission.request.resources.memory_bytes += 1;
    assert!(PlannedResourceUsageGraph::from_execution(plan, &foreign, 1).is_err());
}

#[test]
fn complete_usage_charges_reported_failed_retry_through_cold_portable_evidence() {
    let mut coordinator =
        crate::FluidStudyCoordinator::new(crate::fluid_study::tests::preparation()).unwrap();
    let plan = &coordinator.prepare_solve().unwrap().resource_plan;
    let original = original(plan, FirstAttempt::AccountedFailure);
    let cold: TaskExecutionEvidence =
        serde_json::from_slice(&serde_json::to_vec(&original).unwrap()).unwrap();
    let graph = PlannedResourceUsageGraph::from_execution(plan, &cold, 3).unwrap();
    assert_eq!(graph.attempts.len(), 2);
    let aggregate = AttemptResourceAggregate::new(
        graph
            .attempts
            .iter()
            .map(|attempt| attempt.accounting.clone())
            .collect(),
        graph.wall_millis,
    )
    .unwrap();
    assert_eq!(aggregate.summary.artifact_bytes, 17);
    assert_eq!(aggregate.summary.usage.egress_bytes, 19);
    assert_eq!(aggregate.summary.usage.relay_bytes, 23);
    assert_eq!(plan.request.cpu_millicores, 1000);
    assert_eq!(aggregate.summary.usage.cpu_core_millis, 3);
    assert_eq!(cold, original);
}

#[test]
fn graph_constructor_rejects_internally_valid_accounting_for_a_foreign_allocation() {
    let mut coordinator =
        crate::FluidStudyCoordinator::new(crate::fluid_study::tests::preparation()).unwrap();
    let plan = coordinator.prepare_solve().unwrap().resource_plan.clone();
    let evidence = original(&plan, FirstAttempt::Succeeded);
    let mut accounting = evidence
        .task
        .committed
        .unwrap()
        .result
        .resources()
        .unwrap()
        .clone();
    let mut foreign = accounting.resources.allocation.clone();
    foreign.memory_bytes += 1;
    accounting.resources =
        runmat_execution::resource::ResourceAccounting::for_reserved_interval(foreign, 1, 0, 0, 0)
            .unwrap();
    accounting.validate().unwrap();
    assert!(PlannedResourceUsageGraph::new(
        vec![PlannedResourceUsageAttempt { plan, accounting }],
        1
    )
    .is_err());
}
