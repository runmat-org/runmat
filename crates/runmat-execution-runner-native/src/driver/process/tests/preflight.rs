//! A real provider preflight failure: no artifact producer or process was invoked.
use super::*;
use runmat_execution::{
    identity::ArtifactId,
    resource::ResourceRequest,
    task::{Callable, RetryPolicy, TaskRequest},
    Digest, ExecutionScopeId, OutputContract, TaskId,
};
use runmat_execution_runner::{DriverAction, DriverCommand, TaskSubmission};
use std::collections::BTreeSet;

#[test]
fn unreadable_program_preserves_known_prelaunch_consumption() {
    let directory = tempfile::tempdir().unwrap();
    let mut config = crate::NativeExecutionConfig::for_current_executable().unwrap();
    config.store_root = directory.path().to_path_buf();
    config.max_workers = 1;
    let driver = LocalDriver::new(config, ExecutionScopeId::derive(&[b"preflight"])).unwrap();
    let task = TaskRequest {
        id: TaskId::derive(&[b"preflight"]),
        scope_id: driver.scope_id(),
        pool_id: driver.pool_id(),
        program_artifact_id: ArtifactId::derive(&[b"missing-program"]),
        callable: Callable {
            owner_identity: "test".into(),
            qualified_name: "preflight".into(),
            entrypoint_digest: Digest::sha256(b"preflight"),
        },
        inputs: Vec::new(),
        outputs: OutputContract {
            requested_outputs: 0,
        },
        resources: ResourceRequest {
            cpu_millicores: 1000,
            memory_bytes: 1024,
            scratch_bytes: 1024,
            max_wall_millis: 10000,
            max_artifact_bytes: 1024,
            max_egress_bytes: 0,
            max_relay_bytes: 0,
            accelerators: Vec::new(),
            required_capabilities: BTreeSet::new(),
        },
        retry: RetryPolicy::Never,
        deadline_unix_millis: None,
    };
    // Admit through the real driver without dispatching its launch action.
    let request = driver
        .driver
        .lock()
        .unwrap()
        .handle(DriverCommand::Submit(Box::new(TaskSubmission {
            request: task,
            dependencies: BTreeSet::new(),
            priority: 0,
        })))
        .unwrap()
        .into_iter()
        .find_map(|action| match action {
            DriverAction::Launch(request) => Some(request),
            _ => None,
        })
        .unwrap();
    let observed = execute_attempt(&driver, &request, &TaskCompletion::new());
    assert!(observed
        .result
        .unwrap_err()
        .contains("local artifact store failed"));
    assert_eq!(
        observed.execution.termination,
        AttemptExecutionTermination::NotStarted
    );
    assert_eq!(observed.execution.local_ipc_sent_bytes, Some(0));
    assert_eq!(observed.execution.local_ipc_received_bytes, Some(0));
    let resources = observed.unsuccessful_resources.unwrap();
    resources.validate_against(&request).unwrap();
    assert_eq!(
        resources.resources.usage.wall_millis,
        observed.execution.wall_millis
    );
    assert_eq!(resources.resources.artifact_bytes, 0);
    assert_eq!(resources.resources.usage.egress_bytes, 0);
    assert_eq!(resources.resources.usage.relay_bytes, 0);
}
