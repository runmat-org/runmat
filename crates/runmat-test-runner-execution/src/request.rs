use std::collections::BTreeSet;

use runmat_execution::identity::ArtifactId;
use runmat_execution::task::{Callable, RetryPolicy, TaskRequest};
use runmat_execution::OutputContract;
use runmat_execution_runner::TaskSubmission;
use runmat_test_runner::worker::ExecutionRequest;

use crate::{ExecutionBackendConfig, ExecutionWorkerSession};

pub(crate) fn task<S>(
    session: &ExecutionWorkerSession<S>,
    request: &ExecutionRequest,
    config: &ExecutionBackendConfig,
) -> TaskSubmission {
    let revision = session
        .revision
        .canonical_bytes()
        .expect("validated test revision must have a canonical encoding");
    TaskSubmission {
        request: TaskRequest {
            console: None,
            environment: runmat_execution::EnvironmentSnapshot::default(),
            id: session.task_id(request.test_id.as_str(), request.attempt),
            scope_id: session.scope_id,
            pool_id: session.pool_id,
            schema_version: runmat_execution::task::TASK_REQUEST_SCHEMA_VERSION,
            executable: runmat_execution::TaskExecutableIdentity::ProgramArtifact(
                ArtifactId::derive(&[b"runmat-test-plan-v1", &revision]),
            ),
            callable: Callable::for_program(
                "runmat.test",
                &runmat_execution::ProgramCallable::semantic(
                    runmat_execution::ProgramFunctionId(0),
                    None,
                ),
            ),
            host: crate::capability::host_requirement(&session.revision),
            invocation_context: runmat_execution::ProgramInvocationContext::Direct,
            inputs: Vec::new(),
            outputs: OutputContract {
                requested_outputs: 1,
                owned_work: None,
            },
            resources: config.attempt_resources.clone(),
            retry: RetryPolicy::Never,
            // The test coordinator's deadline is expressed in its injected
            // host clock, not necessarily Unix time. It remains authoritative
            // for timeout/cancellation; do not mislabel it in the scheduler.
            deadline_unix_millis: None,
        },
        dependencies: BTreeSet::new(),
        priority: 0,
    }
}
