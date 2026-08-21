use std::collections::BTreeSet;

use runmat_execution::identity::ArtifactId;
use runmat_execution::task::{Callable, RetryPolicy, TaskRequest};
use runmat_execution::{Digest, OutputContract};
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
    let qualified_name = format!("test:{}", request.test_id.as_str());
    TaskSubmission {
        request: TaskRequest {
            id: session.task_id(request.test_id.as_str(), request.attempt),
            scope_id: session.scope_id,
            pool_id: session.pool_id,
            program_artifact_id: ArtifactId::derive(&[b"runmat-test-plan-v1", &revision]),
            callable: Callable {
                owner_identity: "runmat.test".into(),
                qualified_name: qualified_name.clone(),
                entrypoint_digest: Digest::sha256(qualified_name),
            },
            inputs: Vec::new(),
            outputs: OutputContract {
                requested_outputs: 1,
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
