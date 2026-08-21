//! Raw portable-program session over the existing native scheduler and process backend.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use runmat_execution::{CancellationReason, ExecutionScopeId, PoolId};
use runmat_execution_artifact::ProgramExecutionRequest;
use runmat_execution_runner::{AttemptSuccess, TaskSubmission};

use crate::config::fresh_scope_id;
use crate::driver::{LocalDriver, TaskCompletion};
use crate::{
    NativeExecutionConfig, NativeExecutionError, NativeExecutionResult, NativeObjectStore,
    ProgramProgress,
};

static NEXT_PROGRAM_SCOPE: AtomicU64 = AtomicU64::new(1);

pub struct NativeProgramSession {
    driver: Arc<LocalDriver>,
}

impl NativeProgramSession {
    pub fn new(mut config: NativeExecutionConfig) -> NativeExecutionResult<Self> {
        let nonce = NEXT_PROGRAM_SCOPE.fetch_add(1, Ordering::Relaxed);
        let scope_id = fresh_scope_id(b"native-program-session", nonce);
        config.store_root.push(scope_id.to_string());
        Ok(Self {
            driver: LocalDriver::new(config, scope_id)?,
        })
    }

    pub fn scope_id(&self) -> ExecutionScopeId {
        self.driver.scope_id()
    }

    pub fn pool_id(&self) -> PoolId {
        self.driver.pool_id()
    }

    pub fn object_store(&self) -> NativeObjectStore {
        self.driver.object_store()
    }

    pub fn submit(
        &self,
        program: ProgramExecutionRequest,
        submission: TaskSubmission,
    ) -> NativeExecutionResult<NativeProgramTask> {
        program
            .validate_for_portable_host()
            .map_err(|error| NativeExecutionError::Protocol(error.to_string()))?;
        if submission.request.inputs != program.arguments
            || submission.request.outputs.requested_outputs != program.requested_outputs
        {
            return Err(NativeExecutionError::Protocol(
                "native task inputs or outputs differ from its exact program request".into(),
            ));
        }
        let completion = self
            .driver
            .submit_task(submission, program.recipe, program.artifact)?;
        Ok(NativeProgramTask { completion })
    }

    pub fn cancel(&self, reason: CancellationReason) {
        self.driver.cancel_all(reason);
    }
}

pub struct NativeProgramTask {
    completion: Arc<TaskCompletion>,
}

impl NativeProgramTask {
    pub fn try_result(&self) -> Option<Result<AttemptSuccess, String>> {
        self.completion.try_value()
    }

    pub fn drain_progress(&self) -> Vec<ProgramProgress> {
        self.completion.drain_progress()
    }
}
