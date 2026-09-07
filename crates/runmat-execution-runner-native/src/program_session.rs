//! Raw portable-program session over the existing native scheduler and process backend.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use runmat_execution::identity::ValueId;
use runmat_execution::value::{ValuePayload, ValueRef, ValueRefKind};
use runmat_execution::{CancellationReason, ExecutionScopeId, PoolId};
use runmat_execution_artifact::{ExecutableForm, ProgramExecutionObject, ProgramExecutionRequest};
use runmat_execution_runner::{AttemptSuccess, TaskSubmission};

use crate::config::fresh_scope_id;
use crate::driver::{LocalDriver, TaskCompletion};
use crate::{
    FilesystemObjectStore, NativeExecutionConfig, NativeExecutionError, NativeExecutionResult,
    ProgramProgress,
};

static NEXT_PROGRAM_SCOPE: AtomicU64 = AtomicU64::new(1);

pub struct NativeProgramSession {
    driver: Arc<LocalDriver>,
    values: Mutex<HashMap<ValueId, (ValueRef, ValuePayload)>>,
}

impl NativeProgramSession {
    pub fn new(mut config: NativeExecutionConfig) -> NativeExecutionResult<Self> {
        let nonce = NEXT_PROGRAM_SCOPE.fetch_add(1, Ordering::Relaxed);
        let scope_id = fresh_scope_id(b"native-program-session", nonce);
        config.store_root.push(scope_id.to_string());
        Ok(Self {
            driver: LocalDriver::new(config, scope_id)?,
            values: Mutex::new(HashMap::new()),
        })
    }

    pub fn scope_id(&self) -> ExecutionScopeId {
        self.driver.scope_id()
    }

    pub fn pool_id(&self) -> PoolId {
        self.driver.pool_id()
    }

    pub fn object_store(&self) -> FilesystemObjectStore {
        self.driver.object_store()
    }

    pub fn register_value_object(
        &self,
        object: ProgramExecutionObject,
    ) -> NativeExecutionResult<()> {
        object.validate().map_err(protocol)?;
        if object.reference.kind != ValueRefKind::DriverObject {
            return Err(protocol(
                "local input value object has the wrong reference kind",
            ));
        }
        let value = object
            .decode_externalized_value()
            .map_err(protocol)?
            .ok_or_else(|| protocol("local input is not an externalized RunMat value"))?;
        let mut values = self.values.lock().expect("local value registry poisoned");
        match values.get(&object.reference.id) {
            Some((reference, stored)) if reference == &object.reference && stored == &value => {
                Ok(())
            }
            Some(_) => Err(protocol(
                "local value object id was reused for different content",
            )),
            None => {
                values.insert(object.reference.id, (object.reference, value));
                Ok(())
            }
        }
    }

    pub fn submit(
        &self,
        program: ProgramExecutionRequest,
        submission: TaskSubmission,
    ) -> NativeExecutionResult<NativeProgramTask> {
        self.submit_graph(vec![(program, submission)])?
            .pop()
            .ok_or_else(|| {
                NativeExecutionError::Protocol("native task admission returned no handle".into())
            })
    }

    /// Atomically admits a topologically ordered graph before dispatching any task.
    pub fn submit_graph(
        &self,
        mut tasks: Vec<(ProgramExecutionRequest, TaskSubmission)>,
    ) -> NativeExecutionResult<Vec<NativeProgramTask>> {
        if tasks.is_empty() {
            return Err(NativeExecutionError::Protocol(
                "native task graph must not be empty".into(),
            ));
        }
        let values = self.values.lock().expect("local value registry poisoned");
        let registered = values
            .iter()
            .map(|(id, (_, value))| (*id, value.clone()))
            .collect::<HashMap<_, _>>();
        for (program, submission) in &mut tasks {
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
            if !matches!(
                program.artifact.form,
                ExecutableForm::MeshingWorkload | ExecutableForm::AnalysisWorkload
            ) {
                let arguments = materialize_arguments(&program.arguments, &registered)?;
                program.arguments = arguments.clone();
                submission.request.inputs = arguments;
            }
        }
        drop(values);
        let completions = self.driver.submit_tasks(
            tasks
                .into_iter()
                .map(|(program, submission)| {
                    (
                        submission,
                        program.recipe,
                        program.artifact,
                        program.function,
                    )
                })
                .collect(),
        )?;
        Ok(completions
            .into_iter()
            .map(|completion| NativeProgramTask { completion })
            .collect())
    }

    pub fn cancel(&self, reason: CancellationReason) {
        self.driver.cancel_all(reason);
    }
}

fn protocol(error: impl std::fmt::Display) -> NativeExecutionError {
    NativeExecutionError::Protocol(error.to_string())
}

fn materialize_arguments(
    arguments: &[ValuePayload],
    values: &HashMap<ValueId, ValuePayload>,
) -> NativeExecutionResult<Vec<ValuePayload>> {
    arguments
        .iter()
        .map(|argument| crate::remote::value_transfer::materialize(argument, values))
        .collect()
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

#[cfg(test)]
mod tests {
    use runmat_execution::value::{InlineValue, ValueRefKind};
    use runmat_execution::Digest;
    use runmat_execution_artifact::{ArtifactAccess, ProgramExecutionObject};

    use super::*;

    #[test]
    fn registered_externalized_values_materialize_before_local_execution() {
        let value = ValuePayload::Inline(Box::new(InlineValue::String("immutable".into())));
        let object = ProgramExecutionObject::externalized_value(
            &ArtifactAccess {
                authorization_scope: "local-value-test".into(),
                encryption_context: Digest::sha256(b"local-value-context"),
            },
            ValueRefKind::DriverObject,
            value.clone(),
        )
        .unwrap();
        let reference = object.reference.clone();
        let stored = HashMap::from([(
            reference.id,
            object.decode_externalized_value().unwrap().unwrap(),
        )]);

        let materialized =
            materialize_arguments(&[ValuePayload::Object(Box::new(reference))], &stored).unwrap();

        assert_eq!(materialized, vec![value]);
    }
}
