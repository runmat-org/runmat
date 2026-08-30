use runmat_execution::{PoolBackend, ProgramExecutionAssignment};
use runmat_value::{ObjectInstance, Tensor, Value};

use crate::context::RuntimeContext;

const TASK_CLASS: runmat_types::StaticClassIdentity =
    runmat_types::StaticClassIdentity::new("parallel.Task");
const WORKER_CLASS: runmat_types::StaticClassIdentity =
    runmat_types::StaticClassIdentity::new("parallel.Worker");
const JOB_CLASS: runmat_types::StaticClassIdentity =
    runmat_types::StaticClassIdentity::new("parallel.Job");

pub fn task(context: &RuntimeContext) -> Value {
    context
        .execution_assignment()
        .map(task_object)
        .unwrap_or_else(empty_double)
}

pub fn worker(context: &RuntimeContext) -> Value {
    context
        .execution_assignment()
        .map(worker_object)
        .unwrap_or_else(empty_double)
}

pub fn job(context: &RuntimeContext) -> Value {
    context
        .execution_job()
        .map(job_object)
        .unwrap_or_else(empty_double)
}

fn task_object(assignment: ProgramExecutionAssignment) -> Value {
    let mut object = ObjectInstance::new(TASK_CLASS.to_string());
    insert_text(&mut object, "ID", assignment.task_id);
    insert_text(&mut object, "AttemptID", assignment.attempt_id);
    insert_text(&mut object, "PoolID", assignment.pool_id);
    insert_text(&mut object, "WorkerID", assignment.worker_id);
    object
        .properties
        .insert("State".into(), Value::String("running".into()));
    Value::Object(object)
}

fn worker_object(assignment: ProgramExecutionAssignment) -> Value {
    let mut object = ObjectInstance::new(WORKER_CLASS.to_string());
    insert_text(&mut object, "ID", assignment.worker_id);
    insert_text(&mut object, "PoolID", assignment.pool_id);
    object.properties.insert(
        "Backend".into(),
        Value::String(backend_name(assignment.backend).into()),
    );
    Value::Object(object)
}

fn job_object(job_id: runmat_execution::JobId) -> Value {
    let mut object = ObjectInstance::new(JOB_CLASS.to_string());
    insert_text(&mut object, "ID", job_id);
    Value::Object(object)
}

fn insert_text(object: &mut ObjectInstance, property: &str, value: impl std::fmt::Display) {
    object
        .properties
        .insert(property.into(), Value::String(value.to_string()));
}

const fn backend_name(backend: PoolBackend) -> &'static str {
    match backend {
        PoolBackend::Serial => "serial",
        PoolBackend::LocalProcesses => "local_processes",
        PoolBackend::BrowserWorkers => "browser_workers",
        PoolBackend::Remote => "remote",
    }
}

fn empty_double() -> Value {
    Value::Tensor(Tensor::new(Vec::new(), vec![0, 0]).expect("empty matrix shape is valid"))
}

#[cfg(test)]
mod tests {
    use std::rc::Rc;

    use runmat_execution::identity::{AttemptId, WorkerId};
    use runmat_execution::{ExecutionScopeId, JobId, PoolId, ProgramExecutionAssignment, TaskId};

    use super::{job, task, worker};
    use crate::context::RuntimeContext;
    use crate::execution::RuntimeExecutionService;

    fn assignment() -> ProgramExecutionAssignment {
        ProgramExecutionAssignment {
            scope_id: ExecutionScopeId::derive(&[b"scope"]),
            pool_id: PoolId::derive(&[b"pool"]),
            task_id: TaskId::derive(&[b"task"]),
            attempt_id: AttemptId::derive(&[b"attempt"]),
            worker_id: WorkerId::derive(&[b"worker"]),
            backend: runmat_execution::PoolBackend::LocalProcesses,
            resources: Default::default(),
        }
    }

    fn runtime() -> RuntimeContext {
        RuntimeContext::new(Rc::new(RuntimeExecutionService::new()))
    }

    #[test]
    fn driver_introspection_is_empty() {
        let runtime = runtime();
        for value in [task(&runtime), worker(&runtime), job(&runtime)] {
            assert!(
                matches!(value, runmat_value::Value::Tensor(tensor) if tensor.shape == vec![0, 0] && tensor.is_empty())
            );
        }
    }

    #[test]
    fn worker_introspection_projects_the_scheduler_assignment() {
        let runtime = runtime();
        let assignment = assignment();
        let _guard = runtime.enter_execution_assignment(Some(assignment.clone()));
        let _job_guard = runtime.enter_execution_job(Some(JobId::derive(&[b"job"])));

        let runmat_value::Value::Object(task) = task(&runtime) else {
            panic!("task assignment should produce an object");
        };
        assert_eq!(task.class_name.display_name(), "parallel.Task");
        assert_eq!(
            task.properties.get("ID"),
            Some(&runmat_value::Value::String(assignment.task_id.to_string()))
        );

        let runmat_value::Value::Object(worker) = worker(&runtime) else {
            panic!("worker assignment should produce an object");
        };
        assert_eq!(worker.class_name.display_name(), "parallel.Worker");
        assert_eq!(
            worker.properties.get("Backend"),
            Some(&runmat_value::Value::String("local_processes".into()))
        );

        let runmat_value::Value::Object(job) = job(&runtime) else {
            panic!("job assignment should produce an object");
        };
        assert_eq!(job.class_name.display_name(), "parallel.Job");
    }

    #[test]
    fn nested_execution_identity_is_scoped_and_restored() {
        let runtime = runtime();
        let outer_assignment = assignment();
        let outer_job = JobId::derive(&[b"outer-job"]);
        let _outer_assignment_guard =
            runtime.enter_execution_assignment(Some(outer_assignment.clone()));
        let _outer_job_guard = runtime.enter_execution_job(Some(outer_job));

        {
            let inner_assignment = ProgramExecutionAssignment {
                task_id: TaskId::derive(&[b"inner-task"]),
                ..assignment()
            };
            let _inner_assignment_guard =
                runtime.enter_execution_assignment(Some(inner_assignment.clone()));
            let _inner_job_guard =
                runtime.enter_execution_job(Some(JobId::derive(&[b"inner-job"])));
            assert_eq!(runtime.execution_assignment(), Some(inner_assignment));
        }

        assert_eq!(runtime.execution_assignment(), Some(outer_assignment));
        assert_eq!(runtime.execution_job(), Some(outer_job));
    }
}
