use runmat_execution::TaskHandle;
use runmat_value::{ObjectArray, ObjectInstance, Value};

pub const FEVAL_FUTURE_CLASS: runmat_types::StaticClassIdentity =
    runmat_types::StaticClassIdentity::new("parallel.FevalFuture");
pub const FEVAL_ON_ALL_FUTURE_CLASS: runmat_types::StaticClassIdentity =
    runmat_types::StaticClassIdentity::new("parallel.FevalOnAllFuture");
const EXECUTION_HANDLE_PROPERTY: &str = "__runmat_execution_handle";

pub fn wrap_task(task: TaskHandle) -> Value {
    Value::Object(future_object(task))
}

pub fn wrap_tasks(tasks: Vec<TaskHandle>) -> Result<Value, String> {
    let length = tasks.len();
    ObjectArray::new(
        FEVAL_FUTURE_CLASS.to_string(),
        tasks
            .into_iter()
            .map(|task| Value::Object(future_object(task)))
            .collect(),
        vec![1, length],
    )
    .map(Value::ObjectArray)
}

pub fn wrap_on_all_tasks(tasks: Vec<TaskHandle>) -> Result<Value, String> {
    let mut object = ObjectInstance::new(FEVAL_ON_ALL_FUTURE_CLASS.to_string());
    object
        .properties
        .insert(EXECUTION_HANDLE_PROPERTY.to_string(), wrap_tasks(tasks)?);
    Ok(Value::Object(object))
}

pub fn execution_value(value: &Value) -> Option<&Value> {
    match value {
        Value::Object(object) if object.class_name.is(FEVAL_FUTURE_CLASS) => {
            object.properties.get(EXECUTION_HANDLE_PROPERTY)
        }
        Value::Future(_) | Value::Task(_) | Value::Job(_) => Some(value),
        _ => None,
    }
}

pub fn tasks(value: &Value) -> Option<Vec<TaskHandle>> {
    match value {
        Value::Object(object) => task_from_object(object).map(|task| vec![task]),
        Value::ObjectArray(array) if array.class_name().is(FEVAL_FUTURE_CLASS) => array
            .data()
            .iter()
            .map(|value| match value {
                Value::Object(object) => task_from_object(object),
                _ => None,
            })
            .collect::<Option<Vec<_>>>(),
        Value::Task(task) => Some(vec![task.clone()]),
        _ => None,
    }
}

pub fn output_tasks(value: &Value) -> Option<Vec<TaskHandle>> {
    match value {
        Value::Object(object) if object.class_name.is(FEVAL_ON_ALL_FUTURE_CLASS) => object
            .properties
            .get(EXECUTION_HANDLE_PROPERTY)
            .and_then(tasks),
        _ => tasks(value),
    }
}

fn future_object(task: TaskHandle) -> ObjectInstance {
    let mut object = ObjectInstance::new(FEVAL_FUTURE_CLASS.to_string());
    object
        .properties
        .insert(EXECUTION_HANDLE_PROPERTY.to_string(), Value::Task(task));
    object
}

fn task_from_object(object: &ObjectInstance) -> Option<TaskHandle> {
    if !object.class_name.is(FEVAL_FUTURE_CLASS) {
        return None;
    }
    match object.properties.get(EXECUTION_HANDLE_PROPERTY) {
        Some(Value::Task(task)) => Some(task.clone()),
        _ => None,
    }
}
