use runmat_execution::TaskHandle;
use runmat_value::{ObjectArray, ObjectInstance, SharedTaskHandle, UnboundedObjectBuilder, Value};

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
    let mut object = UnboundedObjectBuilder::new(FEVAL_ON_ALL_FUTURE_CLASS.to_string());
    object.insert(EXECUTION_HANDLE_PROPERTY.to_string(), wrap_tasks(tasks)?);
    Ok(Value::Object(object.finish()))
}

pub fn execution_value(value: &Value) -> Option<&Value> {
    match value {
        Value::Object(object) if object.class_name().is(FEVAL_FUTURE_CLASS) => {
            object.properties().get(EXECUTION_HANDLE_PROPERTY)
        }
        Value::Future(_) | Value::Task(_) | Value::Job(_) => Some(value),
        _ => None,
    }
}

pub fn tasks(value: &Value) -> Option<Vec<SharedTaskHandle>> {
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

pub fn output_tasks(value: &Value) -> Option<Vec<SharedTaskHandle>> {
    match value {
        Value::Object(object) if object.class_name().is(FEVAL_ON_ALL_FUTURE_CLASS) => object
            .properties()
            .get(EXECUTION_HANDLE_PROPERTY)
            .and_then(tasks),
        _ => tasks(value),
    }
}

/// Borrow the same validated task targets as output_tasks, without cloning
/// their handles into a second Vec. Validation finishes before iteration, so a
/// malformed aggregate cannot cause a partially applied cancellation.
#[derive(Clone)]
pub struct OutputTaskValues<'a> {
    values: std::slice::Iter<'a, Value>,
    wrapped: bool,
}
impl<'a> Iterator for OutputTaskValues<'a> {
    type Item = &'a Value;
    fn next(&mut self) -> Option<Self::Item> {
        let value = self.values.next()?;
        if self.wrapped {
            execution_value(value)
        } else {
            Some(value)
        }
    }
}
pub fn output_task_values(value: &Value) -> Option<OutputTaskValues<'_>> {
    let value = match value {
        Value::Object(object) if object.class_name().is(FEVAL_ON_ALL_FUTURE_CLASS) => {
            object.properties().get(EXECUTION_HANDLE_PROPERTY)?
        }
        value => value,
    };
    let (values, wrapped) = match value {
        Value::Task(_) => (std::slice::from_ref(value), false),
        Value::Object(object) if object.class_name().is(FEVAL_FUTURE_CLASS) => {
            let value = execution_value(value)?;
            if !matches!(value, Value::Task(_)) {
                return None;
            }
            (std::slice::from_ref(value), false)
        }
        Value::ObjectArray(array) if array.class_name().is(FEVAL_FUTURE_CLASS) => {
            if !array.data().iter().all(|value| {
                matches!(value, Value::Object(_))
                    && matches!(execution_value(value), Some(Value::Task(_)))
            }) {
                return None;
            }
            (array.data(), true)
        }
        _ => return None,
    };
    Some(OutputTaskValues {
        values: values.iter(),
        wrapped,
    })
}

fn future_object(task: TaskHandle) -> ObjectInstance {
    let mut object = UnboundedObjectBuilder::new(FEVAL_FUTURE_CLASS.to_string());
    object.insert(
        EXECUTION_HANDLE_PROPERTY.to_string(),
        Value::Task(task.into()),
    );
    object.finish()
}

fn task_from_object(object: &ObjectInstance) -> Option<SharedTaskHandle> {
    if !object.class_name().is(FEVAL_FUTURE_CLASS) {
        return None;
    }
    match object.properties().get(EXECUTION_HANDLE_PROPERTY) {
        Some(Value::Task(task)) => Some(task.clone()),
        _ => None,
    }
}
