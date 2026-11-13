//! Owned payloads that provenance projection or host gathering can transform.
//! Managed handles and listeners publish shared referents; neither operation
//! rewrites those referents, so they must not force a new callable instance.

use runmat_value::Value;

pub(super) fn has_transformable_gpu(value: &Value) -> bool {
    match value {
        Value::GpuTensor(_) => true,
        Value::Cell(cell) => cell.data().iter().any(has_transformable_gpu),
        Value::Struct(object) => object.fields.values().any(has_transformable_gpu),
        Value::StructArray(array) => {
            let mut found = false;
            array.for_each_value(|value| found |= has_transformable_gpu(value));
            found
        }
        Value::Object(object) => object.properties().values().any(has_transformable_gpu),
        Value::ObjectArray(array) => array.data().iter().any(has_transformable_gpu),
        Value::Closure(closure) => closure.captures.iter().any(has_transformable_gpu),
        Value::OutputList(values) => values.iter().any(has_transformable_gpu),
        Value::HandleObject(_) | Value::Listener(_) => false,
        _ => false,
    }
}
