use runmat_value::{ObjectInstance, Value};

use crate::BuiltinResult;

use super::{ensure_vector, numeric, GroupingInput, KeyOrder};

pub(super) fn datetime(value: ObjectInstance) -> BuiltinResult<GroupingInput> {
    let tensor =
        crate::builtins::datetime::serials_from_datetime_value(&Value::Object(value.clone()))?;
    prepare(value, tensor, "datetime")
}

pub(super) fn duration(value: ObjectInstance) -> BuiltinResult<GroupingInput> {
    let tensor = crate::builtins::duration::duration_tensor_from_duration_value(&Value::Object(
        value.clone(),
    ))?;
    prepare(value, tensor, "duration")
}

fn prepare(
    value: ObjectInstance,
    tensor: runmat_value::Tensor,
    kind: &str,
) -> BuiltinResult<GroupingInput> {
    ensure_vector(&tensor.shape, kind)?;
    Ok(GroupingInput::new(
        Value::Object(value),
        numeric::rows(&tensor)?,
        KeyOrder::FirstAppearance,
    ))
}
