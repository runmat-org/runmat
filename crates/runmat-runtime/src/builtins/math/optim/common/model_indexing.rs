//! Named selectors for actual model-array axes. Original variable projections
//! and immutable expression metadata supply labels; canonical index planning
//! alone owns numeric/logical selection and physical stride arithmetic.
use super::{model_policy::Policy, ScalarCooperation};
use crate::BuiltinResult;
use runmat_value::{HostNumericBuffer, NumericStorage, Tensor, Utf8Text, Value};
pub(in crate::builtins::math::optim) trait Axes {
    fn shape(&self) -> &[usize];
    fn has_names(&self, axis: usize) -> bool;
    fn name(&self, axis: usize, position: usize) -> Option<&Utf8Text>;
}
pub(in crate::builtins::math::optim) fn is_name_selector(value: &Value) -> bool {
    matches!(
        value,
        Value::String(_) | Value::StringArray(_) | Value::CharArray(_) | Value::Cell(_)
    )
}
pub(in crate::builtins::math::optim) fn vector_axis(axes: &impl Axes) -> usize {
    axes.shape().iter().position(|&n| n > 1).unwrap_or_else(|| {
        (0..axes.shape().len())
            .find(|&axis| axes.has_names(axis))
            .unwrap_or(0)
    })
}
pub(in crate::builtins::math::optim) fn axis(
    axes: &impl Axes,
    count: usize,
    index: usize,
    policy: Policy,
) -> BuiltinResult<usize> {
    if count == 1 {
        if axes.shape().len() != 2 || axes.shape().iter().filter(|&&n| n > 1).count() > 1 {
            return Err(policy.invalid("named linear indexing requires a vector"));
        }
        Ok(vector_axis(axes))
    } else if index < axes.shape().len() {
        Ok(index)
    } else {
        Err(policy.invalid("named selector exceeds model array rank"))
    }
}
pub(in crate::builtins::math::optim) async fn indices(
    axes: &impl Axes,
    axis: usize,
    value: &Value,
    cooperation: &mut ScalarCooperation,
    policy: Policy,
) -> BuiltinResult<Value> {
    let mut output = Vec::new();
    let count = match value {
        Value::Cell(cell) => {
            cell.validate_layout()
                .map_err(|source| policy.invalid(source))?;
            cell.data().len()
        }
        Value::StringArray(array) => {
            array
                .validate_layout()
                .map_err(|source| policy.invalid(source))?;
            array.data().len()
        }
        _ => 1,
    };
    for ordinal in 0..count {
        let name = match value {
            Value::Cell(cell) => policy.text(&cell.data()[ordinal], cooperation.runtime())?,
            Value::StringArray(array) => policy.text(
                &Value::String(array.data()[ordinal].clone()),
                cooperation.runtime(),
            )?,
            _ => policy.text(value, cooperation.runtime())?,
        };
        let mut found = false;
        for position in 0..axes.shape()[axis] {
            if axes
                .name(axis, position)
                .is_some_and(|known| known.as_str() == name.as_str())
            {
                let index = u64::try_from(position)
                    .ok()
                    .and_then(|value| value.checked_add(1))
                    .ok_or_else(|| policy.invalid("named index exceeds uint64"))?;
                output
                    .try_reserve(1)
                    .map_err(|source| policy.reserve(source))?;
                output.push(index);
                found = true;
            }
            cooperation.completed(1).await?;
        }
        if !found {
            return Err(policy.invalid("unknown model array index name"));
        }
    }
    let len = output.len();
    let tensor = Tensor::from_host_buffer(
        HostNumericBuffer::from_numeric_storage(NumericStorage::U64(output)),
        vec![1, len],
    )
    .map_err(|source| policy.invalid(source))?;
    Ok(Value::Tensor(tensor))
}
