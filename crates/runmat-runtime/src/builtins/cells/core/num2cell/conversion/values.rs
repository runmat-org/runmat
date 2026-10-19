use runmat_value::{LogicalArray, StringArray, SymbolicArray, Value};

use super::super::{error, plan::PartitionPlan};

pub(super) fn logical(array: LogicalArray, dimensions: &[usize]) -> crate::BuiltinResult<Value> {
    let plan = PartitionPlan::new(&array.shape, dimensions)?;
    let cells = plan
        .groups()
        .map(|indices| {
            let indices = indices?;
            if indices.len() == 1 {
                return array
                    .data
                    .get(indices[0])
                    .map(|value| Value::Bool(*value != 0))
                    .ok_or_else(|| error::internal("logical index is out of bounds"));
            }
            let data = select_copy(&array.data, &indices, "logical")?;
            LogicalArray::new(data, plan.slice_shape.clone())
                .map(Value::LogicalArray)
                .map_err(error::internal)
        })
        .collect::<crate::BuiltinResult<Vec<_>>>()?;
    super::output(&plan, cells)
}

pub(super) fn strings(array: StringArray, dimensions: &[usize]) -> crate::BuiltinResult<Value> {
    let plan = PartitionPlan::new(&array.shape, dimensions)?;
    let mut values = options(array.data);
    let cells = plan
        .groups()
        .map(|indices| {
            let mut group = take_group(&mut values, &indices?, "string")?;
            if group.len() == 1 {
                return group
                    .pop()
                    .map(Value::String)
                    .ok_or_else(|| error::internal("string group is empty"));
            }
            StringArray::new(group, plan.slice_shape.clone())
                .map(Value::StringArray)
                .map_err(error::internal)
        })
        .collect::<crate::BuiltinResult<Vec<_>>>()?;
    super::output(&plan, cells)
}

pub(super) fn symbolic(array: SymbolicArray, dimensions: &[usize]) -> crate::BuiltinResult<Value> {
    let plan = PartitionPlan::new(&array.shape, dimensions)?;
    let mut values = options(array.data);
    let cells = plan
        .groups()
        .map(|indices| {
            let mut group = take_group(&mut values, &indices?, "symbolic")?;
            if group.len() == 1 {
                return group
                    .pop()
                    .map(Value::Symbolic)
                    .ok_or_else(|| error::internal("symbolic group is empty"));
            }
            SymbolicArray::new(group, plan.slice_shape.clone())
                .map(Value::SymbolicArray)
                .map_err(error::internal)
        })
        .collect::<crate::BuiltinResult<Vec<_>>>()?;
    super::output(&plan, cells)
}

fn select_copy<T: Copy>(data: &[T], indices: &[usize], kind: &str) -> crate::BuiltinResult<Vec<T>> {
    indices
        .iter()
        .map(|index| {
            data.get(*index)
                .copied()
                .ok_or_else(|| error::internal(format!("{kind} index is out of bounds")))
        })
        .collect()
}

pub(super) fn options<T>(values: Vec<T>) -> Vec<Option<T>> {
    values.into_iter().map(Some).collect()
}

pub(super) fn take_group<T>(
    values: &mut [Option<T>],
    indices: &[usize],
    kind: &str,
) -> crate::BuiltinResult<Vec<T>> {
    indices
        .iter()
        .map(|index| {
            values
                .get_mut(*index)
                .and_then(Option::take)
                .ok_or_else(|| {
                    error::internal(format!("{kind} index is out of bounds or repeated"))
                })
        })
        .collect()
}
