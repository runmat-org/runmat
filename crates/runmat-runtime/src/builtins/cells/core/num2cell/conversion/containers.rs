use runmat_value::{CellArray, ObjectArray, Value};

use super::super::{error, plan::PartitionPlan};
use super::values::{options, take_group};

pub(super) fn cells(array: CellArray, dimensions: &[usize]) -> crate::BuiltinResult<Value> {
    // Unlike scalar numeric arrays, an authoritative cell shape of [] has no
    // elements to partition. Explicit dimensions still use normal rank admission.
    if array.shape.is_empty() && dimensions.is_empty() {
        let (data, shape, _, _) = array.into_parts();
        return CellArray::from_shared_storage(data, shape)
            .map(Value::Cell)
            .map_err(error::internal);
    }
    let shape = array.shape.clone();
    let plan = PartitionPlan::new(&shape, dimensions)?;
    let mut values = options(array.into_column_major().map_err(error::internal)?);
    let cells = plan
        .groups()
        .map(|indices| {
            let group = take_group(&mut values, &indices?, "cell")?;
            CellArray::from_column_major(group, plan.slice_shape.clone())
                .map(Value::Cell)
                .map_err(error::internal)
        })
        .collect::<crate::BuiltinResult<Vec<_>>>()?;
    super::output(&plan, cells)
}

pub(super) fn objects(array: ObjectArray, dimensions: &[usize]) -> crate::BuiltinResult<Value> {
    let plan = PartitionPlan::new(array.shape(), dimensions)?;
    let context = crate::context::legacy::active();
    let cells = plan
        .groups()
        .map(|indices| {
            let indices = indices?;
            if let [index] = indices.as_slice() {
                return array
                    .data()
                    .get(*index)
                    .cloned()
                    .ok_or_else(|| error::internal("object group index is out of bounds"));
            }
            crate::execution::object_array::select(
                &array,
                indices.iter().copied(),
                &plan.slice_shape,
                context.as_ref(),
            )
            .map(|output| output.with_owned(|value| value))
        })
        .collect::<crate::BuiltinResult<Vec<_>>>()?;
    super::output(&plan, cells)
}
