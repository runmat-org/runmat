use runmat_value::{CellArray, ObjectArray, Value};

use super::super::{error, plan::PartitionPlan};
use super::values::{options, take_group};

pub(super) fn cells(array: CellArray, dimensions: &[usize]) -> crate::BuiltinResult<Value> {
    // Unlike scalar numeric arrays, an authoritative cell shape of [] has no
    // elements to partition. Explicit dimensions still use normal rank admission.
    if array.shape.is_empty() && dimensions.is_empty() {
        return CellArray::new_with_shape(array.data, array.shape)
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
    let shape = array.shape().to_vec();
    let class = array.class_name().clone();
    let plan = PartitionPlan::new(&shape, dimensions)?;
    let mut values = options(array.into_data());
    let cells = plan
        .groups()
        .map(|indices| {
            let mut group = take_group(&mut values, &indices?, "object")?;
            if group.len() == 1 {
                return group
                    .pop()
                    .ok_or_else(|| error::internal("object group is empty"));
            }
            ObjectArray::new(class.clone(), group, plan.slice_shape.clone())
                .map(Value::ObjectArray)
                .map_err(error::internal)
        })
        .collect::<crate::BuiltinResult<Vec<_>>>()?;
    super::output(&plan, cells)
}
