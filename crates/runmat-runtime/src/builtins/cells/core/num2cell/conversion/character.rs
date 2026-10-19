use runmat_value::{CharArray, Value};

use super::super::{
    error,
    plan::{column_major_coords, PartitionPlan},
};

pub(super) fn convert(array: CharArray, dimensions: &[usize]) -> crate::BuiltinResult<Value> {
    let shape = vec![array.rows, array.cols];
    let plan = PartitionPlan::new(&shape, dimensions)?;
    let cells = plan
        .groups()
        .map(|indices| {
            let indices = indices?;
            let rows = plan.slice_shape.first().copied().unwrap_or(1);
            let columns = plan.slice_shape.get(1).copied().unwrap_or(1);
            let mut data = vec!['\0'; indices.len()];
            for (slice_linear, index) in indices.iter().enumerate() {
                let row = index % array.rows.max(1);
                let column = index / array.rows.max(1);
                let source_offset = row
                    .checked_mul(array.cols)
                    .and_then(|offset| offset.checked_add(column))
                    .ok_or_else(|| error::internal("character index overflow"))?;
                let coords = column_major_coords(slice_linear, &plan.slice_shape);
                let destination = coords[0]
                    .checked_mul(columns)
                    .and_then(|offset| offset.checked_add(coords[1]))
                    .ok_or_else(|| error::internal("character slice index overflow"))?;
                data[destination] = *array
                    .data
                    .get(source_offset)
                    .ok_or_else(|| error::internal("character index is out of bounds"))?;
            }
            CharArray::new(data, rows, columns)
                .map(Value::CharArray)
                .map_err(error::internal)
        })
        .collect::<crate::BuiltinResult<Vec<_>>>()?;
    super::output(&plan, cells)
}
