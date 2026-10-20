use super::*;

pub fn reshape(array: StructArray, shape: Vec<usize>) -> Result<StructArray, RuntimeError> {
    if shape.len() < 2 || count(&shape)? != array.len() {
        return Err(invalid(
            "structure reshape must preserve element count and rank",
        ));
    }
    let runtime = crate::context::legacy::active();
    let original = Value::StructArray(array.clone());
    let _roots = replacement::lease(&original, runtime.as_ref())?;
    let shape = shape_owner(runtime.as_ref(), &shape)?;
    let output = if let Some(runtime) = runtime.as_ref() {
        crate::execution::construct_struct_array_from_columns_in_context(
            runtime,
            array.column_owners(),
            &shape,
        )?
    } else {
        let mut output = runmat_value::UnboundedStructArrayBuilder::with_capacity(
            shape,
            array.field_names().len(),
        )
        .map_err(|e| invalid(e.to_string()))?;
        for (name, values) in array.column_owners() {
            output
                .insert(name.to_owned(), values.clone())
                .map_err(|e| invalid(e.to_string()))?;
        }
        RootedValue::new(Value::StructArray(output.finish())).map_err(|e| e.into_runtime_error())?
    };
    array_result(output)
}
pub fn permute(array: StructArray, order: &[usize]) -> Result<StructArray, RuntimeError> {
    let effective = array
        .shape()
        .iter()
        .rposition(|&d| d != 1)
        .map_or(2, |axis| (axis + 1).max(2));
    if order.len() < effective
        || order.iter().any(|&axis| axis >= order.len())
        || order
            .iter()
            .enumerate()
            .any(|(index, axis)| order[..index].contains(axis))
    {
        return Err(invalid(
            "structure permutation must name each dimension exactly once",
        ));
    }
    let runtime = crate::context::legacy::active();
    let rank = order.len();
    let mut scratch = Scratch::new(
        runtime.as_ref(),
        rank.checked_mul(2)
            .ok_or_else(|| invalid("permutation rank overflow"))?,
    )?;
    let (shape, strides) = scratch.values_mut().split_at_mut(rank);
    let mut stride = 1usize;
    for axis in 0..rank {
        shape[axis] = array.shape().get(order[axis]).copied().unwrap_or(1);
        strides[axis] = stride;
        stride = stride
            .checked_mul(array.shape().get(axis).copied().unwrap_or(1))
            .ok_or_else(|| invalid("permutation stride overflow"))?;
    }
    scratch.checkpoint()?;
    let (shape, strides) = scratch.values().split_at(rank);
    reindex(&array, shape, |mut index| {
        let mut source = 0usize;
        for axis in 0..rank {
            let coordinate = index % shape[axis];
            index /= shape[axis];
            source = source
                .checked_add(
                    coordinate
                        .checked_mul(strides[order[axis]])
                        .ok_or_else(|| invalid("permutation index overflow"))?,
                )
                .ok_or_else(|| invalid("permutation index overflow"))?;
        }
        Ok(source)
    })
}
pub fn flip(array: StructArray, dimension: usize) -> Result<StructArray, RuntimeError> {
    let extent = *array
        .shape()
        .get(dimension)
        .ok_or_else(|| invalid("structure flip dimension is out of range"))?;
    if extent <= 1 || array.is_empty() {
        return Ok(array);
    }
    let stride = count(&array.shape()[..dimension])?;
    reindex(&array, array.shape(), |index| {
        let coordinate = index / stride % extent;
        Ok(index - coordinate * stride + (extent - 1 - coordinate) * stride)
    })
}
pub fn circular_shift(array: StructArray, shifts: &[isize]) -> Result<StructArray, RuntimeError> {
    let runtime = crate::context::legacy::active();
    let rank = array.shape().len().max(shifts.len());
    let mut scratch = Scratch::new(runtime.as_ref(), rank)?;
    for (axis, extent) in scratch.values_mut().iter_mut().enumerate() {
        *extent = array.shape().get(axis).copied().unwrap_or(1);
    }
    scratch.checkpoint()?;
    reindex(&array, scratch.values(), |mut index| {
        let mut source = 0usize;
        let mut stride = 1usize;
        for (axis, &extent) in scratch.values().iter().enumerate() {
            let coordinate = index % extent;
            index /= extent;
            let shift = shifts.get(axis).copied().unwrap_or(0);
            let coordinate =
                ((coordinate as i128) - (shift as i128)).rem_euclid(extent as i128) as usize;
            source = source
                .checked_add(
                    coordinate
                        .checked_mul(stride)
                        .ok_or_else(|| invalid("shift index overflow"))?,
                )
                .ok_or_else(|| invalid("shift index overflow"))?;
            stride = stride
                .checked_mul(extent)
                .ok_or_else(|| invalid("shift stride overflow"))?;
        }
        Ok(source)
    })
}
pub fn tile(array: &StructArray, repetitions: &[usize]) -> Result<StructArray, RuntimeError> {
    let runtime = crate::context::legacy::active();
    let rank = array.shape().len().max(repetitions.len()).max(2);
    let mut scratch = Scratch::new(runtime.as_ref(), rank)?;
    for (axis, extent) in scratch.values_mut().iter_mut().enumerate() {
        *extent = array
            .shape()
            .get(axis)
            .copied()
            .unwrap_or(1)
            .checked_mul(repetitions.get(axis).copied().unwrap_or(1))
            .ok_or_else(|| invalid("structure replication dimensions overflow"))?;
    }
    scratch.checkpoint()?;
    reindex(array, scratch.values(), |mut index| {
        let mut source = 0usize;
        let mut stride = 1usize;
        for (axis, &extent) in scratch.values().iter().enumerate() {
            let coordinate = index % extent;
            index /= extent;
            let input = array.shape().get(axis).copied().unwrap_or(1);
            source = source
                .checked_add(
                    (coordinate % input)
                        .checked_mul(stride)
                        .ok_or_else(|| invalid("replication index overflow"))?,
                )
                .ok_or_else(|| invalid("replication index overflow"))?;
            stride = stride
                .checked_mul(input)
                .ok_or_else(|| invalid("replication stride overflow"))?;
        }
        Ok(source)
    })
}
