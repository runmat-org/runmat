use super::*;
use crate::indexing::plan::IndexPlan;

/// Schema and RHS arity are validated by the indexing adapter. Duplicate
/// destination indices use the final RHS occurrence, as ordinary assignment.
pub fn assign(base: &Value, rhs: &Value, plan: &IndexPlan) -> Result<Value, RuntimeError> {
    let source_shape = shape(base)?;
    if source_shape
        .iter()
        .enumerate()
        .any(|(axis, &extent)| plan.base_shape.get(axis).copied().unwrap_or(1) < extent)
    {
        return Err(invalid("structure growth cannot shrink a dimension"));
    }
    let runtime = crate::context::legacy::active();
    let _roots = match runtime.as_ref() {
        Some(runtime) => crate::execution::RootedValueLease::for_value_slices_in_context(
            &[std::slice::from_ref(base), std::slice::from_ref(rhs)],
            runtime,
        )?,
        None => {
            HostAllocationReceipt::capture_owner(None).map_err(|e| allocation(None, e))?;
            crate::execution::RootedValueLease::for_values([base, rhs].into_iter())
                .map_err(|e| e.into_runtime_error())?
        }
    };
    let length = count(&plan.base_shape)?;
    let mut positions = Scratch::new(runtime.as_ref(), length)?;
    for offset in (0..length).step_by(4096) {
        let end = length.min(offset.saturating_add(4096));
        positions.values_mut()[offset..end].fill(usize::MAX);
        positions.checkpoint()?;
    }
    for (source, &destination) in plan.indices.iter().enumerate() {
        if source % 4096 == 0 {
            positions.checkpoint()?;
        }
        *positions
            .values_mut()
            .get_mut(destination as usize)
            .ok_or_else(|| invalid("structure assignment index is out of bounds"))? = source;
    }
    positions.checkpoint()?;
    let fill = if count(source_shape)? < length {
        Some(empty(runtime.as_ref())?)
    } else {
        None
    };
    let value = |name: &str, destination: usize| {
        let replacement = positions.values()[destination];
        if replacement != usize::MAX {
            let source = if matches!(rhs, Value::Struct(_)) {
                0
            } else {
                replacement
            };
            return field(rhs, name, source);
        }
        let mut remainder = destination;
        let mut source = 0usize;
        let mut stride = 1usize;
        for (axis, &extent) in plan.base_shape.iter().enumerate() {
            let coordinate = remainder % extent;
            remainder /= extent;
            let old_extent = source_shape.get(axis).copied().unwrap_or(1);
            if coordinate >= old_extent {
                return fill
                    .as_ref()
                    .ok_or_else(|| invalid("structure growth default is missing"));
            }
            source = source
                .checked_add(
                    coordinate
                        .checked_mul(stride)
                        .ok_or_else(|| invalid("structure growth index overflow"))?,
                )
                .ok_or_else(|| invalid("structure growth index overflow"))?;
            stride = stride
                .checked_mul(old_extent)
                .ok_or_else(|| invalid("structure growth stride overflow"))?;
        }
        field(base, name, source)
    };
    let output = match base {
        Value::Struct(base) => construct_generated_structure(
            runtime.as_ref(),
            base.field_names(),
            &plan.base_shape,
            value,
        ),
        Value::StructArray(base) => construct_generated_structure(
            runtime.as_ref(),
            base.field_names(),
            &plan.base_shape,
            value,
        ),
        _ => Err(invalid("structure assignment requires a structure")),
    }?;
    positions.checkpoint()?;
    Ok(raw(output))
}
pub(super) fn shape(value: &Value) -> Result<&[usize], RuntimeError> {
    match value {
        Value::Struct(_) => Ok(&[1, 1]),
        Value::StructArray(value) => Ok(value.shape()),
        _ => Err(invalid("structure value expected")),
    }
}
pub(super) fn field<'a>(
    value: &'a Value,
    name: &str,
    index: usize,
) -> Result<&'a Value, RuntimeError> {
    let value = match value {
        Value::Struct(value) if index == 0 => value.fields.get(name),
        Value::StructArray(value) => value
            .field_values(name)
            .and_then(|values| values.get(index)),
        _ => None,
    };
    value.ok_or_else(|| invalid("structure field or index is missing"))
}
fn empty(runtime: Option<&RuntimeContext>) -> Result<Value, RuntimeError> {
    let shape = [0, 0];
    let dtype = runmat_value::NumericDType::F64;
    let bytes = HostAllocationPlan::retained_tensor_bytes(dtype, &shape)
        .map_err(|e| allocation(runtime, e))?;
    let plan = HostAllocationPlan::new(0, bytes).map_err(|e| allocation(runtime, e))?;
    let receipt =
        HostAllocationReceipt::admit(runtime, plan).map_err(|e| allocation(runtime, e))?;
    let tensor = if let Some(receipt) = receipt {
        let runtime = runtime.expect("managed context");
        let output = receipt
            .exact_output_tensor(runtime, dtype, &shape)
            .map_err(|e| allocation(Some(runtime), e))?;
        let output = output.into_tensor();
        receipt
            .checkpoint(runtime)
            .map_err(|e| allocation(Some(runtime), e))?;
        output.map_err(|e| allocation(Some(runtime), e.into()))?
    } else {
        runmat_value::Tensor::zeros(shape.to_vec())
    };
    Ok(Value::Tensor(tensor))
}
