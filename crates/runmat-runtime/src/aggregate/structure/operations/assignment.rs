use super::*;
use crate::indexing::plan::IndexPlan;

/// Schema and RHS arity are validated by the indexing adapter. Duplicate
/// destination indices use the final RHS occurrence, as ordinary assignment.
pub fn assign(base: &Value, rhs: &Value, plan: &IndexPlan) -> Result<Value, RuntimeError> {
    let runtime = crate::context::legacy::active();
    assign_in_context(runtime.as_ref(), base, rhs, plan)
}

pub(crate) fn assign_in_context(
    runtime: Option<&RuntimeContext>,
    base: &Value,
    rhs: &Value,
    plan: &IndexPlan,
) -> Result<Value, RuntimeError> {
    assign_with_schema(runtime, base, rhs, plan, Schema::Strict).map(raw)
}

#[derive(Clone, Copy)]
enum Schema {
    Strict,
    InitializingMemberPath,
}

/// Only a captured, initializing member path may add fields while writing its
/// selected structure child back. Ordinary structure-array assignment remains
/// schema-strict at its existing indexing adapter.
pub(crate) fn assign_initializing_in_context(
    runtime: Option<&RuntimeContext>,
    base: &Value,
    rhs: &Value,
    plan: &IndexPlan,
) -> Result<RootedValue, RuntimeError> {
    let owner = crate::indexing::plan::admission::Owner::new(runtime)?;
    if plan.indices.is_empty() {
        let unchanged = RootedValue::in_optional_context(base.clone(), runtime)?;
        owner.check()?;
        return Ok(unchanged);
    }
    shape(base)?;
    let rhs_shape = shape(rhs)?;
    if !matches!(rhs, Value::Struct(_)) && count(rhs_shape)? != plan.indices.len() {
        return Err(invalid(
            "structure path writeback has an invalid replacement count",
        ));
    }
    if names(base).any(|name| !has_field(rhs, name)) {
        return Err(invalid(
            "structure path writeback cannot remove existing fields",
        ));
    }
    assign_with_schema(runtime, base, rhs, plan, Schema::InitializingMemberPath)
}

fn assign_with_schema(
    runtime: Option<&RuntimeContext>,
    base: &Value,
    rhs: &Value,
    plan: &IndexPlan,
    schema: Schema,
) -> Result<RootedValue, RuntimeError> {
    let source_shape = shape(base)?;
    if source_shape
        .iter()
        .enumerate()
        .any(|(axis, &extent)| plan.base_shape.get(axis).copied().unwrap_or(1) < extent)
    {
        return Err(invalid("structure growth cannot shrink a dimension"));
    }
    let _roots = match runtime {
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
    let mut positions = Scratch::new(runtime, length)?;
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
    let extending = matches!(schema, Schema::InitializingMemberPath);
    let fill = if count(source_shape)? < length
        || (extending && names(rhs).any(|name| !has_field(base, name)))
    {
        Some(empty(runtime)?)
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
        if let Some(source) = source_index(destination, source_shape, &plan.base_shape)? {
            if has_field(base, name) {
                return field(base, name, source);
            }
        }
        fill.as_ref()
            .ok_or_else(|| invalid("structure growth default is missing"))
    };
    let fields = names(base).chain(names(rhs).filter(|name| extending && !has_field(base, name)));
    let output = construct_generated_structure(runtime, fields, &plan.base_shape, value)?;
    positions.checkpoint()?;
    Ok(output)
}

pub(super) fn names(value: &Value) -> impl Iterator<Item = &str> + Clone {
    let scalar = match value {
        Value::Struct(value) => Some(value.field_names()),
        _ => None,
    };
    let array = match value {
        Value::StructArray(value) => Some(value.field_names()),
        _ => None,
    };
    scalar
        .into_iter()
        .flatten()
        .chain(array.into_iter().flatten())
}

fn has_field(value: &Value, name: &str) -> bool {
    match value {
        Value::Struct(value) => value.fields.contains_key(name),
        Value::StructArray(value) => value.column_owner(name).is_some(),
        _ => false,
    }
}

/// Canonical column-major relocation for growth. Both projection and final
/// assignment use this mapping; missing coordinates are new empty elements.
pub(super) fn source_index(
    destination: usize,
    source_shape: &[usize],
    destination_shape: &[usize],
) -> Result<Option<usize>, RuntimeError> {
    let mut remainder = destination;
    let mut source = 0usize;
    let mut stride = 1usize;
    for (axis, &extent) in destination_shape.iter().enumerate() {
        if extent == 0 {
            return Err(invalid("structure assignment index is out of bounds"));
        }
        let coordinate = remainder % extent;
        remainder /= extent;
        let old_extent = source_shape.get(axis).copied().unwrap_or(1);
        if coordinate >= old_extent {
            return Ok(None);
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
    if remainder != 0 {
        return Err(invalid("structure assignment index is out of bounds"));
    }
    Ok(Some(source))
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
pub(crate) fn empty(runtime: Option<&RuntimeContext>) -> Result<Value, RuntimeError> {
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
