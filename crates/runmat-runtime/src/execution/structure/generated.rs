//! Column-major structure generation into the canonical final backing.
//! The caller retains every borrowed source root through publication. This
//! admits final map, key, shape and slot owners, not the caller's index planner.
use super::*;
use runmat_value::{
    ExactArrayInitializationError, OutputListStorage, SharedShape, StructArrayBuilder,
    UnboundedStructArrayBuilder,
};

pub(crate) fn construct_generated_structure<'n, 'v, I, F>(
    runtime: Option<&RuntimeContext>,
    names: I,
    shape: &[usize],
    value: F,
) -> Result<RootedValue, RuntimeError>
where
    I: Iterator<Item = &'n str> + Clone,
    F: Fn(&str, usize) -> Result<&'v Value, RuntimeError>,
{
    construct_generated_structure_owned(runtime, names, shape, |name, index| {
        value(name, index).cloned()
    })
}

/// Initialize the same final structural backing with newly prepared Values.
/// The caller independently retains all GC children reachable from every
/// returned Value until publication; construction does not create GC objects.
pub(crate) fn construct_generated_structure_owned<'n, I, F>(
    runtime: Option<&RuntimeContext>,
    names: I,
    shape: &[usize],
    value: F,
) -> Result<RootedValue, RuntimeError>
where
    I: Iterator<Item = &'n str> + Clone,
    F: Fn(&str, usize) -> Result<Value, RuntimeError>,
{
    let error = |source| {
        host_allocation_runtime_error(
            runtime,
            "RunMat:Struct:Allocation",
            "structure destination admission failed",
            source,
        )
    };
    let owner = HostAllocationReceipt::capture_owner(runtime).map_err(error)?;
    if shape.len() < 2 {
        return Err(invalid(
            "structure arrays must have at least two dimensions",
        ));
    }
    let length = shape
        .iter()
        .try_fold(1usize, |n, &d| n.checked_mul(d))
        .ok_or_else(|| invalid("structure array dimensions overflow"))?;
    let count = names.clone().try_fold(0usize, |n, _| {
        check_owner(runtime, &owner).map_err(error)?;
        n.checked_add(1)
            .ok_or_else(|| error(HostAllocationError::Overflow))
    })?;
    let receipt = if length == 1 {
        admit_struct_generation_counted(runtime, names.clone(), count).map_err(error)?
    } else {
        let slots = OutputListStorage::requested_bytes(length)
            .map_err(|e| error(e.into()))?
            .checked_mul(count)
            .ok_or_else(|| error(HostAllocationError::Overflow))?;
        let shape_bytes =
            SharedShape::requested_bytes_for_rank(shape.len()).map_err(|e| error(e.into()))?;
        let bytes = slots
            .checked_add(shape_bytes)
            .ok_or_else(|| error(HostAllocationError::Overflow))?;
        let claims = count
            .checked_mul(usize::from(length != 0))
            .and_then(|n| n.checked_add(1))
            .ok_or_else(|| error(HostAllocationError::Overflow))?;
        admit_struct_generation_extra_for::<OutputListStorage, _>(
            runtime,
            names.clone(),
            count,
            bytes,
            claims,
        )
        .map_err(error)?
    };
    check_owner(runtime, &owner).map_err(error)?;
    let mut emitted = 0usize;
    let result = if let Some(receipt) = &receipt {
        let runtime = runtime.expect("managed receipt has selected runtime");
        let permit = receipt.output_permit(runtime).map_err(error)?;
        if length == 1 {
            let builder = StructValueBuilder::try_new(count, &permit);
            receipt.checkpoint(runtime).map_err(error)?;
            let mut builder = builder.map_err(|e| error(e.into()))?;
            for name in names {
                receipt.checkpoint(runtime).map_err(error)?;
                let key = copy_key(runtime, receipt, &permit, name)?;
                let child = value(name, 0)?;
                builder.try_insert_prepared(key, child).map_err(|source| {
                    crate::runtime_error::semantic_error_with_source(
                        "StructConstruction",
                        "structure field insertion failed",
                        source.into_parts().0,
                    )
                })?;
                emitted += 1;
            }
            Value::Struct(builder.finish())
        } else {
            let builder = StructArrayBuilder::try_new(count, shape, &permit);
            receipt.checkpoint(runtime).map_err(error)?;
            let mut builder = builder.map_err(|e| super::array::array_error(runtime, e))?;
            for name in names {
                let key = copy_key(runtime, receipt, &permit, name)?;
                let column = OutputListStorage::try_initialize(length, &permit, |index| {
                    if index % 4096 == 0 {
                        receipt.checkpoint(runtime).map_err(error)?;
                    }
                    value(name, index)
                });
                receipt.checkpoint(runtime).map_err(error)?;
                let column = column.map_err(|e| match e {
                    ExactArrayInitializationError::Allocation(e) => error(e.into()),
                    ExactArrayInitializationError::Element(e) => e,
                })?;
                builder
                    .try_insert_prepared(key, column)
                    .map_err(|source| match source {
                        runmat_value::StructArrayFieldError::Map(source) => {
                            crate::runtime_error::semantic_error_with_source(
                                "StructConstruction",
                                "structure column insertion failed",
                                source.into_parts().0,
                            )
                        }
                        runmat_value::StructArrayFieldError::InvalidColumn { .. } => {
                            iterator_changed()
                        }
                    })?;
                emitted += 1;
            }
            Value::StructArray(builder.finish())
        }
    } else if length == 1 {
        let mut builder = UnboundedStructValueBuilder::with_capacity(count);
        for name in names {
            builder.insert(name, value(name, 0)?);
            emitted += 1;
        }
        Value::Struct(builder.finish())
    } else {
        let mut builder = UnboundedStructArrayBuilder::with_capacity(shape.to_vec().into(), count)
            .map_err(|e| invalid(e.to_string()))?;
        for name in names {
            let mut column =
                runmat_value::exact_capacity_vec(length).map_err(|e| error(e.into()))?;
            for index in 0..length {
                column.push(value(name, index)?);
            }
            builder
                .insert(name.to_owned(), OutputListStorage::shared_unbounded(column))
                .map_err(|e| invalid(e.to_string()))?;
            emitted += 1;
        }
        Value::StructArray(builder.finish())
    };
    if emitted != count {
        return Err(iterator_changed());
    }
    check_owner(runtime, &owner).map_err(error)?;
    let result = match runtime {
        Some(runtime) => RootedValue::in_context(result, runtime)?,
        None => RootedValue::new(result).map_err(|e| e.into_runtime_error())?,
    };
    check_owner(runtime, &owner).map_err(error)?;
    Ok(result)
}
fn invalid(message: impl Into<String>) -> RuntimeError {
    crate::runtime_error::semantic_error("StructConstruction", message)
}
