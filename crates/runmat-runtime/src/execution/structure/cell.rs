//! Exact private Cell slots for structural transformations of rooted sources.
use crate::context::{
    host_allocation::{
        host_allocation_runtime_error, HostAllocationError, HostAllocationPlan,
        HostAllocationPolicy, HostAllocationReceipt,
    },
    RuntimeContext,
};
use crate::execution::{RootedValue, RootedValues};
use crate::RuntimeError;
use runmat_value::{CellArray, SharedShape, Value};

pub(crate) fn from_values(
    runtime: Option<&RuntimeContext>,
    shape: &[usize],
    values: &[Value],
) -> Result<RootedValue, RuntimeError> {
    from_borrowed_values(runtime, shape, values.iter())
}

/// Publish an already-rooted final slot owner as a Cell without copying its
/// Values into a second collection. Only the Cell shape/header remains to be
/// admitted here; every child retains the authority carried by `values`.
pub(crate) fn from_rooted_values(
    runtime: &RuntimeContext,
    shape: &[usize],
    values: RootedValues,
) -> Result<RootedValue, RuntimeError> {
    let expected = shape
        .iter()
        .try_fold(1usize, |count, dimension| count.checked_mul(*dimension))
        .ok_or_else(|| {
            crate::runtime_error::semantic_error("CellConstruction", "Cell shape length overflow")
        })?;
    if expected != values.len() {
        return Err(crate::runtime_error::semantic_error(
            "CellConstruction",
            "Cell payload length does not match its shape",
        ));
    }
    let error = |source| {
        host_allocation_runtime_error(
            Some(runtime),
            "RunMat:CellConstruction",
            "Cell shape admission failed",
            source,
        )
    };
    let raw = SharedShape::requested_bytes_for_rank(shape.len())
        .map_err(|source| error(source.into()))?;
    let retained = HostAllocationPlan::retained_exact_output_bytes(
        u64::try_from(raw).map_err(|_| error(HostAllocationError::Overflow))?,
        1,
    )
    .map_err(error)?;
    let receipt = HostAllocationReceipt::admit(
        Some(runtime),
        HostAllocationPlan::new(0, retained).map_err(error)?,
    )
    .map_err(error)?;
    let shape = match &receipt {
        Some(receipt) => {
            let permit = receipt.output_permit(runtime).map_err(error)?;
            let shape = SharedShape::new_admitted_from_slice(shape, &permit)
                .map_err(|source| error(source.into()))?;
            receipt.checkpoint(runtime).map_err(error)?;
            shape
        }
        None => SharedShape::new_unmanaged(shape.to_vec()),
    };
    let result = values.with_owned_output_storage(|storage| {
        let cell = if receipt.is_some() {
            CellArray::from_admitted(storage, shape)
        } else {
            CellArray::from_shared_storage(storage, shape)
        }
        .map_err(|source| crate::runtime_error::semantic_error("CellConstruction", source))?;
        RootedValue::in_context(Value::Cell(cell), runtime)
    })?;
    if let Some(receipt) = &receipt {
        receipt.checkpoint(runtime).map_err(error)?;
    }
    Ok(result)
}

/// The caller supplies a stable repeatable view of its original rooted values.
/// Concatenated segments fill the one final backing without a flattening Vec.
pub(crate) fn from_borrowed_values<'a>(
    runtime: Option<&RuntimeContext>,
    shape: &[usize],
    mut values: impl Iterator<Item = &'a Value> + Clone,
) -> Result<RootedValue, RuntimeError> {
    let count = values
        .clone()
        .try_fold(0usize, |count, _| count.checked_add(1))
        .ok_or_else(|| {
            crate::runtime_error::semantic_error("CellConstruction", "Cell payload length overflow")
        })?;
    if shape
        .iter()
        .try_fold(1usize, |count, dimension| count.checked_mul(*dimension))
        != Some(count)
    {
        return Err(crate::runtime_error::semantic_error(
            "CellConstruction",
            "Cell payload length does not match its shape",
        ));
    }
    let error = |source| {
        host_allocation_runtime_error(
            runtime,
            "RunMat:CellConstruction",
            "Cell destination admission failed",
            source,
        )
    };
    let owner = HostAllocationReceipt::capture_owner(runtime).map_err(error)?;
    let source_roots = match runtime {
        Some(runtime) => crate::execution::RootedValueLease::for_values_iterator_in_context(
            values.clone(),
            runtime,
        )?,
        None => {
            if HostAllocationReceipt::policy(None).map_err(error)?
                != HostAllocationPolicy::Unbounded
            {
                return Err(error(HostAllocationError::MissingManagedOwner));
            }
            crate::execution::RootedValueLease::for_values(values.clone())
                .map_err(|source| source.into_runtime_error())?
        }
    };
    super::check_owner(runtime, &owner).map_err(error)?;
    let result = if owner.is_some() {
        let runtime = runtime.expect("managed Cell owner has its context");
        let raw = SharedShape::requested_bytes_for_rank(shape.len())
            .map_err(|source| error(source.into()))?;
        let retained = HostAllocationPlan::retained_exact_output_bytes(
            u64::try_from(raw).map_err(|_| error(HostAllocationError::Overflow))?,
            1,
        )
        .map_err(error)?;
        let receipt = HostAllocationReceipt::admit(
            Some(runtime),
            HostAllocationPlan::new(0, retained).map_err(error)?,
        )
        .map_err(error)?
        .ok_or_else(|| error(HostAllocationError::StaleAuthority))?;
        let permit = receipt.output_permit(runtime).map_err(error)?;
        let shape = SharedShape::new_admitted_from_slice(shape, &permit)
            .map_err(|source| error(source.into()))?;
        receipt.checkpoint(runtime).map_err(error)?;
        let slots = RootedValues::try_initialize_in_context(count, runtime, |_| {
            values.next().cloned().ok_or_else(|| {
                crate::runtime_error::semantic_error(
                    "CellConstruction",
                    "Cell source iterator changed",
                )
            })
        })?;
        slots.with_owned_output_storage(|slots| {
            let cell = CellArray::from_admitted(slots, shape).map_err(|source| {
                crate::runtime_error::semantic_error("CellConstruction", source)
            })?;
            RootedValue::in_context(Value::Cell(cell), runtime)
        })?
    } else {
        let cell = CellArray::new_with_shape(values.cloned().collect(), shape.to_vec())
            .map_err(|source| crate::runtime_error::semantic_error("CellConstruction", source))?;
        super::replacement::root(Value::Cell(cell), runtime)?
    };
    super::check_owner(runtime, &owner).map_err(error)?;
    drop(source_roots);
    Ok(result)
}
