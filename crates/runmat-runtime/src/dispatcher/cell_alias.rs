//! Exact replacement owner for a shared Cell whose descendants need an owned
//! transformation. Every new collection, shape and borrowed UTF-8 copy is
//! admitted before a GPU effect; existing numeric and resident owners alias
//! their original physical backing in O(1).

use runmat_value::{CellArray, OutputListStorage, SharedShape, Value};

use crate::context::host_allocation::{
    host_allocation_runtime_error, HostAllocationError, HostAllocationPlan, HostAllocationReceipt,
};
use crate::context::RuntimeContext;
use crate::{build_runtime_error, RuntimeError};

#[derive(Default)]
struct AliasQuote {
    physical: u64,
    claims: usize,
}

impl AliasQuote {
    fn add(&mut self, physical: usize, claims: usize) -> Result<(), RuntimeError> {
        self.physical = self
            .physical
            .checked_add(u64::try_from(physical).map_err(|_| overflow())?)
            .ok_or_else(overflow)?;
        self.claims = self.claims.checked_add(claims).ok_or_else(overflow)?;
        Ok(())
    }
}

fn overflow() -> RuntimeError {
    build_runtime_error("gather: shared Cell replacement extent overflows")
        .with_identifier("RunMat:Placement:ResourceOverflow")
        .build()
}

fn unsupported() -> RuntimeError {
    build_runtime_error("gather: shared Cell child has no admitted alias representation")
        .with_identifier("RunMat:gather:UnsupportedSharedChild")
        .build()
}

fn admission(context: &RuntimeContext, source: HostAllocationError) -> RuntimeError {
    host_allocation_runtime_error(
        Some(context),
        "RunMat:gather:SharedCell",
        "shared Cell replacement admission failed",
        source,
    )
}

fn quote_cell(
    cell: &CellArray,
    quote: &mut AliasQuote,
    context: &RuntimeContext,
    visited: &mut usize,
) -> Result<(), RuntimeError> {
    *visited = visited.checked_add(1).ok_or_else(overflow)?;
    if (*visited).is_multiple_of(4096) && context.cancellation_context().is_cancelled() {
        return Err(admission(context, HostAllocationError::Cancelled));
    }
    quote.add(
        SharedShape::requested_bytes_for_rank(cell.shape.len()).map_err(|_| overflow())?,
        1,
    )?;
    quote_values(cell.data_owner(), quote, context, visited)
}

fn quote_values(
    values: &OutputListStorage,
    quote: &mut AliasQuote,
    context: &RuntimeContext,
    visited: &mut usize,
) -> Result<(), RuntimeError> {
    quote.add(
        OutputListStorage::requested_bytes(values.len()).map_err(|_| overflow())?,
        usize::from(!values.is_empty()),
    )?;
    for child in values.iter() {
        *visited = visited.checked_add(1).ok_or_else(overflow)?;
        if (*visited).is_multiple_of(4096) && context.cancellation_context().is_cancelled() {
            return Err(admission(context, HostAllocationError::Cancelled));
        }
        match child {
            Value::String(text)
                if !text.has_retained_claim() && text.as_str().is_some_and(|s| !s.is_empty()) =>
            {
                quote.add(
                    runmat_value::StringElement::requested_bytes_for_capacity(text.capacity())
                        .map_err(|_| overflow())?,
                    1,
                )?;
            }
            Value::Cell(nested) => quote_cell(nested, quote, context, visited)?,
            _ if supported_shallow_child(child) => {}
            _ => return Err(unsupported()),
        }
    }
    Ok(())
}

fn supported_shallow_child(value: &Value) -> bool {
    match value {
        Value::Int(_)
        | Value::Num(_)
        | Value::Complex(_, _)
        | Value::Bool(_)
        | Value::Tensor(_)
        | Value::SparseTensor(_)
        | Value::ComplexTensor(_)
        | Value::LogicalArray(_)
        | Value::GpuTensor(_) => true,
        Value::String(_) => true,
        Value::CharArray(array) => array.data_owner().is_shallow_clone(),
        Value::StringArray(array) => array.data_owner().is_shallow_clone(),
        Value::Cell(_) => true,
        _ => false,
    }
}

fn prepare_child(
    child: &Value,
    permit: &dyn runmat_value::ExactHostAllocationPermit,
    context: &RuntimeContext,
    receipt: &HostAllocationReceipt,
    visited: &mut usize,
) -> Result<Value, RuntimeError> {
    match child {
        Value::Cell(cell) => prepare_cell(cell, permit, context, receipt, visited).map(Value::Cell),
        Value::String(text)
            if !text.has_retained_claim() && text.as_str().is_some_and(|s| !s.is_empty()) =>
        {
            text.try_copy_exact_with(permit, |_| {
                receipt
                    .checkpoint(context)
                    .map_err(|source| admission(context, source))
            })
            .map(|text| Value::String(text.into()))
            .map_err(|error| match error {
                runmat_value::ExactArrayInitializationError::Allocation(error) => {
                    build_runtime_error(format!("gather: shared Cell text admission: {error}"))
                        .with_identifier("RunMat:Placement:ResourceLimit")
                        .with_source(error)
                        .build()
                }
                runmat_value::ExactArrayInitializationError::Element(error) => error,
            })
        }
        Value::String(text) if !text.has_retained_claim() => Ok(Value::String(text.clone())),
        _ if supported_shallow_child(child) => Ok(child.clone()),
        _ => Err(unsupported()),
    }
}

fn prepare_cell(
    cell: &CellArray,
    permit: &dyn runmat_value::ExactHostAllocationPermit,
    context: &RuntimeContext,
    receipt: &HostAllocationReceipt,
    visited: &mut usize,
) -> Result<CellArray, RuntimeError> {
    *visited = visited.checked_add(1).ok_or_else(overflow)?;
    if (*visited).is_multiple_of(4096) {
        receipt
            .checkpoint(context)
            .map_err(|source| admission(context, source))?;
    }
    let shape = SharedShape::new_admitted_from_slice(cell.shape(), permit).map_err(|error| {
        build_runtime_error(format!("gather: shared Cell shape admission: {error}"))
            .with_identifier("RunMat:Placement:ResourceLimit")
            .with_source(error)
            .build()
    })?;
    let data = prepare_values(cell.data_owner(), permit, context, receipt, visited)?;
    CellArray::from_admitted(data, shape).map_err(|reason| {
        build_runtime_error(format!("gather: shared Cell geometry: {reason}"))
            .with_identifier("RunMat:gather:CellShapeError")
            .build()
    })
}

fn prepare_values(
    values: &OutputListStorage,
    permit: &dyn runmat_value::ExactHostAllocationPermit,
    context: &RuntimeContext,
    receipt: &HostAllocationReceipt,
    visited: &mut usize,
) -> Result<OutputListStorage, RuntimeError> {
    OutputListStorage::try_initialize(values.len(), permit, |index| {
        *visited = visited.checked_add(1).ok_or_else(overflow)?;
        if (*visited).is_multiple_of(4096) {
            receipt
                .checkpoint(context)
                .map_err(|source| admission(context, source))?;
        }
        prepare_child(&values[index], permit, context, receipt, visited)
    })
    .map_err(|error| match error {
        runmat_value::ExactArrayInitializationError::Allocation(error) => {
            build_runtime_error(format!("gather: shared Cell list admission: {error}"))
                .with_identifier("RunMat:Placement:ResourceLimit")
                .with_source(error)
                .build()
        }
        runmat_value::ExactArrayInitializationError::Element(error) => error,
    })
}

/// The receipt stays with the caller through its subsequent async download;
/// each final child/collection claim survives it independently.
pub(super) fn prepare_shared_cell(
    cell: &CellArray,
    context: &RuntimeContext,
) -> Result<(CellArray, HostAllocationReceipt), RuntimeError> {
    let mut quote = AliasQuote::default();
    let mut visited = 0;
    quote_cell(cell, &mut quote, context, &mut visited)?;
    if context.cancellation_context().is_cancelled() {
        return Err(admission(context, HostAllocationError::Cancelled));
    }
    let retained = HostAllocationPlan::retained_exact_output_bytes(quote.physical, quote.claims)
        .map_err(|source| admission(context, source))?;
    let plan = HostAllocationPlan::new(0, retained).map_err(|source| admission(context, source))?;
    let receipt = HostAllocationReceipt::admit(Some(context), plan)
        .map_err(|source| admission(context, source))?
        .ok_or_else(|| admission(context, HostAllocationError::MissingManagedOwner))?;
    let permit = receipt
        .output_permit(context)
        .map_err(|source| admission(context, source))?;
    let mut visited = 0;
    let prepared = prepare_cell(cell, &permit, context, &receipt, &mut visited);
    receipt
        .checkpoint(context)
        .map_err(|source| admission(context, source))?;
    Ok((prepared?, receipt))
}

/// Prepare a shared closure capture or output list without a fictitious Cell
/// shape; children use the same exact alias rules as shared Cell elements.
pub(super) fn prepare_shared_values(
    values: &OutputListStorage,
    context: &RuntimeContext,
) -> Result<(OutputListStorage, HostAllocationReceipt), RuntimeError> {
    let mut quote = AliasQuote::default();
    let mut visited = 0;
    quote_values(values, &mut quote, context, &mut visited)?;
    if context.cancellation_context().is_cancelled() {
        return Err(admission(context, HostAllocationError::Cancelled));
    }
    let retained = HostAllocationPlan::retained_exact_output_bytes(quote.physical, quote.claims)
        .map_err(|source| admission(context, source))?;
    let plan = HostAllocationPlan::new(0, retained).map_err(|source| admission(context, source))?;
    let receipt = HostAllocationReceipt::admit(Some(context), plan)
        .map_err(|source| admission(context, source))?
        .ok_or_else(|| admission(context, HostAllocationError::MissingManagedOwner))?;
    let permit = receipt
        .output_permit(context)
        .map_err(|source| admission(context, source))?;
    let mut visited = 0;
    let prepared = prepare_values(values, &permit, context, &receipt, &mut visited);
    receipt
        .checkpoint(context)
        .map_err(|source| admission(context, source))?;
    Ok((prepared?, receipt))
}
