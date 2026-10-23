//! Borrowed RHS, exact update ordering and the shared canonical CSC merge.
use super::*;
use crate::context::RuntimeContext;
use crate::indexing::{
    plan::admission::{Owner, Scratch},
    read_slice::sparse::output,
};
use output::Element;
use runmat_value::{ComplexElement, SparseCscMergeEvent};
pub(in crate::indexing) mod delete;
pub(super) use delete::delete;

fn dimensions(source: &SparseTensor, plan: &IndexPlan) -> Result<(usize, usize), RuntimeError> {
    let rows = plan.base_shape.first().copied().unwrap_or(source.rows);
    let cols = plan.base_shape.get(1).copied().unwrap_or(source.cols);
    if rows < source.rows || cols < source.cols {
        return Err(map_slice_shape_error(
            "sparse slice expansion",
            format!(
                "SparseTensor cannot shrink shape ({}, {}) to ({rows}, {cols})",
                source.rows, source.cols
            ),
        ));
    }
    cols.checked_add(1).ok_or_else(|| {
        map_slice_shape_error(
            "sparse slice expansion",
            "SparseTensor expanded column count overflow",
        )
    })?;
    Ok((rows, cols))
}

fn rhs_element(source: &SparseTensor, view: &host_rhs::HostRhs<'_>, ordinal: usize) -> Element {
    if source.is_logical() {
        Element::Logical(view.real_at(ordinal) != 0.0)
    } else if let Some(storage) = source.integer_storage() {
        Element::Numeric(
            integer_assignment::scalar_value(storage, &view.integer_at(ordinal)).into(),
        )
    } else if source.is_complex() {
        let (real, imaginary) = view.complex_at(ordinal);
        if source.numeric_dtype() == Some(NumericDType::F32) {
            Element::Complex32(ComplexElement(real as f32, imaginary as f32))
        } else {
            Element::Complex64(ComplexElement(real, imaginary))
        }
    } else if source.numeric_dtype() == Some(NumericDType::F32) {
        let value = match view.numeric_at(ordinal) {
            NumericScalar::F32(value) => value,
            other => other.materialize_f32(),
        };
        Element::Numeric(NumericScalar::F32(value))
    } else {
        Element::Numeric(NumericScalar::F64(view.real_at(ordinal)))
    }
}

pub(super) async fn assign(
    context: Option<&RuntimeContext>,
    source: SparseTensor,
    plan: &IndexPlan,
    rhs: &Value,
) -> Result<Value, RuntimeError> {
    let owner = Owner::new(context)?;
    preflight_sparse_assignment(&source, plan, rhs)?;
    dimensions(&source, plan)?;
    if matches!(rhs, Value::GpuTensor(_)) && !plan.indices.is_empty() && !source.is_complex() {
        let host = owner
            .await_stage(gather_assignment_rhs_in_context(
                context,
                rhs,
                source.integer_storage().is_some(),
            ))
            .await??;
        assign_host(context, source, plan, &host)
    } else {
        assign_host(context, source, plan, rhs)
    }
}

pub(super) fn assign_host(
    context: Option<&RuntimeContext>,
    source: SparseTensor,
    plan: &IndexPlan,
    rhs: &Value,
) -> Result<Value, RuntimeError> {
    let owner = Owner::new(context)?;
    let mut work = 0usize;
    preflight_sparse_assignment(&source, plan, rhs)?;
    let (rows, cols) = dimensions(&source, plan)?;
    if plan.indices.is_empty() {
        return grow(context, source, rows, cols).map(Value::SparseTensor);
    }
    // Preserve the sparse complex family's existing multidimensional RHS
    // interpretation and rejection of logical RHS, even for linear plans.
    let view = if source.is_complex() {
        host_rhs::HostRhs::complex(rhs, &plan.selection_lengths, false)?
    } else {
        host_rhs::HostRhs::for_plan(rhs, plan)?
    };
    if view.len() != plan.indices.len() {
        return Err(mex("ShapeMismatch", "shape mismatch for slice assign"));
    }
    let mut order = Scratch::<(usize, usize)>::new(context, plan.indices.len())?;
    for (ordinal, &index) in plan.indices.iter().enumerate() {
        output::step(&owner, &mut work)?;
        order.as_mut_slice()[ordinal] = (index as usize, ordinal);
    }
    owner.check()?;
    order.as_mut_slice().sort_unstable();
    owner.check()?;
    merge(
        context,
        source,
        rows,
        cols,
        order.as_slice(),
        |source, ordinal| rhs_element(source, &view, ordinal),
    )
}

pub(in crate::indexing) fn merge(
    context: Option<&RuntimeContext>,
    source: SparseTensor,
    rows: usize,
    cols: usize,
    order: &[(usize, usize)],
    updated: impl Fn(&SparseTensor, usize) -> Element,
) -> Result<Value, RuntimeError> {
    let owner = Owner::new(context)?;
    let mut work = 0usize;
    let mut nnz = 0usize;
    source.visit_merged_linear_updates(
        rows,
        cols,
        order,
        |event| {
            let retained = match event {
                SparseCscMergeEvent::Stored { .. } => true,
                SparseCscMergeEvent::Update { ordinal, .. } => !updated(&source, ordinal).is_zero(),
                SparseCscMergeEvent::ColumnEnd { .. } => false,
            };
            if retained {
                nnz = nnz
                    .checked_add(1)
                    .ok_or_else(|| mex("ShapeMismatch", "SparseTensor assignment nnz overflow"))?;
            }
            Ok(())
        },
        |detail| map_slice_shape_error("sparse slice assign", detail),
        || output::step(&owner, &mut work),
    )?;
    owner.check()?;
    let output = output::build(context, &source, rows, cols, nnz, |destination| {
        destination.pointer(0, 0)?;
        let mut written = 0;
        source.visit_merged_linear_updates(
            rows,
            cols,
            order,
            |event| {
                let selected = match event {
                    SparseCscMergeEvent::Stored { row, entry } => {
                        Some((row, output::element(&source, entry)))
                    }
                    SparseCscMergeEvent::Update { row, ordinal } => {
                        let value = updated(&source, ordinal);
                        (!value.is_zero()).then_some((row, value))
                    }
                    SparseCscMergeEvent::ColumnEnd { column } => {
                        destination.pointer(column + 1, written)?;
                        None
                    }
                };
                if let Some((row, value)) = selected {
                    destination.set(written, row, value)?;
                    written += 1;
                }
                Ok(())
            },
            |detail| map_slice_shape_error("sparse slice assign", detail),
            || output::step(&owner, &mut work),
        )?;
        owner.check()
    })?;
    owner.check()?;
    Ok(Value::SparseTensor(output))
}

fn grow(
    context: Option<&RuntimeContext>,
    source: SparseTensor,
    rows: usize,
    cols: usize,
) -> Result<SparseTensor, RuntimeError> {
    if rows == source.rows && cols == source.cols {
        Owner::new(context)?.check()?;
        return Ok(source);
    }
    // Growth with no updates shares original row/value owners; only the
    // column pointer backing actually changes. No full source payload copy.
    let count = cols.checked_add(1).ok_or_else(|| {
        output::allocation(
            context,
            crate::context::host_allocation::HostAllocationError::Overflow,
        )
    })?;
    let pointers = output::indices(context, count, |column| {
        Ok(source.col_ptrs.get(column).copied().unwrap_or(source.nnz()))
    })?;
    source
        .with_expanded_column_pointers(rows, cols, pointers)
        .map_err(|detail| map_slice_shape_error("sparse slice expansion", detail))
}
