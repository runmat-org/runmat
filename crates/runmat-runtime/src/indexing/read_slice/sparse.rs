//! Canonical sparse reads over borrowed selectors and actual CSC final owners.
use super::*;
use crate::context::RuntimeContext;
use crate::indexing::plan::admission::{Owner, Scratch};
pub(in crate::indexing) mod output;

/// Physical stored-zero policy, independent of selector evaluation or class.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SparseReadZeroPolicy {
    /// Existing slice semantics: colon-linear preserves all stored entries;
    /// singleton selection preserves typed/complex components; other selections
    /// elide numerical zeros.
    BySelector,
    /// Existing plan semantics: preserve stored integer/complex components,
    /// but elide ordinary real floating zeros, including for colon selection.
    PreserveTypedComponents,
}

impl SparseReadZeroPolicy {
    fn zeros(self, by_selector: Zeros) -> Zeros {
        match self {
            Self::BySelector => by_selector,
            Self::PreserveTypedComponents => Zeros::ExactComponents,
        }
    }
}

#[derive(Clone, Copy)]
enum Zeros {
    Drop,
    ExactComponents,
    AllStored,
}
fn keep(source: &SparseTensor, entry: usize, zeros: Zeros) -> bool {
    matches!(zeros, Zeros::AllStored)
        || (matches!(zeros, Zeros::ExactComponents)
            && (source.is_complex() || source.integer_storage().is_some()))
        || !output::element(source, entry).is_zero()
}
fn bounds() -> RuntimeError {
    crate::runtime_error::semantic_error("IndexOutOfBounds", "Index out of bounds")
}
fn entry(source: &SparseTensor, linear: usize) -> Option<usize> {
    let row = linear % source.rows;
    let column = linear / source.rows;
    let begin = source.col_ptrs[column];
    source.row_indices[begin..source.col_ptrs[column + 1]]
        .binary_search(&row)
        .ok()
        .map(|offset| begin + offset)
}
fn selected(
    context: Option<&RuntimeContext>,
    source: &SparseTensor,
    rows: usize,
    cols: usize,
    indices: impl ExactSizeIterator<Item = usize> + Clone,
    zeros: Zeros,
) -> Result<Value, RuntimeError> {
    let owner = Owner::new(context)?;
    let mut work = 0usize;
    let total = checked_sparse_numel(source)?;
    let count = rows.checked_mul(cols).ok_or_else(bounds)?;
    if indices.len() != count {
        return Err(crate::runtime_error::semantic_error(
            "ShapeMismatch",
            "sparse slice plan output shape does not match selected indices",
        ));
    }
    let mut nnz = 0usize;
    for linear in indices.clone() {
        output::step(&owner, &mut work)?;
        if linear >= total || source.rows == 0 {
            return Err(bounds());
        }
        if entry(source, linear).is_some_and(|index| keep(source, index, zeros)) {
            nnz += 1;
        }
    }
    owner.check()?;
    let result = output::build(context, source, rows, cols, nnz, |destination| {
        destination.pointer(0, 0)?;
        let mut written = 0;
        let mut indices = indices;
        for column in 0..cols {
            output::step(&owner, &mut work)?;
            for row in 0..rows {
                output::step(&owner, &mut work)?;
                let linear = indices.next().expect("validated exact sparse selection");
                if let Some(index) =
                    entry(source, linear).filter(|&index| keep(source, index, zeros))
                {
                    destination.set(written, row, output::element(source, index))?;
                    written += 1;
                }
            }
            destination.pointer(column + 1, written)?;
        }
        owner.check()
    })?;
    owner.check()?;
    Ok(Value::SparseTensor(result))
}

pub(super) fn from_plan(
    context: Option<&RuntimeContext>,
    source: &SparseTensor,
    plan: &IndexPlan,
) -> Result<Value, RuntimeError> {
    if source.integer_storage().is_some() {
        crate::compatibility::ensure_sparse_integer_extension_enabled("indexed access")?;
    }
    let (rows, cols) = if plan.indices.len() == 1 {
        (1, 1)
    } else {
        sparse_output_shape(plan)?
    };
    selected(
        context,
        source,
        rows,
        cols,
        plan.indices.iter().map(|&index| index as usize),
        Zeros::ExactComponents,
    )
}

pub(super) fn linear(
    context: Option<&RuntimeContext>,
    source: &SparseTensor,
    selector: &SliceSelector,
    policy: SparseReadZeroPolicy,
) -> Result<Value, RuntimeError> {
    let owner = Owner::new(context)?;
    let mut work = 0usize;
    let total = checked_sparse_numel(source)?;
    if matches!(selector, SliceSelector::Colon) {
        let zeros = policy.zeros(Zeros::AllStored);
        let nnz = if matches!(zeros, Zeros::AllStored)
            || (matches!(zeros, Zeros::ExactComponents)
                && (source.is_complex() || source.integer_storage().is_some()))
        {
            source.nnz()
        } else {
            let mut count = 0;
            for index in 0..source.nnz() {
                output::step(&owner, &mut work)?;
                if keep(source, index, zeros) {
                    count += 1;
                }
            }
            count
        };
        owner.check()?;
        let result = output::build(context, source, total, 1, nnz, |destination| {
            destination.pointer(0, 0)?;
            let mut written = 0;
            for column in 0..source.cols {
                output::step(&owner, &mut work)?;
                for index in source.col_ptrs[column]..source.col_ptrs[column + 1] {
                    output::step(&owner, &mut work)?;
                    if !keep(source, index, zeros) {
                        continue;
                    }
                    destination.set(
                        written,
                        source.row_indices[index] + column * source.rows,
                        output::element(source, index),
                    )?;
                    written += 1;
                }
            }
            destination.pointer(1, written)?;
            owner.check()
        })?;
        owner.check()?;
        return Ok(Value::SparseTensor(result));
    }
    let indices = selector.iter_indices(total);
    let shape: Option<&[usize]> = match selector {
        SliceSelector::LinearIndices { output_shape, .. } => Some(output_shape),
        _ => None,
    };
    let count = indices.len();
    for index in indices.clone() {
        output::step(&owner, &mut work)?;
        if index == 0 || index > total {
            return Err(bounds());
        }
    }
    owner.check()?;
    let (rows, cols) = if count == 1 {
        (1, 1)
    } else if let Some(shape) = shape {
        match shape {
            [rows, cols] => (*rows, *cols),
            [length] => (*length, 1),
            _ => {
                return Err(crate::runtime_error::semantic_error(
                    "UnsupportedSparseIndexRank",
                    "Sparse indexing currently supports two-dimensional outputs",
                ))
            }
        }
    } else if count == 0 {
        (0, 1)
    } else if source.rows == 1 && source.cols > 1 {
        (1, count)
    } else {
        (count, 1)
    };
    selected(
        context,
        source,
        rows,
        cols,
        indices.map(|index| index - 1),
        policy.zeros(if count == 1 {
            Zeros::ExactComponents
        } else {
            Zeros::Drop
        }),
    )
}

mod matrix;
pub(super) use matrix::matrix;
