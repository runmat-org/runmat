//! Same sparse deletion-axis policy, admitted metadata and canonical rebuild.
use super::*;

#[derive(Clone, Copy)]
pub(in crate::indexing) enum Axis {
    Rows,
    Columns,
    All,
}

pub(in crate::indexing::write_slice) fn delete(
    context: Option<&RuntimeContext>,
    source: SparseTensor,
    plan: &IndexPlan,
    rhs: &Value,
) -> Result<Value, RuntimeError> {
    let owner = Owner::new(context)?;
    let mut work = 0usize;
    if source.integer_storage().is_some() {
        crate::compatibility::ensure_sparse_integer_extension_enabled("indexed assignment")?;
    }
    if !is_empty_delete_rhs(rhs) {
        return Err(mex(
            "DeletionRequiresEmptyRhs",
            "Indexed deletion requires empty RHS",
        ));
    }
    if plan.indices.is_empty() {
        owner.check()?;
        return Ok(Value::SparseTensor(source));
    }
    let selected_rows = plan.selection_lengths.first().copied().unwrap_or(0);
    let selected_cols = plan.selection_lengths.get(1).copied().unwrap_or(0);
    let (axis, count) = if plan.dims == 1 {
        if source.rows == 1 {
            (Axis::Columns, plan.indices.len())
        } else if source.cols == 1 {
            (Axis::Rows, plan.indices.len())
        } else {
            return Err(mex(
                "UnsupportedDeletion",
                "Linear sparse deletion is only supported for vectors",
            ));
        }
    } else if plan.dims != 2 {
        return Err(mex(
            "UnsupportedDeletion",
            "Sparse deletion currently supports vectors and complete matrix rows or columns",
        ));
    } else if selected_rows == source.rows && selected_cols == source.cols {
        (Axis::All, 0)
    } else if selected_rows == source.rows {
        (Axis::Columns, selected_cols)
    } else if selected_cols == source.cols {
        (Axis::Rows, selected_rows)
    } else {
        return Err(mex(
            "UnsupportedDeletion",
            "Sparse deletion requires selecting complete rows or columns",
        ));
    };
    if matches!(axis, Axis::All) {
        return output::build(context, &source, 0, 0, 0, |destination| {
            destination.pointer(0, 0)
        })
        .map(Value::SparseTensor);
    }
    let mut removed = Scratch::<usize>::new(context, count)?;
    for (ordinal, slot) in removed.as_mut_slice().iter_mut().enumerate() {
        output::step(&owner, &mut work)?;
        *slot = if plan.dims == 1 {
            plan.indices[ordinal] as usize
        } else if matches!(axis, Axis::Rows) {
            plan.indices[ordinal] as usize % source.rows
        } else {
            plan.indices[ordinal * source.rows] as usize / source.rows
        };
    }
    owner.check()?;
    removed.as_mut_slice().sort_unstable();
    owner.check()?;
    rebuild(context, source, axis, removed.as_slice())
}

pub(in crate::indexing) fn rebuild(
    context: Option<&RuntimeContext>,
    source: SparseTensor,
    axis: Axis,
    removed: &[usize],
) -> Result<Value, RuntimeError> {
    let owner = Owner::new(context)?;
    let mut work = 0usize;
    let (bound, label) = if matches!(axis, Axis::Rows) {
        (source.rows, "row")
    } else {
        (source.cols, "column")
    };
    if removed.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(map_slice_shape_error(
            "sparse deletion",
            format!("SparseTensor {label} deletion indices must be unique"),
        ));
    }
    if removed.iter().any(|&index| index >= bound) {
        return Err(map_slice_shape_error(
            "sparse deletion",
            format!("SparseTensor {label} deletion index exceeds dimension"),
        ));
    }
    let (rows, cols) = if matches!(axis, Axis::Rows) {
        (source.rows - removed.len(), source.cols)
    } else {
        (source.rows, source.cols - removed.len())
    };
    let source_columns = || {
        (0..source.cols)
            .filter(|column| matches!(axis, Axis::Rows) || removed.binary_search(column).is_err())
    };
    let map_row = |row| {
        if matches!(axis, Axis::Columns) {
            Some(row)
        } else {
            match removed.binary_search(&row) {
                Ok(_) => None,
                Err(before) => Some(row - before),
            }
        }
    };
    let mut nnz = 0usize;
    source.visit_rebuilt_csc(
        source_columns(),
        map_row,
        |event| {
            if matches!(event, SparseCscMergeEvent::Stored { .. }) {
                nnz = nnz
                    .checked_add(1)
                    .ok_or_else(|| mex("ShapeMismatch", "SparseTensor deletion nnz overflow"))?;
            }
            Ok(())
        },
        || output::step(&owner, &mut work),
        |detail| map_slice_shape_error("sparse deletion", detail),
    )?;
    owner.check()?;
    let result = output::build(context, &source, rows, cols, nnz, |destination| {
        let mut written = 0;
        destination.pointer(0, 0)?;
        source.visit_rebuilt_csc(
            source_columns(),
            map_row,
            |event| {
                match event {
                    SparseCscMergeEvent::Stored { row, entry } => {
                        destination.set(written, row, output::element(&source, entry))?;
                        written += 1;
                    }
                    SparseCscMergeEvent::ColumnEnd { column } => {
                        destination.pointer(column + 1, written)?
                    }
                    SparseCscMergeEvent::Update { .. } => {
                        unreachable!("rebuild has no update input")
                    }
                }
                Ok(())
            },
            || output::step(&owner, &mut work),
            |detail| map_slice_shape_error("sparse deletion", detail),
        )?;
        owner.check()
    })?;
    owner.check()?;
    Ok(Value::SparseTensor(result))
}
