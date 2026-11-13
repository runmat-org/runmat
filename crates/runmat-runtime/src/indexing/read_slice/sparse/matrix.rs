use super::*;

pub(in crate::indexing::read_slice) fn matrix(
    context: Option<&RuntimeContext>,
    source: &SparseTensor,
    selectors: &[SliceSelector],
    policy: SparseReadZeroPolicy,
) -> Result<Value, RuntimeError> {
    let owner = Owner::new(context)?;
    let mut work = 0usize;
    let zeros = policy.zeros(Zeros::Drop);
    let row_selector = selectors.first().unwrap_or(&SliceSelector::Colon);
    let col_selector = selectors.get(1).unwrap_or(&SliceSelector::Colon);
    let all_rows = matches!(row_selector, SliceSelector::Colon);
    let rows = row_selector.iter_indices(source.rows);
    let cols = col_selector.iter_indices(source.cols);
    if !all_rows {
        for row in rows.clone() {
            output::step(&owner, &mut work)?;
            if row == 0 || row > source.rows {
                return Err(bounds());
            }
        }
    }
    for col in cols.clone() {
        output::step(&owner, &mut work)?;
        if col == 0 || col > source.cols {
            return Err(bounds());
        }
    }
    owner.check()?;
    let (out_rows, out_cols) = (rows.len(), cols.len());
    if out_rows == 0 || out_cols == 0 {
        return output::build(context, source, out_rows, out_cols, 0, |destination| {
            destination.pointer(0, 0)
        })
        .map(Value::SparseTensor);
    }
    if out_rows == 1 && out_cols == 1 {
        return selected(
            context,
            source,
            1,
            1,
            std::iter::once(
                rows.clone().next().expect("one row") - 1
                    + (cols.clone().next().expect("one column") - 1) * source.rows,
            ),
            policy.zeros(Zeros::ExactComponents),
        );
    }
    if all_rows {
        let mut nnz = 0usize;
        for column in cols.clone() {
            output::step(&owner, &mut work)?;
            for index in source.col_ptrs[column - 1]..source.col_ptrs[column] {
                output::step(&owner, &mut work)?;
                if keep(source, index, zeros) {
                    nnz = nnz.checked_add(1).ok_or_else(bounds)?;
                }
            }
        }
        owner.check()?;
        return output::build(context, source, out_rows, out_cols, nnz, |destination| {
            let mut offset = 0;
            destination.pointer(0, 0)?;
            for (out_col, column) in cols.enumerate() {
                output::step(&owner, &mut work)?;
                for index in source.col_ptrs[column - 1]..source.col_ptrs[column] {
                    output::step(&owner, &mut work)?;
                    if keep(source, index, zeros) {
                        destination.set(
                            offset,
                            source.row_indices[index],
                            output::element(source, index),
                        )?;
                        offset += 1;
                    }
                }
                destination.pointer(out_col + 1, offset)?;
            }
            owner.check()
        })
        .map(Value::SparseTensor);
    }
    let mut row_order =
        Scratch::<(usize, usize)>::new(context, if all_rows { 0 } else { out_rows })?;
    if !all_rows {
        for (ordinal, row) in rows.enumerate() {
            output::step(&owner, &mut work)?;
            row_order.as_mut_slice()[ordinal] = (row - 1, ordinal);
        }
        owner.check()?;
        row_order.as_mut_slice().sort_unstable();
        owner.check()?;
    }
    let mut pointers = Scratch::<usize>::new(context, out_cols.checked_add(1).ok_or_else(bounds)?)?;
    let mut nnz = 0usize;
    for (out_col, column) in cols.clone().enumerate() {
        output::step(&owner, &mut work)?;
        for index in source.col_ptrs[column - 1]..source.col_ptrs[column] {
            output::step(&owner, &mut work)?;
            if !keep(source, index, zeros) {
                continue;
            }
            let matches = if all_rows {
                1
            } else {
                let row = source.row_indices[index];
                let begin = row_order
                    .as_slice()
                    .partition_point(|&(base, _)| base < row);
                let end = row_order
                    .as_slice()
                    .partition_point(|&(base, _)| base <= row);
                end - begin
            };
            nnz = nnz.checked_add(matches).ok_or_else(bounds)?;
        }
        pointers.as_mut_slice()[out_col + 1] = nnz;
    }
    owner.check()?;
    // Only index/entry metadata is materialized, never source values. Sorting
    // this exact count preserves output-row order for repeated/unsorted rows.
    let mut selected_entries = Scratch::<(usize, usize)>::new(context, nnz)?;
    for (out_col, column) in cols.enumerate() {
        output::step(&owner, &mut work)?;
        let mut offset = pointers.as_slice()[out_col];
        for index in source.col_ptrs[column - 1]..source.col_ptrs[column] {
            output::step(&owner, &mut work)?;
            if !keep(source, index, zeros) {
                continue;
            }
            let row = source.row_indices[index];
            if all_rows {
                selected_entries.as_mut_slice()[offset] = (row, index);
                offset += 1;
            } else {
                let begin = row_order
                    .as_slice()
                    .partition_point(|&(base, _)| base < row);
                let end = row_order
                    .as_slice()
                    .partition_point(|&(base, _)| base <= row);
                for &(_, out_row) in &row_order.as_slice()[begin..end] {
                    output::step(&owner, &mut work)?;
                    selected_entries.as_mut_slice()[offset] = (out_row, index);
                    offset += 1;
                }
            }
        }
        owner.check()?;
        selected_entries.as_mut_slice()
            [pointers.as_slice()[out_col]..pointers.as_slice()[out_col + 1]]
            .sort_unstable();
        owner.check()?;
    }
    owner.check()?;
    let result = output::build(context, source, out_rows, out_cols, nnz, |destination| {
        for column in 0..=out_cols {
            output::step(&owner, &mut work)?;
            destination.pointer(column, pointers.as_slice()[column])?;
        }
        for (offset, &(row, index)) in selected_entries.as_slice().iter().enumerate() {
            output::step(&owner, &mut work)?;
            destination.set(offset, row, output::element(source, index))?;
        }
        owner.check()
    })?;
    owner.check()?;
    Ok(Value::SparseTensor(result))
}
