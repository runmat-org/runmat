use std::collections::BTreeMap;

use crate::operator::CsrMatrix;

pub(super) fn empty_rows(dof_count: usize) -> Vec<BTreeMap<usize, f64>> {
    (0..dof_count).map(|_| BTreeMap::new()).collect()
}

pub(super) fn scatter_csr<const N: usize>(
    rows: &mut [BTreeMap<usize, f64>],
    dof_offsets: &[usize],
    element_stiffness: &[[f64; N]; N],
) {
    for (local_row_node, row_offset) in dof_offsets.iter().copied().enumerate() {
        for local_row_axis in 0..3 {
            let local_row = local_row_node * 3 + local_row_axis;
            let global_row = row_offset + local_row_axis;
            for (local_column_node, column_offset) in dof_offsets.iter().copied().enumerate() {
                for local_column_axis in 0..3 {
                    let local_column = local_column_node * 3 + local_column_axis;
                    let global_column = column_offset + local_column_axis;
                    *rows[global_row].entry(global_column).or_insert(0.0) +=
                        element_stiffness[local_row][local_column];
                }
            }
        }
    }
}

pub(super) fn rows_to_csr(rows: Vec<BTreeMap<usize, f64>>) -> CsrMatrix {
    let mut row_offsets = Vec::with_capacity(rows.len() + 1);
    let mut column_indices = Vec::new();
    let mut values = Vec::new();
    row_offsets.push(0);
    for row in rows {
        for (column, value) in row {
            if value.abs() > 0.0 {
                column_indices.push(column);
                values.push(value);
            }
        }
        row_offsets.push(values.len());
    }
    CsrMatrix {
        row_offsets,
        column_indices,
        values,
    }
}
