use anyhow::{anyhow, ensure, Result};
use runmat_accelerate_api::{SortComparison, SortOrder, SortRowsColumnSpec};
use std::cmp::Ordering;

pub struct SortRowsHostOutputs {
    pub values: Vec<f64>,
    pub indices: Vec<f64>,
    pub indices_shape: Vec<usize>,
}

pub fn sort_rows_host(
    data: &[f64],
    shape: &[usize],
    columns: &[SortRowsColumnSpec],
    comparison: SortComparison,
) -> Result<SortRowsHostOutputs> {
    ensure!(
        is_matrix_shape(shape),
        "sortrows: input must be a 2-D matrix on the provider path"
    );

    let (rows, cols) = rows_cols_for_shape(shape);
    let expected = rows
        .checked_mul(cols)
        .ok_or_else(|| anyhow!("sortrows: dimension product exceeds limits"))?;
    ensure!(
        expected == data.len(),
        "sortrows: tensor data length {} does not match shape {:?}",
        data.len(),
        shape
    );

    if rows <= 1 || cols == 0 || data.is_empty() || columns.is_empty() {
        return Ok(SortRowsHostOutputs {
            values: data.to_vec(),
            indices: identity_indices(rows),
            indices_shape: vec![rows, 1],
        });
    }

    let mut order: Vec<usize> = (0..rows).collect();
    order.sort_by(|&a, &b| compare_rows(data, rows, cols, columns, comparison, a, b));

    let mut sorted = vec![0.0f64; data.len()];
    for col in 0..cols {
        for (dest_row, &src_row) in order.iter().enumerate() {
            let src_idx = src_row + col * rows;
            let dst_idx = dest_row + col * rows;
            sorted[dst_idx] = data[src_idx];
        }
    }

    Ok(SortRowsHostOutputs {
        values: sorted,
        indices: order.into_iter().map(|idx| (idx + 1) as f64).collect(),
        indices_shape: vec![rows, 1],
    })
}

fn is_matrix_shape(shape: &[usize]) -> bool {
    if shape.len() <= 2 {
        return true;
    }
    shape.iter().skip(2).all(|&dim| dim == 1)
}

fn rows_cols_for_shape(shape: &[usize]) -> (usize, usize) {
    match shape.len() {
        0 => (1, 1),
        1 => (shape[0].max(1), 1),
        _ => (shape[0], shape[1]),
    }
}

fn compare_rows(
    data: &[f64],
    rows: usize,
    cols: usize,
    columns: &[SortRowsColumnSpec],
    comparison: SortComparison,
    a: usize,
    b: usize,
) -> Ordering {
    for spec in columns {
        if spec.index >= cols {
            continue;
        }
        let idx_a = a + spec.index * rows;
        let idx_b = b + spec.index * rows;
        let va = data[idx_a];
        let vb = data[idx_b];
        let ord = compare_scalar(va, vb, spec.order, comparison);
        if ord != Ordering::Equal {
            return ord;
        }
    }
    Ordering::Equal
}

fn compare_scalar(a: f64, b: f64, order: SortOrder, comparison: SortComparison) -> Ordering {
    match (a.is_nan(), b.is_nan()) {
        (true, true) => Ordering::Equal,
        (true, false) => match order {
            SortOrder::Ascend => Ordering::Greater,
            SortOrder::Descend => Ordering::Less,
        },
        (false, true) => match order {
            SortOrder::Ascend => Ordering::Less,
            SortOrder::Descend => Ordering::Greater,
        },
        (false, false) => compare_finite(a, b, order, comparison),
    }
}

fn compare_finite(a: f64, b: f64, order: SortOrder, comparison: SortComparison) -> Ordering {
    if matches!(comparison, SortComparison::Abs) {
        let abs_cmp = a.abs().partial_cmp(&b.abs()).unwrap_or(Ordering::Equal);
        if abs_cmp != Ordering::Equal {
            return match order {
                SortOrder::Ascend => abs_cmp,
                SortOrder::Descend => abs_cmp.reverse(),
            };
        }
    }
    match order {
        SortOrder::Ascend => a.partial_cmp(&b).unwrap_or(Ordering::Equal),
        SortOrder::Descend => b.partial_cmp(&a).unwrap_or(Ordering::Equal),
    }
}

fn identity_indices(rows: usize) -> Vec<f64> {
    (0..rows).map(|idx| (idx + 1) as f64).collect()
}
