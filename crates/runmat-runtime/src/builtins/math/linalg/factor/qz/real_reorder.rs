//! Real adjacent-block exchange for a retained generalized Schur form.
//!
//! Blocks have size at most two. A tiny generalized Sylvester system gives
//! the invariant subspace of the trailing block; orthogonal bases move that
//! subspace to the leading position. Local two-by-two QZ reductions restore
//! the real quasi-triangular/triangular diagonal-block convention.

use faer::Mat;

use super::algorithm::{real_qz, RealQz};
use super::reorder::ClusterOrder;

fn diagonal_blocks(form: &RealQz) -> Result<Vec<(usize, usize)>, &'static str> {
    let n = form.aa.nrows();
    let mut blocks = Vec::new();
    let mut i = 0;
    while i < n {
        let size = if i + 1 < n && form.aa[(i + 1, i)] != 0.0 {
            if i + 2 < n && form.aa[(i + 2, i + 1)] != 0.0 {
                return Err("overlapping real generalized Schur blocks");
            }
            2
        } else {
            1
        };
        blocks.push((i, size));
        i += size;
    }
    Ok(blocks)
}

fn small_solve(mut matrix: Vec<Vec<f64>>, mut rhs: Vec<f64>) -> Result<Vec<f64>, &'static str> {
    let n = rhs.len();
    let scale = matrix
        .iter()
        .flatten()
        .copied()
        .map(f64::abs)
        .fold(0.0_f64, f64::max);
    if scale == 0.0 {
        return Err("generalized Schur block exchange is singular");
    }
    for (row, value) in matrix.iter_mut().zip(rhs.iter_mut()) {
        for element in row {
            *element /= scale;
        }
        *value /= scale;
    }
    let threshold = 128.0 * f64::EPSILON;
    for col in 0..n {
        let pivot = (col..n)
            .max_by(|&a, &b| matrix[a][col].abs().total_cmp(&matrix[b][col].abs()))
            .ok_or("generalized Schur block exchange has no pivot")?;
        if matrix[pivot][col].abs() <= threshold {
            return Err("generalized Schur blocks are inseparable");
        }
        matrix.swap(col, pivot);
        rhs.swap(col, pivot);
        for row in (col + 1)..n {
            let factor = matrix[row][col] / matrix[col][col];
            for j in col..n {
                matrix[row][j] -= factor * matrix[col][j];
            }
            rhs[row] -= factor * rhs[col];
        }
    }
    let mut solution = vec![0.0; n];
    for row in (0..n).rev() {
        let correction = ((row + 1)..n)
            .map(|col| matrix[row][col] * solution[col])
            .sum::<f64>();
        solution[row] = (rhs[row] - correction) / matrix[row][row];
    }
    if solution.iter().any(|value| !value.is_finite()) {
        return Err("generalized Schur block exchange overflowed");
    }
    Ok(solution)
}

fn invariant_subspace(
    form: &RealQz,
    start: usize,
    p: usize,
    q: usize,
) -> Result<(Vec<f64>, Vec<f64>), &'static str> {
    let size = 2 * p * q;
    let mut system = vec![vec![0.0; size]; size];
    let mut rhs = vec![0.0; size];
    let local_scale = (0..(p + q))
        .flat_map(|j| (0..(p + q)).map(move |i| (i, j)))
        .fold(0.0_f64, |maximum, (i, j)| {
            maximum
                .max(form.aa[(start + i, start + j)].abs())
                .max(form.bb[(start + i, start + j)].abs())
        });
    if local_scale == 0.0 {
        return Err("zero generalized pencil has no ordered invariant subspace");
    }
    let coupling =
        (0..p)
            .flat_map(|i| (0..q).map(move |j| (i, j)))
            .fold(0.0_f64, |maximum, (i, j)| {
                maximum
                    .max(form.aa[(start + i, start + p + j)].abs())
                    .max(form.bb[(start + i, start + p + j)].abs())
            });
    if coupling == 0.0 {
        return Ok((vec![0.0; p * q], vec![0.0; p * q]));
    }
    let x = |i: usize, j: usize| i * q + j;
    let y = |i: usize, j: usize| p * q + i * q + j;
    for i in 0..p {
        for j in 0..q {
            for (equation, factor) in [&form.aa, &form.bb].into_iter().enumerate() {
                let row = equation * p * q + i * q + j;
                rhs[row] = -factor[(start + i, start + p + j)] / local_scale;
                for k in 0..p {
                    system[row][x(k, j)] += factor[(start + i, start + k)] / local_scale;
                }
                for k in 0..q {
                    system[row][y(i, k)] -= factor[(start + p + k, start + p + j)] / local_scale;
                }
            }
        }
    }
    if rhs.iter().all(|&value| value == 0.0) {
        return Err("generalized Schur coupling underflowed during block admission");
    }
    let solution = small_solve(system, rhs)?;
    Ok((solution[..p * q].to_vec(), solution[p * q..].to_vec()))
}

fn orthogonal_basis(top: &[f64], p: usize, q: usize) -> Result<Mat<f64>, &'static str> {
    let n = p + q;
    let mut basis = Mat::<f64>::zeros(n, n);
    for col in 0..q {
        let column_scale = (0..p)
            .map(|row| top[row * q + col].abs())
            .fold(1.0_f64, f64::max);
        for row in 0..p {
            basis[(row, col)] = top[row * q + col] / column_scale;
        }
        basis[(p + col, col)] = 1.0 / column_scale;
    }
    for col in q..n {
        basis[(col - q, col)] = 1.0;
    }
    for col in 0..n {
        // Two passes keep the tiny basis orthogonal even if X/Y are large.
        for _ in 0..2 {
            for earlier in 0..col {
                let projection = (0..n)
                    .map(|row| basis[(row, col)] * basis[(row, earlier)])
                    .sum::<f64>();
                for row in 0..n {
                    basis[(row, col)] -= projection * basis[(row, earlier)];
                }
            }
        }
        let norm = (0..n)
            .map(|row| basis[(row, col)] * basis[(row, col)])
            .sum::<f64>()
            .sqrt();
        if !norm.is_finite() || norm <= 128.0 * f64::EPSILON {
            return Err("generalized Schur invariant subspace is ill-conditioned");
        }
        for row in 0..n {
            basis[(row, col)] /= norm;
        }
    }
    Ok(basis)
}

fn rotated_entry(old: &[f64; 4], rotation: &Mat<f64>, size: usize, column: usize) -> f64 {
    let scale = old[..size]
        .iter()
        .copied()
        .map(f64::abs)
        .fold(0.0_f64, f64::max);
    if scale == 0.0 {
        return 0.0;
    }
    (0..size)
        .map(|i| (old[i] / scale) * rotation[(i, column)])
        .sum::<f64>()
        * scale
}

fn apply_local(form: &mut RealQz, start: usize, left: &Mat<f64>, right: &Mat<f64>) {
    let n = form.aa.nrows();
    let size = left.nrows();
    for matrix in [&mut form.aa, &mut form.bb] {
        for row in 0..n {
            let mut old = [0.0; 4];
            for j in 0..size {
                old[j] = matrix[(row, start + j)];
            }
            for col in 0..size {
                matrix[(row, start + col)] = rotated_entry(&old, right, size, col);
            }
        }
        for col in 0..n {
            let mut old = [0.0; 4];
            for i in 0..size {
                old[i] = matrix[(start + i, col)];
            }
            for row in 0..size {
                matrix[(start + row, col)] = rotated_entry(&old, left, size, row);
            }
        }
    }
    for (matrix, rotation) in [(&mut form.left, left), (&mut form.right, right)] {
        for row in 0..n {
            let mut old = [0.0; 4];
            for j in 0..size {
                old[j] = matrix[(row, start + j)];
            }
            for col in 0..size {
                matrix[(row, start + col)] = rotated_entry(&old, rotation, size, col);
            }
        }
    }
}

fn restore_block(form: &mut RealQz, start: usize, size: usize) -> Result<(), &'static str> {
    if size != 2 {
        return Ok(());
    }
    let a = Mat::from_fn(2, 2, |i, j| form.aa[(start + i, start + j)]);
    let b = Mat::from_fn(2, 2, |i, j| form.bb[(start + i, start + j)]);
    let reduced = real_qz(a, b)?;
    apply_local(form, start, &reduced.left, &reduced.right);
    Ok(())
}

fn exchange(form: &mut RealQz, start: usize, p: usize, q: usize) -> Result<(), &'static str> {
    let (x, y) = invariant_subspace(form, start, p, q)?;
    let right = orthogonal_basis(&x, p, q)?;
    let left = orthogonal_basis(&y, p, q)?;
    apply_local(form, start, &left, &right);
    let order = form.aa.nrows();
    if [&form.aa, &form.bb, &form.left, &form.right]
        .into_iter()
        .any(|matrix| (0..order).any(|j| (0..order).any(|i| !matrix[(i, j)].is_finite())))
    {
        return Err("generalized Schur block exchange produced nonfinite factors");
    }
    let scale = (0..(p + q))
        .flat_map(|j| (0..(p + q)).map(move |i| (i, j)))
        .fold(0.0_f64, |maximum, (i, j)| {
            maximum
                .max(form.aa[(start + i, start + j)].abs())
                .max(form.bb[(start + i, start + j)].abs())
        });
    let tolerance = 128.0 * f64::EPSILON * (form.aa.nrows() as f64).max(1.0) * scale;
    for row in q..(p + q) {
        for col in 0..q {
            if form.aa[(start + row, start + col)].abs() > tolerance
                || form.bb[(start + row, start + col)].abs() > tolerance
            {
                return Err("generalized Schur blocks are too ill-conditioned to exchange");
            }
            form.aa[(start + row, start + col)] = 0.0;
            form.bb[(start + row, start + col)] = 0.0;
        }
    }
    restore_block(form, start, q)?;
    restore_block(form, start + q, p)?;
    if [&form.aa, &form.bb, &form.left, &form.right]
        .into_iter()
        .any(|matrix| (0..order).any(|j| (0..order).any(|i| !matrix[(i, j)].is_finite())))
    {
        return Err("generalized Schur block restoration produced nonfinite factors");
    }
    Ok(())
}

pub(super) fn reorder_real_clusters(
    form: &mut RealQz,
    labels: &[u64],
    order: ClusterOrder,
) -> Result<(), &'static str> {
    let n = form.aa.nrows();
    if labels.len() != n {
        return Err("generalized Schur cluster length does not match pencil order");
    }
    let blocks = diagonal_blocks(form)?;
    let mut blocks = blocks
        .into_iter()
        .map(|(start, size)| (size, labels[start]))
        .collect::<Vec<_>>();
    let mut offset = 0;
    for destination in 0..blocks.len() {
        let mut best = destination;
        for candidate in (destination + 1)..blocks.len() {
            if order.precedes(blocks[candidate].1, blocks[best].1) {
                best = candidate;
            }
        }
        for position in (destination..best).rev() {
            let left_size = blocks[position].0;
            let right_size = blocks[position + 1].0;
            let start = offset
                + blocks[destination..position]
                    .iter()
                    .map(|block| block.0)
                    .sum::<usize>();
            exchange(form, start, left_size, right_size)?;
            blocks.swap(position, position + 1);
        }
        offset += blocks[destination].0;
    }
    Ok(())
}
