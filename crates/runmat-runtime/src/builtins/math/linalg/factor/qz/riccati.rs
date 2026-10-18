//! Stable invariant-subspace extraction for standard real CARE/DARE problems.
//!
//! This is an internal consumer of the same QZ and ordered-Schur kernels as
//! the public factorization. The extended pencil is deflated by its input
//! columns before QZ; factoring the singular full pencil directly does not
//! provide the required two-block Riccati subspace.

use faer::col::ColMut;
use faer::dyn_stack::{MemBuffer, MemStack, StackReq};
use faer::linalg::{householder, qr::no_pivoting::factor as qr};
use faer::mat::MatMut;
use faer::{c64, Conj, Par, Spec};
use nalgebra::DMatrix;
use runmat_value::{exact_capacity_vec, exact_capacity_vec_with};

use super::algorithm::{complex_qz_in_place, complex_qz_scratch, ComplexQzView};
use super::reorder::{reorder_clusters_in_place, ClusterOrder};
use super::work::QzWork;
use crate::context::RuntimeContext;
use crate::RuntimeError;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RiccatiTimeDomain {
    Continuous,
    Discrete,
}

pub(crate) struct RiccatiFactorization {
    pub(crate) a: DMatrix<f64>,
    pub(crate) b: DMatrix<f64>,
    pub(crate) q: DMatrix<f64>,
    pub(crate) r: DMatrix<f64>,
    pub(crate) cross: DMatrix<f64>,
    pub(crate) solution: DMatrix<f64>,
}

/// All matrices are already finite, shape-checked and host-owned by the
/// caller. QZ work moves into the installed blocking worker; dropping the
/// awaiting caller never drops matrices while that worker is still running.
pub(crate) async fn stabilizing_subspace(
    a: DMatrix<f64>,
    b: DMatrix<f64>,
    q: DMatrix<f64>,
    r: DMatrix<f64>,
    cross: DMatrix<f64>,
    domain: RiccatiTimeDomain,
    runtime: Option<&RuntimeContext>,
    name: &'static str,
    numerical_error: impl FnOnce(&'static str) -> RuntimeError,
) -> Result<RiccatiFactorization, RuntimeError> {
    let mut work = QzWork::new(runtime, None, name);
    work.checkpoint(0).await?;
    let result = work
        .blocking(move || solve_owned(a, b, q, r, cross, domain))
        .await?;
    result.map_err(numerical_error)
}

fn solve_owned(
    a: DMatrix<f64>,
    b: DMatrix<f64>,
    q: DMatrix<f64>,
    r: DMatrix<f64>,
    cross: DMatrix<f64>,
    domain: RiccatiTimeDomain,
) -> Result<RiccatiFactorization, &'static str> {
    let n = a.nrows();
    let m = b.ncols();
    let doubled = n.checked_mul(2).ok_or("Riccati pencil order overflow")?;
    let order = doubled
        .checked_add(m)
        .ok_or("Riccati pencil order overflow")?;
    if n == 0
        || m == 0
        || a.ncols() != n
        || b.nrows() != n
        || q.shape() != (n, n)
        || r.shape() != (m, m)
        || cross.shape() != (n, m)
    {
        return Err("Riccati pencil dimensions are invalid");
    }
    let zero = c64::new(0.0, 0.0);
    let one = c64::new(1.0, 0.0);
    let pencil_len = order
        .checked_mul(order)
        .ok_or("Riccati pencil extent overflow")?;
    let mut h_data = complex_buffer(pencil_len)?;
    let mut j_data = complex_buffer(pencil_len)?;
    let mut h = MatMut::from_column_major_slice_mut(&mut h_data, order, order);
    let mut j = MatMut::from_column_major_slice_mut(&mut j_data, order, order);
    for col in 0..n {
        for row in 0..n {
            h[(row, col)] = c64::new(a[(row, col)], 0.0);
            h[(n + row, col)] = c64::new(-q[(row, col)], 0.0);
            h[(n + row, n + col)] = match domain {
                RiccatiTimeDomain::Continuous => c64::new(-a[(col, row)], 0.0),
                RiccatiTimeDomain::Discrete => {
                    if row == col {
                        one
                    } else {
                        zero
                    }
                }
            };
            j[(row, col)] = if row == col { one } else { zero };
            j[(n + row, n + col)] = match domain {
                RiccatiTimeDomain::Continuous => {
                    if row == col {
                        one
                    } else {
                        zero
                    }
                }
                RiccatiTimeDomain::Discrete => c64::new(a[(col, row)], 0.0),
            };
        }
    }
    for col in 0..m {
        for row in 0..n {
            h[(row, doubled + col)] = c64::new(b[(row, col)], 0.0);
            h[(n + row, doubled + col)] = c64::new(-cross[(row, col)], 0.0);
            h[(doubled + col, row)] = c64::new(cross[(row, col)], 0.0);
            match domain {
                RiccatiTimeDomain::Continuous => {
                    h[(doubled + col, n + row)] = c64::new(b[(row, col)], 0.0);
                }
                RiccatiTimeDomain::Discrete => {
                    j[(doubled + col, n + row)] = c64::new(-b[(row, col)], 0.0);
                }
            }
        }
        for row in 0..m {
            h[(doubled + row, doubled + col)] = c64::new(r[(row, col)], 0.0);
        }
    }
    let (mut aa_data, mut bb_data) = deflate_input_columns(h, j, n, m)?;
    let reduced_len = doubled
        .checked_mul(doubled)
        .ok_or("Riccati reduced extent overflow")?;
    let mut left_data = complex_buffer(reduced_len)?;
    let mut right_data = complex_buffer(reduced_len)?;
    let mut alpha = complex_buffer(doubled)?;
    let mut beta = complex_buffer(doubled)?;
    let mut qz_memory = MemBuffer::try_new(complex_qz_scratch(doubled))
        .map_err(|_| "Riccati QZ workspace allocation failed")?;
    let mut form = ComplexQzView {
        aa: MatMut::from_column_major_slice_mut(&mut aa_data, doubled, doubled),
        bb: MatMut::from_column_major_slice_mut(&mut bb_data, doubled, doubled),
        left: MatMut::from_column_major_slice_mut(&mut left_data, doubled, doubled),
        right: MatMut::from_column_major_slice_mut(&mut right_data, doubled, doubled),
        alpha: ColMut::from_slice_mut(&mut alpha),
        beta: ColMut::from_slice_mut(&mut beta),
    };
    complex_qz_in_place(&mut form, &mut qz_memory)?;
    let mut labels =
        exact_capacity_vec(doubled).map_err(|_| "Riccati cluster metadata allocation failed")?;
    let boundary = 128.0 * f64::EPSILON * (doubled as f64).max(1.0);
    for index in 0..doubled {
        let alpha = form.alpha[index];
        let beta = form.beta[index];
        let a_norm = alpha.re.hypot(alpha.im);
        let b_norm = beta.re.hypot(beta.im);
        if !a_norm.is_finite() || !b_norm.is_finite() || b_norm == 0.0 {
            return Err("Riccati pencil has a nonfinite or infinite eigenvalue");
        }
        let stable = match domain {
            RiccatiTimeDomain::Continuous => {
                let a_scale = alpha.re.abs().max(alpha.im.abs());
                let b_scale = beta.re.abs().max(beta.im.abs());
                if a_scale == 0.0 {
                    return Err("Hamiltonian spectrum touches the imaginary axis");
                }
                let real = (alpha.re / a_scale) * (beta.re / b_scale)
                    + (alpha.im / a_scale) * (beta.im / b_scale);
                if real.abs() <= boundary {
                    return Err("Hamiltonian spectrum is too near the imaginary axis");
                }
                real < 0.0
            }
            RiccatiTimeDomain::Discrete => {
                let scale = a_norm.max(b_norm);
                if ((a_norm / scale) - (b_norm / scale)).abs() <= boundary {
                    return Err("symplectic spectrum is too near the unit circle");
                }
                a_norm < b_norm
            }
        };
        labels.push(if stable { 0 } else { 1 });
    }
    if labels.iter().filter(|label| **label == 0).count() != n {
        return Err("Riccati pencil has no n-dimensional stable subspace");
    }
    reorder_clusters_in_place(&mut form, &mut labels, ClusterOrder::SelectedFirst)?;
    let solution = real_symmetric_subspace_solution(form.right.as_ref(), n)?;
    Ok(RiccatiFactorization {
        a,
        b,
        q,
        r,
        cross,
        solution,
    })
}

/// Solve `U0^T X^T = U1^T` using one pivoted complex LU. This avoids an
/// opaque matrix inverse and duplicate SVD backing; the subsequent original-
/// equation residual and closed-loop test remain the publication authority.
fn real_symmetric_subspace_solution(
    right: faer::mat::MatRef<'_, c64>,
    n: usize,
) -> Result<DMatrix<f64>, &'static str> {
    let extent = n.checked_mul(n).ok_or("Riccati subspace extent overflow")?;
    let mut lu = complex_buffer(extent)?;
    let mut rhs = complex_buffer(extent)?;
    for col in 0..n {
        for row in 0..n {
            lu[col * n + row] = right[(col, row)];
            rhs[col * n + row] = right[(n + col, row)];
        }
    }
    let scale = lu
        .iter()
        .fold(0.0_f64, |maximum, value| maximum.max(value.norm()));
    if !scale.is_finite() || scale == 0.0 {
        return Err("stable Riccati invariant subspace is singular");
    }
    let pivot_floor = f64::EPSILON * (n as f64).max(1.0) * scale;
    for k in 0..n {
        let pivot = (k..n)
            .max_by(|&a, &b| lu[k * n + a].norm().total_cmp(&lu[k * n + b].norm()))
            .ok_or("stable Riccati invariant subspace is empty")?;
        if !lu[k * n + pivot].norm().is_finite() || lu[k * n + pivot].norm() <= pivot_floor {
            return Err("stable Riccati invariant subspace is ill-conditioned");
        }
        if pivot != k {
            for col in 0..n {
                lu.swap(col * n + k, col * n + pivot);
                rhs.swap(col * n + k, col * n + pivot);
            }
        }
        let diagonal = lu[k * n + k];
        for row in (k + 1)..n {
            let multiplier = lu[k * n + row] / diagonal;
            lu[k * n + row] = multiplier;
            for col in (k + 1)..n {
                let prior = lu[col * n + k];
                lu[col * n + row] -= multiplier * prior;
            }
        }
    }
    for col in 0..n {
        for row in 0..n {
            let mut value = rhs[col * n + row];
            for k in 0..row {
                value -= lu[k * n + row] * rhs[col * n + k];
            }
            rhs[col * n + row] = value;
        }
        for row in (0..n).rev() {
            let mut value = rhs[col * n + row];
            for k in (row + 1)..n {
                value -= lu[k * n + row] * rhs[col * n + k];
            }
            rhs[col * n + row] = value / lu[row * n + row];
        }
    }
    let value_scale = rhs
        .iter()
        .fold(1.0_f64, |maximum, value| maximum.max(value.norm()));
    let tolerance = 1024.0 * f64::EPSILON * (n as f64).max(1.0) * value_scale;
    let mut solution = exact_capacity_vec_with(extent, |_| 0.0_f64)
        .map_err(|_| "Riccati solution allocation failed")?;
    for col in 0..n {
        for row in 0..n {
            let value = rhs[row * n + col];
            let transpose = rhs[col * n + row];
            if !value.re.is_finite()
                || !value.im.is_finite()
                || value.im.abs() > tolerance
                || (value.re - transpose.re).abs() > tolerance
            {
                return Err("stable Riccati subspace does not yield a real symmetric solution");
            }
            solution[col * n + row] = (value.re + transpose.re) * 0.5;
        }
    }
    Ok(DMatrix::<f64>::from_vec(n, n, solution))
}

fn deflate_input_columns(
    mut h: MatMut<'_, c64>,
    mut j: MatMut<'_, c64>,
    n: usize,
    m: usize,
) -> Result<(Vec<c64>, Vec<c64>), &'static str> {
    let doubled = n.checked_mul(2).ok_or("Riccati deflation order overflow")?;
    let order = doubled
        .checked_add(m)
        .ok_or("Riccati deflation order overflow")?;
    let column_len = order.checked_mul(m).ok_or("Riccati QR extent overflow")?;
    let mut column_data = complex_buffer(column_len)?;
    for col in 0..m {
        for row in 0..order {
            column_data[col * order + row] = h[(row, doubled + col)];
        }
    }
    let mut columns = MatMut::from_column_major_slice_mut(&mut column_data, order, m);
    let block = qr::recommended_block_size::<c64>(order, m);
    let householder_len = block.checked_mul(m).ok_or("Riccati QR extent overflow")?;
    let mut householder_data = complex_buffer(householder_len)?;
    let mut householder = MatMut::from_column_major_slice_mut(&mut householder_data, block, m);
    let params: Spec<qr::QrParams, c64> = Default::default();
    let scratch = StackReq::any_of(&[
        qr::qr_in_place_scratch::<c64>(order, m, block, Par::Seq, params),
        householder::apply_block_householder_sequence_transpose_on_the_left_in_place_scratch::<c64>(
            order, block, order,
        ),
    ]);
    let mut memory =
        MemBuffer::try_new(scratch).map_err(|_| "Riccati QR workspace allocation failed")?;
    let stack = MemStack::new(&mut memory);
    let info = qr::qr_in_place(
        columns.as_mut(),
        householder.as_mut(),
        Par::Seq,
        stack,
        params,
    );
    if info.rank < m {
        return Err("Riccati pencil input column is rank deficient");
    }
    householder::apply_block_householder_sequence_transpose_on_the_left_in_place_with_conj(
        columns.as_ref(),
        householder.as_ref(),
        Conj::Yes,
        h.as_mut(),
        Par::Seq,
        stack,
    );
    householder::apply_block_householder_sequence_transpose_on_the_left_in_place_with_conj(
        columns.as_ref(),
        householder.as_ref(),
        Conj::Yes,
        j.as_mut(),
        Par::Seq,
        stack,
    );
    let reduced_len = doubled
        .checked_mul(doubled)
        .ok_or("Riccati reduced extent overflow")?;
    let mut aa = complex_buffer(reduced_len)?;
    let mut bb = complex_buffer(reduced_len)?;
    for col in 0..doubled {
        for row in 0..doubled {
            aa[col * doubled + row] = h[(m + row, col)];
            bb[col * doubled + row] = j[(m + row, col)];
        }
    }
    Ok((aa, bb))
}

fn complex_buffer(len: usize) -> Result<Vec<c64>, &'static str> {
    exact_capacity_vec_with(len, |_| c64::new(0.0, 0.0))
        .map_err(|_| "Riccati workspace allocation failed")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scalar(value: f64) -> DMatrix<f64> {
        DMatrix::from_element(1, 1, value)
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn ordered_subspace_selects_stabilizing_care_and_dare_branches() {
        let care = solve_owned(
            scalar(-1.0),
            scalar(1.0),
            scalar(1.0),
            scalar(1.0),
            scalar(0.0),
            RiccatiTimeDomain::Continuous,
        )
        .expect("stable CARE subspace");
        assert!((care.solution[(0, 0)] - (2.0_f64.sqrt() - 1.0)).abs() < 1.0e-8);

        let dare = solve_owned(
            scalar(1.0),
            scalar(1.0),
            scalar(1.0),
            scalar(1.0),
            scalar(0.0),
            RiccatiTimeDomain::Discrete,
        )
        .expect("stable DARE subspace");
        assert!((dare.solution[(0, 0)] - (1.0 + 5.0_f64.sqrt()) * 0.5).abs() < 1.0e-8);

        // Q=0 makes the old fixed-point iteration stop at the unstable zero
        // root. An ordered pencil must either select the stabilizing root or
        // explicitly decline; it must never publish zero as stabilizing.
        let weakly_detectable = solve_owned(
            scalar(2.0),
            scalar(1.0),
            scalar(0.0),
            scalar(1.0),
            scalar(0.0),
            RiccatiTimeDomain::Discrete,
        );
        assert!(
            weakly_detectable
                .as_ref()
                .is_ok_and(|solution| (solution.solution[(0, 0)] - 3.0).abs() < 1.0e-8)
                || weakly_detectable.is_err()
        );

        // U0 = [0 1; 1 1] requires an LU row pivot. With U1 = X U0 for
        // X = [2 .5; .5 1], both right-hand columns must survive that pivot.
        let right = [
            c64::new(0.0, 0.0),
            c64::new(1.0, 0.0),
            c64::new(0.5, 0.0),
            c64::new(1.0, 0.0),
            c64::new(1.0, 0.0),
            c64::new(1.0, 0.0),
            c64::new(2.5, 0.0),
            c64::new(1.5, 0.0),
            c64::new(0.0, 0.0),
            c64::new(0.0, 0.0),
            c64::new(0.0, 0.0),
            c64::new(0.0, 0.0),
            c64::new(0.0, 0.0),
            c64::new(0.0, 0.0),
            c64::new(0.0, 0.0),
            c64::new(0.0, 0.0),
        ];
        let solution = real_symmetric_subspace_solution(
            faer::mat::MatRef::from_column_major_slice(&right, 4, 4),
            2,
        )
        .expect("pivoted two-column subspace solve");
        for (row, col, expected) in [(0, 0, 2.0), (0, 1, 0.5), (1, 0, 0.5), (1, 1, 1.0)] {
            assert!((solution[(row, col)] - expected).abs() < 1.0e-12);
        }
    }
}
