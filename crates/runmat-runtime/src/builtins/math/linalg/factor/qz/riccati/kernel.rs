//! One borrowed numerical body for ordinary and exactly admitted Riccati work.

use faer::col::ColMut;
use faer::dyn_stack::MemStack;
use faer::linalg::{householder, qr::no_pivoting::factor as qr};
use faer::mat::{MatMut, MatRef};
use faer::{c64, Conj, Par, Spec};

use super::balance::balance_extended_pencil;
use super::managed_workspace::{RiccatiScratch, RiccatiWorkspacePlan};
use super::RiccatiTimeDomain;
use crate::builtins::math::linalg::factor::qz::algorithm::{complex_qz_in_place, ComplexQzView};
use crate::builtins::math::linalg::factor::qz::reorder::{reorder_clusters_in_place, ClusterOrder};

pub(super) fn solve(
    a: &[f64],
    b: &[f64],
    q: &[f64],
    r: &[f64],
    cross: &[f64],
    plan: &RiccatiWorkspacePlan,
    domain: RiccatiTimeDomain,
    scratch: RiccatiScratch<'_>,
) -> Result<(), &'static str> {
    let RiccatiScratch {
        h,
        j,
        columns,
        householder,
        aa,
        bb,
        alpha,
        beta,
        lu,
        rhs,
        labels,
        solution,
        state_scales,
        qr_scratch,
        qz_scratch,
    } = scratch;
    let (n, m, doubled, order) = (plan.n, plan.m, plan.doubled, plan.order);
    if n == 0
        || m == 0
        || a.len() != plan.subspace_len
        || b.len() != n * m
        || q.len() != plan.subspace_len
        || r.len() != m * m
        || cross.len() != n * m
        || h.len() != plan.pencil_len
        || j.len() != plan.pencil_len
        || columns.len() != plan.column_len
        || householder.len() != plan.householder_len
        || aa.len() != plan.reduced_len
        || bb.len() != plan.reduced_len
        || alpha.len() != doubled
        || beta.len() != doubled
        || labels.len() != doubled
        || lu.len() != plan.subspace_len
        || rhs.len() != plan.subspace_len
        || solution.len() != plan.subspace_len
        || state_scales.len() != n
    {
        return Err("Riccati pencil dimensions are invalid");
    }
    let zero = c64::new(0.0, 0.0);
    let one = c64::new(1.0, 0.0);
    h.fill(zero);
    j.fill(zero);
    {
        let mut h = MatMut::from_column_major_slice_mut(&mut *h, order, order);
        let mut j = MatMut::from_column_major_slice_mut(&mut *j, order, order);
        for col in 0..n {
            for row in 0..n {
                h[(row, col)] = c64::new(a[col * n + row], 0.0);
                h[(n + row, col)] = c64::new(-q[col * n + row], 0.0);
                h[(n + row, n + col)] = match domain {
                    RiccatiTimeDomain::Continuous => c64::new(-a[row * n + col], 0.0),
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
                    RiccatiTimeDomain::Discrete => c64::new(a[row * n + col], 0.0),
                };
            }
        }
        for col in 0..m {
            for row in 0..n {
                h[(row, doubled + col)] = c64::new(b[col * n + row], 0.0);
                h[(n + row, doubled + col)] = c64::new(-cross[col * n + row], 0.0);
                h[(doubled + col, row)] = c64::new(cross[col * n + row], 0.0);
                match domain {
                    RiccatiTimeDomain::Continuous => {
                        h[(doubled + col, n + row)] = c64::new(b[col * n + row], 0.0);
                    }
                    RiccatiTimeDomain::Discrete => {
                        j[(doubled + col, n + row)] = c64::new(-b[col * n + row], 0.0);
                    }
                }
            }
            for row in 0..m {
                h[(doubled + row, doubled + col)] = c64::new(r[col * m + row], 0.0);
            }
        }
        // One reciprocal state/costate similarity covers both the continuous
        // Hamiltonian and discrete symplectic pencils, including cross weights.
        // The original physical inputs remain untouched for the independent
        // residual and closed-loop stability publication checks.
        balance_extended_pencil(&mut h, &mut j, n, state_scales)?;
        deflate_into(
            h.as_mut(),
            j.as_mut(),
            plan,
            columns,
            householder,
            aa,
            bb,
            qr_scratch,
        )?;
    }
    let mut form = ComplexQzView {
        aa: MatMut::from_column_major_slice_mut(aa, doubled, doubled),
        bb: MatMut::from_column_major_slice_mut(bb, doubled, doubled),
        left: MatMut::from_column_major_slice_mut(&mut h[..plan.reduced_len], doubled, doubled),
        right: MatMut::from_column_major_slice_mut(&mut j[..plan.reduced_len], doubled, doubled),
        alpha: ColMut::from_slice_mut(alpha),
        beta: ColMut::from_slice_mut(beta),
    };
    complex_qz_in_place(&mut form, qz_scratch)?;
    let boundary = 128.0 * f64::EPSILON * (doubled as f64).max(1.0);
    for (index, label) in labels.iter_mut().enumerate() {
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
        *label = if stable { 0 } else { 1 };
    }
    if labels.iter().filter(|label| **label == 0).count() != n {
        return Err("Riccati pencil has no n-dimensional stable subspace");
    }
    reorder_clusters_in_place(&mut form, labels, ClusterOrder::SelectedFirst)?;
    subspace_into(form.right.as_ref(), n, lu, rhs, solution, state_scales)
}

fn deflate_into(
    mut h: MatMut<'_, c64>,
    mut j: MatMut<'_, c64>,
    plan: &RiccatiWorkspacePlan,
    column_data: &mut [c64],
    householder_data: &mut [c64],
    aa: &mut [c64],
    bb: &mut [c64],
    memory: &mut [std::mem::MaybeUninit<u8>],
) -> Result<(), &'static str> {
    let (m, doubled, order) = (plan.m, plan.doubled, plan.order);
    for col in 0..m {
        for row in 0..order {
            column_data[col * order + row] = h[(row, doubled + col)];
        }
    }
    let mut columns = MatMut::from_column_major_slice_mut(column_data, order, m);
    let block = qr::recommended_block_size::<c64>(order, m);
    let mut householder = MatMut::from_column_major_slice_mut(householder_data, block, m);
    let params: Spec<qr::QrParams, c64> = Default::default();
    let stack = MemStack::new(memory);
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
    for col in 0..doubled {
        for row in 0..doubled {
            aa[col * doubled + row] = h[(m + row, col)];
            bb[col * doubled + row] = j[(m + row, col)];
        }
    }
    Ok(())
}

pub(super) fn subspace_into(
    right: MatRef<'_, c64>,
    n: usize,
    lu: &mut [c64],
    rhs: &mut [c64],
    solution: &mut [f64],
    state_scales: &[f64],
) -> Result<(), &'static str> {
    let extent = n.checked_mul(n).ok_or("Riccati subspace extent overflow")?;
    if lu.len() != extent
        || rhs.len() != extent
        || solution.len() != extent
        || state_scales.len() != n
    {
        return Err("Riccati subspace workspace dimensions are invalid");
    }
    for col in 0..n {
        for row in 0..n {
            lu[col * n + row] = right[(col, row)];
            rhs[col * n + row] = right[(n + col, row)];
        }
    }
    let scale = lu.iter().fold(0.0_f64, |max, value| max.max(value.norm()));
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
    // S = diag(D, D^-1, I) acted by S (H,J) S^-1. Its balanced graph is
    // D^-1 X D^-1, so recover the physical graph before the unchanged
    // real/symmetry acceptance gate. Both triangular solves share this RHS.
    for col in 0..n {
        for row in 0..n {
            let value = rhs[col * n + row];
            let factor = state_scales[row] * state_scales[col];
            let real = value.re * factor;
            let imaginary = value.im * factor;
            if !real.is_finite()
                || !imaginary.is_finite()
                || (value.re != 0.0 && real == 0.0)
                || (value.im != 0.0 && imaginary == 0.0)
            {
                return Err("Riccati balanced subspace cannot be restored finitely");
            }
            rhs[col * n + row] = c64::new(real, imaginary);
        }
    }
    let value_scale = rhs.iter().fold(1.0_f64, |max, value| max.max(value.norm()));
    let tolerance = 1024.0 * f64::EPSILON * (n as f64).max(1.0) * value_scale;
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
    Ok(())
}
