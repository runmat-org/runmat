//! Symmetric factorizations: LLᴴ, LDLᴴ (with dynamic regularization) and intranode LBLᴴ.

use faer::sparse::linalg::cholesky::{
    factorize_symbolic_cholesky, CholeskySymbolicParams, IntranodeLbltRef, LdltRef, LltRef,
    SymbolicCholesky, SymbolicCholeskyRaw, SymmetricOrdering,
};
use faer::sparse::linalg::SupernodalThreshold;
use faer::{Conj, MatMut, Side};

use super::{
    memory, stack_bytes, try_vec, DirectOptions, DirectWorkspacePolicy, FactorEstimate, FactorKind,
    FactorScalar, Inertia, NumericInner, SparseInput, SymbolicFactorization, SymbolicInner,
};
use crate::error::{SolverError, SolverResult};
use crate::ordering::Permutation;
use crate::par;
use crate::scalar::Scalar;

fn real<T: FactorScalar>(value: f64) -> T::Real {
    faer::traits::math_utils::from_f64::<T::Real>(value)
}

pub(super) fn analyze<T: FactorScalar>(
    matrix: SparseInput<'_, T>,
    kind: FactorKind,
    workspace: DirectWorkspacePolicy,
    permutation: &Permutation,
) -> SolverResult<SymbolicFactorization> {
    let (columns, transposed) = matrix.as_columns();
    let n = columns.ncols();
    let perm = faer::perm::PermRef::new_checked(&permutation.perm, &permutation.inverse, n);
    let params = CholeskySymbolicParams {
        supernodal_flop_ratio_threshold: if workspace == DirectWorkspacePolicy::CallerOwned {
            SupernodalThreshold::FORCE_SIMPLICIAL
        } else if kind == FactorKind::BunchKaufman {
            // Pivoting happens inside supernodes only; the simplicial path would not pivot.
            SupernodalThreshold::FORCE_SUPERNODAL
        } else {
            SupernodalThreshold::AUTO
        },
        ..Default::default()
    };
    let symbolic = factorize_symbolic_cholesky(
        columns.faer_symbolic(),
        Side::Lower,
        SymmetricOrdering::Custom(perm),
        params,
    )?;
    let estimate = estimate::<T>(&symbolic, kind);
    Ok(SymbolicFactorization {
        kind,
        workspace,
        fingerprint: matrix.pattern_fingerprint(),
        input_transposed: transposed,
        estimate,
        scalar_bytes: std::mem::size_of::<T>(),
        complex: T::IS_COMPLEX,
        inner: SymbolicInner::Cholesky(symbolic),
    })
}

fn estimate<T: FactorScalar>(
    symbolic: &SymbolicCholesky<usize>,
    kind: FactorKind,
) -> FactorEstimate {
    let n = symbolic.nrows();
    let entries = symbolic.len_val() as u64;
    let (structure, flops) = match symbolic.raw() {
        SymbolicCholeskyRaw::Simplicial(this) => {
            let col_ptr = this.col_ptr();
            let flops: f64 = (0..n)
                .map(|j| {
                    let c = (col_ptr[j + 1] - col_ptr[j]) as f64;
                    c * c
                })
                .sum();
            // Column offsets, row indices and retained elimination tree.
            (((2 * n + 1) + this.row_idx().len()) as u64 * 8, flops)
        }
        SymbolicCholeskyRaw::Supernodal(this) => {
            let begin = this.supernode_begin();
            let end = this.supernode_end();
            let pattern = this.nnz_per_super();
            let mut flops = 0.0f64;
            for s in 0..this.n_supernodes() {
                let width = end[s] - begin[s];
                let p = pattern[s];
                for k in 0..width {
                    let c = ((width - k) + p) as f64;
                    flops += c * c;
                }
            }
            (
                (this.row_idx().len() + 4 * this.n_supernodes() + n) as u64 * 8,
                flops,
            )
        }
    };
    let par = par::faer_par();
    let workspace = match kind {
        FactorKind::Cholesky => {
            symbolic.factorize_numeric_llt_scratch::<T>(par, Default::default())
        }
        FactorKind::Ldlt => symbolic.factorize_numeric_ldlt_scratch::<T>(par, Default::default()),
        _ => symbolic.factorize_numeric_intranode_lblt_scratch::<T>(par, Default::default()),
    };
    let scalar = std::mem::size_of::<T>() as u64;
    let pivoting = if kind == FactorKind::BunchKaufman {
        n as u64 * (scalar + 16)
    } else {
        0
    };
    FactorEstimate {
        dimension: n,
        factor_entries: entries,
        factor_bytes: entries * scalar
            + structure
            + pivoting
            + if symbolic.perm().is_some() {
                n as u64 * 16
            } else {
                0
            },
        // LDLT verifies the extracted diagonal while its explicit factorization
        // scratch remains live. This is wrapper-owned numerical storage too.
        workspace_bytes: stack_bytes(workspace)
            + if kind == FactorKind::Ldlt {
                n as u64 * scalar
            } else {
                0
            },
        flops: flops * if T::IS_COMPLEX { 4.0 } else { 1.0 },
        nominal: false,
    }
}

fn symbolic_of(symbolic: &SymbolicFactorization) -> &SymbolicCholesky<usize> {
    match &symbolic.inner {
        SymbolicInner::Cholesky(inner) => inner,
        SymbolicInner::Lu { .. } => unreachable!("symmetric factorization on an LU analysis"),
    }
}

fn conj_of<T: Scalar>(symbolic: &SymbolicFactorization) -> Conj {
    // A CSR array viewed as CSC is Aᵀ = conj(A) for Hermitian A; solve with the conjugate factor.
    if symbolic.input_transposed && T::IS_COMPLEX {
        Conj::Yes
    } else {
        Conj::No
    }
}

pub(super) fn factorize<T: FactorScalar>(
    symbolic: &SymbolicFactorization,
    matrix: SparseInput<'_, T>,
    options: &DirectOptions,
    previous: Option<NumericInner<T>>,
) -> SolverResult<(NumericInner<T>, usize)> {
    let inner = symbolic_of(symbolic);
    let (columns, _) = matrix.as_columns();
    let n = inner.nrows();
    let a = columns.faer();
    let par = par::faer_par();
    let reuse = |previous: Option<NumericInner<T>>| -> SolverResult<Vec<T>> {
        match previous {
            Some(NumericInner::Llt(values) | NumericInner::Ldlt(values))
                if values.len() == inner.len_val() =>
            {
                Ok(values)
            }
            Some(NumericInner::Lblt { values, .. }) if values.len() == inner.len_val() => {
                Ok(values)
            }
            _ => try_vec(inner.len_val(), T::zero()),
        }
    };
    if let Some(values) = matrix_values_nonfinite(columns.values()) {
        return Err(SolverError::NonFinite { entry: values });
    }
    match symbolic.kind {
        FactorKind::Cholesky => {
            let mut values = reuse(previous)?;
            let mut buffer =
                memory(inner.factorize_numeric_llt_scratch::<T>(par, Default::default()))?;
            inner
                .factorize_numeric_llt::<T>(
                    &mut values,
                    a,
                    Side::Lower,
                    Default::default(),
                    par,
                    faer::dyn_stack::MemStack::new(&mut buffer),
                    Default::default(),
                )
                .map_err(|error| match error {
                    faer::linalg::cholesky::llt::factor::LltError::NonPositivePivot { index } => {
                        SolverError::NotPositiveDefinite { pivot: index }
                    }
                })?;
            Ok((NumericInner::Llt(values), 0))
        }
        FactorKind::Ldlt => {
            let mut values = reuse(previous)?;
            if let Some(signs) = &options.pivot_signs {
                if signs.len() != n {
                    return Err(SolverError::Dimension("pivot sign count".into()));
                }
            }
            let regularization = faer::linalg::cholesky::ldlt::factor::LdltRegularization {
                dynamic_regularization_signs: options.pivot_signs.as_deref(),
                dynamic_regularization_delta: real::<T>(options.regularization_delta),
                dynamic_regularization_epsilon: real::<T>(options.regularization_epsilon),
            };
            let mut buffer =
                memory(inner.factorize_numeric_ldlt_scratch::<T>(par, Default::default()))?;
            inner
                .factorize_numeric_ldlt::<T>(
                    &mut values,
                    a,
                    Side::Lower,
                    regularization,
                    par,
                    faer::dyn_stack::MemStack::new(&mut buffer),
                    Default::default(),
                )
                .map_err(|error| match error {
                    faer::linalg::cholesky::ldlt::factor::LdltError::ZeroPivot { index } => {
                        SolverError::Singular { pivot: index }
                    }
                })?;
            let diagonal = factor_diagonal(inner, &values);
            if let Some(pivot) = diagonal
                .iter()
                .position(|d| !d.is_finite() || d.abs() == 0.0)
            {
                return Err(SolverError::Singular { pivot });
            }
            Ok((NumericInner::Ldlt(values), 0))
        }
        FactorKind::BunchKaufman => {
            let mut values = reuse(previous)?;
            let mut subdiag = try_vec(n, T::zero())?;
            let mut perm = try_vec(n, 0usize)?;
            let mut inverse = try_vec(n, 0usize)?;
            let mut buffer = memory(
                inner.factorize_numeric_intranode_lblt_scratch::<T>(par, Default::default()),
            )?;
            inner.factorize_numeric_intranode_lblt::<T>(
                &mut values,
                &mut subdiag,
                &mut perm,
                &mut inverse,
                a,
                Side::Lower,
                par,
                faer::dyn_stack::MemStack::new(&mut buffer),
                Default::default(),
            );
            let diagonal = factor_diagonal(inner, &values);
            for j in 0..n {
                let in_block = subdiag[j].abs() != 0.0 || (j > 0 && subdiag[j - 1].abs() != 0.0);
                if !diagonal[j].is_finite() || (!in_block && diagonal[j].abs() == 0.0) {
                    return Err(SolverError::Singular { pivot: j });
                }
            }
            Ok((
                NumericInner::Lblt {
                    values,
                    subdiag,
                    perm,
                    inverse,
                },
                0,
            ))
        }
        FactorKind::Lu => unreachable!(),
    }
}

fn matrix_values_nonfinite<T: Scalar>(values: &[T]) -> Option<usize> {
    values.iter().position(|value| !value.is_finite())
}

/// Diagonal of the stored factor (`D` for LDLᴴ/LBLᴴ, `diag(L)` for LLᴴ) in factor order.
fn factor_diagonal<T: FactorScalar>(symbolic: &SymbolicCholesky<usize>, values: &[T]) -> Vec<T> {
    let n = symbolic.nrows();
    let mut diagonal = vec![T::zero(); n];
    match symbolic.raw() {
        SymbolicCholeskyRaw::Simplicial(this) => {
            let (col_ptr, row_idx) = (this.col_ptr(), this.row_idx());
            for j in 0..n {
                for entry in col_ptr[j]..col_ptr[j + 1] {
                    if row_idx[entry] == j {
                        diagonal[j] = values[entry];
                        break;
                    }
                }
            }
        }
        SymbolicCholeskyRaw::Supernodal(this) => {
            let (begin, end) = (this.supernode_begin(), this.supernode_end());
            let (pattern, offsets) = (this.nnz_per_super(), this.col_ptr_for_val());
            for s in 0..this.n_supernodes() {
                let width = end[s] - begin[s];
                let rows = width + pattern[s];
                for k in 0..width {
                    diagonal[begin[s] + k] = values[offsets[s] + k * rows + k];
                }
            }
        }
    }
    diagonal
}

pub(super) fn solve<T: FactorScalar>(
    symbolic: &SymbolicFactorization,
    numeric: &NumericInner<T>,
    rhs: &mut [T],
    k: usize,
) {
    let inner = symbolic_of(symbolic);
    let n = inner.nrows();
    let par = par::faer_par();
    let conj = conj_of::<T>(symbolic);
    let mut buffer = faer::dyn_stack::MemBuffer::new(inner.solve_in_place_scratch::<T>(k, par));
    let stack = faer::dyn_stack::MemStack::new(&mut buffer);
    let rhs = MatMut::from_column_major_slice_mut(rhs, n, k);
    match numeric {
        NumericInner::Llt(values) => {
            LltRef::<usize, T>::new(inner, values).solve_in_place_with_conj(conj, rhs, par, stack)
        }
        NumericInner::Ldlt(values) => {
            LdltRef::<usize, T>::new(inner, values).solve_in_place_with_conj(conj, rhs, par, stack)
        }
        NumericInner::Lblt {
            values,
            subdiag,
            perm,
            inverse,
        } => {
            let perm = faer::perm::PermRef::new_checked(perm, inverse, n);
            IntranodeLbltRef::<usize, T>::new(inner, values, subdiag, perm)
                .solve_in_place_with_conj(conj, rhs, par, stack)
        }
        NumericInner::Lu(_) => unreachable!(),
    }
}

pub(super) fn inertia<T: FactorScalar>(
    symbolic: &SymbolicFactorization,
    numeric: &NumericInner<T>,
) -> Option<Inertia> {
    let SymbolicInner::Cholesky(inner) = &symbolic.inner else {
        return None;
    };
    let n = inner.nrows();
    let mut inertia = Inertia::default();
    let classify = |value: f64, inertia: &mut Inertia| {
        if value > 0.0 {
            inertia.positive += 1;
        } else if value < 0.0 {
            inertia.negative += 1;
        } else {
            inertia.zero += 1;
        }
    };
    match numeric {
        NumericInner::Llt(_) => {
            inertia.positive = n;
        }
        NumericInner::Ldlt(values) => {
            for d in factor_diagonal(inner, values) {
                classify(d.re(), &mut inertia);
            }
        }
        NumericInner::Lblt {
            values, subdiag, ..
        } => {
            let d = factor_diagonal(inner, values);
            let mut j = 0;
            while j < n {
                if j + 1 < n && subdiag[j].abs() != 0.0 {
                    let (a, c) = (d[j].re(), d[j + 1].re());
                    let determinant = a * c - subdiag[j].abs2();
                    if determinant < 0.0 {
                        inertia.positive += 1;
                        inertia.negative += 1;
                    } else if determinant > 0.0 {
                        classify(a + c, &mut inertia);
                        classify(a + c, &mut inertia);
                    } else {
                        classify(a + c, &mut inertia);
                        inertia.zero += 1;
                    }
                    j += 2;
                } else {
                    classify(d[j].re(), &mut inertia);
                    j += 1;
                }
            }
        }
        NumericInner::Lu(_) => return None,
    }
    Some(inertia)
}

pub(super) fn pivots<T: FactorScalar>(
    symbolic: &SymbolicFactorization,
    numeric: &NumericInner<T>,
) -> Option<Vec<f64>> {
    let SymbolicInner::Cholesky(inner) = &symbolic.inner else {
        return None;
    };
    let (values, squared) = match numeric {
        NumericInner::Llt(values) => (values, true),
        NumericInner::Ldlt(values) => (values, false),
        _ => return None,
    };
    let diagonal = factor_diagonal(inner, values);
    let n = inner.nrows();
    let mut pivots = vec![0.0; n];
    match inner.perm() {
        Some(perm) => {
            let (forward, _) = perm.arrays();
            for (position, &original) in forward.iter().enumerate() {
                let d = diagonal[position].re();
                pivots[original] = if squared { d * d } else { d };
            }
        }
        None => {
            for (j, d) in diagonal.iter().enumerate() {
                pivots[j] = if squared { d.re() * d.re() } else { d.re() };
            }
        }
    }
    Some(pivots)
}

/// The [`FactorEstimate`] the simplicial analysis of [`analyze`] reports, from the elimination
/// tree and column counts alone: no factor structure is built. Requested heap is the permuted
/// upper triangle (one word per stored entry and per column) and four words per row.
pub(super) fn simplicial_estimate<T: FactorScalar>(
    matrix: SparseInput<'_, T>,
    kind: FactorKind,
    permutation: &Permutation,
) -> SolverResult<FactorEstimate> {
    let (columns, _) = matrix.as_columns();
    let n = columns.ncols();
    let a = columns.faer_symbolic();
    let inverse = &permutation.inverse;
    // Upper triangle of P A Pᵀ from the input's lower triangle, as `analyze` reads it.
    let mut col_ptr = try_vec(n + 1, 0usize)?;
    for j in 0..n {
        for &i in a.row_idx_of_col_raw(j) {
            if i >= j {
                col_ptr[inverse[i].max(inverse[j]) + 1] += 1;
            }
        }
    }
    for j in 0..n {
        col_ptr[j + 1] += col_ptr[j];
    }
    let mut row_idx = try_vec(col_ptr[n], 0usize)?;
    {
        let mut next = try_vec(n, 0usize)?;
        next.copy_from_slice(&col_ptr[..n]);
        for j in 0..n {
            for &i in a.row_idx_of_col_raw(j) {
                if i >= j {
                    let (row, column) = (
                        inverse[i].min(inverse[j]),
                        inverse[i].max(inverse[j]),
                    );
                    row_idx[next[column]] = row;
                    next[column] += 1;
                }
            }
        }
    }
    let upper =
        faer::sparse::SymbolicSparseColMatRef::new_unsorted_checked(n, n, &col_ptr, None, &row_idx);
    let mut etree = try_vec(n, 0isize)?;
    let mut col_counts = try_vec(n, 0usize)?;
    let mut buffer = faer::dyn_stack::MemBuffer::try_new(
        faer::sparse::linalg::cholesky::simplicial::prefactorize_symbolic_cholesky_scratch::<usize>(
            n,
            row_idx.len(),
        ),
    )
    .map_err(|_| SolverError::OutOfMemory {
        bytes: (n * std::mem::size_of::<usize>()) as u64,
    })?;
    faer::sparse::linalg::cholesky::simplicial::prefactorize_symbolic_cholesky::<usize>(
        &mut etree,
        &mut col_counts,
        upper,
        faer::dyn_stack::MemStack::new(&mut buffer),
    );
    let entries: u64 = col_counts.iter().map(|&c| c as u64).sum();
    let flops: f64 = col_counts.iter().map(|&c| (c as f64) * (c as f64)).sum();
    let structure = ((2 * n + 1) as u64 + entries) * 8;
    // The layout of `SymbolicCholesky::factorize_numeric_{llt,ldlt}_scratch` for a simplicial
    // factor: the permuted input copy, then the permutation or the factorization scratch.
    use faer::dyn_stack::StackReq;
    let a_nnz = a.compute_nnz();
    let factor = match kind {
        FactorKind::Ldlt => faer::sparse::linalg::cholesky::simplicial::factorize_simplicial_numeric_ldlt_scratch::<usize, T>(n),
        _ => faer::sparse::linalg::cholesky::simplicial::factorize_simplicial_numeric_llt_scratch::<usize, T>(n),
    };
    let signs = if kind == FactorKind::Ldlt {
        StackReq::new::<i8>(n)
    } else {
        StackReq::empty()
    };
    let workspace = StackReq::all_of(&[
        signs,
        StackReq::all_of(&[
            faer::linalg::temp_mat_scratch::<T>(a_nnz, 1),
            StackReq::new::<usize>(n + 1),
            StackReq::new::<usize>(a_nnz),
        ]),
        StackReq::or(StackReq::new::<usize>(n), factor),
    ]);
    let scalar = std::mem::size_of::<T>() as u64;
    Ok(FactorEstimate {
        dimension: n,
        factor_entries: entries,
        factor_bytes: entries * scalar + structure + n as u64 * 16,
        workspace_bytes: stack_bytes(workspace)
            + if kind == FactorKind::Ldlt {
                n as u64 * scalar
            } else {
                0
            },
        flops: flops * if T::IS_COMPLEX { 4.0 } else { 1.0 },
        nominal: false,
    })
}
