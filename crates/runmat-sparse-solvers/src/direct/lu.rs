//! LU with partial pivoting (COLAMD column ordering) for general sparse matrices.

use faer::sparse::linalg::lu::{factorize_symbolic_lu, LuRef, NumericLu, SymbolicLu};
use faer::{Conj, MatMut};

use super::{
    memory, stack_bytes, DirectOptions, FactorEstimate, FactorKind, FactorScalar, NumericInner,
    SparseInput, SymbolicFactorization, SymbolicInner,
};
use crate::error::{SolverError, SolverResult};
use crate::par;

pub(super) fn analyze<T: FactorScalar>(
    matrix: SparseInput<'_, T>,
    options: &DirectOptions,
) -> SolverResult<SymbolicFactorization> {
    let (columns, transposed) = matrix.as_columns();
    let symbolic = factorize_symbolic_lu(columns.faer_symbolic(), Default::default())?;
    let (estimate, reference) = nominal_estimate::<T>(matrix, &symbolic)?;
    Ok(SymbolicFactorization {
        kind: FactorKind::Lu,
        workspace: options.workspace,
        fingerprint: matrix.pattern_fingerprint(),
        input_transposed: transposed,
        estimate,
        scalar_bytes: std::mem::size_of::<T>(),
        complex: T::IS_COMPLEX,
        inner: SymbolicInner::Lu {
            symbolic,
            reference,
        },
    })
}

/// LU fill depends on pivoting; quote twice the Cholesky fill of the symmetrized pattern under
/// minimum degree (exact for structurally symmetric matrices pivoting on the diagonal, and a
/// documented nominal figure otherwise). The same symmetrized fill fixes the LU reference
/// schedule used for simulation-work metering.
fn nominal_estimate<T: FactorScalar>(
    matrix: SparseInput<'_, T>,
    symbolic: &SymbolicLu<usize>,
) -> SolverResult<(
    FactorEstimate,
    Result<super::DirectReferenceWork, super::ReferenceWorkError>,
)> {
    let (columns, _) = matrix.as_columns();
    let n = columns.ncols();
    let graph = crate::ordering::Graph::from_matrix(columns.as_transpose_csr());
    // Build the symmetric pattern with diagonal as CSC for the symbolic Cholesky count.
    let mut col_ptr = Vec::with_capacity(n + 1);
    let mut row_idx = Vec::with_capacity(graph.adjncy.len() + n);
    col_ptr.push(0usize);
    for v in 0..n {
        let neighbors = graph.neighbors(v);
        let mut inserted = false;
        for &u in neighbors {
            if !inserted && u > v {
                row_idx.push(v);
                inserted = true;
            }
            row_idx.push(u);
        }
        if !inserted {
            row_idx.push(v);
        }
        col_ptr.push(row_idx.len());
    }
    let pattern =
        faer::sparse::SymbolicSparseColMatRef::new_checked(n, n, &col_ptr, None, &row_idx);
    let cholesky = faer::sparse::linalg::cholesky::factorize_symbolic_cholesky(
        pattern,
        faer::Side::Lower,
        faer::sparse::linalg::cholesky::SymmetricOrdering::Amd,
        Default::default(),
    )?;
    let reference = super::reference_work::general_columns(
        super::reference_work::cholesky_column_heights(&cholesky),
        T::IS_COMPLEX,
    );
    let entries = 2 * cholesky.len_val() as u64;
    let scalar = std::mem::size_of::<T>() as u64;
    let par = par::faer_par();
    let workspace =
        stack_bytes(symbolic.factorize_numeric_lu_scratch::<T>(par, Default::default()));
    let lower_flops = {
        // Same structure-based flop count as the Cholesky estimate, doubled for L and U.
        let est = cholesky.len_val() as f64;
        est * est / n.max(1) as f64
    };
    let estimate = FactorEstimate {
        dimension: n,
        factor_entries: entries,
        factor_bytes: entries * (scalar + 8) + (n as u64) * 64,
        workspace_bytes: workspace,
        flops: 2.0 * lower_flops * if T::IS_COMPLEX { 4.0 } else { 1.0 },
        nominal: true,
    };
    Ok((estimate, reference))
}

fn symbolic_of(symbolic: &SymbolicFactorization) -> &SymbolicLu<usize> {
    match &symbolic.inner {
        SymbolicInner::Lu { symbolic, .. } => symbolic,
        SymbolicInner::Cholesky(_) => unreachable!("LU on a symmetric analysis"),
    }
}

pub(super) fn factorize<T: FactorScalar>(
    symbolic: &SymbolicFactorization,
    matrix: SparseInput<'_, T>,
    previous: Option<NumericInner<T>>,
) -> SolverResult<NumericInner<T>> {
    let inner = symbolic_of(symbolic);
    let (columns, _) = matrix.as_columns();
    if let Some(entry) = columns.values().iter().position(|v| !v.is_finite()) {
        return Err(SolverError::NonFinite { entry });
    }
    let mut numeric = match previous {
        Some(NumericInner::Lu(numeric)) => numeric,
        _ => Box::new(NumericLu::new()),
    };
    let par = par::faer_par();
    let mut buffer = memory(inner.factorize_numeric_lu_scratch::<T>(par, Default::default()))?;
    inner
        .factorize_numeric_lu::<T>(
            &mut numeric,
            columns.faer(),
            par,
            faer::dyn_stack::MemStack::new(&mut buffer),
            Default::default(),
        )
        .map_err(|error| match error {
            faer::sparse::linalg::LuError::SymbolicSingular { index } => {
                SolverError::Singular { pivot: index }
            }
            faer::sparse::linalg::LuError::Generic(error) => error.into(),
        })?;
    Ok(NumericInner::Lu(numeric))
}

pub(super) fn solve<T: FactorScalar>(
    symbolic: &SymbolicFactorization,
    numeric: &NumericLu<usize, T>,
    rhs: &mut [T],
    k: usize,
) -> SolverResult<()> {
    let inner = symbolic_of(symbolic);
    let n = inner.nrows();
    let par = par::faer_par();
    let lu = LuRef::new_unchecked(inner, numeric);
    let view = MatMut::from_column_major_slice_mut(rhs, n, k);
    if symbolic.input_transposed {
        // The factor is of Aᵀ (CSR arrays read as CSC): solve (Aᵀ)ᵀ x = b.
        let mut buffer =
            faer::dyn_stack::MemBuffer::new(inner.solve_transpose_in_place_scratch::<T>(k, par));
        lu.solve_transpose_in_place_with_conj(
            Conj::No,
            view,
            par,
            faer::dyn_stack::MemStack::new(&mut buffer),
        );
    } else {
        let mut buffer = faer::dyn_stack::MemBuffer::new(inner.solve_in_place_scratch::<T>(k, par));
        lu.solve_in_place_with_conj(
            Conj::No,
            view,
            par,
            faer::dyn_stack::MemStack::new(&mut buffer),
        );
    }
    if let Some(position) = rhs.iter().position(|value| !value.is_finite()) {
        return Err(SolverError::Singular {
            pivot: position % n.max(1),
        });
    }
    Ok(())
}

/// Solves `Aᵀ X = B` with the factorization of `A`.
pub(super) fn solve_transpose<T: FactorScalar>(
    symbolic: &SymbolicFactorization,
    numeric: &NumericLu<usize, T>,
    rhs: &mut [T],
    k: usize,
) -> SolverResult<()> {
    let inner = symbolic_of(symbolic);
    let n = inner.nrows();
    let par = par::faer_par();
    let lu = LuRef::new_unchecked(inner, numeric);
    let view = MatMut::from_column_major_slice_mut(rhs, n, k);
    if symbolic.input_transposed {
        // The factor is of Aᵀ: a plain solve is a transpose solve of A.
        let mut buffer = faer::dyn_stack::MemBuffer::new(inner.solve_in_place_scratch::<T>(k, par));
        lu.solve_in_place_with_conj(
            Conj::No,
            view,
            par,
            faer::dyn_stack::MemStack::new(&mut buffer),
        );
    } else {
        let mut buffer =
            faer::dyn_stack::MemBuffer::new(inner.solve_transpose_in_place_scratch::<T>(k, par));
        lu.solve_transpose_in_place_with_conj(
            Conj::No,
            view,
            par,
            faer::dyn_stack::MemStack::new(&mut buffer),
        );
    }
    if let Some(position) = rhs.iter().position(|value| !value.is_finite()) {
        return Err(SolverError::Singular {
            pivot: position % n.max(1),
        });
    }
    Ok(())
}
