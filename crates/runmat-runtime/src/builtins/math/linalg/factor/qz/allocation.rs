//! Checked QZ geometry against the installed managed host-allocation ledger.
//! faer's pinned 0.24.4 dense Mat pads each column to a 64-byte boundary;
//! its own `gevd_scratch` supplies the algorithm's stack requirement.

use faer::linalg::evd::ComputeEigenvectors;
use faer::linalg::gevd::{self, GevdParams};
use faer::{c64, Par, Spec};
use runmat_value::NumericDType;

use crate::context::host_allocation::{HostAllocationError, HostAllocationPlan};

use super::Mode;

fn add(a: u64, b: u64) -> Result<u64, HostAllocationError> {
    a.checked_add(b).ok_or(HostAllocationError::Overflow)
}

fn times(a: u64, b: usize) -> Result<u64, HostAllocationError> {
    a.checked_mul(u64::try_from(b).map_err(|_| HostAllocationError::Overflow)?)
        .ok_or(HostAllocationError::Overflow)
}

fn usize_bytes<T>(count: usize) -> Result<u64, HostAllocationError> {
    let layout =
        std::alloc::Layout::array::<T>(count).map_err(|_| HostAllocationError::Overflow)?;
    u64::try_from(layout.size()).map_err(|_| HostAllocationError::Overflow)
}

/// Exact payload capacity of one faer 0.24.4 column, including its 64-byte
/// SIMD row-padding rule for the f64/c64 fields used here.
fn column_bytes<T>(n: usize) -> Result<u64, HostAllocationError> {
    let element = std::mem::size_of::<T>();
    let per_column = n
        .checked_mul(element)
        .ok_or(HostAllocationError::Overflow)?;
    let padded = per_column
        .checked_add(63)
        .ok_or(HostAllocationError::Overflow)?
        / 64
        * 64;
    std::alloc::Layout::from_size_align(padded, 64).map_err(|_| HostAllocationError::Overflow)?;
    u64::try_from(padded).map_err(|_| HostAllocationError::Overflow)
}

fn matrix_bytes<T>(n: usize) -> Result<u64, HostAllocationError> {
    let bytes = times(column_bytes::<T>(n)?, n)?;
    let size = usize::try_from(bytes).map_err(|_| HostAllocationError::Overflow)?;
    std::alloc::Layout::from_size_align(size, 64).map_err(|_| HostAllocationError::Overflow)?;
    Ok(bytes)
}

fn qz_stack(n: usize, mode: Mode) -> Result<u64, HostAllocationError> {
    let req = match mode {
        Mode::Real => {
            let params: GevdParams = faer::auto!(f64);
            gevd::gevd_scratch::<f64>(
                n,
                ComputeEigenvectors::Yes,
                ComputeEigenvectors::Yes,
                Par::Seq,
                Spec::new(params),
            )
        }
        Mode::Complex => {
            let params: GevdParams = faer::auto!(c64);
            gevd::gevd_scratch::<c64>(
                n,
                ComputeEigenvectors::Yes,
                ComputeEigenvectors::Yes,
                Par::Seq,
                Spec::new(params),
            )
        }
    };
    let layout = req.layout().map_err(|_| HostAllocationError::Overflow)?;
    u64::try_from(layout.size()).map_err(|_| HostAllocationError::Overflow)
}

fn local_real_qz_stack(n: usize) -> Result<u64, HostAllocationError> {
    let params: GevdParams = faer::auto!(f64);
    let req = gevd::qz_real::hessenberg_to_qz_scratch::<f64>(n, Par::Seq, params.schur);
    let layout = req.layout().map_err(|_| HostAllocationError::Overflow)?;
    u64::try_from(layout.size()).map_err(|_| HostAllocationError::Overflow)
}

fn output_bytes(n: usize, dtype: NumericDType, mode: Mode) -> Result<u64, HostAllocationError> {
    let count = n.checked_mul(n).ok_or(HostAllocationError::Overflow)?;
    match mode {
        Mode::Real => HostAllocationPlan::retained_tensor_bytes_for_dimensions(dtype, count, 2),
        Mode::Complex => {
            HostAllocationPlan::retained_complex_tensor_bytes_for_dimensions(dtype, count, 2)
        }
    }
}

/// Count simultaneously live matrices and temporary column vectors at each
/// stage, then reserve output headroom separately. No OS-memory estimate is
/// converted into a finite quota by this calculation.
pub(super) fn factor_plan(
    n: usize,
    dtype: NumericDType,
    mode: Mode,
    requested: usize,
) -> Result<HostAllocationPlan, HostAllocationError> {
    let vectors = requested.saturating_sub(4).min(2);
    let matrix = match mode {
        Mode::Real => matrix_bytes::<f64>(n)?,
        Mode::Complex => matrix_bytes::<c64>(n)?,
    };
    // AA, BB, left, right; requested eigenvector matrices are built while
    // all four remain live. Column solves retain at most four n-element c64
    // vectors and the real block table retains n pairs of indices.
    let matrices = times(matrix, 4 + vectors)?;
    let columns = match mode {
        Mode::Real => times(column_bytes::<f64>(n)?, 3)?,
        Mode::Complex => times(column_bytes::<c64>(n)?, 2)?,
    };
    let vector_scratch = add(
        times(usize_bytes::<c64>(n)?, 4)?,
        usize_bytes::<(usize, usize)>(n)?,
    )?;
    let publication_metadata = add(
        usize_bytes::<runmat_value::Value>(requested)?,
        times(usize_bytes::<usize>(2)?, requested)?,
    )?;
    let scratch = add(
        add(matrices, columns)?,
        add(
            add(qz_stack(n, mode)?, vector_scratch)?,
            publication_metadata,
        )?,
    )?;
    HostAllocationPlan::new(scratch, times(output_bytes(n, dtype, mode)?, requested)?)
}

pub(super) fn reorder_plan(
    n: usize,
    dtype: NumericDType,
    mode: Mode,
    requested: usize,
) -> Result<HostAllocationPlan, HostAllocationError> {
    let matrix = match mode {
        Mode::Real => matrix_bytes::<f64>(n)?,
        Mode::Complex => matrix_bytes::<c64>(n)?,
    };
    // Four supplied factors and the transformed left factor coexist. Real
    // adjacent exchanges also run a local (at most 4-by-4) QZ and retain its
    // two-sided factors and stack; this is constant geometry, not n^3.
    let local = if mode == Mode::Real {
        // Two 4-by-4 exchange bases, four retained 2-by-2 QZ factors,
        // and the bounded 8-equation Sylvester system with its row headers,
        // right-hand side, solution, and split invariant-subspace vectors.
        let bases = times(matrix_bytes::<f64>(4)?, 2)?;
        let reduced = times(matrix_bytes::<f64>(2)?, 4)?;
        let system = add(
            times(usize_bytes::<f64>(8)?, 8)?,
            usize_bytes::<Vec<f64>>(8)?,
        )?;
        let vectors = times(usize_bytes::<f64>(8)?, 4)?;
        add(
            add(add(bases, reduced)?, add(system, vectors)?)?,
            qz_stack(2, Mode::Real)?,
        )?
    } else {
        0
    };
    let labels = add(
        add(usize_bytes::<u64>(n)?, usize_bytes::<(c64, c64)>(n)?)?,
        times(usize_bytes::<(usize, usize)>(n)?, 2)?,
    )?;
    let real_columns = if mode == Mode::Real {
        times(column_bytes::<f64>(n)?, 3)?
    } else {
        times(column_bytes::<c64>(n)?, 2)?
    };
    let publication_metadata = add(
        usize_bytes::<runmat_value::Value>(requested)?,
        times(usize_bytes::<usize>(2)?, requested)?,
    )?;
    let scratch = add(
        add(times(matrix, 5)?, local)?,
        add(add(labels, real_columns)?, publication_metadata)?,
    )?;
    HostAllocationPlan::new(scratch, times(output_bytes(n, dtype, mode)?, requested)?)
}

pub(super) fn eigenvalues_plan(
    n: usize,
    dtype: NumericDType,
    mode: Mode,
    requested: bool,
) -> Result<HostAllocationPlan, HostAllocationError> {
    let matrix = match mode {
        Mode::Real => matrix_bytes::<f64>(n)?,
        Mode::Complex => matrix_bytes::<c64>(n)?,
    };
    let local = if mode == Mode::Real {
        add(times(matrix_bytes::<f64>(2)?, 2)?, local_real_qz_stack(2)?)?
    } else {
        0
    };
    let pairs = times(usize_bytes::<(c64, c64)>(n)?, 2)?;
    let small_columns = if mode == Mode::Real {
        times(column_bytes::<f64>(2)?, 3)?
    } else {
        0
    };
    let scratch = add(add(times(matrix, 2)?, local)?, add(pairs, small_columns)?)?;
    let output = if requested {
        // A real quasi-triangular block may still have complex eigenvalues.
        HostAllocationPlan::retained_complex_tensor_bytes_for_dimensions(dtype, n, 2)?
    } else {
        0
    };
    HostAllocationPlan::new(scratch, output)
}
