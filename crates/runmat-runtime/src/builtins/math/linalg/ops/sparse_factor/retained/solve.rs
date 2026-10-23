//! Invocation-local execution over immutable native factor DATA.
use super::super::managed_lu::FactorSolveOptions;
use super::super::managed_storage::{ClaimedVec, Work};
use super::super::{ManagedSparseCholeskyError, ManagedSparseLuError, SparseCompleteQrError};
use super::{RetainedFactorError, RetainedFactorScalar};
use crate::context::host_allocation::HostAllocationError;
use runmat_value::{NumericFactorCandidate, NumericFactorData};
#[derive(Clone, Copy, Debug, Default)]
pub(in crate::builtins::math::linalg) struct RetainedSolveStatus {
    pub singular: bool,
    pub deficient_rank: Option<usize>,
}

fn qr_view<T: RetainedFactorScalar>(
    data: &runmat_value::OrthogonalFactorData,
) -> Result<super::super::managed_qr::BorrowedQr<'_, T>, RetainedFactorError> {
    use super::super::managed_qr::{BorrowedCsc, BorrowedQr};
    Ok(BorrowedQr {
        rows: data.r.rows,
        columns: data.r.cols,
        rank: data.rank,
        r: BorrowedCsc {
            pointers: &data.r.col_ptrs,
            rows: &data.r.row_indices,
            values: T::borrowed(&data.r).ok_or(RetainedFactorError::UnsupportedClass)?,
        },
        reflectors: BorrowedCsc {
            pointers: &data.reflectors.col_ptrs,
            rows: &data.reflectors.row_indices,
            values: T::borrowed(&data.reflectors).ok_or(RetainedFactorError::UnsupportedClass)?,
        },
        permutation: &data.permutation.forward,
    })
}

/// Existing immutable DATA can be read after the factor-construction context
/// ends. Only the current caller admits, controls, and owns solve scratch and
/// output. The sink is private and receives at most one write per result lane.
#[cfg(test)]
pub(in crate::builtins::math::linalg) async fn solve_factor_native<
    T: RetainedFactorScalar,
    S: Copy,
>(
    data: &NumericFactorData,
    input: &[S],
    columns: usize,
    options: FactorSolveOptions,
    runtime: &crate::context::RuntimeContext,
    mut convert: impl FnMut(usize, S) -> Option<T>,
    output: impl FnMut(usize, T) -> Result<(), HostAllocationError>,
) -> Result<RetainedSolveStatus, RetainedFactorError> {
    solve_factor_indexed(
        data,
        super::super::managed_lu::FactorRhsGeometry {
            len: input.len(),
            columns,
        },
        options,
        runtime,
        |index| convert(index, input[index]),
        output,
    )
    .await
}
pub(in crate::builtins::math::linalg) async fn solve_factor_indexed<T: RetainedFactorScalar>(
    data: &NumericFactorData,
    geometry: super::super::managed_lu::FactorRhsGeometry,
    options: FactorSolveOptions,
    runtime: &crate::context::RuntimeContext,
    read: impl FnMut(usize) -> Option<T>,
    output: impl FnMut(usize, T) -> Result<(), HostAllocationError>,
) -> Result<RetainedSolveStatus, RetainedFactorError> {
    if data.storage_class() != T::storage_class() {
        return Err(RetainedFactorError::UnsupportedClass);
    }
    solve_factor_indexed_scaled::<T>(data, geometry, options, (1.0, 0.0), runtime, read, output)
        .await
}

pub(in crate::builtins::math::linalg) async fn solve_factor_indexed_scaled<
    T: RetainedFactorScalar,
>(
    data: &NumericFactorData,
    geometry: super::super::managed_lu::FactorRhsGeometry,
    options: FactorSolveOptions,
    scale: (f64, f64),
    runtime: &crate::context::RuntimeContext,
    read: impl FnMut(usize) -> Option<T>,
    output: impl FnMut(usize, T) -> Result<(), HostAllocationError>,
) -> Result<RetainedSolveStatus, RetainedFactorError> {
    use runmat_value::{ComplexElement, ExactHostSparseClass as C, NumericDType as D};
    let source = data.storage_class();
    let target = T::storage_class();
    if source != target
        && !matches!(
            (source, target),
            (C::Numeric(D::F32), C::Complex(D::F32)) | (C::Numeric(D::F64), C::Complex(D::F64))
        )
    {
        return Err(RetainedFactorError::UnsupportedClass);
    }
    if !scale.0.is_finite()
        || !scale.1.is_finite()
        || (scale == (0.0, 0.0) && !matches!(data.parts(), NumericFactorCandidate::Empty { .. }))
    {
        return Err(RetainedFactorError::Invalid(
            "scaled solve requires a finite nonzero scalar".into(),
        ));
    }
    match source {
        C::Numeric(D::F32) => {
            solve_typed::<f32, T>(data, geometry, options, scale, runtime, read, output).await
        }
        C::Numeric(D::F64) => {
            solve_typed::<f64, T>(data, geometry, options, scale, runtime, read, output).await
        }
        C::Complex(D::F32) => {
            solve_typed::<ComplexElement<f32>, T>(
                data, geometry, options, scale, runtime, read, output,
            )
            .await
        }
        C::Complex(D::F64) => {
            solve_typed::<ComplexElement<f64>, T>(
                data, geometry, options, scale, runtime, read, output,
            )
            .await
        }
        _ => Err(RetainedFactorError::UnsupportedClass),
    }
}
async fn solve_typed<S: RetainedFactorScalar, T: RetainedFactorScalar>(
    data: &NumericFactorData,
    geometry: super::super::managed_lu::FactorRhsGeometry,
    options: FactorSolveOptions,
    scale: (f64, f64),
    runtime: &crate::context::RuntimeContext,
    mut read: impl FnMut(usize) -> Option<T>,
    mut output: impl FnMut(usize, T) -> Result<(), HostAllocationError>,
) -> Result<RetainedSolveStatus, RetainedFactorError> {
    let super::super::managed_lu::FactorRhsGeometry { len, columns } = geometry;
    use super::super::{
        managed_cholesky::BorrowedCholesky, managed_lu::BorrowedLu, managed_qr::BorrowedCompleteQr,
    };
    use crate::context::host_allocation::{HostAllocationPlan, HostAllocationReceipt};
    let (m, n) = data.dimensions();
    let adjoint = options.adjoint;
    if data.storage_class() != S::storage_class() {
        return Err(RetainedFactorError::UnsupportedClass);
    }
    if data.kind() == runmat_value::NumericFactorKind::Qr && adjoint {
        return Err(RetainedFactorError::Invalid(
            "basic QR does not support conjugate-transpose solves; use COD".into(),
        ));
    }
    let (input_rows, output_rows) = if adjoint { (n, m) } else { (m, n) };
    if input_rows.checked_mul(columns) != Some(len) || output_rows.checked_mul(columns).is_none() {
        return Err(RetainedFactorError::Invalid(
            "retained solve right-hand-side extent mismatch".into(),
        ));
    }
    let receipt = HostAllocationReceipt::admit(Some(runtime), HostAllocationPlan::new(0, 0)?)?;
    let mut work = Work::for_policy(runtime, receipt.as_ref())?;
    work.flush().await?;
    let mut status = RetainedSolveStatus::default();
    if columns == 0 {
        return Ok(status);
    }
    match data.parts() {
        NumericFactorCandidate::DenseLu { .. }
        | NumericFactorCandidate::DenseCholesky { .. }
        | NumericFactorCandidate::DenseQr(_)
        | NumericFactorCandidate::DenseCod { .. } => {
            return Err(RetainedFactorError::UnsupportedClass);
        }
        NumericFactorCandidate::Empty { kind, .. } => {
            if *kind == runmat_value::NumericFactorKind::Qr && adjoint {
                return Err(RetainedFactorError::Invalid(
                    "basic QR does not support conjugate-transpose solves; use COD".into(),
                ));
            }
            for index in 0..len {
                if !read(index).is_some_and(|value| value.finite()) {
                    return Err(RetainedFactorError::Invalid(
                        "retained empty solve requires finite right-hand sides".into(),
                    ));
                }
                work.tick().await?;
            }
            for index in 0..output_rows * columns {
                output(index, T::default())?;
                work.tick().await?;
            }
        }
        NumericFactorCandidate::Lu {
            lower,
            upper,
            rows,
            columns: permutation,
            row_scale,
        } => {
            let mut rhs = ClaimedVec::<T>::zeroed(n, &mut work).await?;
            status.singular = BorrowedLu {
                order: n,
                l_ptr: &lower.col_ptrs,
                l_rows: &lower.row_indices,
                l_values: S::borrowed(lower).ok_or(RetainedFactorError::UnsupportedClass)?,
                u_ptr: &upper.col_ptrs,
                u_rows: &upper.row_indices,
                u_values: S::borrowed(upper).ok_or(RetainedFactorError::UnsupportedClass)?,
                row_perm: &rows.forward,
                col_perm: &permutation.forward,
                row_scale: row_scale.as_ref(),
            }
            .solve_indexed_scaled(
                geometry,
                options,
                scale,
                rhs.as_mut_slice(),
                read,
                |index, value| output(index, value).map_err(ManagedSparseLuError::Admission),
                &mut work,
            )
            .await?;
        }
        NumericFactorCandidate::Cholesky { lower, permutation } => {
            let mut rhs = ClaimedVec::<T>::zeroed(n, &mut work).await?;
            BorrowedCholesky {
                order: n,
                pointers: &lower.col_ptrs,
                rows: &lower.row_indices,
                values: S::borrowed(lower).ok_or(RetainedFactorError::UnsupportedClass)?,
                permutation: &permutation.forward,
            }
            .solve_indexed_scaled(
                geometry,
                scale,
                rhs.as_mut_slice(),
                read,
                |index, value| output(index, value).map_err(ManagedSparseCholeskyError::Admission),
                &mut work,
            )
            .await?;
        }
        NumericFactorCandidate::Qr(first) => {
            status.deficient_rank = (first.rank < m.min(n)).then_some(first.rank);
            if adjoint {
                return Err(RetainedFactorError::Invalid(
                    "basic QR does not support conjugate-transpose solves; use COD".into(),
                ));
            }
            BorrowedCompleteQr {
                first: qr_view::<S>(first)?,
                completion: None,
            }
            .solve_indexed_scaled(
                geometry,
                false,
                scale,
                read,
                |index, value| output(index, value).map_err(SparseCompleteQrError::Admission),
                &mut work,
            )
            .await?;
        }
        NumericFactorCandidate::Cod { first, completion } => {
            status.deficient_rank = (first.rank < m.min(n)).then_some(first.rank);
            BorrowedCompleteQr {
                first: qr_view::<S>(first)?,
                completion: completion.as_ref().map(qr_view::<S>).transpose()?,
            }
            .solve_indexed_scaled(
                geometry,
                adjoint,
                scale,
                read,
                |index, value| output(index, value).map_err(SparseCompleteQrError::Admission),
                &mut work,
            )
            .await?;
        }
    }
    work.flush().await?;
    Ok(status)
}
