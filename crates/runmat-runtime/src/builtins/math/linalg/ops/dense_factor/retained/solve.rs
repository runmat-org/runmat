//! Shared immutable dense DATA, fresh RHS/coordinates/StackReq per invocation.
use super::storage::{run_owned, scratch_bytes, DenseScalar, Scratch};
use super::*;
use crate::builtins::math::linalg::ops::sparse_factor::{
    FactorSolveOptions, RetainedSolveStatus, SingularSolvePolicy,
};
use crate::builtins::math::linalg::ops::triangular_operator::TriangularScalar;
use faer::linalg::lu::partial_pivoting::solve as lu_solve;
use faer::linalg::{cholesky::llt::solve as chol, householder as hh};
use faer::prelude::ReborrowMut;
use faer::Conj;
use runmat_value::DenseOrthogonalFactorData;
#[path = "solve/cholesky.rs"]
mod cholesky;
#[path = "solve/lu.rs"]
mod lu;
#[path = "solve/orthogonal.rs"]
mod orthogonal;
#[path = "solve/real_lanes.rs"]
mod real_lanes;
#[path = "solve/scaled_triangle.rs"]
mod scaled_triangle;
use real_lanes::RealLanes;

#[derive(Clone, Copy)]
struct SelectedSolve {
    adjoint: bool,
    columns: usize,
    scale: (f64, f64),
    singular: bool,
    cholesky_stages: Option<((f64, f64), (f64, f64))>,
}

enum Buffer<E: HostTypedElement> {
    Empty,
    Managed(HostTypedWorkspace<E>),
    Unbounded(Vec<E>),
}
impl<E: HostTypedElement> Buffer<E> {
    async fn new(
        len: usize,
        receipt: Option<&HostAllocationReceipt>,
        runtime: &RuntimeContext,
    ) -> Result<Self, DenseFactorError> {
        if len == 0 {
            return Ok(Self::Empty);
        }
        Ok(if let Some(r) = receipt {
            Self::Managed(r.temporary_typed(runtime, len)?)
        } else {
            Self::Unbounded(unbounded_vec(len, Some(runtime)).await?)
        })
    }
    fn get(&self) -> &[E] {
        match self {
            Self::Empty => &[],
            Self::Managed(v) => v.as_slice(),
            Self::Unbounded(v) => v,
        }
    }
    fn get_mut(&mut self) -> &mut [E] {
        match self {
            Self::Empty => &mut [],
            Self::Managed(v) => v.as_mut_slice(),
            Self::Unbounded(v) => v,
        }
    }
}
#[cfg(test)]
pub(in crate::builtins::math::linalg) async fn solve_dense<
    T: DenseScalar + TriangularScalar,
    S: Copy,
>(
    data: &NumericFactorData,
    input: &[S],
    columns: usize,
    options: FactorSolveOptions,
    runtime: &RuntimeContext,
    mut convert: impl FnMut(usize, S) -> Option<T>,
    output: impl FnMut(usize, T) -> Result<(), HostAllocationError>,
) -> Result<RetainedSolveStatus, DenseFactorError> {
    solve_dense_indexed(
        data,
        crate::builtins::math::linalg::ops::sparse_factor::FactorRhsGeometry {
            len: input.len(),
            columns,
        },
        options,
        (1.0, 0.0),
        runtime,
        |index| convert(index, input[index]),
        output,
    )
    .await
}
pub(in crate::builtins::math::linalg) async fn solve_dense_indexed<
    T: DenseScalar + TriangularScalar,
>(
    data: &NumericFactorData,
    geometry: crate::builtins::math::linalg::ops::sparse_factor::FactorRhsGeometry,
    options: FactorSolveOptions,
    scale: (f64, f64),
    runtime: &RuntimeContext,
    mut read: impl FnMut(usize) -> Option<T>,
    mut output: impl FnMut(usize, T) -> Result<(), HostAllocationError>,
) -> Result<RetainedSolveStatus, DenseFactorError> {
    let crate::builtins::math::linalg::ops::sparse_factor::FactorRhsGeometry { len, columns } =
        geometry;
    if !scale.0.is_finite() || !scale.1.is_finite() || scale == (0.0, 0.0) {
        return Err(DenseFactorError::NonFinite);
    }
    let (m, n) = data.dimensions();
    let (input_rows, output_rows) = if options.adjoint { (n, m) } else { (m, n) };
    if input_rows.checked_mul(columns) != Some(len) || output_rows.checked_mul(columns).is_none() {
        return Err(DenseFactorError::InvalidShape);
    }
    let widened_real = scale.1 != 0.0
        && T::COMPLEX
        && matches!(
            data.storage_class(),
            runmat_value::ExactHostSparseClass::Numeric(dtype) if dtype == T::DTYPE
        );
    let real_householder_lanes = widened_real
        && matches!(
            data.parts(),
            NumericFactorCandidate::DenseQr(_) | NumericFactorCandidate::DenseCod { .. }
        );
    if T::storage_class() != data.storage_class() && !widened_real {
        return Err(DenseFactorError::UnsupportedClass);
    }
    if data.kind() == NumericFactorKind::Qr && options.adjoint {
        return Err(DenseFactorError::InvalidShape);
    }
    let rank = match data.parts() {
        NumericFactorCandidate::DenseLu { .. } => n,
        NumericFactorCandidate::DenseCholesky { .. } => n,
        NumericFactorCandidate::DenseQr(first) | NumericFactorCandidate::DenseCod { first, .. } => {
            first.rank
        }
        _ => return Err(DenseFactorError::UnsupportedClass),
    };
    let cholesky_stages = if data.kind() == NumericFactorKind::Cholesky && scale != (1.0, 0.0) {
        Some(
            crate::builtins::math::linalg::ops::sparse_factor::cholesky_stage_scales(
                scale,
                !T::COMPLEX,
            )
            .ok_or(DenseFactorError::NonFinite)?,
        )
    } else {
        None
    };
    if columns == 0 {
        let receipt = HostAllocationReceipt::admit(Some(runtime), HostAllocationPlan::new(0, 0)?)?;
        work::Work::new(Some(runtime), receipt.as_ref())?
            .flush()
            .await?;
        return Ok(RetainedSolveStatus::default());
    }
    if rank == 0 && matches!(data.kind(), NumericFactorKind::Qr | NumericFactorKind::Cod) {
        let receipt = HostAllocationReceipt::admit(Some(runtime), HostAllocationPlan::new(0, 0)?)?;
        let mut work = work::Work::new(Some(runtime), receipt.as_ref())?;
        for index in 0..len {
            if !read(index).is_some_and(|v| {
                crate::builtins::math::linalg::ops::sparse_factor::ManagedSparseScalar::finite(v)
            }) {
                return Err(DenseFactorError::NonFinite);
            }
            work.tick().await?;
        }
        for index in 0..output_rows * columns {
            output(index, T::default())?;
            work.tick().await?;
        }
        work.flush().await?;
        return Ok(RetainedSolveStatus {
            singular: false,
            deficient_rank: (m.min(n) > 0).then_some(0),
        });
    }
    let extent = m.max(n);
    let count = extent
        .checked_mul(columns)
        .ok_or(HostAllocationError::Overflow)?;
    let coordinate_count = if matches!(data.kind(), NumericFactorKind::Qr | NumericFactorKind::Cod)
    {
        rank.checked_mul(columns)
            .ok_or(HostAllocationError::Overflow)?
    } else {
        0
    };
    let req = if real_householder_lanes {
        match T::DTYPE {
            runmat_value::NumericDType::F32 => StackReq::any_of(&[
                solve_req::<T>(data, columns),
                solve_req::<f32>(data, columns),
            ]),
            runmat_value::NumericDType::F64 => StackReq::any_of(&[
                solve_req::<T>(data, columns),
                solve_req::<f64>(data, columns),
            ]),
            _ => return Err(DenseFactorError::UnsupportedClass),
        }
    } else {
        solve_req::<T>(data, columns)
    };
    let rhs_bytes = HostTypedWorkspace::<T::Element>::quoted_bytes(count)?;
    let coordinate_bytes = if coordinate_count == 0 {
        0
    } else {
        HostTypedWorkspace::<T::Element>::quoted_bytes(coordinate_count)?
    };
    let stack_bytes = scratch_bytes(req)?;
    let real_lane_bytes = RealLanes::quoted_bytes(data, real_householder_lanes, count)?;
    let scratch = rhs_bytes
        .checked_add(coordinate_bytes)
        .and_then(|bytes| bytes.checked_add(stack_bytes))
        .and_then(|bytes| bytes.checked_add(real_lane_bytes))
        .ok_or(HostAllocationError::Overflow)?;
    let receipt =
        HostAllocationReceipt::admit(Some(runtime), HostAllocationPlan::new(scratch, 0)?)?;
    let mut work = work::Work::new(Some(runtime), receipt.as_ref())?;
    let mut singular = false;
    if let NumericFactorCandidate::DenseLu { packed, .. } = data.parts() {
        for i in 0..n {
            let diagonal = scaled_triangle::borrowed_coefficient::<T>(packed, i * n + i)
                .ok_or(DenseFactorError::UnsupportedClass)?;
            let oriented = if options.adjoint {
                diagonal.conjugated()
            } else {
                diagonal
            };
            singular |= oriented
                .scaled_complex(scale)
                .ok_or(DenseFactorError::NonFinite)?
                == T::default();
            work.tick().await?;
        }
    }
    if singular && options.singular == SingularSolvePolicy::Reject {
        return Err(DenseFactorError::Singular);
    }
    let status = RetainedSolveStatus {
        singular,
        deficient_rank: (rank < m.min(n)).then_some(rank),
    };
    if columns == 0 {
        work.flush().await?;
        return Ok(status);
    }
    let mut rhs = Buffer::<T::Element>::new(count, receipt.as_ref(), runtime).await?;
    let mut coordinates =
        Buffer::<T::Element>::new(coordinate_count, receipt.as_ref(), runtime).await?;
    let mut real_lanes = RealLanes::new(
        data,
        real_householder_lanes,
        count,
        receipt.as_ref(),
        runtime,
    )
    .await?;
    for i in 0..len {
        let value = read(i)
            .filter(|v| {
                crate::builtins::math::linalg::ops::sparse_factor::ManagedSparseScalar::finite(*v)
            })
            .ok_or(DenseFactorError::NonFinite)?;
        rhs.get_mut()[(i / input_rows) * extent + i % input_rows] = value.to_factor();
        work.tick().await?;
    }
    let mut scratch = Scratch::new(req, receipt.as_ref(), runtime)?;
    work.flush().await?;
    // This clone retains only existing shared buffer/shape owners while the
    // owned CPU job runs; it never copies factor coefficients or refactors.
    let retained = data.clone();
    let (rhs, solved) = run_owned(runtime, receipt.as_ref(), move || {
        let solved = solve_body::<T>(
            &retained,
            SelectedSolve {
                adjoint: options.adjoint,
                columns,
                scale,
                singular,
                cholesky_stages,
            },
            rhs.get_mut(),
            coordinates.get_mut(),
            &mut real_lanes,
            scratch.bytes(),
        );
        (rhs, solved)
    })
    .await?;
    solved?;
    let allow_ieee = singular && options.singular == SingularSolvePolicy::PublicDecompositionIeee;
    for column in 0..columns {
        for row in 0..output_rows {
            let value = rhs.get()[column * extent + row];
            if !allow_ieee && !<T as FaerFactorScalar>::finite(value) {
                return Err(DenseFactorError::NonFinite);
            }
            output(column * output_rows + row, T::from_element(value))?;
            work.tick().await?;
        }
    }
    work.flush().await?;
    Ok(status)
}
fn solve_req<T: DenseScalar>(data: &NumericFactorData, columns: usize) -> StackReq {
    let (m, n) = data.dimensions();
    match data.parts() {
        NumericFactorCandidate::DenseLu { .. } => StackReq::any_of(&[
            lu_solve::solve_in_place_scratch::<usize, T::Element>(n, columns, Par::Seq),
            lu_solve::solve_transpose_in_place_scratch::<usize, T::Element>(n, columns, Par::Seq),
        ]),
        NumericFactorCandidate::DenseCholesky { .. } => {
            chol::solve_in_place_scratch::<T::Element>(n, columns, Par::Seq)
        }
        NumericFactorCandidate::DenseQr(first) | NumericFactorCandidate::DenseCod { first, .. } => {
            let block = first.coefficients.dimensions().0;
            let a = StackReq::any_of(&[
                hh::apply_block_householder_sequence_on_the_left_in_place_scratch::<T::Element>(
                    m, block, columns,
                ),
                faer::perm::permute_rows_in_place_scratch::<usize, T::Element>(n, columns),
            ]);
            if let NumericFactorCandidate::DenseCod {
                completion: Some(second),
                ..
            } = data.parts()
            {
                StackReq::any_of(&[
                    a,
                    hh::apply_block_householder_sequence_on_the_left_in_place_scratch::<T::Element>(
                        n,
                        second.coefficients.dimensions().0,
                        columns,
                    ),
                ])
            } else {
                a
            }
        }
        _ => StackReq::EMPTY,
    }
}
fn packed<T: DenseScalar>(m: &NativeDenseFactorMatrix) -> MatRef<'_, T::Element> {
    let (rows, cols) = m.dimensions();
    MatRef::from_column_major_slice(
        T::faer(T::elements(m).expect("validated dense native class")),
        rows,
        cols,
    )
}
fn apply<T: DenseScalar>(
    qr: &DenseOrthogonalFactorData,
    adjoint: bool,
    rhs: MatMut<'_, T::Element>,
    stack: &mut MemStack,
) {
    if qr.rank == 0 {
        return;
    }
    // Truncation is on complete block coefficient columns, with the partial
    // final block interpreted by faer's sequence owner; never sparse tau.
    let basis = packed::<T>(&qr.packed).subcols(0, qr.rank);
    let coeff = packed::<T>(&qr.coefficients).subcols(0, qr.rank);
    if adjoint {
        hh::apply_block_householder_sequence_transpose_on_the_left_in_place_with_conj(
            basis,
            coeff,
            Conj::Yes,
            rhs,
            Par::Seq,
            stack,
        );
    } else {
        hh::apply_block_householder_sequence_on_the_left_in_place_with_conj(
            basis,
            coeff,
            Conj::No,
            rhs,
            Par::Seq,
            stack,
        );
    }
}
fn solve_body<T: DenseScalar + TriangularScalar>(
    data: &NumericFactorData,
    selected: SelectedSolve,
    rhs: &mut [T::Element],
    coordinates: &mut [T::Element],
    real_lanes: &mut RealLanes,
    scratch: &mut [std::mem::MaybeUninit<u8>],
) -> Result<(), DenseFactorError> {
    let (m, n) = data.dimensions();
    let extent = m.max(n);
    let rhs = MatMut::from_column_major_slice_mut(rhs, extent, selected.columns);
    let mut stack = MemStack::new(scratch);
    match data.parts() {
        NumericFactorCandidate::DenseLu { .. } => lu::run::<T>(data, selected, rhs, stack.rb_mut()),
        NumericFactorCandidate::DenseCholesky { .. } => {
            cholesky::run::<T>(data, selected, rhs, stack.rb_mut())
        }
        NumericFactorCandidate::DenseQr(_) | NumericFactorCandidate::DenseCod { .. } => {
            orthogonal::run::<T>(data, selected, rhs, coordinates, real_lanes, stack.rb_mut())
        }
        _ => unreachable!("validated dense factor"),
    }
}
