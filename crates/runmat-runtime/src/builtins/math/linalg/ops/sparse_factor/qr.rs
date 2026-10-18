//! Reusable tall-CSC QR factor for the ordinary host least-squares route.
//! faer's symbolic and numeric fill remains privately allocated, so this is
//! deliberately not a Managed factor owner.

use faer::prelude::SolveLstsq;
use faer::sparse::linalg::solvers::Qr;
use faer::{c32, c64, MatMut};
use runmat_value::SparseTensor;

use crate::builtins::math::linalg::ops::sparse_operator::{
    SparseLinearOperator, SparseOperatorError,
};
use crate::context::host_allocation::{
    HostAllocationError, HostAllocationPolicy, HostAllocationReceipt,
};
use crate::context::RuntimeContext;

use super::faer_adapter::borrowed_csc;

mod input;
use input::{copy_complex, validate_finite};

#[derive(Debug)]
pub enum SparseQrError {
    Admission(HostAllocationError),
    Operator(SparseOperatorError),
    NotTall,
    DimensionMismatch,
    TemporaryAllocationFailed,
    NonFiniteInput,
    NumericalBreakdown,
    Library(faer::sparse::FaerError),
}

impl From<HostAllocationError> for SparseQrError {
    fn from(error: HostAllocationError) -> Self {
        Self::Admission(error)
    }
}

impl From<SparseOperatorError> for SparseQrError {
    fn from(error: SparseOperatorError) -> Self {
        Self::Operator(error)
    }
}

fn require_unbounded(context: &RuntimeContext) -> Result<(), SparseQrError> {
    match HostAllocationReceipt::policy(Some(context))? {
        HostAllocationPolicy::Unbounded => Ok(()),
        HostAllocationPolicy::Managed => Err(SparseQrError::Admission(
            HostAllocationError::MissingManagedOwner,
        )),
    }
}

pub enum SparseQrFactor {
    RealF32 {
        rows: usize,
        cols: usize,
        factor: Qr<usize, f32>,
    },
    RealF64 {
        rows: usize,
        cols: usize,
        factor: Qr<usize, f64>,
    },
    ComplexF32 {
        rows: usize,
        cols: usize,
        factor: Qr<usize, c32>,
    },
    ComplexF64 {
        rows: usize,
        cols: usize,
        factor: Qr<usize, c64>,
    },
}

impl SparseQrFactor {
    pub fn factor_unbounded(
        sparse: &SparseTensor,
        context: &RuntimeContext,
    ) -> Result<Self, SparseQrError> {
        require_unbounded(context)?;
        let operator = SparseLinearOperator::borrow_with_context(sparse, context)?;
        let (rows, cols) = (operator.rows(), operator.cols());
        if rows <= cols || cols == 0 {
            return Err(SparseQrError::NotTall);
        }
        let factor = match operator {
            SparseLinearOperator::RealF32(view) => {
                validate_finite(view.as_slice(), context, |value| value.is_finite())?;
                require_unbounded(context)?;
                Self::RealF32 {
                    rows,
                    cols,
                    factor: borrowed_csc(sparse, view.as_slice())
                        .sp_qr()
                        .map_err(SparseQrError::Library)?,
                }
            }
            SparseLinearOperator::RealF64(view) => {
                validate_finite(view.as_slice(), context, |value| value.is_finite())?;
                require_unbounded(context)?;
                Self::RealF64 {
                    rows,
                    cols,
                    factor: borrowed_csc(sparse, view.as_slice())
                        .sp_qr()
                        .map_err(SparseQrError::Library)?,
                }
            }
            SparseLinearOperator::ComplexF32(view) => {
                validate_finite(view.as_slice(), context, |value| {
                    value.0.is_finite() && value.1.is_finite()
                })?;
                let converted =
                    copy_complex(view.as_slice(), context, |value| c32::new(value.0, value.1))?;
                require_unbounded(context)?;
                Self::ComplexF32 {
                    rows,
                    cols,
                    factor: borrowed_csc(sparse, &converted)
                        .sp_qr()
                        .map_err(SparseQrError::Library)?,
                }
            }
            SparseLinearOperator::ComplexF64(view) => {
                validate_finite(view.as_slice(), context, |value| {
                    value.0.is_finite() && value.1.is_finite()
                })?;
                let converted =
                    copy_complex(view.as_slice(), context, |value| c64::new(value.0, value.1))?;
                require_unbounded(context)?;
                Self::ComplexF64 {
                    rows,
                    cols,
                    factor: borrowed_csc(sparse, &converted)
                        .sp_qr()
                        .map_err(SparseQrError::Library)?,
                }
            }
        };
        require_unbounded(context)?;
        Ok(factor)
    }
}

macro_rules! solve_native {
    ($method:ident, $scalar:ty, $variant:ident, $finite:expr) => {
        impl SparseQrFactor {
            /// `rhs` has `rows * columns` entries. The first `cols` entries of
            /// each right-hand column contain the least-squares solution.
            pub fn $method(
                &self,
                rhs: &mut [$scalar],
                columns: usize,
                context: &RuntimeContext,
            ) -> Result<usize, SparseQrError> {
                let Self::$variant { rows, cols, factor } = self else {
                    return Err(SparseQrError::DimensionMismatch);
                };
                if rows.checked_mul(columns) != Some(rhs.len()) {
                    return Err(SparseQrError::DimensionMismatch);
                }
                require_unbounded(context)?;
                factor
                    .solve_lstsq_in_place(MatMut::from_column_major_slice_mut(rhs, *rows, columns));
                require_unbounded(context)?;
                for column in rhs.chunks(*rows) {
                    for chunk in column[..*cols].chunks(1024) {
                        require_unbounded(context)?;
                        if !chunk.iter().all($finite) {
                            return Err(SparseQrError::NumericalBreakdown);
                        }
                    }
                }
                Ok(*cols)
            }
        }
    };
}

solve_native!(solve_real_f32, f32, RealF32, |value: &f32| value
    .is_finite());
solve_native!(solve_real_f64, f64, RealF64, |value: &f64| value
    .is_finite());
solve_native!(solve_complex_f32, c32, ComplexF32, |value: &c32| value
    .re
    .is_finite()
    && value.im.is_finite());
solve_native!(solve_complex_f64, c64, ComplexF64, |value: &c64| value
    .re
    .is_finite()
    && value.im.is_finite());
