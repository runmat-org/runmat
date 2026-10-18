use faer::prelude::Solve;
use faer::sparse::linalg::{solvers::Lu, LuError};
use faer::{c32, c64, MatMut};
use runmat_value::{ComplexElement, SparseTensor};

use crate::builtins::math::linalg::ops::sparse_operator::{
    SparseLinearOperator, SparseOperatorError,
};
use crate::context::host_allocation::{
    HostAllocationError, HostAllocationPolicy, HostAllocationReceipt,
};
use crate::context::RuntimeContext;

use super::faer_adapter::borrowed_csc;

#[derive(Debug)]
pub enum SparseLuError {
    Admission(HostAllocationError),
    Operator(SparseOperatorError),
    NotSquare,
    DimensionMismatch,
    TemporaryAllocationFailed,
    Singular { column: usize },
    NonFiniteInput,
    NumericalBreakdown,
    Library(faer::sparse::FaerError),
}

impl From<HostAllocationError> for SparseLuError {
    fn from(error: HostAllocationError) -> Self {
        Self::Admission(error)
    }
}

impl From<SparseOperatorError> for SparseLuError {
    fn from(error: SparseOperatorError) -> Self {
        Self::Operator(error)
    }
}

impl From<LuError> for SparseLuError {
    fn from(error: LuError) -> Self {
        match error {
            LuError::SymbolicSingular { index } => Self::Singular { column: index },
            LuError::Generic(error) => Self::Library(error),
        }
    }
}

/// One reusable numeric factor. The faer-owned symbolic/fill storage has no
/// Managed claim, so construction and every solve are gated Unbounded.
pub enum SparseLuFactor {
    RealF32 {
        order: usize,
        factor: Lu<usize, f32>,
    },
    RealF64 {
        order: usize,
        factor: Lu<usize, f64>,
    },
    ComplexF32 {
        order: usize,
        factor: Lu<usize, c32>,
    },
    ComplexF64 {
        order: usize,
        factor: Lu<usize, c64>,
    },
}

fn require_unbounded(context: &RuntimeContext) -> Result<(), SparseLuError> {
    match HostAllocationReceipt::policy(Some(context))? {
        HostAllocationPolicy::Unbounded => Ok(()),
        HostAllocationPolicy::Managed => Err(SparseLuError::Admission(
            HostAllocationError::MissingManagedOwner,
        )),
    }
}

fn copy_complex<T: Copy, U>(
    values: &[ComplexElement<T>],
    context: &RuntimeContext,
    mut convert: impl FnMut(ComplexElement<T>) -> U,
) -> Result<Vec<U>, SparseLuError> {
    let mut result = Vec::new();
    result
        .try_reserve_exact(values.len())
        .map_err(|_| SparseLuError::TemporaryAllocationFailed)?;
    for chunk in values.chunks(1024) {
        require_unbounded(context)?;
        result.extend(chunk.iter().copied().map(&mut convert));
    }
    Ok(result)
}

fn validate_finite<T>(
    values: &[T],
    context: &RuntimeContext,
    finite: impl Fn(&T) -> bool,
) -> Result<(), SparseLuError> {
    for chunk in values.chunks(1024) {
        require_unbounded(context)?;
        if !chunk.iter().all(&finite) {
            return Err(SparseLuError::NonFiniteInput);
        }
    }
    Ok(())
}

impl SparseLuFactor {
    pub fn factor_unbounded(
        sparse: &SparseTensor,
        context: &RuntimeContext,
    ) -> Result<Self, SparseLuError> {
        // This precedes the complex conversion and faer's private allocations.
        require_unbounded(context)?;
        let operator = SparseLinearOperator::borrow_with_context(sparse, context)?;
        if operator.rows() != operator.cols() {
            return Err(SparseLuError::NotSquare);
        }
        let order = operator.rows();
        let factor = match operator {
            SparseLinearOperator::RealF32(view) => {
                validate_finite(view.as_slice(), context, |value| value.is_finite())?;
                require_unbounded(context)?;
                Self::RealF32 {
                    order,
                    factor: borrowed_csc(
                        sparse,
                        sparse.as_f32_slice().ok_or(SparseLuError::Operator(
                            SparseOperatorError::UnsupportedClass,
                        ))?,
                    )
                    .sp_lu()?,
                }
            }
            SparseLinearOperator::RealF64(view) => {
                validate_finite(view.as_slice(), context, |value| value.is_finite())?;
                require_unbounded(context)?;
                Self::RealF64 {
                    order,
                    factor: borrowed_csc(
                        sparse,
                        sparse.as_f64_slice().ok_or(SparseLuError::Operator(
                            SparseOperatorError::UnsupportedClass,
                        ))?,
                    )
                    .sp_lu()?,
                }
            }
            SparseLinearOperator::ComplexF32(view) => {
                validate_finite(view.as_slice(), context, |value| {
                    value.0.is_finite() && value.1.is_finite()
                })?;
                let converted = copy_complex(
                    sparse
                        .as_complex_f32_slice()
                        .ok_or(SparseLuError::Operator(
                            SparseOperatorError::UnsupportedClass,
                        ))?,
                    context,
                    |value| c32::new(value.0, value.1),
                )?;
                require_unbounded(context)?;
                Self::ComplexF32 {
                    order,
                    factor: borrowed_csc(sparse, &converted).sp_lu()?,
                }
            }
            SparseLinearOperator::ComplexF64(view) => {
                validate_finite(view.as_slice(), context, |value| {
                    value.0.is_finite() && value.1.is_finite()
                })?;
                let converted = copy_complex(
                    sparse
                        .as_complex_f64_slice()
                        .ok_or(SparseLuError::Operator(
                            SparseOperatorError::UnsupportedClass,
                        ))?,
                    context,
                    |value| c64::new(value.0, value.1),
                )?;
                require_unbounded(context)?;
                Self::ComplexF64 {
                    order,
                    factor: borrowed_csc(sparse, &converted).sp_lu()?,
                }
            }
        };
        require_unbounded(context)?;
        Ok(factor)
    }
}

macro_rules! solve_native {
    ($method:ident, $scalar:ty, $variant:ident, $finite:expr) => {
        impl SparseLuFactor {
            pub fn $method(
                &self,
                rhs: &mut [$scalar],
                columns: usize,
                context: &RuntimeContext,
            ) -> Result<(), SparseLuError> {
                let Self::$variant { order, factor } = self else {
                    return Err(SparseLuError::DimensionMismatch);
                };
                if order.checked_mul(columns) != Some(rhs.len()) {
                    return Err(SparseLuError::DimensionMismatch);
                }
                require_unbounded(context)?;
                factor.solve_in_place(MatMut::from_column_major_slice_mut(rhs, *order, columns));
                require_unbounded(context)?;
                // faer reports missing structural pivots, but its numeric LU
                // may divide by a zero pivot and return non-finite entries.
                // Never publish such a solve as a successful finite result.
                for chunk in rhs.chunks(1024) {
                    require_unbounded(context)?;
                    if !chunk.iter().all($finite) {
                        return Err(SparseLuError::NumericalBreakdown);
                    }
                }
                Ok(())
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
