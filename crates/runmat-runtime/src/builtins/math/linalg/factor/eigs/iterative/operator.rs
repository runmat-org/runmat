//! One selected operator and reusable inverse action for the entire solve.
mod residual;
pub(super) mod cholesky;
mod normalization;
use super::{request::Request, scalar::Scalar, work::Work};
use crate::{
    builtins::math::linalg::{
        factor::iterative::callback::{self, CallbackErrorPolicy},
        ops::{
            callable_linear_operator::RepeatedCallableVector,
            dense_factor::{DenseFactor, FaerFactorScalar},
            iterative_preconditioner::{MatrixFactorError, PreparedMatrixFactor},
            linear_operator::LinearOperator,
            sparse_factor::ManagedSparseLu,
        },
    },
    context::RuntimeContext,
    execution::RootedValueSequence,
    BuiltinResult,
};
use runmat_value::Value;
enum Coefficient<'a> {
    Matrix(LinearOperator<'a>),
    Callable {
        selected: &'a RepeatedCallableVector,
        inverse: bool,
    },
}
enum Inverse<'a, T: Scalar> {
    None,
    Generalized(PreparedMatrixFactor<'a, T>),
    DenseB(Box<DenseFactor<T>>),
    Cholesky(cholesky::Cholesky<'a>),
    SparseB(Box<ManagedSparseLu<T::SparseNative>>),
    Shifted(Box<ManagedSparseLu<T::SparseNative>>),
}
pub(super) struct Operator<'a, T: Scalar> {
    coefficient: Coefficient<'a>,
    b: Option<LinearOperator<'a>>,
    inverse: Inverse<'a, T>,
    first: Option<RootedValueSequence>,
    metric: bool,
    range_seed: bool,
}
const ERRORS: CallbackErrorPolicy = CallbackErrorPolicy {
    name: "eigs",
    invalid: super::invalid,
};
pub(super) async fn borrow<'a>(
    value: &'a Value,
    runtime: Option<&RuntimeContext>,
) -> BuiltinResult<LinearOperator<'a>> {
    match runtime {
        Some(runtime) => LinearOperator::borrow_cooperatively(value, runtime)
            .await
            .map_err(|e| super::operator_error(Some(runtime), e)),
        None => LinearOperator::borrow_unbounded(value).map_err(|e| super::matrix_error(None, e)),
    }
}
impl<'a, T: Scalar> Operator<'a, T> {
    pub async fn prepare(
        a: &'a Value,
        request: &mut Request<'a>,
        selected: Option<&'a RepeatedCallableVector>,
        first: Option<RootedValueSequence>,
        work: &mut Work<'_>,
    ) -> BuiltinResult<Self> {
        let runtime = work.runtime;
        let coefficient = if let Some(selected) = selected {
            Coefficient::Callable {
                selected,
                inverse: matches!(
                    request.sigma,
                    super::super::Sigma::Near(_) | super::super::Sigma::SmallestAbs
                ),
            }
        } else {
            Coefficient::Matrix(
                request
                    .matrix
                    .ok_or_else(|| super::invalid("matrix operator proof is missing"))?,
            )
        };
        let b = request.mass;
        let shift = match request.sigma {
            super::super::Sigma::SmallestAbs => Some(num_complex::Complex64::new(0.0, 0.0)),
            super::super::Sigma::Near(value) => Some(value),
            _ => None,
        };
        let inverse = if request.cholesky {
            Inverse::Cholesky(
                request
                    .cholesky_owner
                    .take()
                    .ok_or_else(|| super::invalid("supplied Cholesky proof is missing"))?,
            )
        } else if let (Some(shift), Coefficient::Matrix(_)) = (shift, &coefficient) {
            let Value::SparseTensor(a) = a else {
                return Err(super::unsupported(
                    "shifted iterative route requires borrowed CSC A",
                ));
            };
            let runtime = runtime.ok_or_else(|| {
                super::unsupported("shifted sparse eigs requires an execution context")
            })?;
            let shift = T::native(shift)
                .ok_or_else(|| super::invalid("shift exceeds selected native range"))?;
            Inverse::Shifted(
                ManagedSparseLu::<T::SparseNative>::factor_pencil(
                    a,
                    request.b,
                    T::to_sparse(shift),
                    runtime,
                )
                .await
                .map_err(|e| {
                    super::factor_error(Some(runtime), MatrixFactorError::ManagedSparse(e))
                })?,
            )
        } else if shift.is_none() {
            match request.b {
                Some(Value::SparseTensor(b)) => {
                    let runtime = runtime.ok_or_else(|| {
                        super::unsupported("generalized sparse eigs requires an execution context")
                    })?;
                    Inverse::SparseB(
                        ManagedSparseLu::<T::SparseNative>::factor_shifted(
                            b,
                            None,
                            T::to_sparse(T::zero()),
                            runtime,
                        )
                        .await
                        .map_err(|e| {
                            super::factor_error(Some(runtime), MatrixFactorError::ManagedSparse(e))
                        })?,
                    )
                }
                Some(b) if <T as FaerFactorScalar>::source(b).is_some() => Inverse::Generalized(
                    PreparedMatrixFactor::<T>::prepare(b, request.n, runtime)
                        .await
                        .map_err(|e| super::factor_error(runtime, e))?,
                ),
                Some(b) => Inverse::DenseB(
                    DenseFactor::<T>::factor_converted(b, request.n, runtime, T::native)
                        .await
                        .map_err(|e| super::factor_error(runtime, MatrixFactorError::Dense(e)))?,
                ),
                None => Inverse::None,
            }
        } else {
            Inverse::None
        };
        Ok(Self {
            coefficient,
            b,
            inverse,
            first,
            metric: request.cholesky || request.definite,
            range_seed: request.definite,
        })
    }
    pub async fn apply(
        &mut self,
        input: &[T],
        output: &mut [T],
        temporary: &mut [T],
        work: &mut Work<'_>,
    ) -> BuiltinResult<()> {
        work.check()?;
        if let Some(first) = self.first.take() {
            let [value] = first.value().as_slice() else {
                return Err(super::invalid(
                    "initial callback product must contain one vector",
                ));
            };
            let values = T::input(value).ok_or_else(|| {
                super::invalid("initial callback product changed its inferred class")
            })?;
            for (destination, &source) in output.iter_mut().zip(values.as_slice()) {
                *destination = source;
                work.tick().await?;
            }
        } else if let Inverse::Shifted(factor) = &mut self.inverse {
            let runtime = work
                .runtime
                .ok_or_else(|| super::unsupported("shifted factor lost execution context"))?;
            let input = if let Some(b) = &self.b {
                b.apply_converted(
                    input,
                    &mut temporary[..input.len()],
                    T::native,
                    Some(runtime),
                )
                .await
                .map_err(|e| super::operator_error(Some(runtime), e))?;
                &temporary[..input.len()]
            } else {
                input
            };
            factor
                .solve_converted_into(input, output, 1, runtime, T::to_sparse, T::from_sparse)
                .await
                .map_err(|e| {
                    super::factor_error(Some(runtime), MatrixFactorError::ManagedSparse(e))
                })?;
            return work.check();
        } else {
            match &self.coefficient {
                Coefficient::Matrix(matrix) => matrix
                    .apply_converted(input, output, T::native, work.runtime)
                    .await
                    .map_err(|e| super::operator_error(work.runtime, e))?,
                Coefficient::Callable { selected, inverse } => {
                    let input = if *inverse {
                        if let Some(b) = &self.b {
                            b.apply_converted(
                                input,
                                &mut temporary[..input.len()],
                                T::native,
                                work.runtime,
                            )
                            .await
                            .map_err(|e| super::operator_error(work.runtime, e))?;
                            &temporary[..input.len()]
                        } else {
                            input
                        }
                    } else {
                        input
                    };
                    callback::apply::<T>(
                        selected,
                        work.runtime
                            .ok_or_else(|| super::unsupported("callable lost execution context"))?,
                        input,
                        output,
                        ERRORS,
                    )
                    .await?
                }
            }
        }
        match &mut self.inverse {
            Inverse::Cholesky(factor) => {
                let (result, scratch) = temporary.split_at_mut(output.len());
                factor.solve(output, result, scratch, work).await?;
            }
            Inverse::DenseB(factor) => factor
                .solve_into(output, &mut temporary[..output.len()], work.runtime)
                .await
                .map_err(|e| super::factor_error(work.runtime, MatrixFactorError::Dense(e)))?,
            Inverse::Generalized(factor) => factor
                .solve_into(output, &mut temporary[..output.len()], work.runtime)
                .await
                .map_err(|e| super::factor_error(work.runtime, e))?,
            Inverse::SparseB(factor) => {
                let runtime = work.runtime.ok_or_else(|| {
                    super::unsupported("generalized factor lost execution context")
                })?;
                factor
                    .solve_converted_into(
                        output,
                        &mut temporary[..output.len()],
                        1,
                        runtime,
                        T::to_sparse,
                        T::from_sparse,
                    )
                    .await
                    .map_err(|e| {
                        super::factor_error(Some(runtime), MatrixFactorError::ManagedSparse(e))
                    })?;
            }
            Inverse::None | Inverse::Shifted(_) => return work.check(),
        }
        for (destination, &source) in output.iter_mut().zip(temporary.iter()) {
            *destination = source;
            work.tick().await?;
        }
        work.check()
    }
    pub fn uses_metric(&self) -> bool {
        self.metric
    }
    pub fn needs_range_seed(&self) -> bool {
        self.range_seed
    }
    pub fn is_generalized(&self) -> bool {
        self.b.is_some() || matches!(self.inverse, Inverse::Cholesky(_))
    }
    pub async fn apply_mass(
        &self,
        input: &[T],
        output: &mut [T],
        scratch: &mut [T],
        work: &mut Work<'_>,
    ) -> BuiltinResult<()> {
        if let Inverse::Cholesky(factor) = &self.inverse {
            factor.apply(input, output, scratch, work).await?;
        } else if let Some(b) = &self.b {
            b.apply_converted(input, output, T::native, work.runtime)
                .await
                .map_err(|e| super::operator_error(work.runtime, e))?;
        } else {
            if input.len() != output.len() {
                return Err(super::invalid("mass product dimensions differ"));
            }
            for (destination, &source) in output.iter_mut().zip(input) {
                *destination = source;
                work.tick().await?;
            }
        }
        work.flush().await
    }
}
