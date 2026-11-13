//! The accepted Jacobian retains its actual immutable numeric backing.
//! Dense row-major workspace exists only for finite differences or an explicit
//! dense algorithm/output boundary, never as a mirror of a sparse CG source.
use super::*;
mod columns;
mod storage;
use storage::{append, append_typed, copy, finite};
mod opaque;
use crate::builtins::math::linalg::ops::linear_operator::{
    CooperativeOperatorError, LinearOperator, LinearOperatorError, ProductOrientation,
};
use crate::execution::RootedValue;
pub(crate) use opaque::Opaque;
use runmat_value::Value;
pub(crate) enum Jacobian {
    Opaque(Opaque),
    RowMajor {
        values: Vec<f64>,
        rows: usize,
        columns: usize,
    },
    Matrix {
        source: RootedValue,
        rows: usize,
        columns: usize,
    },
}
pub(super) enum Products<'a> {
    Opaque(&'a Opaque),
    RowMajor {
        values: &'a [f64],
        rows: usize,
        columns: usize,
    },
    Matrix(LinearOperator<'a>),
}
impl Jacobian {
    pub(crate) async fn matrix(
        source: RootedValue,
        rows: usize,
        columns: usize,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<Self> {
        let source = if matches!(source.value(), Value::GpuTensor(_)) {
            let runtime = c.runtime().cloned();
            source
                .with_owned_async(|value| async move {
                    let value = crate::dispatcher::gather_if_needed_in_context_async(
                        &value,
                        runtime.as_ref(),
                    )
                    .await?;
                    RootedValue::in_optional_context(value, runtime.as_ref())
                })
                .await?
        } else {
            source
        };
        let operator = borrow(source.value(), c).await?;
        if operator.rows() != rows
            || operator.cols() != columns
            || !matches!(
                operator,
                LinearOperator::RealF32(_) | LinearOperator::RealF64(_)
            )
        {
            return Err(invalid(
                "supplied Jacobian must be a real m-by-n single or double matrix",
            ));
        }
        match operator {
            LinearOperator::RealF32(view) => finite(view, c).await?,
            LinearOperator::RealF64(view) => finite(view, c).await?,
            _ => unreachable!("real operator was checked"),
        }
        Ok(Self::Matrix {
            source,
            rows,
            columns,
        })
    }
    pub(super) fn dimensions(&self) -> (usize, usize) {
        match self {
            Self::Opaque(source) => source.dimensions(),
            Self::RowMajor { rows, columns, .. } | Self::Matrix { rows, columns, .. } => {
                (*rows, *columns)
            }
        }
    }
    pub(super) async fn products<'a>(
        &'a self,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<Products<'a>> {
        match self {
            Self::Opaque(source) => Ok(Products::Opaque(source)),
            Self::RowMajor {
                values,
                rows,
                columns,
            } => {
                if rows.checked_mul(*columns) != Some(values.len()) {
                    return Err(invalid(
                        "accepted Jacobian workspace geometry is inconsistent",
                    ));
                }
                Ok(Products::RowMajor {
                    values,
                    rows: *rows,
                    columns: *columns,
                })
            }
            Self::Matrix { source, .. } => borrow(source.value(), c).await.map(Products::Matrix),
        }
    }
    /// Transfer the original numeric output owner; normalization changes
    /// only a trailing-singleton dense shape header, never its backing.
    pub(super) async fn into_published(
        self,
        numerical: Option<Vec<f64>>,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<PublishedJacobian> {
        match self {
            Self::Opaque(source) => {
                let (rows, columns) = source.dimensions();
                match numerical.filter(|values| rows.checked_mul(columns) == Some(values.len())) {
                    Some(values) => Ok(PublishedJacobian::RowMajor(values)),
                    None => source.published(c).await,
                }
            }
            Self::RowMajor { values, .. } => Ok(PublishedJacobian::RowMajor(values)),
            Self::Matrix {
                source,
                rows,
                columns,
            } => {
                let source = if matches!(source.value(), Value::Tensor(value) if value.shape() != [rows, columns])
                {
                    let runtime = c.runtime().cloned();
                    source
                        .with_owned_async(|value| async move {
                            let Value::Tensor(value) = value else {
                                unreachable!("dense Jacobian shape header");
                            };
                            let value = value
                                .reshape(vec![rows, columns])
                                .map(Value::Tensor)
                                .map_err(|error| invalid(&error))?;
                            RootedValue::in_optional_context(value, runtime.as_ref())
                        })
                        .await?
                } else {
                    source
                };
                Ok(PublishedJacobian::Supplied(source))
            }
        }
    }
    /// LM alone requires a dense numerical matrix; the accepted source owner
    /// can remain rooted independently until terminal products are selected.
    pub(in crate::builtins::math::optim) async fn copy_row_major(
        &self,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<Vec<f64>> {
        if let Self::Opaque(source) = self {
            return source.row_major(c).await;
        }
        let (rows, columns) = self.dimensions();
        let length = checked_jacobian_work("lsqnonlin", rows, columns)?;
        let mut values = runmat_value::exact_capacity_vec(length).map_err(allocation)?;
        match self {
            Self::RowMajor { values: source, .. } => {
                for chunk in source.chunks(1024) {
                    values.extend_from_slice(chunk);
                    c.completed(chunk.len() as u64).await?;
                }
            }
            Self::Opaque(_) => unreachable!("opaque materialization selected above"),
            Self::Matrix { source, .. } => {
                while values.len() < length {
                    let end = values.len().saturating_add(1024).min(length);
                    let count = end - values.len();
                    values.resize(end, 0.);
                    c.completed(count as u64).await?;
                }
                let operator = borrow(source.value(), c).await?;
                match operator {
                    LinearOperator::RealF32(view) => copy(view, &mut values, c).await?,
                    LinearOperator::RealF64(view) => copy(view, &mut values, c).await?,
                    _ => return Err(invalid("accepted Jacobian changed numeric class")),
                }
            }
        }
        Ok(values)
    }
    pub(super) async fn into_row_major(self, c: &mut ScalarCooperation) -> BuiltinResult<Vec<f64>> {
        match self {
            Self::RowMajor { values, .. } => Ok(values),
            source => source.copy_row_major(c).await,
        }
    }
}
impl Products<'_> {
    pub(super) fn is_opaque(&self) -> bool {
        matches!(self, Self::Opaque(_))
    }
    pub(super) async fn apply_normal(
        &mut self,
        input: &[f64],
        scratch: &mut [f64],
        output: &mut [f64],
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<()>> {
        if let Self::Opaque(source) = self {
            return source.normal(input, output, c).await;
        }
        if self
            .apply(ProductOrientation::Normal, input, scratch, c)
            .await?
            .is_none()
        {
            return Ok(None);
        }
        self.apply(ProductOrientation::Adjoint, scratch, output, c)
            .await
    }

    pub(super) fn dimensions(&self) -> (usize, usize) {
        match self {
            Self::Opaque(source) => source.dimensions(),
            Self::RowMajor { rows, columns, .. } => (*rows, *columns),
            Self::Matrix(operator) => (operator.rows(), operator.cols()),
        }
    }
    pub(super) async fn selected_entries(
        &self,
        axes: &[usize],
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<usize> {
        use crate::builtins::math::linalg::ops::linear_operator::TypedLinearOperator as V;
        let (rows, columns) = self.dimensions();
        let mut total = 0usize;
        for &axis in axes {
            if axis >= columns {
                return Err(invalid("selected Jacobian column is out of range"));
            }
            let count = match self {
                Self::Matrix(LinearOperator::RealF32(V::Sparse(view))) => view.column(axis).0.len(),
                Self::Matrix(LinearOperator::RealF64(V::Sparse(view))) => view.column(axis).0.len(),
                _ => rows,
            };
            total = total
                .checked_add(count)
                .ok_or_else(|| invalid("selected Jacobian storage extent overflow"))?;
            c.completed(1).await?;
        }
        Ok(total)
    }
    /// Append one physical factorization column, without a dense CSC mirror.
    pub(super) async fn append_column(
        &self,
        axis: usize,
        scale: f64,
        offset: usize,
        indices: &mut Vec<usize>,
        values: &mut Vec<f64>,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<()>> {
        let (rows, columns) = self.dimensions();
        if axis >= columns {
            return Err(invalid("selected Jacobian column is out of range"));
        }
        match self {
            Self::Opaque(source) => {
                source
                    .append_column(axis, scale, offset, indices, values, c)
                    .await
            }
            Self::RowMajor { values: source, .. } => {
                for row in 0..rows {
                    if !append(
                        row,
                        source[row * columns + axis],
                        scale,
                        offset,
                        indices,
                        values,
                    )? {
                        return Ok(None);
                    }
                    c.completed(1).await?;
                }
                Ok(Some(()))
            }
            Self::Matrix(LinearOperator::RealF32(view)) => {
                append_typed(*view, axis, scale, offset, indices, values, c).await
            }
            Self::Matrix(LinearOperator::RealF64(view)) => {
                append_typed(*view, axis, scale, offset, indices, values, c).await
            }
            _ => Err(invalid("accepted Jacobian changed numeric class")),
        }
    }
    pub(super) async fn apply(
        &mut self,
        orientation: ProductOrientation,
        input: &[f64],
        output: &mut [f64],
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<()>> {
        let (rows, columns) = self.dimensions();
        let (to, from) = match orientation {
            ProductOrientation::Normal => (rows, columns),
            ProductOrientation::Adjoint => (columns, rows),
        };
        if input.len() != from || output.len() != to {
            return Err(invalid("Jacobian product dimensions disagree"));
        }
        match self {
            Self::Opaque(source) => return source.apply(orientation, input, output, c).await,
            Self::RowMajor { values, .. } => {
                for value in output.iter_mut() {
                    *value = 0.;
                    c.completed(1).await?;
                }
                for row in 0..rows {
                    for column in 0..columns {
                        let coefficient = values[row * columns + column];
                        let (to, from) = match orientation {
                            ProductOrientation::Normal => (row, column),
                            ProductOrientation::Adjoint => (column, row),
                        };
                        output[to] = coefficient.mul_add(input[from], output[to]);
                        c.completed(1).await?;
                    }
                }
            }
            Self::Matrix(operator) => {
                let (runtime, budget) = c.runtime_budget();
                let result = operator
                    .apply_real_with_budget(input, output, runtime, budget, orientation)
                    .await;
                result.map_err(|error| operator_error(error, c.runtime()))?;
            }
        }
        for chunk in output.chunks(1024) {
            if chunk.iter().any(|value| !value.is_finite()) {
                return Ok(None);
            }
            c.completed(chunk.len() as u64).await?;
        }
        Ok(Some(()))
    }
}
fn invalid(detail: &str) -> crate::RuntimeError {
    optim_error("lsqnonlin", detail)
}
fn allocation(source: runmat_value::ExactHostNumericAllocationError) -> crate::RuntimeError {
    crate::build_runtime_error("lsqnonlin: Jacobian workspace allocation failed")
        .with_source(source)
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .build()
}
fn operator_error(
    source: CooperativeOperatorError,
    runtime: Option<&crate::context::RuntimeContext>,
) -> crate::RuntimeError {
    match source {
        CooperativeOperatorError::Execution(error) => error.into_runtime_error_in_context(runtime),
        CooperativeOperatorError::Operator(LinearOperatorError::Admission(
            crate::context::host_allocation::HostAllocationError::Cancelled,
        )) => crate::execution::ExecutionServiceError::Cancelled
            .into_runtime_error_in_context(runtime),
        CooperativeOperatorError::Operator(LinearOperatorError::Admission(source)) => {
            crate::context::host_allocation::host_allocation_runtime_error(
                runtime,
                "RunMat:lsqnonlin:JacobianOwnership",
                "accepted Jacobian",
                source,
            )
        }
        CooperativeOperatorError::Operator(source) => {
            crate::build_runtime_error(format!("lsqnonlin: invalid Jacobian operator: {source}"))
                .with_source(source)
                .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                .build()
        }
    }
}
async fn borrow<'a>(
    value: &'a Value,
    c: &mut ScalarCooperation,
) -> BuiltinResult<LinearOperator<'a>> {
    let result = {
        let (runtime, budget) = c.runtime_budget();
        LinearOperator::borrow_with_budget(value, runtime, budget).await
    };
    result.map_err(|error| operator_error(error, c.runtime()))
}
trait Real: Copy {
    fn real(self) -> f64;
}
impl Real for f32 {
    fn real(self) -> f64 {
        self as f64
    }
}
impl Real for f64 {
    fn real(self) -> f64 {
        self
    }
}
