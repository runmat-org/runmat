//! Sparse products borrow original floating or logical column storage.
//! Full output or touched-row CSC accumulation owns the only numerical result.
use super::*;
use crate::builtins::math::linalg::ops::linear_operator::{
    scalar::LinearOperatorScalar, CooperativeOperatorError, LinearOperator, OperatorWork,
    TypedLinearOperator,
};
use crate::context::RuntimeContext;
use num_complex::Complex64;
use runmat_value::{
    ComplexElement, ComplexStorage, ComplexTensor, NumericStorage, SparseTensor, Tensor,
};
mod csc;
mod element;
mod full;
mod operand;
use element::Element;
use operand::Operand;

fn failure(
    error: impl std::error::Error + Send + Sync + 'static,
    runtime: Option<&RuntimeContext>,
) -> crate::RuntimeError {
    let _ = runtime;
    crate::build_runtime_error("mtimes: sparse product allocation failed")
        .with_builtin(NAME)
        .with_source(error)
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .build()
}
fn operator_error(
    error: CooperativeOperatorError,
    runtime: Option<&RuntimeContext>,
) -> crate::RuntimeError {
    match error {
        CooperativeOperatorError::Execution(error) => error.into_runtime_error_in_context(runtime),
        CooperativeOperatorError::Operator(
            crate::builtins::math::linalg::ops::sparse_operator::SparseOperatorError::Admission(
                error,
            ),
        ) => crate::builtin::builtin_control_error(
            runtime.map(|r| r.cancellation_context()).as_ref(),
            NAME,
            error,
        ),
        CooperativeOperatorError::Operator(error) => {
            crate::build_runtime_error(format!("mtimes: {error:?}"))
                .with_builtin(NAME)
                .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                .build()
        }
    }
}
fn checked<T>(
    value: Result<T, CooperativeOperatorError>,
    runtime: Option<&RuntimeContext>,
) -> crate::BuiltinResult<ProductResult<T>> {
    match value {
        Ok(value) => Ok(Ok(value)),
        Err(CooperativeOperatorError::Operator(error))
            if !matches!(
                error,
                crate::builtins::math::linalg::ops::sparse_operator::SparseOperatorError::Admission(
                    _
                )
            ) =>
        {
            Ok(Err(errors::invalid_input(format!(
                "mtimes: invalid sparse matrix operand: {error:?}"
            ))))
        }
        Err(error) => Err(operator_error(error, runtime)),
    }
}
async fn tick(
    work: &mut Option<OperatorWork<'_>>,
    runtime: Option<&RuntimeContext>,
) -> crate::BuiltinResult<()> {
    if let Some(work) = work {
        work.tick().await.map_err(|e| operator_error(e, runtime))?;
    }
    Ok(())
}
pub(super) async fn evaluate(
    context: Option<&RuntimeContext>,
    lhs: &Value,
    rhs: &Value,
) -> crate::BuiltinResult<ProductResult<Value>> {
    // An explicitly supplied original context wins over the legacy fallback.
    // No provider is selected and no input matrix is converted or copied.
    let runtime = context.cloned().or_else(crate::context::legacy::active);
    crate::context::host_allocation::require_unbounded_workspace(
        runtime.as_ref(),
        NAME,
        "finite host-memory sparse matrix product output and CSC growth are not certified",
    )?;
    let runtime = runtime.map(RuntimeContext::with_pinned_cancellation);
    let left = product_try!(checked(
        Operand::borrow(lhs, runtime.as_ref()).await,
        runtime.as_ref()
    )?);
    let right = product_try!(checked(
        Operand::borrow(rhs, runtime.as_ref()).await,
        runtime.as_ref()
    )?);
    if left.cols() != right.rows() {
        return Ok(Err(errors::invalid_input(
            "Inner matrix dimensions must agree.",
        )));
    }
    let complex = left.complex() || right.complex();
    let single = left.single() || right.single();
    let sparse = left.sparse() && right.sparse();
    let output = match (single, complex) {
        (false, false) => {
            if sparse {
                csc::multiply::<f64>(left, right, runtime.as_ref()).await?
            } else {
                full::multiply::<f64>(left, right, runtime.as_ref()).await?
            }
        }
        (true, false) => {
            if sparse {
                csc::multiply::<f32>(left, right, runtime.as_ref()).await?
            } else {
                full::multiply::<f32>(left, right, runtime.as_ref()).await?
            }
        }
        (false, true) => {
            if sparse {
                csc::multiply::<ComplexElement<f64>>(left, right, runtime.as_ref()).await?
            } else {
                full::multiply::<ComplexElement<f64>>(left, right, runtime.as_ref()).await?
            }
        }
        (true, true) => {
            if sparse {
                csc::multiply::<ComplexElement<f32>>(left, right, runtime.as_ref()).await?
            } else {
                full::multiply::<ComplexElement<f32>>(left, right, runtime.as_ref()).await?
            }
        }
    };
    Ok(Ok(output))
}
