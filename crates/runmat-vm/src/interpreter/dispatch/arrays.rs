use crate::ops::arrays as array_ops;
use runmat_runtime::execution::RootedValue;
use runmat_runtime::RuntimeError;
use runmat_value::Value;
use std::future::Future;

/// An aggregate instruction failed its own numeric row-shape construction at
/// this exact issued source. A called function's error is never stamped here.
#[derive(Debug)]
pub(crate) struct MatrixConstructionAtSource {
    source: runmat_runtime::source_context::RuntimeSourceRef,
    invocation_arity: runmat_runtime::context::RuntimeInvocationArity,
    cause: RuntimeError,
}

impl MatrixConstructionAtSource {
    pub(crate) fn in_source_script(
        &self,
        source: runmat_runtime::source_context::RuntimeSourceRef,
    ) -> bool {
        self.source == source
            && self.invocation_arity == runmat_runtime::context::RuntimeInvocationArity::Script
    }
}

impl std::fmt::Display for MatrixConstructionAtSource {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.cause.fmt(formatter)
    }
}

impl std::error::Error for MatrixConstructionAtSource {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.cause)
    }
}

pub(crate) fn qualify_matrix_construction_shape(
    error: RuntimeError,
    source: Option<runmat_runtime::source_context::RuntimeSourceRef>,
    invocation_arity: runmat_runtime::context::RuntimeInvocationArity,
) -> RuntimeError {
    use runmat_runtime::builtins::common::concatenation::MatrixConstructionShapeError;

    let Some(source) = source else { return error };
    if !error.source.as_deref().is_some_and(|cause| {
        cause
            .downcast_ref::<MatrixConstructionShapeError>()
            .is_some()
    }) {
        return error;
    }
    let identifier = error.identifier.clone();
    let span = error.span;
    let context = error.context.clone();
    let gpu_gather_retry = error.gpu_gather_retry;
    let failure_origin = error.failure_origin;
    let mut bound = runmat_runtime::build_runtime_error(error.message.clone())
        .with_source(MatrixConstructionAtSource {
            source,
            invocation_arity,
            cause: error,
        })
        .build();
    bound.identifier = identifier;
    bound.span = span;
    bound.context = context;
    bound.gpu_gather_retry = gpu_gather_retry;
    bound.failure_origin = failure_origin;
    bound
}

pub fn create_matrix(
    stack: &mut crate::interpreter::stack::InterpreterStack,
    rows: usize,
    cols: usize,
) -> Result<(), RuntimeError> {
    array_ops::create_matrix(stack, rows, cols)
}

pub async fn create_range<F, Fut>(
    stack: &mut crate::interpreter::stack::InterpreterStack,
    has_step: bool,
    call_colon: F,
) -> Result<(), RuntimeError>
where
    F: FnMut(Vec<Value>) -> Fut,
    Fut: Future<Output = Result<RootedValue, RuntimeError>>,
{
    array_ops::create_range(stack, has_step, call_colon).await
}

pub fn pack_to_row(
    stack: &mut crate::interpreter::stack::InterpreterStack,
    count: usize,
) -> Result<(), RuntimeError> {
    array_ops::pack_to_row(stack, count)
}

pub fn pack_to_col(
    stack: &mut crate::interpreter::stack::InterpreterStack,
    count: usize,
) -> Result<(), RuntimeError> {
    array_ops::pack_to_col(stack, count)
}

pub fn unpack(
    stack: &mut crate::interpreter::stack::InterpreterStack,
    out_count: usize,
) -> Result<(), RuntimeError> {
    array_ops::unpack(stack, out_count)
}
