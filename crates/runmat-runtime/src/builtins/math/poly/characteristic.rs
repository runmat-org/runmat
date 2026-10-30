//! Monic polynomial coefficients from a root vector or square host matrix.

mod input;
mod output;
mod recurrence;
#[cfg(test)]
mod tests;

use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::context::host_allocation::{
    HostAllocationError, HostAllocationPolicy, HostAllocationReceipt,
};
use crate::context::RuntimeContext;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

use input::{MatrixInput, PolyInput};

const NAME: &str = "poly";

fn invalid(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(NAME)
        .with_identifier("RunMat:poly:InvalidInput")
        .build()
}

fn unsupported_provider(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(NAME)
        .with_identifier("RunMat:poly:UnsupportedProvider")
        .build()
}

fn numerical(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(NAME)
        .with_identifier("RunMat:poly:Numerical")
        .build()
}

fn admission(context: Option<&RuntimeContext>, error: HostAllocationError) -> RuntimeError {
    let cancellation = context.map(RuntimeContext::cancellation_context);
    crate::builtin::builtin_control_error(cancellation.as_ref(), NAME, error)
}

fn checkpoint(context: Option<&RuntimeContext>) -> BuiltinResult<()> {
    HostAllocationReceipt::policy(context)
        .map(|_| ())
        .map_err(|error| admission(context, error))
}

fn require_unbounded_host_work(context: Option<&RuntimeContext>) -> BuiltinResult<()> {
    match HostAllocationReceipt::policy(context).map_err(|error| admission(context, error))? {
        HostAllocationPolicy::Unbounded => Ok(()),
        // Provider readback and Schur's private factor workspace have no
        // complete finite admitted owner on their respective paths.
        HostAllocationPolicy::Managed => {
            Err(admission(context, HostAllocationError::MissingManagedOwner))
        }
    }
}

#[runtime_builtin(
    name = "poly",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::poly"
)]
async fn poly_builtin(context: Option<RuntimeContext>, input: Value) -> BuiltinResult<Value> {
    if crate::output_count::current_output_count().is_some_and(|count| count > 1) {
        return Err(invalid("poly returns one coefficient row"));
    }
    checkpoint(context.as_ref())?;
    let input = if let Value::GpuTensor(handle) = &input {
        if runmat_accelerate_api::handle_is_explicit(handle) {
            return Err(unsupported_provider("poly cannot restore explicit device output with the current host coefficient owner"));
        }
        require_unbounded_host_work(context.as_ref())?;
        if runmat_accelerate_api::handle_integer_type(handle).is_some()
            || runmat_accelerate_api::handle_is_logical(handle)
        {
            return Err(invalid("poly requires single or double real/complex input"));
        }
        let readback =
            runmat_accelerate_api::numeric_readback_for_handle(handle).map_err(|source| {
                crate::runtime_error::with_provider_execution_source(
                    unsupported_provider("poly cannot resolve the resident input's exact provider"),
                    source.into(),
                )
            })?;
        crate::builtins::common::gpu_helpers::download_readback_value_in_context(
            context.as_ref(),
            &readback,
        )
        .await
        .map_err(|mut error| {
            error.gpu_gather_retry = crate::GpuGatherRetry::Never;
            error
        })?
    } else {
        input
    };
    match PolyInput::classify(&input)? {
        PolyInput::Vector(roots) => output::from_vector(roots, context.as_ref()),
        PolyInput::Matrix(MatrixInput::Real(matrix)) => {
            require_unbounded_host_work(context.as_ref())?;
            let factors = input::real_schur(matrix, context.as_ref())?;
            output::from_real_schur(&factors, matrix.numeric_dtype(), context.as_ref())
        }
        PolyInput::Matrix(MatrixInput::Complex(matrix)) => {
            require_unbounded_host_work(context.as_ref())?;
            let factors = input::complex_schur(matrix, context.as_ref())?;
            output::from_complex_schur(&factors, matrix.numeric_dtype(), context.as_ref())
        }
    }
}
