//! Pole-zero cancellation for ordinary SISO transfer functions.
//!
//! The model parser owns coefficient/domain validation. This leaf works with
//! physical (descending-power) polynomials, then leaves serialization of an
//! inverse-basis model to `TfModel`.

use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::builtins::common::tensor::scalar_f64_from_host_value;
use crate::builtins::control::tf_model::{control_error, TfModel, TF_CLASS};
use crate::context::host_allocation::{
    HostAllocationError, HostAllocationPolicy, HostAllocationReceipt,
};
use crate::context::RuntimeContext;
use crate::{BuiltinResult, RuntimeError};

const NAME: &str = "minreal";

fn invalid(message: impl Into<String>) -> RuntimeError {
    control_error(NAME, "RunMat:minreal:InvalidArgument", message)
}

fn unsupported(message: impl Into<String>) -> RuntimeError {
    control_error(NAME, "RunMat:minreal:UnsupportedModel", message)
}

fn numerical(message: impl Into<String>) -> RuntimeError {
    control_error(NAME, "RunMat:minreal:NumericalFailure", message)
}

fn policy(runtime: Option<&RuntimeContext>) -> BuiltinResult<HostAllocationPolicy> {
    HostAllocationReceipt::policy(runtime).map_err(|error| {
        let cancellation = runtime.map(RuntimeContext::cancellation_context);
        crate::builtin::builtin_control_error(cancellation.as_ref(), NAME, error)
    })
}

fn check(runtime: Option<&RuntimeContext>) -> BuiltinResult<()> {
    policy(runtime).map(|_| ())
}

#[runtime_builtin(
    name = "minreal",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::control::minreal"
)]
async fn minreal_builtin(
    runtime: Option<RuntimeContext>,
    system: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    if rest.len() > 1 || crate::output_count::current_output_count().is_some_and(|n| n > 1) {
        return Err(invalid(
            "minreal expects a model, an optional positive tolerance, and at most one output",
        ));
    }
    let tolerance = match rest.first() {
        None => Some(f64::EPSILON.sqrt()),
        Some(value)
            if matches!(value, Value::Num(_))
                || matches!(value, Value::Tensor(tensor) if tensor.len() == 1
                    && (tensor.as_f32_slice().is_some() || tensor.as_f64_slice().is_some())) =>
        {
            scalar_f64_from_host_value(value).map_err(invalid)?
        }
        _ => None,
    }
    .ok_or_else(|| invalid("minreal tolerance requires a real single/double scalar"))?;
    if !tolerance.is_finite() || tolerance <= 0.0 {
        return Err(invalid("minreal tolerance must be finite and positive"));
    }
    let Value::Object(object) = system else {
        return Err(invalid("minreal expects a SISO tf model"));
    };
    if !object.is_class(TF_CLASS) {
        return Err(unsupported(format!(
            "minreal does not yet support model class '{}'",
            object.class_name()
        )));
    }
    if policy(runtime.as_ref())? == HostAllocationPolicy::Managed {
        // The existing companion-eigenvalue owner has opaque host allocations.
        // Refuse before parsing a resident property or publishing a result.
        return Err(crate::builtin::builtin_control_error(
            runtime
                .as_ref()
                .map(RuntimeContext::cancellation_context)
                .as_ref(),
            NAME,
            HostAllocationError::MissingManagedOwner,
        ));
    }
    let mut model = TfModel::from_value_async(Value::Object(object), NAME).await?;
    reduce(&mut model, tolerance, runtime.as_ref())?;
    check(runtime.as_ref())?;
    let value = model.to_value(NAME)?;
    check(runtime.as_ref())?;
    Ok(value)
}

mod reduction;
#[cfg(test)]
mod tests;

use reduction::reduce;
