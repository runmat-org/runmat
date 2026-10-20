//! Polar/cylindrical coordinate transform; domain-owned admission and execution.
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{broadcast, gpu_helpers, tensor};
use crate::context::RuntimeContext;
use crate::execution::{RootedValue, RootedValueLease};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};
use runmat_accelerate_api::{AccelProvider, GpuTensorHandle, GpuTensorStorage};
use runmat_builtins::{
    BuiltinErrorDescriptor, POL2CART_ERROR_COMPLEX_UNSUPPORTED as ERROR_COMPLEX_UNSUPPORTED,
    POL2CART_ERROR_INTERNAL as ERROR_INTERNAL, POL2CART_ERROR_INVALID_INPUT as ERROR_INVALID_INPUT,
    POL2CART_ERROR_OUTPUT_COUNT as ERROR_OUTPUT_COUNT,
    POL2CART_ERROR_SIZE_MISMATCH as ERROR_SIZE_MISMATCH, POL2CART_INTEGER_RHO_EXTENSION,
    POL2CART_INTEGER_THETA_EXTENSION, POL2CART_INTEGER_Z_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::{Tensor, Value};
mod admission;
mod host;
mod owned;
mod provider;
mod preparation;
mod specs;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod provider_tests;
#[cfg(test)]
mod preparation_tests;
use host::*;
use provider::*;
#[cfg(test)]
use runmat_builtins::POL2CART_DESCRIPTOR;
#[cfg(target_arch = "wasm32")]
pub(crate) use specs::{
    __runmat_wasm_register_fusion_spec_FUSION_SPEC, __runmat_wasm_register_gpu_spec_GPU_SPEC,
};
const NAME: &str = "pol2cart";

#[runtime_builtin(
    name = "pol2cart",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::trigonometry::pol2cart"
)]
async fn pol2cart_builtin(
    runtime: Option<RuntimeContext>,
    theta: Value,
    rho: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<RootedValue> {
    let requested_outputs = crate::output_count::current_output_count();
    let mut inputs = vec![theta, rho];
    inputs.extend(rest);
    let _inputs =
        RootedValueLease::for_values(inputs.iter()).map_err(|cause| cause.into_runtime_error())?;
    let shape = admission::admit(&inputs, requested_outputs)?;
    for ((value, extension), role) in inputs
        .iter()
        .zip([
            &POL2CART_INTEGER_THETA_EXTENSION,
            &POL2CART_INTEGER_RHO_EXTENSION,
            &POL2CART_INTEGER_Z_EXTENSION,
        ])
        .zip(["theta", "rho", "z"])
    {
        crate::builtins::common::validation::ensure_runmat_integer_f64_boundary(
            value, extension, NAME, role,
        )
        .await?;
    }
    let sources: Vec<_> = inputs
        .iter()
        .filter_map(|value| match value {
            Value::GpuTensor(handle) => Some(handle.clone()),
            _ => None,
        })
        .collect();
    let result = preparation::execute(runtime.as_ref(), inputs, &shape, requested_outputs).await?;
    // A terminal coordinate transform ends fusion use only after successful
    // semantic work; this never frees inputs or changes explicit GPU intent.
    gpu_helpers::clear_input_residency(&sources, NAME)?;
    Ok(result)
}

async fn execute(
    inputs: Vec<Value>,
    shape: &[usize],
    requested_outputs: Option<usize>,
) -> BuiltinResult<RootedValue> {
    let wants_z_output = matches!(requested_outputs, Some(count) if count >= 3);
    if let Some(output) = try_pol2cart_gpu(&inputs, shape, requested_outputs).await? {
        return Ok(output);
    }
    let mut inputs: Vec<Value> =
        futures::future::try_join_all(inputs.into_iter().map(gather_if_gpu)).await?;

    let eval = match inputs.len() {
        2 => {
            let rho = inputs.pop().expect("rho");
            let theta = inputs.pop().expect("theta");
            Pol2CartEval::polar(theta, rho)?
        }
        3 => {
            let z = inputs.pop().expect("z");
            let rho = inputs.pop().expect("rho");
            let theta = inputs.pop().expect("theta");
            Pol2CartEval::cylindrical(theta, rho, z, wants_z_output)?
        }
        _ => {
            return Err(pol2cart_error(
                &ERROR_INVALID_INPUT,
                "expected two or three inputs",
            ))
        }
    };

    let published = match requested_outputs {
        Some(0) => Ok(Value::output_list(Vec::new())),
        Some(count) => eval.output_list(count),
        None => eval.x_value(),
    }?;
    RootedValue::new(published).map_err(|cause| cause.into_runtime_error())
}

async fn gather_if_gpu(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::GpuTensor(handle) => gather_gpu(handle).await,
        other => Ok(other),
    }
}

async fn gather_gpu(handle: GpuTensorHandle) -> BuiltinResult<Value> {
    gpu_helpers::gather_value_async(&Value::GpuTensor(handle))
        .await
        .map_err(|error| crate::builtins::common::map_control_flow_with_builtin(error, NAME))
}

fn pol2cart_error(
    error: &'static BuiltinErrorDescriptor,
    detail: impl std::fmt::Display,
) -> RuntimeError {
    let mut builder =
        build_runtime_error(format!("{}: {}", error.message, detail)).with_builtin(NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}
