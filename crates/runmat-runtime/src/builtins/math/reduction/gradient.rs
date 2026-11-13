//! MATLAB-compatible `gradient` builtin with scalar and coordinate-vector spacing support.

use runmat_accelerate_api::{GpuTensorHandle, GpuTensorStorage};
use runmat_builtins::{
    BuiltinErrorDescriptor, GRADIENT_DESCRIPTOR, GRADIENT_ERROR_INTERNAL,
    GRADIENT_ERROR_INVALID_ARGUMENT, GRADIENT_ERROR_INVALID_INPUT, GRADIENT_INTEGER_DATA_EXTENSION,
    GRADIENT_INTEGER_SPACING_EXTENSION, GRADIENT_LOGICAL_DATA_EXTENSION,
    GRADIENT_LOGICAL_SPACING_EXTENSION,
};
#[cfg(test)]
use runmat_builtins::{GRADIENT_EXTENSIONS, GRADIENT_INTEGER_CAPABILITIES};
use runmat_macros::runtime_builtin;
use runmat_value::{ComplexTensor, NumericDType, Tensor, Value};

use crate::builtins::common::gpu_helpers;
use crate::builtins::common::random_args::complex_tensor_into_value;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::tensor;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

mod arguments;
use arguments::*;
mod geometry;
pub use geometry::matlab_gradient_shape;
use geometry::*;
mod provider;
use provider::*;
mod host;
use host::*;
pub use host::{
    gradient_complex_tensor_host, gradient_complex_tensor_host_with_coordinates,
    gradient_real_tensor_host, gradient_real_tensor_host_with_coordinates,
};
const NAME: &str = "gradient";

fn gradient_descriptor_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .build()
}

fn gradient_descriptor_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    gradient_descriptor_error_with_message(format!("{}: {}", error.message, detail.as_ref()), error)
}

fn gradient_invalid_argument(detail: impl AsRef<str>) -> RuntimeError {
    gradient_descriptor_error_with_detail(&GRADIENT_ERROR_INVALID_ARGUMENT, detail)
}

fn gradient_invalid_input(detail: impl AsRef<str>) -> RuntimeError {
    gradient_descriptor_error_with_detail(&GRADIENT_ERROR_INVALID_INPUT, detail)
}

fn gradient_internal_error(detail: impl AsRef<str>) -> RuntimeError {
    gradient_descriptor_error_with_detail(&GRADIENT_ERROR_INTERNAL, detail)
}

#[path = "gradient/kernel_error.rs"]
pub mod kernel_error;
pub use kernel_error::GradientKernelError;
use kernel_error::GradientKernelResult;

#[cfg(test)]
#[path = "gradient/kernel_fault_tests.rs"]
mod kernel_fault_tests;

#[derive(Clone, Debug, PartialEq)]
enum GradientSpacing {
    Scalar(f64),
    Coordinates(Vec<f64>),
}

impl GradientSpacing {
    fn is_scalar(&self) -> bool {
        matches!(self, Self::Scalar(_))
    }
}

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::reduction::gradient")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "gradient",
    op_kind: GpuOpKind::Custom("numerical-gradient"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[
        ProviderHook::Custom("gradient_dim"),
        ProviderHook::Custom("gradient_dim_with_coordinates"),
    ],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "Eligible floating real or complex samples use gradient_dim with unit or scalar spacing. Public coordinate-spacing calls use admitted host evaluation only for automatic placement; the coordinate provider hook is not called by this Runtime path.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::reduction::gradient")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "gradient",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Gradient preserves input shape and uses edge-aware finite differences, so providers expose it through a custom sink hook.",
};

#[runtime_builtin(
    name = "gradient",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::reduction::gradient"
)]
async fn gradient_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    value: Value,
    rest: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    crate::output_count::validate_bounded_outputs(
        &GRADIENT_DESCRIPTOR,
        &GRADIENT_ERROR_INVALID_ARGUMENT,
        NAME,
    )
    .map_err(super::numerical_execution::terminal)?;
    let call = super::numerical_execution::NumericalCall::new(
        runmat_builtins::NumericalReductionKind::Gradient,
        runtime.as_ref(),
    )?;
    let requested_outputs = crate::output_count::current_output_count().unwrap_or(1);
    if !matches!(
        value,
        Value::Tensor(_)
            | Value::LogicalArray(_)
            | Value::Num(_)
            | Value::Int(_)
            | Value::Bool(_)
            | Value::Complex(..)
            | Value::ComplexTensor(_)
            | Value::GpuTensor(_)
    ) {
        return Err(gradient_invalid_input(
            "expected dense numeric or logical sampled data",
        ));
    }
    if crate::builtins::common::validation::is_typed_complex_integer(&value) {
        return Err(gradient_invalid_input(
            "operations involving complex numbers with integer types are not supported",
        ));
    }

    ensure_gradient_extensions(&value, rest.view())?;
    let shape = value_shape(&value);
    let available_count = if shape.is_empty() {
        if value_len(&value) == 0 {
            2
        } else {
            1
        }
    } else if shape.len() == 1 {
        1
    } else if shape.len() == 2 {
        if shape[0] == 1 || shape[1] == 1 {
            1
        } else {
            2
        }
    } else {
        shape.len()
    };
    if requested_outputs > available_count {
        return Err(gradient_invalid_argument(format!("gradient: requested {requested_outputs} outputs, but input supports at most {available_count}")));
    }
    if rest.len() > 1 && rest.len() != available_count {
        return Err(gradient_invalid_argument(
            "spacing count must be zero, one, or one per available dimension",
        ));
    }
    call.host()?; // output/spacing collection backing has no complete finite owner
    ensure_exact_integer_boundary(&value, "input data")?;

    let available_outputs = gradient_output_dims(value_shape(&value), value_len(&value));
    if requested_outputs > available_outputs.len() {
        return Err(gradient_invalid_argument(format!(
            "gradient: requested {requested_outputs} outputs, but input supports at most {}",
            available_outputs.len()
        )));
    }

    let dim_lengths =
        gradient_dim_lengths(value_shape(&value), value_len(&value), &available_outputs);
    let requires_host_gpu =
        value_is_integer_or_logical(&value) || rest.iter().any(value_is_integer_or_logical);
    let spacings = parse_spacings(rest.view(), &available_outputs, &dim_lengths, &call).await?;
    let outputs = evaluate_gradient_outputs(
        value,
        &available_outputs[..requested_outputs],
        &spacings,
        requires_host_gpu,
        &call,
    )
    .await?;

    if crate::output_count::current_output_count().is_some() {
        return call.finish_value(Ok(Value::output_list(outputs)));
    }

    call.finish_value(Ok(outputs
        .into_iter()
        .next()
        .expect("single-output gradient result")))
}

fn ensure_gradient_extensions(
    value: &Value,
    spacings: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<()> {
    if value_is_typed_integer(value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &GRADIENT_INTEGER_DATA_EXTENSION,
            NAME,
        )?;
    }
    if value_is_logical(value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &GRADIENT_LOGICAL_DATA_EXTENSION,
            NAME,
        )?;
    }
    for spacing in spacings {
        if value_is_typed_integer(spacing) {
            crate::compatibility::ensure_builtin_extension_enabled(
                &GRADIENT_INTEGER_SPACING_EXTENSION,
                NAME,
            )?;
        }
        if value_is_logical(spacing) {
            crate::compatibility::ensure_builtin_extension_enabled(
                &GRADIENT_LOGICAL_SPACING_EXTENSION,
                NAME,
            )?;
        }
    }
    Ok(())
}

fn value_is_typed_integer(value: &Value) -> bool {
    matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_integer_type(handle).is_some())
}

fn value_is_logical(value: &Value) -> bool {
    matches!(value, Value::Bool(_) | Value::LogicalArray(_))
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_is_logical(handle))
}

fn value_is_integer_or_logical(value: &Value) -> bool {
    value_is_typed_integer(value) || value_is_logical(value)
}

fn ensure_exact_integer_boundary(value: &Value, role: &str) -> BuiltinResult<()> {
    if !matches!(value, Value::GpuTensor(_))
        && !crate::builtins::common::validation::native_integer_value_is_exact_f64(value)
    {
        Err(gradient_invalid_input(format!(
            "integer {role} must be exactly representable as double"
        )))
    } else {
        Ok(())
    }
}

async fn evaluate_gradient_outputs(
    value: Value,
    requested_dims: &[usize],
    all_spacings: &[GradientSpacing],
    requires_host_gpu: bool,
    call: &super::numerical_execution::NumericalCall<'_>,
) -> BuiltinResult<Vec<Value>> {
    if let Value::GpuTensor(handle) = value {
        return gradient_gpu_outputs(
            handle,
            requested_dims,
            all_spacings,
            requires_host_gpu,
            call,
        )
        .await;
    }

    evaluate_host_gradient_outputs(value, requested_dims, all_spacings)
}

fn evaluate_host_gradient_outputs(
    value: Value,
    requested_dims: &[usize],
    all_spacings: &[GradientSpacing],
) -> BuiltinResult<Vec<Value>> {
    ensure_exact_integer_boundary(&value, "input data")?;
    match value {
        Value::Tensor(tensor) => {
            let mut outputs = Vec::with_capacity(requested_dims.len());
            for &dim in requested_dims {
                let spacing = spacing_for_dim(dim, requested_dims, all_spacings);
                outputs.push(tensor::tensor_into_value(
                    gradient_real_tensor_host_with_spacing(tensor.clone(), dim, spacing)?,
                ));
            }
            Ok(outputs)
        }
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(&logical).map_err(gradient_invalid_input)?;
            let mut outputs = Vec::with_capacity(requested_dims.len());
            for &dim in requested_dims {
                let spacing = spacing_for_dim(dim, requested_dims, all_spacings);
                outputs.push(tensor::tensor_into_value(
                    gradient_real_tensor_host_with_spacing(tensor.clone(), dim, spacing)?,
                ));
            }
            Ok(outputs)
        }
        Value::Num(_) | Value::Int(_) | Value::Bool(_) => {
            let tensor =
                tensor::value_into_tensor_for(NAME, value).map_err(gradient_invalid_input)?;
            let mut outputs = Vec::with_capacity(requested_dims.len());
            for &dim in requested_dims {
                let spacing = spacing_for_dim(dim, requested_dims, all_spacings);
                outputs.push(tensor::tensor_into_value(
                    gradient_real_tensor_host_with_spacing(tensor.clone(), dim, spacing)?,
                ));
            }
            Ok(outputs)
        }
        Value::Complex(re, im) => {
            let tensor =
                ComplexTensor::new(vec![(re, im)], vec![1, 1]).map_err(gradient_invalid_input)?;
            let mut outputs = Vec::with_capacity(requested_dims.len());
            for &dim in requested_dims {
                let spacing = spacing_for_dim(dim, requested_dims, all_spacings);
                outputs.push(complex_tensor_into_value(
                    gradient_complex_tensor_host_with_spacing(tensor.clone(), dim, spacing)?,
                ));
            }
            Ok(outputs)
        }
        Value::ComplexTensor(tensor) => {
            let mut outputs = Vec::with_capacity(requested_dims.len());
            for &dim in requested_dims {
                let spacing = spacing_for_dim(dim, requested_dims, all_spacings);
                outputs.push(complex_tensor_into_value(
                    gradient_complex_tensor_host_with_spacing(tensor.clone(), dim, spacing)?,
                ));
            }
            Ok(outputs)
        }
        other => Err(gradient_invalid_input(format!(
            "gradient: unsupported input type {:?}; expected numeric or logical data",
            other
        ))),
    }
}

#[cfg(test)]
#[path = "gradient/owning_tests.rs"]
mod owning_tests;

#[allow(dead_code)]
#[cfg(test)]
pub(crate) mod tests;
