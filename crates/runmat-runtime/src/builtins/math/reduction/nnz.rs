//! MATLAB-compatible `nnz` builtin with GPU-aware semantics for RunMat.

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, FusionError,
    FusionKernelTemplate, GpuOpKind, ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType,
    ShapeRequirements,
};
use crate::builtins::common::{
    gpu_helpers,
    shape::{canonical_scalar_shape, is_scalar_shape, normalize_scalar_shape},
    tensor,
};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};
use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{
    BuiltinErrorDescriptor, NNZ_DESCRIPTOR, NNZ_DIMENSION_EXTENSION, NNZ_ERROR_INTERNAL,
    NNZ_ERROR_INVALID_ARGUMENT, NNZ_ERROR_INVALID_INPUT,
};
use runmat_macros::runtime_builtin;
use runmat_value::{CharArray, ComplexTensor, LogicalArray, SparseTensor, Tensor, Value};

const NAME: &str = "nnz";
mod provider;
mod host;
use host::*;
#[cfg(test)]
use provider::validate_provider_result;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::reduction::nnz")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "nnz",
    op_kind: GpuOpKind::Reduction,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[
        ProviderHook::Reduction {
            name: "reduce_nnz_dim",
        },
        ProviderHook::Reduction {
            name: "reduce_nnz",
        },
    ],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: Some(512),
    workgroup_size: Some(256),
    accepts_nan_mode: false,
    notes: "Eligible real floating input is counted on its owning provider, then only the declared host double result is read back. Single-precision slices larger than 2^24 and native integer resident counts have no exact resident route.",
};

fn nnz_descriptor_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    nnz_descriptor_error_with_message(format!("{}: {}", error.message, detail.as_ref()), error)
}

fn nnz_descriptor_error_with_message(
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

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::reduction::nnz")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "nnz",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: crate::builtins::common::spec::FusionExprBuilder::new(|ctx, writer| {
            let input = ctx.inputs.first().ok_or(FusionError::MissingInput(0))?;
            let (zero, one) = match ctx.scalar_ty {
                ScalarType::F64 => ("f64(0.0)", "f64(1.0)"),
                _ => ("0.0", "1.0"),
            };
            writer.write_formatted(format_args!(
                "if (isNan({input}) || {input} != {zero}) {{ accumulator += {one}; }}"
            ))
        }),
    }),
    emits_nan: false,
    notes: "Count shader templates are retained, but no typed nonzero-count reduction is admitted by the selected fusion binding. Calls execute through the Runtime boundary.",
};

#[runtime_builtin(
    name = "nnz",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::reduction::nnz"
)]
async fn nnz_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    value: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &NNZ_DESCRIPTOR,
        &runmat_builtins::NNZ_ERROR_TOO_MANY_OUTPUTS,
        NAME,
    )
    .map_err(super::numerical_execution::terminal)?;
    let call = super::numerical_execution::NumericalCall::new(
        runmat_builtins::NumericalReductionKind::CountNonzero,
        runtime.as_ref(),
    )?;
    if !rest.is_empty() {
        crate::compatibility::ensure_builtin_extension_enabled(&NNZ_DIMENSION_EXTENSION, NAME)?;
    }
    if rest.len() > 1 {
        return Err(nnz_descriptor_error_with_detail(
            &NNZ_ERROR_INVALID_ARGUMENT,
            "too many input arguments",
        ));
    }
    // Scalar counts borrow native storage and require neither a mask nor output payload.
    let dim = match rest.first() {
        Some(control) => {
            call.host()?;
            if matches!(control, Value::GpuTensor(_)) {
                call.fallback([control])?;
            }
            parse_dimension_arg(rest.view(), &call).await?
        }
        None => None,
    };
    if let Value::GpuTensor(handle) = &value {
        gpu_helpers::require_resident_compute_owner(handle, || {
            super::numerical_execution::error(
                runmat_builtins::NumericalReductionKind::CountNonzero,
                "resident count input has no exact owning provider",
            )
        })?;
    }
    let result = match &value {
        Value::GpuTensor(handle) => provider::execute(handle, dim, &call).await,
        _ => {
            if dim.is_some() {
                call.host()?;
            }
            nnz_host_value(value, dim)
        }
    };
    call.finish_value(result)
}

async fn parse_dimension_arg(
    args: crate::call::FunctionArgumentView<'_>,
    call: &super::numerical_execution::NumericalCall<'_>,
) -> BuiltinResult<Option<usize>> {
    let Some(control) = args.first() else {
        return Ok(None);
    };
    let gathered;
    let boolean;
    let control = if let Value::GpuTensor(handle) = control {
        let readback =
            runmat_accelerate_api::numeric_readback_for_handle(handle).map_err(|source| {
                let error = super::numerical_execution::error(
                    call.kind,
                    "dimension has no exact owning provider",
                );
                crate::runtime_error::with_provider_execution_source(error, source.into())
            })?;
        gathered = gpu_helpers::download_readback_value_in_context(call.runtime, &readback)
            .await
            .map_err(super::numerical_execution::terminal)?;
        call.check()?;
        &gathered
    } else {
        control
    };
    let control = if let Value::Bool(value) = control {
        boolean = Value::Num(if *value { 1. } else { 0. });
        &boolean
    } else {
        control
    };
    let mut values = tensor::borrowed_host_shape_dimensions(control)
        .filter(|v| v.len() == 1)
        .ok_or_else(|| {
            nnz_descriptor_error_with_detail(
                &NNZ_ERROR_INVALID_ARGUMENT,
                "dimension must be a numeric scalar",
            )
        })?;
    let dim = values.next().expect("one dimension").map_err(|source| {
        let error =
            nnz_descriptor_error_with_detail(&NNZ_ERROR_INVALID_ARGUMENT, source.to_string());
        source.attach_cause(error)
    })?;
    if dim == 0 {
        return Err(nnz_descriptor_error_with_detail(
            &NNZ_ERROR_INVALID_ARGUMENT,
            "dimension must be >= 1",
        ));
    }
    Ok(Some(dim))
}

fn describe_value_kind(value: &Value) -> String {
    match value {
        Value::Int(_) => "integer scalar".to_string(),
        Value::Num(_) => "numeric scalar".to_string(),
        Value::Complex(_, _) => "complex scalar".to_string(),
        Value::Bool(_) => "logical scalar".to_string(),
        Value::LogicalArray(_) => "logical array".to_string(),
        Value::String(_) => "string scalar".to_string(),
        Value::Symbolic(_) => "symbolic scalar".to_string(),
        Value::SymbolicArray(_) => "symbolic array".to_string(),
        Value::StringArray(_) => "string array".to_string(),
        Value::CharArray(_) => "char array".to_string(),
        Value::Tensor(_) => "numeric tensor".to_string(),
        Value::SparseTensor(_) => "sparse tensor".to_string(),
        Value::ComplexTensor(_) => "complex tensor".to_string(),
        Value::Cell(_) => "cell array".to_string(),
        Value::Struct(_) | Value::StructArray(_) => "struct".to_string(),
        Value::GpuTensor(_) => "GPU tensor".to_string(),
        Value::ObjectArray(array) => format!("{} object array", array.class_name()),
        Value::Object(obj) => format!("{} object", obj.class_name()),
        Value::HandleObject(h) => format!("handle object ({})", h.class_name),
        Value::Listener(l) => format!("listener for {}", l.event_name),
        Value::ScopedFunctionHandle(_)
        | Value::FunctionHandle(_)
        | Value::ExternalFunctionHandle(_)
        | Value::MethodFunctionHandle(_)
        | Value::BoundFunctionHandle(_)
        | Value::Closure(_) => "function handle".to_string(),
        Value::ClassRef(_) => "class reference".to_string(),
        Value::MException(_) => "exception".to_string(),
        Value::OutputList(_) => "output list".to_string(),
        Value::Future(_) => "future".to_string(),
        Value::Task(_) => "task".to_string(),
        Value::Pool(_) => "pool".to_string(),
        Value::Job(_) => "job".to_string(),
        Value::Distributed(_) => "distributed".to_string(),
        Value::Composite(_) => "Composite".to_string(),
        Value::Foreign(_) => "foreign reference".to_string(),
    }
}

#[cfg(test)]
pub(crate) mod tests;
