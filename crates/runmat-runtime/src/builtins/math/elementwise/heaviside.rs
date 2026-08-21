//! MATLAB-compatible `heaviside` builtin with GPU-aware semantics for RunMat.

use runmat_accelerate_api::{GpuTensorHandle, GpuTensorStorage, ProviderPrecision};
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};
use runmat_macros::runtime_builtin;
use runmat_value::{CharArray, NumericStorage, Tensor, Value};

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, FusionError,
    FusionExprContext, FusionKernelTemplate, GpuOpKind, ProviderHook, ReductionNaN,
    ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, map_control_flow_with_builtin, tensor};
use crate::builtins::math::symbolic::symbolic_function;
use crate::builtins::math::type_resolvers::numeric_unary_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};
use runmat_value::SymbolicFunction;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::elementwise::heaviside")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "heaviside",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Unary {
        name: "unary_heaviside",
    }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "Providers may execute the real Heaviside step function on-device via unary_heaviside; the runtime gathers to the host when the hook is unavailable.",
};

#[runmat_macros::register_fusion_spec(
    builtin_path = "crate::builtins::math::elementwise::heaviside"
)]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "heaviside",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: |ctx: &FusionExprContext| {
            let input = ctx
                .inputs
                .first()
                .ok_or(FusionError::MissingInput(0))?;
            let (zero, half, one) = match ctx.scalar_ty {
                ScalarType::F32 => ("0.0", "0.5", "1.0"),
                ScalarType::F64 => ("f64(0.0)", "f64(0.5)", "f64(1.0)"),
                ScalarType::I32 | ScalarType::Bool => {
                    return Err(FusionError::UnsupportedPrecision(ctx.scalar_ty));
                }
            };
            Ok(format!(
                "select(select(select({zero}, {one}, ({input} > {zero})), {half}, ({input} == {zero})), {input}, isNan({input}))"
            ))
        },
    }),
    reduction: None,
    emits_nan: true,
    notes:
        "Fusion kernels emit the MATLAB Heaviside step function, including H(0) = 0.5 and NaN propagation.",
};

const BUILTIN_NAME: &str = "heaviside";

const HEAVISIDE_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Elementwise Heaviside step result.",
}];

const HEAVISIDE_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Real numeric, logical, or character input.",
}];

const HEAVISIDE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "Y = heaviside(X)",
    inputs: &HEAVISIDE_INPUTS,
    outputs: &HEAVISIDE_OUTPUT,
}];

const HEAVISIDE_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.HEAVISIDE.INVALID_INPUT",
    identifier: Some("RunMat:heaviside:InvalidInput"),
    when: "Input is not a supported real numeric/logical/character value.",
    message: "heaviside: invalid input",
};

const HEAVISIDE_ERROR_PROVIDER_FAILED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.HEAVISIDE.PROVIDER_FAILED",
    identifier: Some("RunMat:heaviside:ProviderFailed"),
    when: "Provider unary_heaviside dispatch fails with a non-unsupported error.",
    message: "heaviside: GPU provider unary_heaviside failed",
};

const HEAVISIDE_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.HEAVISIDE.INTERNAL",
    identifier: Some("RunMat:heaviside:Internal"),
    when: "Internal gather/provider/tensor construction failed.",
    message: "heaviside: internal error",
};

const HEAVISIDE_ERRORS: [BuiltinErrorDescriptor; 3] = [
    HEAVISIDE_ERROR_INVALID_INPUT,
    HEAVISIDE_ERROR_PROVIDER_FAILED,
    HEAVISIDE_ERROR_INTERNAL,
];

const HEAVISIDE_INTEGER_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "heaviside-integer-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "heaviside with an integer input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:HeavisideIntegerInputExtension"),
};
const HEAVISIDE_LOGICAL_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "heaviside-logical-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "heaviside with a logical input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:HeavisideLogicalInputExtension"),
};
const HEAVISIDE_CHARACTER_INPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "heaviside-character-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "heaviside with a character input is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:HeavisideCharacterInputExtension"),
    };
const HEAVISIDE_GPU_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "heaviside-gpu-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "heaviside with a gpuArray input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:HeavisideGpuInputExtension"),
};
const HEAVISIDE_EXTENSIONS: [BuiltinExtensionDescriptor; 4] = [
    HEAVISIDE_INTEGER_INPUT_EXTENSION,
    HEAVISIDE_LOGICAL_INPUT_EXTENSION,
    HEAVISIDE_CHARACTER_INPUT_EXTENSION,
    HEAVISIDE_GPU_INPUT_EXTENSION,
];

const HEAVISIDE_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "X",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "RunMat extension mode accepts all eight real integer classes and classifies sign and zero directly from authoritative integer storage.",
    }];
pub const INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "Y = heaviside(integer_X)",
        inputs: &HEAVISIDE_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
        notes: "Integer sign classification is exact and returns double values 0, 0.5, or 1. Resident integer input is gated before exact owner gather and never reaches a floating provider hook.",
    }];

pub const HEAVISIDE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &HEAVISIDE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &HEAVISIDE_ERRORS,
};

fn heaviside_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl std::fmt::Display,
) -> RuntimeError {
    let mut builder =
        build_runtime_error(format!("{}: {}", error.message, detail)).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[derive(Debug)]
struct ProviderAnyhowError {
    message: String,
}

impl std::fmt::Display for ProviderAnyhowError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for ProviderAnyhowError {}

fn provider_error_is_unsupported(err: &anyhow::Error) -> bool {
    err.chain()
        .any(|cause| cause.to_string() == "unary_heaviside not supported by provider")
}

fn provider_error(err: anyhow::Error) -> RuntimeError {
    let message = err.to_string();
    let source = ProviderAnyhowError {
        message: format!("{err:?}"),
    };
    build_runtime_error(format!(
        "{}: {message}",
        HEAVISIDE_ERROR_PROVIDER_FAILED.message
    ))
    .with_builtin(BUILTIN_NAME)
    .with_identifier(
        HEAVISIDE_ERROR_PROVIDER_FAILED
            .identifier
            .expect("provider failure identifier"),
    )
    .with_source(source)
    .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
    .build()
}

#[runtime_builtin(
    name = "heaviside",
    category = "math/elementwise",
    summary = "Compute the element-wise Heaviside step function.",
    keywords = "heaviside,step,unit step,elementwise,gpu",
    accel = "unary",
    type_resolver(numeric_unary_type),
    descriptor(crate::builtins::math::elementwise::heaviside::HEAVISIDE_DESCRIPTOR),
    extensions(HEAVISIDE_EXTENSIONS),
    integer_capabilities(crate::builtins::math::elementwise::heaviside::INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::math::elementwise::heaviside"
)]
async fn heaviside_builtin(value: Value) -> BuiltinResult<Value> {
    if let Some(symbolic) = symbolic_function(&value, SymbolicFunction::Heaviside) {
        return Ok(symbolic);
    }
    ensure_heaviside_extensions(&value)?;
    match value {
        Value::GpuTensor(handle) => heaviside_gpu(handle).await,
        Value::Complex(_, _) | Value::ComplexTensor(_) => Err(heaviside_error_with_detail(
            &HEAVISIDE_ERROR_INVALID_INPUT,
            "complex inputs are not supported for the real Heaviside step function",
        )),
        Value::CharArray(ca) => heaviside_char_array(ca),
        Value::String(_) | Value::StringArray(_) => Err(heaviside_error_with_detail(
            &HEAVISIDE_ERROR_INVALID_INPUT,
            "expected real numeric, logical, or character input",
        )),
        other => heaviside_real(other),
    }
}

async fn heaviside_gpu(handle: GpuTensorHandle) -> BuiltinResult<Value> {
    let provider = runmat_accelerate_api::provider_for_handle(&handle).ok_or_else(|| {
        terminal_heaviside_error(
            &HEAVISIDE_ERROR_INTERNAL,
            "GPU provider unavailable for input",
        )
    })?;
    let integer_input = runmat_accelerate_api::handle_integer_type(&handle).is_some();
    let logical_input = runmat_accelerate_api::handle_is_logical(&handle);
    if !integer_input && !logical_input {
        match provider.unary_heaviside(&handle).await {
            Ok(out) if valid_gpu_output(&out, &handle, provider) => {
                return Ok(gpu_helpers::resident_gpu_value(out));
            }
            Ok(out) => {
                free_rejected_gpu_output(&out, &handle);
                return Err(terminal_heaviside_error(
                    &HEAVISIDE_ERROR_INTERNAL,
                    "provider unary_heaviside returned malformed output",
                ));
            }
            Err(err) if provider_error_is_unsupported(&err) => {}
            Err(err) => return Err(provider_error(err)),
        }
    }
    let downloaded = gpu_helpers::download_value_preserving_residency_async(provider, &handle)
        .await
        .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
    let tensor = tensor::value_into_tensor_for(BUILTIN_NAME, downloaded)
        .map_err(|detail| heaviside_error_with_detail(&HEAVISIDE_ERROR_INVALID_INPUT, detail))?;
    let output = heaviside_tensor(tensor)?;
    if (integer_input || logical_input) && provider.precision() != ProviderPrecision::F64 {
        return Ok(tensor::tensor_into_value(output));
    }
    restore_heaviside_gpu_output(provider, &handle, &output)
}

fn ensure_heaviside_extensions(value: &Value) -> BuiltinResult<()> {
    let integer = matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_integer_type(handle).is_some());
    if integer {
        crate::compatibility::ensure_builtin_extension_enabled(
            &HEAVISIDE_INTEGER_INPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    let logical = matches!(value, Value::Bool(_) | Value::LogicalArray(_))
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_is_logical(handle));
    if logical {
        crate::compatibility::ensure_builtin_extension_enabled(
            &HEAVISIDE_LOGICAL_INPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if matches!(value, Value::CharArray(_)) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &HEAVISIDE_CHARACTER_INPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if matches!(value, Value::GpuTensor(_)) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &HEAVISIDE_GPU_INPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    Ok(())
}

fn heaviside_real(value: Value) -> BuiltinResult<Value> {
    let tensor = tensor::value_into_tensor_for(BUILTIN_NAME, value)
        .map_err(|e| heaviside_error_with_detail(&HEAVISIDE_ERROR_INVALID_INPUT, e))?;
    Ok(tensor::tensor_into_value(heaviside_tensor(tensor)?))
}

fn heaviside_tensor(tensor: Tensor) -> BuiltinResult<Tensor> {
    let shape = tensor.shape.clone();
    let storage = tensor
        .into_numeric_storage()
        .map_err(|e| heaviside_error_with_detail(&HEAVISIDE_ERROR_INTERNAL, e))?;
    let output = match storage {
        NumericStorage::F64(values) => {
            NumericStorage::F64(values.into_iter().map(heaviside_scalar).collect())
        }
        NumericStorage::F32(values) => NumericStorage::F32(
            values
                .into_iter()
                .map(|value| heaviside_scalar(f64::from(value)) as f32)
                .collect(),
        ),
        NumericStorage::I8(values) => NumericStorage::F64(map_integer_heaviside(values)),
        NumericStorage::I16(values) => NumericStorage::F64(map_integer_heaviside(values)),
        NumericStorage::I32(values) => NumericStorage::F64(map_integer_heaviside(values)),
        NumericStorage::I64(values) => NumericStorage::F64(map_integer_heaviside(values)),
        NumericStorage::U8(values) => NumericStorage::F64(map_integer_heaviside(values)),
        NumericStorage::U16(values) => NumericStorage::F64(map_integer_heaviside(values)),
        NumericStorage::U32(values) => NumericStorage::F64(map_integer_heaviside(values)),
        NumericStorage::U64(values) => NumericStorage::F64(map_integer_heaviside(values)),
    };
    Tensor::from_numeric_storage(output, shape)
        .map_err(|e| heaviside_error_with_detail(&HEAVISIDE_ERROR_INTERNAL, e))
}

fn map_integer_heaviside<T>(values: Vec<T>) -> Vec<f64>
where
    T: PartialOrd + Default,
{
    let zero = T::default();
    values
        .into_iter()
        .map(|value| {
            if value > zero {
                1.0
            } else if value < zero {
                0.0
            } else {
                0.5
            }
        })
        .collect()
}

fn terminal_heaviside_error(
    error: &'static BuiltinErrorDescriptor,
    detail: impl std::fmt::Display,
) -> RuntimeError {
    let mut builder = build_runtime_error(format!("{}: {detail}", error.message))
        .with_builtin(BUILTIN_NAME)
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn restore_heaviside_gpu_output(
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    input: &GpuTensorHandle,
    tensor: &Tensor,
) -> BuiltinResult<Value> {
    let output = gpu_helpers::upload_tensor(provider, tensor).map_err(|err| {
        terminal_heaviside_error(
            &HEAVISIDE_ERROR_INTERNAL,
            format!("failed to restore fallback result to input provider: {err}"),
        )
    })?;
    if !valid_gpu_output(&output, input, provider) {
        free_rejected_gpu_output(&output, input);
        return Err(terminal_heaviside_error(
            &HEAVISIDE_ERROR_INTERNAL,
            "provider upload returned malformed fallback output",
        ));
    }
    Ok(gpu_helpers::resident_gpu_value(output))
}

fn valid_gpu_output(
    output: &GpuTensorHandle,
    input: &GpuTensorHandle,
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
) -> bool {
    output.shape == input.shape
        && output.device_id == input.device_id
        && !gpu_handles_alias(output, input)
        && runmat_accelerate_api::handle_storage(output) == GpuTensorStorage::Real
        && runmat_accelerate_api::handle_integer_type(output).is_none()
        && !runmat_accelerate_api::handle_is_logical(output)
        && runmat_accelerate_api::handle_precision(output) == Some(provider.precision())
        && runmat_accelerate_api::provider_for_handle(output)
            .is_some_and(|owner| std::ptr::eq(owner, provider))
}

fn gpu_handles_alias(lhs: &GpuTensorHandle, rhs: &GpuTensorHandle) -> bool {
    lhs.device_id == rhs.device_id && lhs.buffer_id == rhs.buffer_id
}

fn free_rejected_gpu_output(output: &GpuTensorHandle, input: &GpuTensorHandle) {
    if gpu_handles_alias(output, input) {
        return;
    }
    if let Some(owner) = runmat_accelerate_api::provider_for_handle(output) {
        if owner.free(output).is_ok() {
            runmat_accelerate_api::clear_residency(output);
        }
    }
}

fn heaviside_char_array(ca: CharArray) -> BuiltinResult<Value> {
    let data = ca
        .data
        .iter()
        .map(|&ch| heaviside_scalar(ch as u32 as f64))
        .collect::<Vec<_>>();
    let tensor = Tensor::new(data, vec![ca.rows, ca.cols])
        .map_err(|e| heaviside_error_with_detail(&HEAVISIDE_ERROR_INTERNAL, e))?;
    Ok(Value::Tensor(tensor))
}

#[inline]
fn heaviside_scalar(value: f64) -> f64 {
    if value > 0.0 {
        1.0
    } else if value < 0.0 {
        0.0
    } else if value == 0.0 {
        0.5
    } else {
        value
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_accelerate_api::{
        AccelDownloadFuture, AccelProvider, AccelProviderFuture, GpuTensorHandle, HostTensorOwned,
        HostTensorView,
    };
    use runmat_builtins::{builtin_function_by_name, AccelTag, ResolveContext, Type};
    use runmat_value::{IntValue, LogicalArray, SymbolicExpr};

    fn heaviside_builtin(value: Value) -> BuiltinResult<Value> {
        block_on(super::heaviside_builtin(value))
    }

    #[test]
    fn heaviside_preserves_native_single_storage() {
        let input = Tensor::from_f32(vec![-1.0, 0.0, 2.0], vec![1, 3]).unwrap();
        let output = heaviside_tensor(input).unwrap();
        assert_eq!(
            output.into_numeric_storage().unwrap(),
            NumericStorage::F32(vec![0.0, 0.5, 1.0])
        );
    }

    struct UnsupportedHeavisideProvider;

    impl AccelProvider for UnsupportedHeavisideProvider {
        fn upload(&self, host: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
            let handle = GpuTensorHandle {
                shape: host.shape.to_vec(),
                device_id: self.device_id(),
                buffer_id: 2,
                descriptor: Default::default(),
            }
            .with_numeric_descriptor(
                match self.precision() {
                    ProviderPrecision::F32 => runmat_accelerate_api::NumericElementType::F32,
                    ProviderPrecision::F64 => runmat_accelerate_api::NumericElementType::F64,
                },
                GpuTensorStorage::Real,
            );
            runmat_accelerate_api::set_handle_logical(&handle, false);
            Ok(handle)
        }

        fn download<'a>(&'a self, handle: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
            Box::pin(async move {
                Ok(HostTensorOwned {
                    data: if handle.buffer_id == 2 {
                        vec![0.0, 0.5, 1.0]
                    } else {
                        vec![-1.0, 0.0, 2.0]
                    },
                    shape: vec![1, 3],
                    storage: runmat_accelerate_api::GpuTensorStorage::Real,
                })
            })
        }

        fn free(&self, _: &GpuTensorHandle) -> anyhow::Result<()> {
            Ok(())
        }

        fn device_info(&self) -> String {
            "unsupported-heaviside-test-provider".to_string()
        }

        fn device_id(&self) -> u32 {
            0
        }
    }

    struct FailingHeavisideProvider;

    impl AccelProvider for FailingHeavisideProvider {
        fn upload(&self, host: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
            Ok(GpuTensorHandle {
                shape: host.shape.to_vec(),
                device_id: self.device_id(),
                buffer_id: 1,
                descriptor: Default::default(),
            })
        }

        fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
            Box::pin(async move { Err(anyhow::anyhow!("download should not be called")) })
        }

        fn free(&self, _: &GpuTensorHandle) -> anyhow::Result<()> {
            Ok(())
        }

        fn device_info(&self) -> String {
            "failing-heaviside-test-provider".to_string()
        }

        fn device_id(&self) -> u32 {
            0
        }

        fn unary_heaviside<'a>(
            &'a self,
            _: &'a GpuTensorHandle,
        ) -> AccelProviderFuture<'a, GpuTensorHandle> {
            Box::pin(
                async move { Err(anyhow::anyhow!("device lost while running unary_heaviside")) },
            )
        }
    }

    #[test]
    fn heaviside_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = HEAVISIDE_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"Y = heaviside(X)"));
    }

    #[test]
    fn heaviside_registers_unary_acceleration_tag() {
        let builtin = builtin_function_by_name(BUILTIN_NAME).expect("registered heaviside builtin");
        assert!(
            builtin
                .accel_tags
                .iter()
                .any(|tag| matches!(tag, AccelTag::Unary)),
            "heaviside must be tagged unary so native auto-promotion can keep large numeric inputs on the GPU"
        );
    }

    #[test]
    fn heaviside_type_preserves_tensor_shape() {
        let out = numeric_unary_type(
            &[Type::Tensor {
                shape: Some(vec![Some(2), Some(3)]),
            }],
            &ResolveContext::new(Vec::new()),
        );
        assert_eq!(
            out,
            Type::Tensor {
                shape: Some(vec![Some(2), Some(3)])
            }
        );
    }

    #[test]
    fn heaviside_type_scalar_tensor_returns_num() {
        let out = numeric_unary_type(
            &[Type::Tensor {
                shape: Some(vec![Some(1), Some(1)]),
            }],
            &ResolveContext::new(Vec::new()),
        );
        assert_eq!(out, Type::Num);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn heaviside_scalar_positive_negative_zero() {
        assert_eq!(heaviside_builtin(Value::Num(3.5)).unwrap(), Value::Num(1.0));
        assert_eq!(
            heaviside_builtin(Value::Num(-2.0)).unwrap(),
            Value::Num(0.0)
        );
        assert_eq!(heaviside_builtin(Value::Num(0.0)).unwrap(), Value::Num(0.5));
        assert_eq!(
            heaviside_builtin(Value::Num(-0.0)).unwrap(),
            Value::Num(0.5)
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn heaviside_scalar_nan_propagates() {
        let result = heaviside_builtin(Value::Num(f64::NAN)).unwrap();
        match result {
            Value::Num(value) => assert!(value.is_nan()),
            other => panic!("expected scalar NaN, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn heaviside_tensor_mixed_values() {
        let tensor = Tensor::new(
            vec![f64::NEG_INFINITY, -2.0, -0.0, 0.0, 2.0, f64::INFINITY],
            vec![2, 3],
        )
        .unwrap();
        let result = heaviside_builtin(Value::Tensor(tensor)).unwrap();
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![2, 3]);
                assert_eq!(out.materialize_f64(), vec![0.0, 0.0, 0.5, 0.5, 1.0, 1.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn heaviside_logical_input_promotes() {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let logical = LogicalArray::new(vec![0, 1, 0, 1], vec![2, 2]).unwrap();
        let result = heaviside_builtin(Value::LogicalArray(logical)).unwrap();
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![2, 2]);
                assert_eq!(out.materialize_f64(), vec![0.5, 1.0, 0.5, 1.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn heaviside_bool_values() {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let t = heaviside_builtin(Value::Bool(true)).unwrap();
        let f = heaviside_builtin(Value::Bool(false)).unwrap();
        assert_eq!(t, Value::Num(1.0));
        assert_eq!(f, Value::Num(0.5));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn heaviside_int_values() {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let negative = heaviside_builtin(Value::Int(IntValue::I32(-7))).unwrap();
        let zero = heaviside_builtin(Value::Int(IntValue::I32(0))).unwrap();
        let positive = heaviside_builtin(Value::Int(IntValue::U16(7))).unwrap();
        assert_eq!(negative, Value::Num(0.0));
        assert_eq!(zero, Value::Num(0.5));
        assert_eq!(positive, Value::Num(1.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn heaviside_character_array() {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let ca = CharArray::new("RunMat".chars().collect(), 1, 6).unwrap();
        let result = heaviside_builtin(Value::CharArray(ca)).unwrap();
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![1, 6]);
                assert!(out.materialize_f64().iter().all(|&value| value == 1.0));
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn heaviside_complex_input_errors() {
        let err =
            heaviside_builtin(Value::Complex(1.0, 1.0)).expect_err("expected complex rejection");
        assert_eq!(err.identifier(), HEAVISIDE_ERROR_INVALID_INPUT.identifier);
        assert!(err.message().contains("complex"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn heaviside_string_input_errors() {
        let err = heaviside_builtin(Value::String("runmat".to_string()))
            .expect_err("expected string rejection");
        assert_eq!(err.identifier(), HEAVISIDE_ERROR_INVALID_INPUT.identifier);
        assert!(err.message().contains("expected real numeric"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn heaviside_symbolic_input_returns_symbolic_function() {
        let result = heaviside_builtin(Value::Symbolic(SymbolicExpr::variable("x")))
            .expect("symbolic heaviside");
        match result {
            Value::Symbolic(expr) => assert_eq!(expr.to_string(), "heaviside(x)"),
            other => panic!("expected symbolic result, got {other:?}"),
        }
    }

    #[test]
    fn heaviside_fusion_expression_preserves_nan_and_zero() {
        let template = FUSION_SPEC.elementwise.expect("fusion template");
        let inputs = ["x"];
        let ctx = FusionExprContext {
            scalar_ty: ScalarType::F64,
            inputs: &inputs,
            constants: &[],
        };
        let expr = (template.wgsl_body)(&ctx).expect("fusion expression");
        assert!(expr.contains("isNan(x)"));
        assert!(expr.contains("f64(0.5)"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn heaviside_gpu_provider_roundtrip() {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![-3.0, -0.0, 0.0, 2.5], vec![2, 2]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.materialize_f64(),
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = heaviside_builtin(Value::GpuTensor(handle)).expect("heaviside");
            assert!(
                matches!(result, Value::GpuTensor(_)),
                "provider-backed heaviside should keep gpuArray output resident"
            );
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![2, 2]);
            assert_eq!(gathered.materialize_f64(), vec![0.0, 0.5, 0.5, 1.0]);
        });
    }

    #[test]
    fn heaviside_gpu_falls_back_only_for_explicit_unsupported_hook() {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let _guard = test_support::accel_test_lock();
        let provider: &'static dyn AccelProvider =
            Box::leak(Box::new(UnsupportedHeavisideProvider));
        let _thread_provider = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
        let handle = GpuTensorHandle {
            shape: vec![1, 3],
            device_id: provider.device_id(),
            buffer_id: 1,
            descriptor: Default::default(),
        }
        .with_numeric_descriptor(
            runmat_accelerate_api::NumericElementType::F64,
            GpuTensorStorage::Real,
        );
        let source_metadata = (
            runmat_accelerate_api::handle_storage(&handle),
            runmat_accelerate_api::handle_precision(&handle),
            runmat_accelerate_api::handle_integer_type(&handle),
            runmat_accelerate_api::handle_is_logical(&handle),
        );
        let result =
            heaviside_builtin(Value::GpuTensor(handle.clone())).expect("heaviside gpu fallback");
        assert!(runmat_accelerate_api::provider_for_handle(&handle).is_some());
        assert_eq!(
            runmat_accelerate_api::handle_storage(&handle),
            source_metadata.0
        );
        assert_eq!(
            runmat_accelerate_api::handle_precision(&handle),
            source_metadata.1
        );
        assert_eq!(
            runmat_accelerate_api::handle_integer_type(&handle),
            source_metadata.2
        );
        assert_eq!(
            runmat_accelerate_api::handle_is_logical(&handle),
            source_metadata.3
        );
        assert!(matches!(result, Value::GpuTensor(_)));
        let out = test_support::gather(result).expect("gather restored fallback");
        assert_eq!(out.shape, vec![1, 3]);
        assert_eq!(out.materialize_f64(), vec![0.0, 0.5, 1.0]);
    }

    #[test]
    fn heaviside_gpu_propagates_provider_execution_errors() {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let _guard = test_support::accel_test_lock();
        let provider: &'static dyn AccelProvider = Box::leak(Box::new(FailingHeavisideProvider));
        let _thread_provider = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
        let handle = GpuTensorHandle {
            shape: vec![1, 3],
            device_id: provider.device_id(),
            buffer_id: 1,
            descriptor: Default::default(),
        };
        let err =
            heaviside_builtin(Value::GpuTensor(handle)).expect_err("provider error should surface");
        assert!(err
            .message()
            .contains("heaviside: GPU provider unary_heaviside failed"));
        assert!(err
            .message()
            .contains("device lost while running unary_heaviside"));
        assert_eq!(err.gpu_gather_retry(), crate::GpuGatherRetry::Never);
    }

    #[test]
    fn heaviside_runmat_extensions_follow_compatibility_mode() {
        for (value, identifier) in [
            (
                Value::Int(IntValue::I32(1)),
                "RunMat:compatibility:HeavisideIntegerInputExtension",
            ),
            (
                Value::Bool(true),
                "RunMat:compatibility:HeavisideLogicalInputExtension",
            ),
            (
                Value::CharArray(CharArray::new(vec!['A'], 1, 1).unwrap()),
                "RunMat:compatibility:HeavisideCharacterInputExtension",
            ),
        ] {
            let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
            let err = heaviside_builtin(value).expect_err("strict mode rejects extension");
            assert_eq!(err.identifier(), Some(identifier));
            assert_eq!(err.gpu_gather_retry(), crate::GpuGatherRetry::Never);
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn heaviside_wgpu_matches_cpu() {
        let _guard = test_support::accel_test_lock();
        let Ok(provider) = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        ) else {
            return;
        };
        let tensor = Tensor::new(vec![-3.0, -0.0, 0.0, 4.0, f64::NAN], vec![1, 5]).unwrap();
        let cpu = heaviside_real(Value::Tensor(tensor.clone())).unwrap();
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.materialize_f64(),
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).unwrap();
        let gpu = block_on(heaviside_gpu(handle)).unwrap();
        let gathered = test_support::gather(gpu).expect("gather");
        match cpu {
            Value::Tensor(cpu_tensor) => {
                assert_eq!(gathered.shape, cpu_tensor.shape);
                for (actual, expected) in gathered
                    .materialize_f64()
                    .iter()
                    .zip(cpu_tensor.materialize_f64().iter())
                {
                    if actual.is_nan() && expected.is_nan() {
                        continue;
                    }
                    assert_eq!(actual, expected);
                }
            }
            other => panic!("unexpected cpu result {other:?}"),
        }
    }
}
