//! MATLAB-compatible `cos` builtin with GPU-aware semantics for RunMat.

use runmat_accelerate_api::{GpuTensorHandle, HostTensorView};
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    CharArray, ComplexTensor, Tensor, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::random_args::{complex_tensor_into_value, keyword_of};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, FusionError,
    FusionExprContext, FusionKernelTemplate, GpuOpKind, ProviderHook, ReductionNaN,
    ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, map_control_flow_with_builtin, tensor};
use crate::builtins::math::symbolic::symbolic_function;
use crate::builtins::math::type_resolvers::numeric_unary_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};
use runmat_builtins::SymbolicFunction;

const BUILTIN_NAME: &str = "cos";

const COS_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Element-wise cosine result.",
}];

const COS_INPUTS_X: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input scalar, array, char array, complex value, or gpuArray.",
}];

const COS_INPUTS_X_LIKE_P: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input scalar, array, char array, complex value, or gpuArray.",
    },
    BuiltinParamDescriptor {
        name: "like",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\"like\""),
        description: "Output template selector keyword.",
    },
    BuiltinParamDescriptor {
        name: "P",
        ty: BuiltinParamType::LikePrototype,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Prototype determining host vs gpuArray output residency.",
    },
];

const COS_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "Y = cos(X)",
        inputs: &COS_INPUTS_X,
        outputs: &COS_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Y = cos(X, \"like\", P)",
        inputs: &COS_INPUTS_X_LIKE_P,
        outputs: &COS_OUTPUT,
    },
];

const COS_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.COS.INVALID_INPUT",
    identifier: Some("RunMat:cos:InvalidInput"),
    when: "Input cannot be interpreted as supported numeric/logical/char/complex data.",
    message: "cos: invalid input",
};

const COS_ERROR_INVALID_OPTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.COS.INVALID_OPTION",
    identifier: Some("RunMat:cos:InvalidOption"),
    when: "Optional arguments after X are malformed or unsupported.",
    message: "cos: invalid option",
};

const COS_ERROR_ARG_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.COS.ARG_COUNT",
    identifier: Some("RunMat:cos:ArgCount"),
    when: "Too many input arguments were supplied.",
    message: "cos: too many input arguments",
};

const COS_ERROR_LIKE_PROTOTYPE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.COS.LIKE_PROTOTYPE",
    identifier: Some("RunMat:cos:LikePrototype"),
    when: "The \"like\" prototype is unsupported for this output conversion path.",
    message: "cos: invalid \"like\" prototype",
};

const COS_ERROR_GPU_UNAVAILABLE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.COS.GPU_UNAVAILABLE",
    identifier: Some("RunMat:cos:GpuUnavailable"),
    when: "GPU output was requested via \"like\" but no active provider is available.",
    message: "cos: GPU provider unavailable",
};

const COS_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.COS.INTERNAL",
    identifier: Some("RunMat:cos:Internal"),
    when: "Internal tensor conversion/allocation/provider flow failed.",
    message: "cos: internal error",
};

const COS_ERRORS: [BuiltinErrorDescriptor; 6] = [
    COS_ERROR_INVALID_INPUT,
    COS_ERROR_INVALID_OPTION,
    COS_ERROR_ARG_COUNT,
    COS_ERROR_LIKE_PROTOTYPE,
    COS_ERROR_GPU_UNAVAILABLE,
    COS_ERROR_INTERNAL,
];

pub const COS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &COS_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &COS_ERRORS,
};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::trigonometry::cos")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "cos",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Unary { name: "unary_cos" }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "Providers may execute cosine directly on device; runtimes gather to host when unary_cos is unavailable.",
};

fn cos_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    let mut builder = build_runtime_error(error.message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn cos_error_with_detail(
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

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::trigonometry::cos")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "cos",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: |ctx: &FusionExprContext| {
            let input = ctx.inputs.first().ok_or(FusionError::MissingInput(0))?;
            Ok(format!("cos({input})"))
        },
    }),
    reduction: None,
    emits_nan: false,
    notes: "Fusion planner emits WGSL `cos` calls; providers can override via fused elementwise kernels.",
};

#[runtime_builtin(
    name = "cos",
    category = "math/trigonometry",
    summary = "Compute cosine element-wise.",
    keywords = "cos,cosine,trigonometry,gpu",
    accel = "unary",
    type_resolver(numeric_unary_type),
    descriptor(crate::builtins::math::trigonometry::cos::COS_DESCRIPTOR),
    builtin_path = "crate::builtins::math::trigonometry::cos"
)]
async fn cos_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let template = parse_output_template(&rest)?;
    if let Some(symbolic) = symbolic_function(&value, SymbolicFunction::Cos) {
        return apply_output_template(symbolic, &template).await;
    }
    let base = match value {
        Value::GpuTensor(handle) => cos_gpu(handle).await?,
        Value::Complex(re, im) => Value::Complex(cos_complex_re(re, im), cos_complex_im(re, im)),
        Value::ComplexTensor(ct) => cos_complex_tensor(ct)?,
        Value::CharArray(ca) => cos_char_array(ca)?,
        Value::String(_) | Value::StringArray(_) => {
            return Err(cos_error_with_detail(
                &COS_ERROR_INVALID_INPUT,
                "expected numeric input, got string",
            ))
        }
        other => cos_real(other)?,
    };
    apply_output_template(base, &template).await
}

async fn cos_gpu(handle: GpuTensorHandle) -> BuiltinResult<Value> {
    if let Some(provider) = runmat_accelerate_api::provider_for_handle(&handle) {
        if let Ok(out) = provider.unary_cos(&handle).await {
            return Ok(Value::GpuTensor(out));
        }
    }
    let gathered = gpu_helpers::gather_value_async(&Value::GpuTensor(handle))
        .await
        .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
    match gathered {
        Value::Complex(re, im) => Ok(Value::Complex(
            cos_complex_re(re, im),
            cos_complex_im(re, im),
        )),
        Value::ComplexTensor(ct) => cos_complex_tensor(ct),
        Value::Tensor(tensor) => cos_tensor(tensor).map(tensor::tensor_into_value),
        Value::Num(n) => Ok(Value::Num(n.cos())),
        other => cos_real(other),
    }
}

fn cos_real(value: Value) -> BuiltinResult<Value> {
    let tensor = tensor::value_into_tensor_for("cos", value)
        .map_err(|e| cos_error_with_detail(&COS_ERROR_INVALID_INPUT, e))?;
    cos_tensor(tensor).map(tensor::tensor_into_value)
}

fn cos_tensor(tensor: Tensor) -> BuiltinResult<Tensor> {
    let data = tensor.data.iter().map(|&v| v.cos()).collect::<Vec<_>>();
    Tensor::new(data, tensor.shape.clone())
        .map_err(|e| cos_error_with_detail(&COS_ERROR_INTERNAL, e))
}

fn cos_complex_tensor(ct: ComplexTensor) -> BuiltinResult<Value> {
    let mapped = ct
        .data
        .iter()
        .map(|&(re, im)| (cos_complex_re(re, im), cos_complex_im(re, im)))
        .collect::<Vec<_>>();
    let tensor = ComplexTensor::new(mapped, ct.shape.clone())
        .map_err(|e| cos_error_with_detail(&COS_ERROR_INTERNAL, e))?;
    Ok(complex_tensor_into_value(tensor))
}

fn cos_char_array(ca: CharArray) -> BuiltinResult<Value> {
    let data = ca
        .data
        .iter()
        .map(|&ch| (ch as u32 as f64).cos())
        .collect::<Vec<_>>();
    let tensor = Tensor::new(data, vec![ca.rows, ca.cols])
        .map_err(|e| cos_error_with_detail(&COS_ERROR_INTERNAL, e))?;
    Ok(tensor::tensor_into_value(tensor))
}

#[inline]
fn cos_complex_re(re: f64, im: f64) -> f64 {
    re.cos() * im.cosh()
}

#[inline]
fn cos_complex_im(re: f64, im: f64) -> f64 {
    -re.sin() * im.sinh()
}

#[derive(Clone)]
enum OutputTemplate {
    Default,
    Like(Value),
}

fn parse_output_template(args: &[Value]) -> BuiltinResult<OutputTemplate> {
    match args.len() {
        0 => Ok(OutputTemplate::Default),
        1 => {
            if matches!(keyword_of(&args[0]).as_deref(), Some("like")) {
                Err(cos_error_with_detail(
                    &COS_ERROR_INVALID_OPTION,
                    "expected prototype after 'like'",
                ))
            } else {
                Err(cos_error_with_detail(
                    &COS_ERROR_INVALID_OPTION,
                    "unrecognised argument for cos",
                ))
            }
        }
        2 => {
            if matches!(keyword_of(&args[0]).as_deref(), Some("like")) {
                Ok(OutputTemplate::Like(args[1].clone()))
            } else {
                Err(cos_error_with_detail(
                    &COS_ERROR_INVALID_OPTION,
                    "unsupported option; only 'like' is accepted",
                ))
            }
        }
        _ => Err(cos_error(&COS_ERROR_ARG_COUNT)),
    }
}

async fn apply_output_template(value: Value, template: &OutputTemplate) -> BuiltinResult<Value> {
    match template {
        OutputTemplate::Default => Ok(value),
        OutputTemplate::Like(proto) => match proto {
            Value::GpuTensor(handle) => {
                if runmat_accelerate_api::handle_storage(handle)
                    == runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
                {
                    convert_to_gpu_complex(value).await
                } else {
                    convert_to_gpu(value)
                }
            }
            Value::Tensor(_)
            | Value::Num(_)
            | Value::Int(_)
            | Value::Bool(_)
            | Value::LogicalArray(_) => convert_to_host_like(value).await,
            Value::Complex(_, _) | Value::ComplexTensor(_) => convert_to_host_complex(value).await,
            _ => Err(cos_error_with_detail(
                &COS_ERROR_LIKE_PROTOTYPE,
                "unsupported prototype; provide a numeric or gpuArray prototype",
            )),
        },
    }
}

fn convert_to_gpu(value: Value) -> BuiltinResult<Value> {
    let provider = runmat_accelerate_api::provider().ok_or_else(|| {
        cos_error_with_detail(
            &COS_ERROR_GPU_UNAVAILABLE,
            "GPU output requested via 'like' but no acceleration provider is active",
        )
    })?;
    match value {
        Value::GpuTensor(handle) => Ok(Value::GpuTensor(handle)),
        Value::Tensor(tensor) => {
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider
                .upload(&view)
                .map_err(|e| cos_error_with_detail(&COS_ERROR_INTERNAL, e))?;
            Ok(Value::GpuTensor(handle))
        }
        Value::Num(n) => {
            let tensor = Tensor::new(vec![n], vec![1, 1])
                .map_err(|e| cos_error_with_detail(&COS_ERROR_INTERNAL, e))?;
            convert_to_gpu(Value::Tensor(tensor))
        }
        Value::Int(i) => convert_to_gpu(Value::Num(i.to_f64())),
        Value::Bool(b) => convert_to_gpu(Value::Num(if b { 1.0 } else { 0.0 })),
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(&logical)
                .map_err(|e| cos_error_with_detail(&COS_ERROR_INTERNAL, e))?;
            convert_to_gpu(Value::Tensor(tensor))
        }
        Value::Complex(_, _) | Value::ComplexTensor(_) => Err(cos_error_with_detail(
            &COS_ERROR_LIKE_PROTOTYPE,
            "GPU prototypes for 'like' only support real numeric outputs",
        )),
        other => Err(cos_error_with_detail(
            &COS_ERROR_INTERNAL,
            format!("unsupported result type for GPU output via 'like' ({other:?})"),
        )),
    }
}

#[async_recursion::async_recursion(?Send)]
async fn convert_to_gpu_complex(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::GpuTensor(handle) => {
            if runmat_accelerate_api::handle_storage(&handle)
                == runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
            {
                Ok(Value::GpuTensor(handle))
            } else if let Some(handle_provider) =
                runmat_accelerate_api::provider_for_handle(&handle)
            {
                match handle_provider.complex_from_real(&handle).await {
                    Ok(out) => Ok(Value::GpuTensor(out)),
                    Err(_) => {
                        let gathered = gpu_helpers::gather_value_async(&Value::GpuTensor(handle))
                            .await
                            .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
                        convert_to_gpu_complex(gathered).await
                    }
                }
            } else {
                Err(cos_error_with_detail(
                    &COS_ERROR_GPU_UNAVAILABLE,
                    "complex GPU output requested but the input handle has no provider",
                ))
            }
        }
        Value::Complex(re, im) => {
            let provider = runmat_accelerate_api::provider().ok_or_else(|| {
                cos_error_with_detail(
                    &COS_ERROR_GPU_UNAVAILABLE,
                    "complex GPU output requested via 'like' but no acceleration provider is active",
                )
            })?;
            let tensor = ComplexTensor::new(vec![(re, im)], vec![1, 1])
                .map_err(|e| cos_error_with_detail(&COS_ERROR_INTERNAL, e))?;
            let handle = gpu_helpers::upload_complex_tensor(provider, &tensor)
                .map_err(|e| cos_error_with_detail(&COS_ERROR_INTERNAL, e))?;
            Ok(Value::GpuTensor(handle))
        }
        Value::ComplexTensor(tensor) => {
            let provider = runmat_accelerate_api::provider().ok_or_else(|| {
                cos_error_with_detail(
                    &COS_ERROR_GPU_UNAVAILABLE,
                    "complex GPU output requested via 'like' but no acceleration provider is active",
                )
            })?;
            let handle = gpu_helpers::upload_complex_tensor(provider, &tensor)
                .map_err(|e| cos_error_with_detail(&COS_ERROR_INTERNAL, e))?;
            Ok(Value::GpuTensor(handle))
        }
        Value::Num(n) => convert_to_gpu_complex(Value::Complex(n, 0.0)).await,
        Value::Tensor(tensor) => {
            let data = tensor.data.iter().map(|&re| (re, 0.0)).collect::<Vec<_>>();
            let complex = ComplexTensor::new(data, tensor.shape.clone())
                .map_err(|e| cos_error_with_detail(&COS_ERROR_INTERNAL, e))?;
            convert_to_gpu_complex(Value::ComplexTensor(complex)).await
        }
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(&logical)
                .map_err(|e| cos_error_with_detail(&COS_ERROR_INTERNAL, e))?;
            convert_to_gpu_complex(Value::Tensor(tensor)).await
        }
        Value::Int(i) => convert_to_gpu_complex(Value::Num(i.to_f64())).await,
        Value::Bool(b) => convert_to_gpu_complex(Value::Num(if b { 1.0 } else { 0.0 })).await,
        other => Err(cos_error_with_detail(
            &COS_ERROR_INTERNAL,
            format!("cannot convert value {other:?} to complex GPU output via 'like'"),
        )),
    }
}

async fn convert_to_host_like(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::GpuTensor(handle) => {
            let proxy = Value::GpuTensor(handle);
            gpu_helpers::gather_value_async(&proxy).await
        }
        other => Ok(other),
    }
}

#[async_recursion::async_recursion(?Send)]
async fn convert_to_host_complex(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::Complex(_, _) | Value::ComplexTensor(_) => Ok(value),
        Value::Num(n) => Ok(Value::Complex(n, 0.0)),
        Value::Tensor(tensor) => {
            let data = tensor.data.iter().map(|&re| (re, 0.0)).collect::<Vec<_>>();
            let complex = ComplexTensor::new(data, tensor.shape.clone())
                .map_err(|e| cos_error_with_detail(&COS_ERROR_INTERNAL, e))?;
            Ok(complex_tensor_into_value(complex))
        }
        Value::GpuTensor(handle) => {
            let gathered = gpu_helpers::gather_value_async(&Value::GpuTensor(handle))
                .await
                .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
            convert_to_host_complex(gathered).await
        }
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(&logical)
                .map_err(|e| cos_error_with_detail(&COS_ERROR_INTERNAL, e))?;
            convert_to_host_complex(Value::Tensor(tensor)).await
        }
        Value::Int(i) => convert_to_host_complex(Value::Num(i.to_f64())).await,
        Value::Bool(b) => convert_to_host_complex(Value::Num(if b { 1.0 } else { 0.0 })).await,
        other => Err(cos_error_with_detail(
            &COS_ERROR_INTERNAL,
            format!("cannot convert value {other:?} to complex output via 'like'"),
        )),
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::{IntValue, ResolveContext, StringArray, Tensor, Type};

    use crate::builtins::common::{gpu_helpers, test_support};

    fn cos_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        block_on(super::cos_builtin(value, rest))
    }

    #[test]
    fn cos_descriptor_signatures_cover_like_overload() {
        let labels: Vec<&str> = COS_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"Y = cos(X)"));
        assert!(labels.contains(&"Y = cos(X, \"like\", P)"));
    }

    fn error_message(err: RuntimeError) -> String {
        err.message().to_string()
    }

    #[test]
    fn cos_type_preserves_tensor_shape() {
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
    fn cos_type_scalar_tensor_returns_num() {
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
    fn cos_scalar_zero() {
        let result = cos_builtin(Value::Num(0.0), Vec::new()).expect("cos");
        match result {
            Value::Num(v) => assert!((v - 1.0).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cos_tensor_elements() {
        let tensor = Tensor::new(vec![0.0, std::f64::consts::PI], vec![2, 1]).unwrap();
        let result = cos_builtin(Value::Tensor(tensor), Vec::new()).expect("cos");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 1]);
                assert!((t.data[0] - 1.0).abs() < 1e-12);
                assert!((t.data[1] + 1.0).abs() < 1e-12);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cos_int_value_promotes() {
        let value = Value::Int(IntValue::I32(1));
        let result = cos_builtin(value, Vec::new()).expect("cos");
        match result {
            Value::Num(v) => assert!((v - 1.0f64.cos()).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cos_complex_scalar() {
        let result = cos_builtin(Value::Complex(1.0, 2.0), Vec::new()).expect("cos");
        match result {
            Value::Complex(re, im) => {
                assert!((re - (1.0f64.cos() * 2.0f64.cosh())).abs() < 1e-12);
                assert!((im + (1.0f64.sin() * 2.0f64.sinh())).abs() < 1e-12);
            }
            other => panic!("expected complex result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cos_char_array_roundtrip() {
        let chars = CharArray::new("abc".chars().collect(), 1, 3).unwrap();
        let result = cos_builtin(Value::CharArray(chars), Vec::new()).expect("cos");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 3]);
                for (idx, ch) in ['a', 'b', 'c'].into_iter().enumerate() {
                    let expected = (ch as u32 as f64).cos();
                    assert!((t.data[idx] - expected).abs() < 1e-12);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cos_gpu_provider_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![0.0, 1.0, 2.0, 3.0], vec![4, 1]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = cos_builtin(Value::GpuTensor(handle), Vec::new()).expect("cos");
            let gathered = test_support::gather(result).expect("gather");
            let expected: Vec<f64> = tensor.data.iter().map(|&v| v.cos()).collect();
            assert_eq!(gathered.shape, vec![4, 1]);
            assert_eq!(gathered.data, expected);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cos_like_missing_prototype_errors() {
        let err =
            cos_builtin(Value::Num(1.0), vec![Value::from("like")]).expect_err("expected error");
        assert_eq!(err.identifier(), COS_ERROR_INVALID_OPTION.identifier);
        let message = error_message(err);
        assert!(message.contains("prototype"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cos_like_complex_prototype_returns_complex() {
        let result = cos_builtin(
            Value::Num(1.0),
            vec![Value::from("like"), Value::Complex(0.0, 1.0)],
        )
        .expect("cos");
        match result {
            Value::Complex(re, im) => {
                assert!((re - 1.0_f64.cos()).abs() < 1e-12);
                assert!(im.abs() < 1e-12);
            }
            other => panic!("expected complex result, got {other:?}"),
        }
    }

    #[test]
    fn cos_gpu_complex_input_preserves_complex_output() {
        test_support::with_test_provider(|provider| {
            let input = ComplexTensor::new(vec![(0.5, 0.75), (2.0, -0.25)], vec![1, 2]).unwrap();
            let handle = gpu_helpers::upload_complex_tensor(provider, &input).expect("upload");
            let result = cos_builtin(Value::GpuTensor(handle), Vec::new()).expect("cos");
            let out = match result {
                Value::GpuTensor(handle) => {
                    assert_eq!(
                        runmat_accelerate_api::handle_storage(&handle),
                        runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
                    );
                    match block_on(gpu_helpers::gather_value_async(&Value::GpuTensor(handle)))
                        .expect("gather")
                    {
                        Value::ComplexTensor(out) => out,
                        other => panic!("expected gathered complex tensor, got {other:?}"),
                    }
                }
                Value::ComplexTensor(out) => out,
                other => panic!("expected complex output, got {other:?}"),
            };
            assert_eq!(out.shape, vec![1, 2]);
            for (idx, &(re, im)) in input.data.iter().enumerate() {
                assert!((out.data[idx].0 - cos_complex_re(re, im)).abs() < 1e-12);
                assert!((out.data[idx].1 - cos_complex_im(re, im)).abs() < 1e-12);
            }
        });
    }

    #[test]
    fn cos_like_complex_gpu_prototype_uploads_complex_result() {
        test_support::with_test_provider(|provider| {
            let input = Tensor::new(vec![0.0, 1.0], vec![2, 1]).unwrap();
            let proto_tensor = ComplexTensor::new(vec![(0.0, 1.0)], vec![1, 1]).unwrap();
            let proto = gpu_helpers::upload_complex_tensor(provider, &proto_tensor)
                .expect("upload complex prototype");
            let result = cos_builtin(
                Value::Tensor(input.clone()),
                vec![Value::from("like"), Value::GpuTensor(proto)],
            )
            .expect("cos");
            let Value::GpuTensor(handle) = result else {
                panic!("expected complex gpu tensor, got {result:?}");
            };
            assert_eq!(
                runmat_accelerate_api::handle_storage(&handle),
                runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
            );
            let gathered =
                block_on(gpu_helpers::gather_value_async(&Value::GpuTensor(handle))).unwrap();
            let Value::ComplexTensor(out) = gathered else {
                panic!("expected gathered complex tensor, got {gathered:?}");
            };
            assert_eq!(out.shape, vec![2, 1]);
            for (idx, &re) in input.data.iter().enumerate() {
                assert!((out.data[idx].0 - re.cos()).abs() < 1e-12);
                assert!(out.data[idx].1.abs() < 1e-12);
            }
        });
    }

    #[test]
    fn cos_like_complex_gpu_prototype_converts_resident_real_gpu_result() {
        test_support::with_test_provider(|provider| {
            let input = Tensor::new(vec![0.0, 1.0], vec![2, 1]).unwrap();
            let input_view = HostTensorView {
                data: &input.data,
                shape: &input.shape,
            };
            let input_handle = provider.upload(&input_view).expect("upload input");
            let proto_tensor = ComplexTensor::new(vec![(0.0, 1.0)], vec![1, 1]).unwrap();
            let proto = gpu_helpers::upload_complex_tensor(provider, &proto_tensor)
                .expect("upload complex prototype");
            let result = cos_builtin(
                Value::GpuTensor(input_handle),
                vec![Value::from("like"), Value::GpuTensor(proto)],
            )
            .expect("cos");
            let Value::GpuTensor(handle) = result else {
                panic!("expected complex gpu tensor, got {result:?}");
            };
            assert_eq!(
                runmat_accelerate_api::handle_storage(&handle),
                runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
            );
            let gathered =
                block_on(gpu_helpers::gather_value_async(&Value::GpuTensor(handle))).unwrap();
            let Value::ComplexTensor(out) = gathered else {
                panic!("expected gathered complex tensor, got {gathered:?}");
            };
            assert_eq!(out.shape, vec![2, 1]);
            for (idx, &re) in input.data.iter().enumerate() {
                assert!((out.data[idx].0 - re.cos()).abs() < 1e-12);
                assert!(out.data[idx].1.abs() < 1e-12);
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cos_like_gpu_prototype() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![0.0, 1.0, 2.0, 3.0], vec![4, 1]).unwrap();
            let proto_view = HostTensorView {
                data: &[0.0],
                shape: &[1, 1],
            };
            let proto = provider.upload(&proto_view).expect("upload");
            let result = cos_builtin(
                Value::Tensor(tensor.clone()),
                vec![Value::from("like"), Value::GpuTensor(proto.clone())],
            )
            .expect("cos");
            match result {
                Value::GpuTensor(handle) => {
                    let gathered = test_support::gather(Value::GpuTensor(handle)).expect("gather");
                    let expected: Vec<f64> = tensor.data.iter().map(|&v| v.cos()).collect();
                    assert_eq!(gathered.shape, vec![4, 1]);
                    assert_eq!(gathered.data, expected);
                }
                other => panic!("expected GPU tensor, got {other:?}"),
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cos_like_host_with_gpu_input_gathers() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![0.0, 1.0], vec![2, 1]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = cos_builtin(
                Value::GpuTensor(handle),
                vec![Value::from("like"), Value::Num(0.0)],
            )
            .expect("cos");
            match result {
                Value::Tensor(t) => {
                    let expected: Vec<f64> = tensor.data.iter().map(|&v| v.cos()).collect();
                    assert_eq!(t.shape, vec![2, 1]);
                    assert_eq!(t.data, expected);
                }
                Value::GpuTensor(_) => panic!("expected host result"),
                Value::Num(_) => panic!("expected vector output"),
                other => panic!("unexpected result {other:?}"),
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cos_like_rejects_extra_arguments() {
        let err = cos_builtin(
            Value::Num(0.0),
            vec![Value::from("like"), Value::Num(0.0), Value::Num(1.0)],
        )
        .expect_err("expected error");
        let message = error_message(err);
        assert!(message.contains("too many input arguments"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cos_like_keyword_case_insensitive() {
        let tensor = Tensor::new(vec![0.0, 1.0], vec![2, 1]).unwrap();
        let result = cos_builtin(
            Value::Tensor(tensor.clone()),
            vec![Value::from("LIKE"), Value::Num(0.0)],
        )
        .expect("cos");
        match result {
            Value::Tensor(out) => {
                let expected: Vec<f64> = tensor.data.iter().map(|&v| v.cos()).collect();
                assert_eq!(out.shape, vec![2, 1]);
                assert_eq!(out.data, expected);
            }
            other => panic!("unexpected result {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cos_like_char_array_keyword() {
        let keyword = CharArray::new_row("like");
        let result = cos_builtin(
            Value::Num(0.0),
            vec![Value::CharArray(keyword), Value::Num(0.0)],
        )
        .expect("cos");
        match result {
            Value::Num(v) => assert!((v - 1.0).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cos_like_string_array_keyword() {
        let keyword = StringArray::new(vec!["LIKE".to_string()], vec![1]).unwrap();
        let result = cos_builtin(
            Value::Num(0.0),
            vec![Value::StringArray(keyword), Value::Num(0.0)],
        )
        .expect("cos");
        match result {
            Value::Num(v) => assert!((v - 1.0).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cos_unrecognised_option_errors() {
        let err =
            cos_builtin(Value::Num(0.0), vec![Value::from("invalid")]).expect_err("expected error");
        let message = error_message(err);
        assert!(message.contains("unrecognised argument"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn cos_wgpu_matches_cpu_elementwise() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let t = Tensor::new(vec![0.0, 1.0, 2.0, 3.0], vec![4, 1]).unwrap();
        let cpu = cos_real(Value::Tensor(t.clone())).unwrap();
        let view = HostTensorView {
            data: &t.data,
            shape: &t.shape,
        };
        let h = runmat_accelerate_api::provider()
            .unwrap()
            .upload(&view)
            .unwrap();
        let gpu = block_on(cos_gpu(h)).unwrap();
        let gathered = test_support::gather(gpu).expect("gather");
        match (cpu, gathered) {
            (Value::Tensor(ct), gt) => {
                assert_eq!(gt.shape, ct.shape);
                let tol = match runmat_accelerate_api::provider().unwrap().precision() {
                    runmat_accelerate_api::ProviderPrecision::F64 => 1e-12,
                    runmat_accelerate_api::ProviderPrecision::F32 => 1e-5,
                };
                for (a, b) in gt.data.iter().zip(ct.data.iter()) {
                    assert!((a - b).abs() < tol, "|{} - {}| >= {}", a, b, tol);
                }
            }
            _ => panic!("unexpected shapes"),
        }
    }
}
