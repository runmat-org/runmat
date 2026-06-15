//! MATLAB-compatible `tan` builtin with GPU-aware semantics for RunMat.

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

const BUILTIN_NAME: &str = "tan";

const TAN_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Element-wise tangent result.",
}];

const TAN_INPUTS_X: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input scalar, array, char array, complex value, or gpuArray.",
}];

const TAN_INPUTS_X_LIKE_P: [BuiltinParamDescriptor; 3] = [
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

const TAN_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "Y = tan(X)",
        inputs: &TAN_INPUTS_X,
        outputs: &TAN_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Y = tan(X, \"like\", P)",
        inputs: &TAN_INPUTS_X_LIKE_P,
        outputs: &TAN_OUTPUT,
    },
];

const TAN_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TAN.INVALID_INPUT",
    identifier: Some("RunMat:tan:InvalidInput"),
    when: "Input cannot be interpreted as supported numeric/logical/char/complex data.",
    message: "tan: invalid input",
};

const TAN_ERROR_INVALID_OPTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TAN.INVALID_OPTION",
    identifier: Some("RunMat:tan:InvalidOption"),
    when: "Optional arguments after X are malformed or unsupported.",
    message: "tan: invalid option",
};

const TAN_ERROR_ARG_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TAN.ARG_COUNT",
    identifier: Some("RunMat:tan:ArgCount"),
    when: "Too many input arguments were supplied.",
    message: "tan: too many input arguments",
};

const TAN_ERROR_LIKE_PROTOTYPE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TAN.LIKE_PROTOTYPE",
    identifier: Some("RunMat:tan:LikePrototype"),
    when: "The \"like\" prototype is unsupported for this output conversion path.",
    message: "tan: invalid \"like\" prototype",
};

const TAN_ERROR_GPU_UNAVAILABLE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TAN.GPU_UNAVAILABLE",
    identifier: Some("RunMat:tan:GpuUnavailable"),
    when: "GPU output was requested via \"like\" but no active provider is available.",
    message: "tan: GPU provider unavailable",
};

const TAN_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TAN.INTERNAL",
    identifier: Some("RunMat:tan:Internal"),
    when: "Internal tensor conversion/allocation/provider flow failed.",
    message: "tan: internal error",
};

const TAN_ERRORS: [BuiltinErrorDescriptor; 6] = [
    TAN_ERROR_INVALID_INPUT,
    TAN_ERROR_INVALID_OPTION,
    TAN_ERROR_ARG_COUNT,
    TAN_ERROR_LIKE_PROTOTYPE,
    TAN_ERROR_GPU_UNAVAILABLE,
    TAN_ERROR_INTERNAL,
];

pub const TAN_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &TAN_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &TAN_ERRORS,
};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::trigonometry::tan")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "tan",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Unary { name: "unary_tan" }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Providers may execute tan in place via unary_tan; runtimes gather to host when the hook is unavailable.",
};

fn tan_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    let mut builder = build_runtime_error(error.message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn tan_error_with_detail(
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

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::trigonometry::tan")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "tan",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: |ctx: &FusionExprContext| {
            let input = ctx.inputs.first().ok_or(FusionError::MissingInput(0))?;
            Ok(format!("tan({input})"))
        },
    }),
    reduction: None,
    emits_nan: false,
    notes:
        "Fusion planner emits WGSL tan calls; providers can override with optimised fused kernels.",
};

#[runtime_builtin(
    name = "tan",
    category = "math/trigonometry",
    summary = "Compute element-wise tangent values in radians.",
    keywords = "tan,tangent,trigonometry,radians,gpu",
    accel = "unary",
    type_resolver(numeric_unary_type),
    descriptor(crate::builtins::math::trigonometry::tan::TAN_DESCRIPTOR),
    builtin_path = "crate::builtins::math::trigonometry::tan"
)]
async fn tan_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let template = parse_output_template(&rest)?;
    if let Some(symbolic) = symbolic_function(&value, SymbolicFunction::Tan) {
        return apply_output_template(symbolic, &template).await;
    }
    let base = match value {
        Value::GpuTensor(handle) => tan_gpu(handle).await?,
        Value::Complex(re, im) => {
            let (out_re, out_im) = tan_complex_components(re, im);
            Value::Complex(out_re, out_im)
        }
        Value::ComplexTensor(ct) => tan_complex_tensor(ct)?,
        Value::CharArray(ca) => tan_char_array(ca)?,
        Value::String(_) | Value::StringArray(_) => {
            return Err(tan_error_with_detail(
                &TAN_ERROR_INVALID_INPUT,
                "expected numeric input, got string",
            ))
        }
        other => tan_real(other)?,
    };
    apply_output_template(base, &template).await
}

async fn tan_gpu(handle: GpuTensorHandle) -> BuiltinResult<Value> {
    if let Some(provider) = runmat_accelerate_api::provider_for_handle(&handle) {
        if let Ok(out) = provider.unary_tan(&handle).await {
            return Ok(Value::GpuTensor(out));
        }
    }
    let tensor = gpu_helpers::gather_tensor_async(&handle)
        .await
        .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
    tan_tensor(tensor).map(tensor::tensor_into_value)
}

fn tan_real(value: Value) -> BuiltinResult<Value> {
    let tensor = tensor::value_into_tensor_for("tan", value)
        .map_err(|e| tan_error_with_detail(&TAN_ERROR_INVALID_INPUT, e))?;
    tan_tensor(tensor).map(tensor::tensor_into_value)
}

fn tan_tensor(tensor: Tensor) -> BuiltinResult<Tensor> {
    let data = tensor.data.iter().map(|&v| v.tan()).collect::<Vec<_>>();
    Tensor::new(data, tensor.shape.clone())
        .map_err(|e| tan_error_with_detail(&TAN_ERROR_INTERNAL, e))
}

fn tan_complex_tensor(ct: ComplexTensor) -> BuiltinResult<Value> {
    let mapped = ct
        .data
        .iter()
        .map(|&(re, im)| tan_complex_components(re, im))
        .collect::<Vec<_>>();
    let tensor = ComplexTensor::new(mapped, ct.shape.clone())
        .map_err(|e| tan_error_with_detail(&TAN_ERROR_INTERNAL, e))?;
    Ok(complex_tensor_into_value(tensor))
}

fn tan_char_array(array: CharArray) -> BuiltinResult<Value> {
    let data = array
        .data
        .iter()
        .map(|&ch| (ch as u32 as f64).tan())
        .collect::<Vec<_>>();
    let tensor = Tensor::new(data, vec![array.rows, array.cols])
        .map_err(|e| tan_error_with_detail(&TAN_ERROR_INTERNAL, e))?;
    Ok(tensor::tensor_into_value(tensor))
}

#[inline]
fn tan_complex_components(re: f64, im: f64) -> (f64, f64) {
    let two_re = 2.0 * re;
    let two_im = 2.0 * im;
    let denom = two_re.cos() + two_im.cosh();
    let real = two_re.sin() / denom;
    let imag = two_im.sinh() / denom;
    (real, imag)
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
                Err(tan_error_with_detail(
                    &TAN_ERROR_INVALID_OPTION,
                    "expected prototype after 'like'",
                ))
            } else {
                Err(tan_error_with_detail(
                    &TAN_ERROR_INVALID_OPTION,
                    "unrecognised argument for tan",
                ))
            }
        }
        2 => {
            if matches!(keyword_of(&args[0]).as_deref(), Some("like")) {
                Ok(OutputTemplate::Like(args[1].clone()))
            } else {
                Err(tan_error_with_detail(
                    &TAN_ERROR_INVALID_OPTION,
                    "unsupported option; only 'like' is accepted",
                ))
            }
        }
        _ => Err(tan_error(&TAN_ERROR_ARG_COUNT)),
    }
}

async fn apply_output_template(value: Value, template: &OutputTemplate) -> BuiltinResult<Value> {
    match template {
        OutputTemplate::Default => Ok(value),
        OutputTemplate::Like(proto) => match proto {
            Value::GpuTensor(_) => convert_to_gpu(value),
            Value::Tensor(_)
            | Value::Num(_)
            | Value::Int(_)
            | Value::Bool(_)
            | Value::LogicalArray(_) => convert_to_host_like(value).await,
            Value::Complex(_, _) | Value::ComplexTensor(_) => Err(tan_error_with_detail(
                &TAN_ERROR_LIKE_PROTOTYPE,
                "complex prototypes for 'like' are not supported yet",
            )),
            _ => Err(tan_error_with_detail(
                &TAN_ERROR_LIKE_PROTOTYPE,
                "unsupported prototype; provide a numeric or gpuArray prototype",
            )),
        },
    }
}

fn convert_to_gpu(value: Value) -> BuiltinResult<Value> {
    let provider = runmat_accelerate_api::provider().ok_or_else(|| {
        tan_error_with_detail(
            &TAN_ERROR_GPU_UNAVAILABLE,
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
                .map_err(|e| tan_error_with_detail(&TAN_ERROR_INTERNAL, e))?;
            Ok(Value::GpuTensor(handle))
        }
        Value::Num(n) => {
            let tensor = Tensor::new(vec![n], vec![1, 1])
                .map_err(|e| tan_error_with_detail(&TAN_ERROR_INTERNAL, e))?;
            convert_to_gpu(Value::Tensor(tensor))
        }
        Value::Int(i) => convert_to_gpu(Value::Num(i.to_f64())),
        Value::Bool(b) => convert_to_gpu(Value::Num(if b { 1.0 } else { 0.0 })),
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(&logical)
                .map_err(|e| tan_error_with_detail(&TAN_ERROR_INTERNAL, e))?;
            convert_to_gpu(Value::Tensor(tensor))
        }
        Value::Complex(_, _) | Value::ComplexTensor(_) => Err(tan_error_with_detail(
            &TAN_ERROR_LIKE_PROTOTYPE,
            "GPU prototypes for 'like' only support real numeric outputs",
        )),
        other => Err(tan_error_with_detail(
            &TAN_ERROR_INTERNAL,
            format!("unsupported result type for GPU output via 'like' ({other:?})"),
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

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::{CharArray, IntValue, ResolveContext, StringArray, Tensor, Type};

    fn error_message(err: RuntimeError) -> String {
        err.message().to_string()
    }

    #[test]
    fn tan_descriptor_signatures_cover_like_overload() {
        let labels: Vec<&str> = TAN_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"Y = tan(X)"));
        assert!(labels.contains(&"Y = tan(X, \"like\", P)"));
    }

    fn tan_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        block_on(super::tan_builtin(value, rest))
    }

    #[test]
    fn tan_type_preserves_tensor_shape() {
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
    fn tan_type_scalar_tensor_returns_num() {
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
    fn tan_scalar_pi_over_four() {
        let result = tan_builtin(Value::Num(std::f64::consts::FRAC_PI_4), Vec::new()).expect("tan");
        match result {
            Value::Num(v) => assert!((v - 1.0).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tan_tensor_elements() {
        let tensor = Tensor::new(vec![0.0, std::f64::consts::FRAC_PI_4], vec![2, 1]).unwrap();
        let result = tan_builtin(Value::Tensor(tensor), Vec::new()).expect("tan");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![2, 1]);
                assert!((out.data[0] - 0.0).abs() < 1e-12);
                assert!((out.data[1] - 1.0).abs() < 1e-12);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tan_string_input_errors() {
        let err = tan_builtin(Value::from("invalid"), Vec::new()).expect_err("expected error");
        let message = error_message(err);
        assert!(message.contains("numeric"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tan_int_promotes() {
        let result = tan_builtin(Value::Int(IntValue::I32(1)), Vec::new()).expect("tan");
        match result {
            Value::Num(v) => assert!((v - 1f64.tan()).abs() < 1e-12),
            other => panic!("expected numeric result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tan_complex_scalar_matches_formula() {
        let result = tan_builtin(Value::Complex(1.0, 0.5), Vec::new()).expect("tan");
        match result {
            Value::Complex(re, im) => {
                let (expected_re, expected_im) = tan_complex_components(1.0, 0.5);
                assert!((re - expected_re).abs() < 1e-12);
                assert!((im - expected_im).abs() < 1e-12);
            }
            other => panic!("expected complex result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tan_complex_on_real_axis_matches_real_value() {
        let angle = std::f64::consts::FRAC_PI_2 * 0.9;
        let result = tan_builtin(Value::Complex(angle, 0.0), Vec::new()).expect("tan");
        match result {
            Value::Complex(re, im) => {
                assert!((re - angle.tan()).abs() < 1e-12);
                assert_eq!(im, 0.0);
            }
            other => panic!("expected complex result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tan_char_array_roundtrip() {
        let chars = CharArray::new("AB".chars().collect(), 1, 2).unwrap();
        let result = tan_builtin(Value::CharArray(chars), Vec::new()).expect("tan");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 2]);
                let expected: Vec<f64> = ['A', 'B']
                    .iter()
                    .map(|&ch| (ch as u32 as f64).tan())
                    .collect();
                for (got, exp) in t.data.iter().zip(expected.iter()) {
                    assert!((got - exp).abs() < 1e-12);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tan_gpu_provider_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![0.0, 0.2, -0.3, 1.0], vec![4, 1]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = tan_builtin(Value::GpuTensor(handle), Vec::new()).expect("tan");
            let gathered = test_support::gather(result).expect("gather");
            let expected: Vec<f64> = tensor.data.iter().map(|&v| v.tan()).collect();
            assert_eq!(gathered.shape, vec![4, 1]);
            assert_eq!(gathered.data, expected);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tan_like_missing_prototype_errors() {
        let err =
            tan_builtin(Value::Num(1.0), vec![Value::from("like")]).expect_err("expected error");
        assert_eq!(err.identifier(), TAN_ERROR_INVALID_OPTION.identifier);
        let message = error_message(err);
        assert!(message.contains("prototype"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tan_like_complex_prototype_errors() {
        let err = tan_builtin(
            Value::Num(1.0),
            vec![Value::from("like"), Value::Complex(0.0, 1.0)],
        )
        .expect_err("expected error");
        let message = error_message(err);
        assert!(message.contains("complex prototypes"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tan_like_gpu_prototype() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![0.0, 0.3, 0.6], vec![3, 1]).unwrap();
            let proto_view = HostTensorView {
                data: &[0.0],
                shape: &[1, 1],
            };
            let proto = provider.upload(&proto_view).expect("upload");
            let result = tan_builtin(
                Value::Tensor(tensor.clone()),
                vec![Value::from("like"), Value::GpuTensor(proto.clone())],
            )
            .expect("tan");
            match result {
                Value::GpuTensor(handle) => {
                    let gathered = test_support::gather(Value::GpuTensor(handle)).expect("gather");
                    let expected: Vec<f64> = tensor.data.iter().map(|&v| v.tan()).collect();
                    assert_eq!(gathered.shape, vec![3, 1]);
                    assert_eq!(gathered.data, expected);
                }
                other => panic!("expected GPU tensor, got {other:?}"),
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tan_like_host_with_gpu_input_gathers() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![0.0, 0.5], vec![2, 1]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = tan_builtin(
                Value::GpuTensor(handle),
                vec![Value::from("like"), Value::Num(0.0)],
            )
            .expect("tan");
            match result {
                Value::Tensor(t) => {
                    let expected: Vec<f64> = tensor.data.iter().map(|&v| v.tan()).collect();
                    assert_eq!(t.shape, vec![2, 1]);
                    assert_eq!(t.data, expected);
                }
                Value::GpuTensor(_) => panic!("expected host result"),
                other => panic!("unexpected result {other:?}"),
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tan_like_rejects_extra_arguments() {
        let err = tan_builtin(
            Value::Num(0.0),
            vec![Value::from("like"), Value::Num(0.0), Value::Num(1.0)],
        )
        .expect_err("expected error");
        let message = error_message(err);
        assert!(message.contains("too many input arguments"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tan_like_keyword_case_insensitive() {
        let tensor = Tensor::new(vec![0.0, 0.1], vec![2, 1]).unwrap();
        let result = tan_builtin(
            Value::Tensor(tensor.clone()),
            vec![Value::from("LIKE"), Value::Num(0.0)],
        )
        .expect("tan");
        match result {
            Value::Tensor(out) => {
                let expected: Vec<f64> = tensor.data.iter().map(|&v| v.tan()).collect();
                assert_eq!(out.shape, vec![2, 1]);
                assert_eq!(out.data, expected);
            }
            other => panic!("unexpected result {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tan_like_char_array_keyword() {
        let keyword = CharArray::new_row("like");
        let result = tan_builtin(
            Value::Num(0.0),
            vec![Value::CharArray(keyword), Value::Num(0.0)],
        )
        .expect("tan");
        match result {
            Value::Num(v) => assert!((v - 0.0).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tan_like_string_array_keyword() {
        let keyword = StringArray::new(vec!["LIKE".to_string()], vec![1]).unwrap();
        let result = tan_builtin(
            Value::Num(0.0),
            vec![Value::StringArray(keyword), Value::Num(0.0)],
        )
        .expect("tan");
        match result {
            Value::Num(v) => assert!((v - 0.0).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tan_unrecognised_option_errors() {
        let err =
            tan_builtin(Value::Num(0.0), vec![Value::from("invalid")]).expect_err("expected error");
        let message = error_message(err);
        assert!(message.contains("unrecognised argument"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn tan_wgpu_matches_cpu_elementwise() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let tensor = Tensor::new(vec![0.0, 0.25, -0.5, 1.0], vec![4, 1]).unwrap();
        let cpu = tan_real(Value::Tensor(tensor.clone())).unwrap();
        let view = HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = runmat_accelerate_api::provider()
            .unwrap()
            .upload(&view)
            .unwrap();
        let gpu = block_on(tan_gpu(handle)).unwrap();
        let gathered = test_support::gather(gpu).expect("gather");
        match (cpu, gathered) {
            (Value::Tensor(ct), gt) => {
                assert_eq!(gt.shape, ct.shape);
                let tol = match runmat_accelerate_api::provider().unwrap().precision() {
                    runmat_accelerate_api::ProviderPrecision::F64 => 1e-12,
                    runmat_accelerate_api::ProviderPrecision::F32 => 1e-5,
                };
                for (a, b) in gt.data.iter().zip(ct.data.iter()) {
                    assert!((a - b).abs() < tol, "|{a} - {b}| >= {tol}");
                }
            }
            _ => panic!("unexpected comparison result"),
        }
    }
}
