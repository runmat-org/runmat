//! MATLAB-compatible `double` builtin with GPU-aware semantics for RunMat.

use log::trace;
use runmat_accelerate_api::{GpuTensorHandle, HostTensorView, ProviderPrecision};
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    CharArray, LogicalArray, SparseTensor, Tensor, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::{
    gpu_helpers,
    random_args::keyword_of,
    spec::{
        BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, FusionError,
        FusionExprContext, FusionKernelTemplate, GpuOpKind, ProviderHook, ReductionNaN,
        ResidencyPolicy, ScalarType, ShapeRequirements,
    },
    tensor,
};
use crate::builtins::math::type_resolvers::numeric_unary_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::elementwise::double")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "double",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Unary {
        name: "unary_double",
    }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Casts inputs to float64. Providers without native float64 support gather to host; float64-capable providers keep results on device.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::elementwise::double")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "double",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F64],
        wgsl_body: |ctx: &FusionExprContext| {
            let input = ctx.inputs.first().ok_or(FusionError::MissingInput(0))?;
            Ok(format!("({input})"))
        },
    }),
    reduction: None,
    emits_nan: false,
    notes: "Fusion treats double as an identity when the execution scalar type is already float64.",
};

const BUILTIN_NAME: &str = "double";

const DOUBLE_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Double-precision output value.",
}];

const DOUBLE_INPUTS_X: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input scalar/array value to convert.",
}];

const DOUBLE_INPUTS_X_LIKE: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input scalar/array value to convert.",
    },
    BuiltinParamDescriptor {
        name: "like",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Literal string \"like\".",
    },
    BuiltinParamDescriptor {
        name: "prototype",
        ty: BuiltinParamType::LikePrototype,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Output class/device prototype.",
    },
];

const DOUBLE_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "Y = double(X)",
        inputs: &DOUBLE_INPUTS_X,
        outputs: &DOUBLE_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Y = double(X, \"like\", prototype)",
        inputs: &DOUBLE_INPUTS_X_LIKE,
        outputs: &DOUBLE_OUTPUT,
    },
];

const DOUBLE_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DOUBLE.INVALID_ARGUMENT",
    identifier: Some("RunMat:double:InvalidArgument"),
    when: "Optional arguments are malformed or unsupported.",
    message: "double: invalid argument",
};

const DOUBLE_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DOUBLE.INVALID_INPUT",
    identifier: Some("RunMat:double:InvalidInput"),
    when: "Input value or prototype cannot be converted to double.",
    message: "double: invalid input",
};

const DOUBLE_ERROR_GPU_UNSUPPORTED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DOUBLE.GPU_UNSUPPORTED",
    identifier: Some("RunMat:double:GpuUnsupported"),
    when: "GPU output via \"like\" is requested but no compatible float64 provider is active.",
    message: "double: gpu output not supported",
};

const DOUBLE_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DOUBLE.INTERNAL",
    identifier: Some("RunMat:double:Internal"),
    when: "Internal conversion, gather, or provider upload failed.",
    message: "double: internal error",
};

const DOUBLE_ERRORS: [BuiltinErrorDescriptor; 4] = [
    DOUBLE_ERROR_INVALID_ARGUMENT,
    DOUBLE_ERROR_INVALID_INPUT,
    DOUBLE_ERROR_GPU_UNSUPPORTED,
    DOUBLE_ERROR_INTERNAL,
];

pub const DOUBLE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &DOUBLE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &DOUBLE_ERRORS,
};

fn double_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl std::fmt::Display,
) -> RuntimeError {
    double_error_with_message(format!("{}: {}", error.message, detail), error)
}

fn double_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn conversion_error(type_name: &str) -> RuntimeError {
    double_error_with_detail(
        &DOUBLE_ERROR_INVALID_INPUT,
        format!("conversion to double from {type_name} is not possible"),
    )
}

#[runtime_builtin(
    name = "double",
    category = "math/elementwise",
    summary = "Convert values to double precision.",
    keywords = "double,float64,cast,gpu",
    accel = "unary",
    type_resolver(numeric_unary_type),
    descriptor(crate::builtins::math::elementwise::double::DOUBLE_DESCRIPTOR),
    builtin_path = "crate::builtins::math::elementwise::double"
)]
async fn double_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let template = parse_output_template(&rest)?;
    let converted = match value {
        Value::Num(n) => Ok(Value::Num(n)),
        Value::Int(i) => Ok(Value::Num(i.to_f64())),
        Value::Bool(flag) => Ok(Value::Num(if flag { 1.0 } else { 0.0 })),
        Value::Tensor(tensor) => Ok(Value::Tensor(tensor)),
        Value::SparseTensor(sparse) => double_from_sparse_tensor(sparse),
        Value::Complex(re, im) => Ok(Value::Complex(re, im)),
        Value::ComplexTensor(tensor) => Ok(Value::ComplexTensor(tensor)),
        Value::LogicalArray(array) => double_from_logical(array),
        Value::CharArray(chars) => double_from_char_array(chars),
        Value::GpuTensor(handle) => double_from_gpu(handle).await,
        Value::String(_) | Value::StringArray(_) => Err(conversion_error("string")),
        Value::Symbolic(expr) => expr
            .numeric_constant_value()
            .map(Value::Num)
            .ok_or_else(|| conversion_error("sym")),
        Value::Cell(_) => Err(conversion_error("cell")),
        Value::Struct(_) => Err(conversion_error("struct")),
        Value::Object(obj) => Err(conversion_error(&obj.class_name)),
        Value::HandleObject(handle) => Err(conversion_error(&handle.class_name)),
        Value::Listener(_) => Err(conversion_error("event.listener")),
        Value::FunctionHandle(_)
        | Value::ExternalFunctionHandle(_)
        | Value::MethodFunctionHandle(_)
        | Value::BoundFunctionHandle { .. }
        | Value::Closure(_) => Err(conversion_error("function_handle")),
        Value::ClassRef(_) => Err(conversion_error("meta.class")),
        Value::MException(_) => Err(conversion_error("MException")),
        Value::OutputList(_) => Err(conversion_error("OutputList")),
    }?;
    apply_output_template(converted, &template).await
}

fn double_from_logical(array: LogicalArray) -> BuiltinResult<Value> {
    let tensor = tensor::logical_to_tensor(&array)
        .map_err(|e| double_error_with_detail(&DOUBLE_ERROR_INTERNAL, e))?;
    Ok(tensor::tensor_into_value(tensor))
}

fn double_from_sparse_tensor(sparse: SparseTensor) -> BuiltinResult<Value> {
    let tensor = sparse.to_dense().map_err(|err| {
        double_error_with_detail(
            &DOUBLE_ERROR_INTERNAL,
            format!("failed to densify sparse input: {err}"),
        )
    })?;
    Ok(Value::Tensor(tensor))
}

fn double_from_char_array(chars: CharArray) -> BuiltinResult<Value> {
    let data: Vec<f64> = chars.data.iter().map(|&ch| ch as u32 as f64).collect();
    let tensor = Tensor::new(data, vec![chars.rows, chars.cols])
        .map_err(|e| double_error_with_detail(&DOUBLE_ERROR_INTERNAL, e))?;
    Ok(tensor::tensor_into_value(tensor))
}

async fn double_from_gpu(handle: GpuTensorHandle) -> BuiltinResult<Value> {
    let provider = runmat_accelerate_api::provider_for_handle(&handle);

    if let Some(provider) = provider {
        if provider.precision() == ProviderPrecision::F64 {
            match provider.unary_double(&handle).await {
                Ok(result) => {
                    return Ok(Value::GpuTensor(result));
                }
                Err(err) => {
                    trace!("double: provider unary_double unavailable ({err}); falling back to host conversion");
                }
            }
        } else {
            trace!(
                "double: provider precision {:?} cannot store float64 values; gathering to host",
                provider.precision()
            );
        }
    }

    let tensor = gpu_helpers::gather_tensor_async(&handle).await?;
    if let Some(provider) = provider {
        if provider.precision() == ProviderPrecision::F64 {
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            match provider.upload(&view) {
                Ok(new_handle) => return Ok(Value::GpuTensor(new_handle)),
                Err(err) => {
                    trace!("double: provider upload failed after gather ({err})");
                }
            }
        } else {
            trace!(
                "double: provider precision {:?} does not support float64 outputs; returning host tensor",
                provider.precision()
            );
        }
    }
    Ok(tensor::tensor_into_value(tensor))
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
                Err(double_error_with_detail(
                    &DOUBLE_ERROR_INVALID_ARGUMENT,
                    "expected prototype after 'like'",
                ))
            } else {
                Err(double_error_with_detail(
                    &DOUBLE_ERROR_INVALID_ARGUMENT,
                    "unrecognised argument for double",
                ))
            }
        }
        2 => {
            if matches!(keyword_of(&args[0]).as_deref(), Some("like")) {
                Ok(OutputTemplate::Like(args[1].clone()))
            } else {
                Err(double_error_with_detail(
                    &DOUBLE_ERROR_INVALID_ARGUMENT,
                    "unsupported option; only 'like' is accepted",
                ))
            }
        }
        _ => Err(double_error_with_detail(
            &DOUBLE_ERROR_INVALID_ARGUMENT,
            "too many input arguments",
        )),
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
            Value::Complex(_, _) | Value::ComplexTensor(_) => Err(double_error_with_detail(
                &DOUBLE_ERROR_INVALID_INPUT,
                "complex prototypes for 'like' are not supported yet",
            )),
            _ => Err(double_error_with_detail(
                &DOUBLE_ERROR_INVALID_INPUT,
                "unsupported prototype for 'like'; provide a numeric or gpuArray prototype",
            )),
        },
    }
}

fn convert_to_gpu(value: Value) -> BuiltinResult<Value> {
    let provider = runmat_accelerate_api::provider().ok_or_else(|| {
        double_error_with_detail(
            &DOUBLE_ERROR_GPU_UNSUPPORTED,
            "GPU output requested via 'like' but no acceleration provider is active",
        )
    })?;
    if provider.precision() != ProviderPrecision::F64 {
        return Err(double_error_with_detail(
            &DOUBLE_ERROR_GPU_UNSUPPORTED,
            "active acceleration provider does not support float64 storage",
        ));
    }
    match value {
        Value::GpuTensor(handle) => Ok(Value::GpuTensor(handle)),
        Value::Tensor(tensor) => {
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider
                .upload(&view)
                .map_err(|e| double_error_with_detail(&DOUBLE_ERROR_INTERNAL, e))?;
            Ok(Value::GpuTensor(handle))
        }
        Value::Num(n) => {
            let tensor = Tensor::new(vec![n], vec![1, 1])
                .map_err(|e| double_error_with_detail(&DOUBLE_ERROR_INTERNAL, e))?;
            convert_to_gpu(Value::Tensor(tensor))
        }
        Value::Int(i) => convert_to_gpu(Value::Num(i.to_f64())),
        Value::Bool(b) => convert_to_gpu(Value::Num(if b { 1.0 } else { 0.0 })),
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(&logical)?;
            convert_to_gpu(Value::Tensor(tensor))
        }
        Value::Complex(_, _) | Value::ComplexTensor(_) => Err(double_error_with_detail(
            &DOUBLE_ERROR_INVALID_INPUT,
            "GPU prototypes for 'like' only support real numeric outputs",
        )),
        other => Err(double_error_with_detail(
            &DOUBLE_ERROR_INVALID_INPUT,
            format!("unsupported result type for GPU output via 'like' ({other:?})"),
        )),
    }
}

async fn convert_to_host_like(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::GpuTensor(handle) => {
            let proxy = Value::GpuTensor(handle);
            gpu_helpers::gather_value_async(&proxy)
                .await
                .map_err(|e| double_error_with_detail(&DOUBLE_ERROR_INTERNAL, e))
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
    #[cfg(feature = "wgpu")]
    use runmat_accelerate_api::ProviderPrecision;
    use runmat_builtins::{IntValue, ResolveContext, SparseTensor, SymbolicExpr, Type};

    fn double_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        block_on(super::double_builtin(value, rest))
    }

    #[test]
    fn double_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = DOUBLE_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"Y = double(X)"));
        assert!(labels.contains(&"Y = double(X, \"like\", prototype)"));
    }

    #[test]
    fn double_type_preserves_tensor_shape() {
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
    fn double_type_scalar_tensor_returns_num() {
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
    fn double_scalar_num_is_identity() {
        let value = Value::Num(std::f64::consts::PI);
        let result = double_builtin(value, Vec::new()).expect("double");
        match result {
            Value::Num(n) => assert_eq!(n, std::f64::consts::PI),
            other => panic!("expected scalar Num, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn double_promotes_integers() {
        let value = Value::Int(IntValue::I32(42));
        let result = double_builtin(value, Vec::new()).expect("double");
        match result {
            Value::Num(n) => assert_eq!(n, 42.0),
            other => panic!("expected scalar Num, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn double_converts_symbolic_constants() {
        let result = double_builtin(Value::Symbolic(SymbolicExpr::constant(42.5)), Vec::new())
            .expect("double");

        assert_eq!(result, Value::Num(42.5));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn double_rejects_symbolic_variables() {
        let err = double_builtin(Value::Symbolic(SymbolicExpr::variable("x")), Vec::new())
            .expect_err("symbolic variable should not convert");

        assert_eq!(err.identifier(), DOUBLE_ERROR_INVALID_INPUT.identifier);
        assert!(err.message().contains("conversion to double from sym"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn double_logical_array_returns_tensor() {
        let logical = LogicalArray::new(vec![0, 1, 1, 0], vec![2, 2]).unwrap();
        let result = double_builtin(Value::LogicalArray(logical), Vec::new()).expect("double");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                assert_eq!(t.data, vec![0.0, 1.0, 1.0, 0.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn double_char_array_converts_to_codes() {
        let chars = CharArray::new_row("AB");
        let result = double_builtin(Value::CharArray(chars), Vec::new()).expect("double");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 2]);
                assert_eq!(t.data, vec![65.0, 66.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn double_complex_scalar_is_identity() {
        let result = double_builtin(Value::Complex(1.5, -2.5), Vec::new()).expect("double");
        match result {
            Value::Complex(re, im) => {
                assert_eq!(re, 1.5);
                assert_eq!(im, -2.5);
            }
            other => panic!("expected complex scalar, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn double_tensor_preserves_shape() {
        let tensor = Tensor::new(vec![1.25, 2.5, 3.75, 4.5], vec![2, 2]).unwrap();
        let result = double_builtin(Value::Tensor(tensor.clone()), Vec::new()).expect("double");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, tensor.shape);
                assert_eq!(t.data, tensor.data);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn double_sparse_tensor_densifies() {
        let sparse = SparseTensor::new(3, 2, vec![0, 1, 2], vec![1, 2], vec![4.0, -1.0]).unwrap();
        let result = double_builtin(Value::SparseTensor(sparse), Vec::new()).expect("double");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![3, 2]);
                assert_eq!(t.data, vec![0.0, 4.0, 0.0, 0.0, 0.0, -1.0]);
            }
            other => panic!("expected dense tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn double_rejects_strings() {
        let err = double_builtin(Value::String("hello".into()), Vec::new()).unwrap_err();
        assert_eq!(err.identifier(), DOUBLE_ERROR_INVALID_INPUT.identifier);
        assert!(err.message().contains("conversion to double from string"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn double_gpu_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0], vec![2, 2]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = double_builtin(Value::GpuTensor(handle), Vec::new()).expect("double");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![2, 2]);
            assert_eq!(gathered.data, tensor.data);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn double_like_gpu_prototype_keeps_residency() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
            let proto = provider
                .upload(&HostTensorView {
                    data: &[0.0],
                    shape: &[1, 1],
                })
                .expect("upload");
            let result = double_builtin(
                Value::Tensor(tensor.clone()),
                vec![Value::from("like"), Value::GpuTensor(proto.clone())],
            )
            .expect("double");
            match result {
                Value::GpuTensor(h) => {
                    let gathered = test_support::gather(Value::GpuTensor(h)).expect("gather");
                    assert_eq!(gathered.data, tensor.data);
                }
                other => panic!("expected gpu tensor, got {other:?}"),
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn double_like_host_gathers_gpu_input() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![3.0], vec![1, 1]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = double_builtin(
                Value::GpuTensor(handle),
                vec![Value::from("like"), Value::Num(0.0)],
            )
            .expect("double");
            match result {
                Value::Num(n) => assert_eq!(n, 3.0),
                Value::Tensor(t) => {
                    assert_eq!(t.shape, vec![1, 1]);
                    assert_eq!(t.data, vec![3.0]);
                }
                other => panic!("expected scalar host value, got {other:?}"),
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn double_like_missing_prototype_errors() {
        let err =
            double_builtin(Value::Num(1.0), vec![Value::from("like")]).expect_err("expected error");
        assert_eq!(err.identifier(), DOUBLE_ERROR_INVALID_ARGUMENT.identifier);
        assert!(err.message().contains("expected prototype"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn double_like_rejects_extra_arguments() {
        let err = double_builtin(
            Value::Num(0.0),
            vec![Value::from("like"), Value::Num(0.0), Value::Num(1.0)],
        )
        .expect_err("expected error");
        assert_eq!(err.identifier(), DOUBLE_ERROR_INVALID_ARGUMENT.identifier);
        assert!(err.message().contains("too many input arguments"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn double_wgpu_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let provider = runmat_accelerate_api::provider().expect("wgpu provider");

        let tensor = Tensor::new(vec![1.0, 2.5, -3.75, 4.125], vec![2, 2]).unwrap();
        let cpu_value = double_builtin(Value::Tensor(tensor.clone()), Vec::new()).unwrap();

        let view = HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload");
        let gpu_value = double_builtin(Value::GpuTensor(handle), Vec::new()).unwrap();

        let gathered = test_support::gather(gpu_value.clone()).expect("gather");
        match cpu_value {
            Value::Tensor(ref ct) => {
                assert_eq!(gathered.shape, ct.shape);
                assert_eq!(gathered.data, ct.data);
            }
            Value::Num(n) => {
                assert_eq!(gathered.data, vec![n]);
            }
            other => panic!("unexpected CPU reference value {other:?}"),
        }

        if provider.precision() == ProviderPrecision::F64 {
            assert!(
                matches!(gpu_value, Value::GpuTensor(_)),
                "expected GPU residency under f64 precision"
            );
        } else {
            assert!(
                !matches!(gpu_value, Value::GpuTensor(_)),
                "expected host fallback when f64 unsupported"
            );
        }
    }
}
