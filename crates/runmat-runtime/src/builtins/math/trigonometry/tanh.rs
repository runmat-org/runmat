//! MATLAB-compatible `tanh` builtin with GPU-aware semantics for RunMat.

use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};
use runmat_macros::runtime_builtin;
use runmat_value::{CharArray, ComplexTensor, Tensor, Value};
use runmat_value::{ComplexStorage, NumericDType};

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, FusionError,
    FusionExprContext, FusionKernelTemplate, GpuOpKind, ProviderHook, ReductionNaN,
    ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, map_control_flow_with_builtin, tensor};
use crate::builtins::math::type_resolvers::numeric_unary_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "tanh";

pub const TANH_INTEGER_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "tanh-integer-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "tanh with typed-integer input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:TanhIntegerInputExtension"),
};
pub const TANH_LOGICAL_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "tanh-logical-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "tanh with logical input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:TanhLogicalInputExtension"),
};
pub const TANH_CHARACTER_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "tanh-character-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "tanh with character input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:TanhCharacterInputExtension"),
};
pub const TANH_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    TANH_INTEGER_INPUT_EXTENSION,
    TANH_LOGICAL_INPUT_EXTENSION,
    TANH_CHARACTER_INPUT_EXTENSION,
];
const TANH_INTEGER_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "X",
    classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "All eight real integer classes require exact binary64 representability before hyperbolic evaluation.",
}];
pub const TANH_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "Y = tanh(integer_X)",
        inputs: &TANH_INTEGER_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
        notes: "RunMat mode validates native integer storage before conversion; resident fallback returns through the owner and finite integer inputs naturally approach unit magnitude.",
    }];

const TANH_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Element-wise hyperbolic tangent result.",
}];

const TANH_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input scalar, array, char array, complex value, or gpuArray.",
}];

const TANH_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "Y = tanh(X)",
    inputs: &TANH_INPUTS,
    outputs: &TANH_OUTPUT,
}];

const TANH_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TANH.INVALID_INPUT",
    identifier: Some("RunMat:tanh:InvalidInput"),
    when: "Input cannot be interpreted as supported numeric/char/complex data.",
    message: "tanh: invalid input",
};

const TANH_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TANH.INTERNAL",
    identifier: Some("RunMat:tanh:Internal"),
    when: "Internal gather/conversion/allocation/provider flow failed.",
    message: "tanh: internal error",
};

const TANH_ERRORS: [BuiltinErrorDescriptor; 2] = [TANH_ERROR_INVALID_INPUT, TANH_ERROR_INTERNAL];

pub const TANH_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &TANH_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &TANH_ERRORS,
};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::trigonometry::tanh")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "tanh",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Unary { name: "unary_tanh" }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "Providers may execute tanh directly on the device; runtimes gather to the host when unary_tanh is unavailable.",
};

fn tanh_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    let mut builder = build_runtime_error(error.message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn tanh_error_with_detail(
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

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::trigonometry::tanh")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "tanh",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: |ctx: &FusionExprContext| {
            let input = ctx.inputs.first().ok_or(FusionError::MissingInput(0))?;
            Ok(format!("tanh({input})"))
        },
    }),
    reduction: None,
    emits_nan: false,
    notes:
        "Fusion planner emits WGSL `tanh` calls; providers may override with specialised kernels.",
};

#[runtime_builtin(
    name = "tanh",
    category = "math/trigonometry",
    summary = "Compute element-wise hyperbolic tangent values.",
    keywords = "tanh,hyperbolic tangent,trigonometry,gpu",
    accel = "unary",
    type_resolver(numeric_unary_type),
    descriptor(crate::builtins::math::trigonometry::tanh::TANH_DESCRIPTOR),
    extensions(TANH_EXTENSIONS),
    integer_capabilities(TANH_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::math::trigonometry::tanh"
)]
async fn tanh_builtin(value: Value) -> BuiltinResult<Value> {
    ensure_tanh_extensions(&value).await?;
    crate::builtins::common::validation::reject_typed_complex_integer(&value, "tanh")?;
    match value {
        Value::GpuTensor(handle) => tanh_gpu(handle).await,
        Value::Complex(re, im) => {
            let (real, imag) = tanh_complex_parts(re, im);
            Ok(Value::Complex(real, imag))
        }
        Value::ComplexTensor(ct) => tanh_complex_tensor(ct),
        Value::CharArray(ca) => tanh_char_array(ca),
        Value::String(_) | Value::StringArray(_) => Err(tanh_error(&TANH_ERROR_INVALID_INPUT)),
        other => tanh_real(other),
    }
}

async fn ensure_tanh_extensions(value: &Value) -> BuiltinResult<()> {
    crate::builtins::common::validation::ensure_runmat_integer_f64_boundary(
        value,
        &TANH_INTEGER_INPUT_EXTENSION,
        BUILTIN_NAME,
        "X",
    )
    .await?;
    if matches!(value, Value::Bool(_) | Value::LogicalArray(_))
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_is_logical(handle))
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &TANH_LOGICAL_INPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if matches!(value, Value::CharArray(_)) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &TANH_CHARACTER_INPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    Ok(())
}

async fn tanh_gpu(handle: GpuTensorHandle) -> BuiltinResult<Value> {
    let exact_fallback = runmat_accelerate_api::handle_integer_type(&handle).is_some()
        || runmat_accelerate_api::handle_is_logical(&handle);
    if !exact_fallback {
        if let Some(provider) = gpu_helpers::exact_provider_for_handle(&handle) {
            if let Ok(out) = provider.unary_tanh(&handle).await {
                return Ok(Value::GpuTensor(out));
            }
        }
    }
    let source = handle.clone();
    let gathered = gpu_helpers::gather_value_async(&Value::GpuTensor(handle))
        .await
        .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
    let host = match gathered {
        Value::Complex(re, im) => {
            let (out_re, out_im) = tanh_complex_parts(re, im);
            Ok(Value::Complex(out_re, out_im))
        }
        Value::ComplexTensor(tensor) => tanh_complex_tensor(tensor),
        Value::Tensor(tensor) => tanh_tensor(tensor).map(tensor::tensor_into_value),
        Value::Num(value) => Ok(Value::Num(value.tanh())),
        other => Err(tanh_error_with_detail(
            &TANH_ERROR_INVALID_INPUT,
            format!("unsupported gathered gpuArray value {other:?}"),
        )),
    }?;
    gpu_helpers::restore_class_preserving_value(&source, host, BUILTIN_NAME)
}

fn tanh_real(value: Value) -> BuiltinResult<Value> {
    let tensor = tensor::value_into_tensor_for("tanh", value)
        .map_err(|e| tanh_error_with_detail(&TANH_ERROR_INVALID_INPUT, e))?;
    tanh_tensor(tensor).map(tensor::tensor_into_value)
}

fn tanh_tensor(tensor: Tensor) -> BuiltinResult<Tensor> {
    if tensor.numeric_dtype() == NumericDType::F32 {
        let data = tensor
            .as_f32_slice()
            .expect("single tensor storage")
            .iter()
            .map(|&v| v.tanh())
            .collect();
        return Tensor::from_f32(data, tensor.shape.clone())
            .map_err(|e| tanh_error_with_detail(&TANH_ERROR_INTERNAL, e));
    }
    let data = tensor::tensor_values_f64_cow(&tensor)
        .iter()
        .map(|&v| v.tanh())
        .collect::<Vec<_>>();
    Tensor::new(data, tensor.shape.clone())
        .map_err(|e| tanh_error_with_detail(&TANH_ERROR_INTERNAL, e))
}

fn tanh_complex_tensor(ct: ComplexTensor) -> BuiltinResult<Value> {
    let shape = ct.shape.clone();
    let tensor = match ct.into_complex_storage() {
        ComplexStorage::F32(values) => ComplexTensor::from_f32(
            values
                .into_iter()
                .map(|(re, im)| {
                    let (out_re, out_im) = tanh_complex_parts(f64::from(re), f64::from(im));
                    (out_re as f32, out_im as f32)
                })
                .collect(),
            shape,
        ),
        ComplexStorage::F64(values) => ComplexTensor::new(
            values
                .into_iter()
                .map(|(re, im)| tanh_complex_parts(re, im))
                .collect(),
            shape,
        ),
        ComplexStorage::Integer(_) => Err("typed complex integer input is unsupported".into()),
    }
    .map_err(|e| tanh_error_with_detail(&TANH_ERROR_INTERNAL, e))?;
    Ok(Value::ComplexTensor(tensor))
}

fn tanh_char_array(ca: CharArray) -> BuiltinResult<Value> {
    let data = ca
        .data
        .iter()
        .map(|&ch| (ch as u32 as f64).tanh())
        .collect::<Vec<_>>();
    let tensor = Tensor::new(data, vec![ca.rows, ca.cols])
        .map_err(|e| tanh_error_with_detail(&TANH_ERROR_INTERNAL, e))?;
    Ok(Value::Tensor(tensor))
}

fn tanh_complex_parts(re: f64, im: f64) -> (f64, f64) {
    // Use tanh(z) = sinh(z) / cosh(z) with explicit real/imag components.
    let sinh_re = re.sinh() * im.cos();
    let sinh_im = re.cosh() * im.sin();
    let cosh_re = re.cosh() * im.cos();
    let cosh_im = re.sinh() * im.sin();
    let denom = cosh_re * cosh_re + cosh_im * cosh_im;
    // Division by zero yields the expected IEEE infinities/NaNs, matching MATLAB's behaviour at poles.
    let real = (sinh_re * cosh_re + sinh_im * cosh_im) / denom;
    let imag = (sinh_im * cosh_re - sinh_re * cosh_im) / denom;
    (real, imag)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use num_complex::Complex64;
    use runmat_builtins::{ResolveContext, Type};
    use runmat_value::{CharArray, Tensor};

    fn tanh_builtin(value: Value) -> BuiltinResult<Value> {
        block_on(super::tanh_builtin(value))
    }

    #[test]
    fn tanh_descriptor_signatures_cover_core_form() {
        let labels: Vec<&str> = TANH_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"Y = tanh(X)"));
    }

    #[test]
    fn tanh_type_preserves_tensor_shape() {
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
    fn tanh_type_scalar_tensor_returns_num() {
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
    fn tanh_scalar_num() {
        let result = tanh_builtin(Value::Num(1.0)).expect("tanh");
        match result {
            Value::Num(v) => assert!((v - 1.0_f64.tanh()).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tanh_tensor_elements() {
        let tensor = Tensor::new(vec![-1.0, 0.0, 1.0], vec![3, 1]).unwrap();
        let result = tanh_builtin(Value::Tensor(tensor)).expect("tanh");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![3, 1]);
                for (value, expected) in out
                    .materialize_f64()
                    .iter()
                    .zip([-1.0_f64.tanh(), 0.0, 1.0_f64.tanh()].iter())
                {
                    assert!((*value - *expected).abs() < 1e-12);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tanh_reads_typed_integer_tensor_storage_exactly() {
        let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
        let tensor = Tensor::new_integer(
            runmat_value::IntegerStorage::I16(vec![-1, 0, 1]),
            vec![3, 1],
        )
        .expect("integer tensor");

        match tanh_builtin(Value::Tensor(tensor)).expect("tanh") {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![3, 1]);
                let expected = [-1.0f64.tanh(), 0.0, 1.0f64.tanh()];
                for (actual, expected) in out.materialize_f64().iter().zip(expected.iter()) {
                    assert!((actual - expected).abs() < 1e-12);
                }
                assert!(out.integer_storage().is_none());
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tanh_complex_scalar() {
        let result = tanh_builtin(Value::Complex(0.5, 1.0)).expect("tanh");
        match result {
            Value::Complex(re, im) => {
                let target = Complex64::new(0.5, 1.0).tanh();
                assert!((re - target.re).abs() < 1e-12);
                assert!((im - target.im).abs() < 1e-12);
            }
            other => panic!("expected complex result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tanh_char_array_roundtrip() {
        let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
        let chars = CharArray::new("Az".chars().collect(), 1, 2).unwrap();
        let result = tanh_builtin(Value::CharArray(chars)).expect("tanh");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 2]);
                let expected: Vec<f64> = "Az".chars().map(|c| (c as u32 as f64).tanh()).collect();
                for (value, expect) in t.materialize_f64().iter().zip(expected.iter()) {
                    assert!((*value - *expect).abs() < 1e-12);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tanh_string_errors() {
        let err = tanh_builtin(Value::from("not numeric")).expect_err("expected error");
        assert!(err.message().contains("invalid input"));
        assert_eq!(err.identifier(), TANH_ERROR_INVALID_INPUT.identifier);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tanh_gpu_provider_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![0.0, 0.5, 1.0, 1.5], vec![4, 1]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.materialize_f64(),
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = tanh_builtin(Value::GpuTensor(handle)).expect("tanh");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![4, 1]);
            for (value, expect) in gathered
                .materialize_f64()
                .iter()
                .zip(tensor.materialize_f64().iter())
            {
                assert!((*value - expect.tanh()).abs() < 1e-12);
            }
        });
    }

    #[test]
    fn tanh_gpu_fallback_preserves_single_and_source_owner() {
        test_support::with_f32_test_provider(|provider| {
            let input = [0.0, 0.5, 1.0];
            let source = provider
                .upload(&runmat_accelerate_api::HostTensorView {
                    data: &input,
                    shape: &[3, 1],
                })
                .expect("upload");
            let source_device = source.device_id;
            let result =
                block_on(super::tanh_builtin(Value::GpuTensor(source))).expect("tanh fallback");
            let Value::GpuTensor(handle) = &result else {
                panic!("expected resident result")
            };
            assert_eq!(handle.device_id, source_device);
            let gathered = test_support::gather(result).expect("gather result");
            assert_eq!(gathered.numeric_dtype(), NumericDType::F32);
            assert_eq!(gathered.shape, vec![3, 1]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn tanh_wgpu_matches_cpu_elementwise() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );

        let tensor = Tensor::new(vec![-1.25, -0.5, 0.0, 0.75, 1.5], vec![5, 1]).unwrap();
        let cpu_value = tanh_real(Value::Tensor(tensor.clone())).expect("cpu tanh");
        let cpu_tensor = test_support::gather(cpu_value).expect("gather cpu");

        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.materialize_f64(),
            shape: &tensor.shape,
        };
        let handle = runmat_accelerate_api::provider()
            .expect("wgpu provider")
            .upload(&view)
            .expect("upload");
        let gpu_value = block_on(tanh_gpu(handle)).expect("gpu tanh");
        let gpu_tensor = test_support::gather(gpu_value).expect("gather gpu");

        assert_eq!(gpu_tensor.shape, cpu_tensor.shape);
        let tol = match runmat_accelerate_api::provider()
            .expect("wgpu provider")
            .precision()
        {
            runmat_accelerate_api::ProviderPrecision::F64 => 1e-12,
            runmat_accelerate_api::ProviderPrecision::F32 => 1e-5,
        };
        for (got, expect) in gpu_tensor
            .materialize_f64()
            .iter()
            .zip(cpu_tensor.materialize_f64().iter())
        {
            assert!(
                (*got - *expect).abs() < tol,
                "tanh mismatch: got {got}, expect {expect}, tol {tol}"
            );
        }
    }
}
