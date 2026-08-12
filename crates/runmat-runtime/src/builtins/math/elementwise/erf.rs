//! MATLAB-compatible `erf` builtin with GPU-aware semantics for RunMat.
//!
//! Provides element-wise error-function evaluation for real inputs. MATLAB documents `erf`
//! for real single and double arrays and rejects sparse inputs, so complex and sparse values
//! are reported as input errors instead of silently applying an analytic continuation.

use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinIntegerBackendRule,
    BuiltinIntegerCapabilityDescriptor, BuiltinIntegerComputationDomain,
    BuiltinIntegerInputAvailability, BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule,
    BuiltinIntegerOverflowRule, BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule,
    BuiltinOutputMode, BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType,
    BuiltinSignatureDescriptor, NumericDType, NumericScalar, NumericStorage, Tensor, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, map_control_flow_with_builtin, tensor};
use crate::builtins::math::type_resolvers::numeric_unary_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "erf";

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::elementwise::erf")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: BUILTIN_NAME,
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Unary { name: "unary_erf" }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "Providers may evaluate erf directly on real device buffers; runtimes gather to host when unary_erf is unavailable.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::elementwise::erf")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: BUILTIN_NAME,
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Fusion planner currently falls back to provider or host elementwise erf evaluation.",
};

const ERF_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Elementwise error-function result.",
}];

const ERF_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Real numeric input.",
}];

const ERF_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "Y = erf(X)",
    inputs: &ERF_INPUTS,
    outputs: &ERF_OUTPUT,
}];

const ERF_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ERF.INVALID_INPUT",
    identifier: Some("RunMat:erf:InvalidInput"),
    when: "Input cannot be interpreted as a real, nonsparse numeric array.",
    message: "erf: invalid input",
};

const ERF_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ERF.INTERNAL",
    identifier: Some("RunMat:erf:Internal"),
    when: "Internal tensor construction or provider interaction failed.",
    message: "erf: internal error",
};

const ERF_ERRORS: [BuiltinErrorDescriptor; 2] = [ERF_ERROR_INVALID_INPUT, ERF_ERROR_INTERNAL];

const ERF_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "X",
    classes: &[],
    availability: BuiltinIntegerInputAvailability::Rejected,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes:
        "Integer and logical inputs are rejected before real floating host or provider dispatch.",
}];

pub const INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "Y = erf(X)",
        inputs: &ERF_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
        notes: "erf has no integer overload; the empty accepted-class mask is intentional and prevents generic numeric coercion from admitting integers.",
    }];

pub const ERF_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ERF_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERF_ERRORS,
};

fn builtin_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .build()
}

fn erf_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    let mut builder = build_runtime_error(format!("{}: {}", error.message, detail.as_ref()))
        .with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runtime_builtin(
    name = "erf",
    category = "math/elementwise",
    summary = "Compute element-wise error-function values.",
    keywords = "erf,error function,special,elementwise,gpu",
    accel = "unary",
    type_resolver(numeric_unary_type),
    descriptor(crate::builtins::math::elementwise::erf::ERF_DESCRIPTOR),
    integer_capabilities(crate::builtins::math::elementwise::erf::INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::math::elementwise::erf"
)]
async fn erf_builtin(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::GpuTensor(handle) => erf_gpu(handle).await,
        Value::Complex(_, _) | Value::ComplexTensor(_) => Err(erf_error_with_detail(
            &ERF_ERROR_INVALID_INPUT,
            "complex inputs are not supported",
        )),
        Value::String(_) | Value::StringArray(_) => Err(erf_error_with_detail(
            &ERF_ERROR_INVALID_INPUT,
            "expected real numeric input, got string",
        )),
        Value::SparseTensor(_) => Err(erf_error_with_detail(
            &ERF_ERROR_INVALID_INPUT,
            "sparse inputs are not supported",
        )),
        Value::Bool(_) | Value::LogicalArray(_) => Err(erf_error_with_detail(
            &ERF_ERROR_INVALID_INPUT,
            "logical inputs are not supported",
        )),
        Value::Int(_) => Err(erf_error_with_detail(
            &ERF_ERROR_INVALID_INPUT,
            "integer-class inputs are not supported",
        )),
        Value::CharArray(_) => Err(erf_error_with_detail(
            &ERF_ERROR_INVALID_INPUT,
            "char inputs are not supported",
        )),
        other => erf_real(other),
    }
}

async fn erf_gpu(handle: GpuTensorHandle) -> BuiltinResult<Value> {
    if runmat_accelerate_api::handle_integer_type(&handle).is_some() {
        return Err(erf_error_with_detail(
            &ERF_ERROR_INVALID_INPUT,
            "integer-class gpuArray inputs are not supported",
        ));
    }
    if runmat_accelerate_api::handle_is_logical(&handle) {
        return Err(erf_error_with_detail(
            &ERF_ERROR_INVALID_INPUT,
            "logical gpuArray inputs are not supported",
        ));
    }
    if runmat_accelerate_api::handle_storage(&handle)
        == runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
    {
        return Err(erf_error_with_detail(
            &ERF_ERROR_INVALID_INPUT,
            "complex gpuArray inputs are not supported",
        ));
    }

    let provider = runmat_accelerate_api::provider_for_handle(&handle).ok_or_else(|| {
        erf_error_with_detail(&ERF_ERROR_INTERNAL, "GPU provider unavailable for input")
    })?;
    match provider.unary_erf(&handle).await {
        Ok(out) if valid_real_gpu_output(&out, &handle, provider) => {
            return Ok(gpu_helpers::resident_gpu_value(out));
        }
        Ok(out) => {
            free_rejected_gpu_output(&out, &handle);
            return Err(erf_error_with_detail(
                &ERF_ERROR_INTERNAL,
                "provider unary_erf returned malformed output",
            ));
        }
        Err(err) if is_unsupported_provider_hook(&err) => {}
        Err(err) => {
            return Err(erf_error_with_detail(
                &ERF_ERROR_INTERNAL,
                format!("provider unary_erf failed: {err}"),
            ))
        }
    }
    let tensor = gpu_helpers::gather_tensor_async(&handle)
        .await
        .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
    let result = erf_tensor(tensor)?;
    let out = gpu_helpers::upload_tensor(provider, &result).map_err(|err| {
        erf_error_with_detail(
            &ERF_ERROR_INTERNAL,
            format!("failed to restore fallback result to input provider: {err}"),
        )
    })?;
    if !valid_real_gpu_output(&out, &handle, provider) {
        free_rejected_gpu_output(&out, &handle);
        return Err(erf_error_with_detail(
            &ERF_ERROR_INTERNAL,
            "provider upload returned malformed fallback output",
        ));
    }
    Ok(gpu_helpers::resident_gpu_value(out))
}

fn erf_real(value: Value) -> BuiltinResult<Value> {
    let tensor = tensor::value_into_tensor_for(BUILTIN_NAME, value)
        .map_err(|e| erf_error_with_detail(&ERF_ERROR_INVALID_INPUT, e))?;
    Ok(erf_tensor_into_value(erf_tensor(tensor)?))
}

fn erf_tensor(tensor: Tensor) -> BuiltinResult<Tensor> {
    let shape = tensor.shape.clone();
    let storage = tensor
        .into_numeric_storage()
        .map_err(|e| builtin_error(format!("erf: {e}")))?;
    let output = match storage {
        NumericStorage::F64(values) => {
            NumericStorage::F64(values.into_iter().map(erf_real_scalar).collect())
        }
        NumericStorage::F32(values) => NumericStorage::F32(
            values
                .into_iter()
                .map(|value| erf_real_scalar(f64::from(value)) as f32)
                .collect(),
        ),
        storage => {
            return Err(erf_error_with_detail(
                &ERF_ERROR_INVALID_INPUT,
                format!("{} inputs are not supported", storage.class_name()),
            ))
        }
    };
    Tensor::from_numeric_storage(output, shape).map_err(|e| builtin_error(format!("erf: {e}")))
}

fn erf_tensor_into_value(tensor: Tensor) -> Value {
    if tensor.len() == 1 && tensor.numeric_dtype() == NumericDType::F64 {
        if let Some(NumericScalar::F64(value)) = tensor.numeric_value_at(0) {
            return Value::Num(value);
        }
    }
    Value::Tensor(tensor)
}

fn erf_real_scalar(value: f64) -> f64 {
    libm::erf(value)
}

fn is_unsupported_provider_hook(err: &anyhow::Error) -> bool {
    err.to_string().contains("unary_erf not supported")
}

fn valid_real_gpu_output(
    output: &GpuTensorHandle,
    input: &GpuTensorHandle,
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
) -> bool {
    output.shape == input.shape
        && output.device_id == input.device_id
        && !gpu_handles_alias(output, input)
        && runmat_accelerate_api::handle_precision(output)
            == runmat_accelerate_api::handle_precision(input)
        && runmat_accelerate_api::handle_storage(output)
            == runmat_accelerate_api::GpuTensorStorage::Real
        && runmat_accelerate_api::handle_integer_type(output).is_none()
        && !runmat_accelerate_api::handle_is_logical(output)
        && runmat_accelerate_api::provider_for_handle(output)
            .filter(|owner| owner.device_id() == output.device_id)
            .is_some_and(|owner| std::ptr::eq(owner, provider))
}

fn gpu_handles_alias(lhs: &GpuTensorHandle, rhs: &GpuTensorHandle) -> bool {
    lhs.device_id == rhs.device_id && lhs.buffer_id == rhs.buffer_id
}

fn free_rejected_gpu_output(output: &GpuTensorHandle, input: &GpuTensorHandle) {
    if gpu_handles_alias(output, input) {
        return;
    }
    if let Some(owner) = runmat_accelerate_api::provider_for_handle(output)
        .filter(|owner| owner.device_id() == output.device_id)
    {
        let _ = owner.free(output);
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::{
        CharArray, ComplexTensor, IntValue, IntegerStorage, LogicalArray, ResolveContext,
        SparseTensor, Type,
    };

    fn erf_builtin(value: Value) -> BuiltinResult<Value> {
        block_on(super::erf_builtin(value))
    }

    fn approx_eq(got: f64, expected: f64, tol: f64) {
        assert!(
            (got - expected).abs() <= tol,
            "got {got}, expected {expected}, tol {tol}"
        );
    }

    fn values_f64(tensor: &Tensor) -> Vec<f64> {
        tensor.materialize_f64()
    }

    #[test]
    fn erf_descriptor_signatures_cover_core_form() {
        let labels = ERF_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect::<Vec<_>>();
        assert!(labels.contains(&"Y = erf(X)"));
    }

    #[test]
    fn erf_type_preserves_tensor_shape() {
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

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn erf_scalar_values() {
        match erf_builtin(Value::Num(1.0)).expect("erf") {
            Value::Num(v) => approx_eq(v, 0.842_700_792_949_714_9, 1e-15),
            other => panic!("expected scalar result, got {other:?}"),
        }
        match erf_builtin(Value::Num(-0.5)).expect("erf") {
            Value::Num(v) => approx_eq(v, -0.520_499_877_813_046_5, 1e-15),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn erf_tensor_shape_and_values() {
        let tensor = Tensor::new(vec![-0.5, 0.0, 1.0, 3.0], vec![2, 2]).unwrap();
        let result = erf_builtin(Value::Tensor(tensor)).expect("erf");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                assert_eq!(t.numeric_dtype(), NumericDType::F64);
                let expected = [
                    -0.520_499_877_813_046_5,
                    0.0,
                    0.842_700_792_949_714_9,
                    0.999_977_909_503_001_4,
                ];
                for (got, expected) in values_f64(&t).iter().zip(expected.iter()) {
                    approx_eq(*got, *expected, 1e-15);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn erf_single_tensor_preserves_single_dtype() {
        let tensor = Tensor::new_with_dtype(vec![0.5, 1.0], vec![1, 2], NumericDType::F32).unwrap();
        let result = erf_builtin(Value::Tensor(tensor)).expect("erf");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.numeric_dtype(), NumericDType::F32);
                let values = values_f64(&t);
                approx_eq(values[0], 0.520_499_885_082_244_9, 1e-7);
                approx_eq(values[1], 0.842_700_779_438_018_8, 1e-7);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn erf_rejects_integer_bool_logical_and_char_inputs() {
        assert!(erf_builtin(Value::Int(IntValue::I32(1))).is_err());
        assert!(erf_builtin(Value::Bool(false)).is_err());
        let logical = LogicalArray::new(vec![1, 0, 1, 0], vec![2, 2]).unwrap();
        assert!(erf_builtin(Value::LogicalArray(logical)).is_err());
        let chars = CharArray::new(vec!['\0', '\u{1}'], 1, 2).unwrap();
        assert!(erf_builtin(Value::CharArray(chars)).is_err());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn erf_rejects_typed_integer_tensor_storage() {
        let scalar =
            Tensor::new_integer(IntegerStorage::I16(vec![1]), vec![1, 1]).expect("int tensor");
        assert!(erf_builtin(Value::Tensor(scalar)).is_err());
        let tensor =
            Tensor::new_integer(IntegerStorage::I16(vec![1, 0]), vec![1, 2]).expect("int tensor");
        assert!(erf_builtin(Value::Tensor(tensor)).is_err());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn erf_nan_and_infinities_follow_real_limits() {
        match erf_builtin(Value::Num(f64::INFINITY)).expect("erf") {
            Value::Num(v) => assert_eq!(v, 1.0),
            other => panic!("expected scalar result, got {other:?}"),
        }
        match erf_builtin(Value::Num(f64::NEG_INFINITY)).expect("erf") {
            Value::Num(v) => assert_eq!(v, -1.0),
            other => panic!("expected scalar result, got {other:?}"),
        }
        match erf_builtin(Value::Num(f64::NAN)).expect("erf") {
            Value::Num(v) => assert!(v.is_nan()),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn erf_rejects_complex_string_and_sparse_inputs() {
        let err = erf_builtin(Value::Complex(1.0, 1.0)).expect_err("complex should error");
        assert_eq!(err.identifier(), ERF_ERROR_INVALID_INPUT.identifier);

        let complex = ComplexTensor::new(vec![(1.0, 0.0)], vec![1, 1]).unwrap();
        let err = erf_builtin(Value::ComplexTensor(complex)).expect_err("complex should error");
        assert_eq!(err.identifier(), ERF_ERROR_INVALID_INPUT.identifier);

        let err = erf_builtin(Value::from("1")).expect_err("string should error");
        assert_eq!(err.identifier(), ERF_ERROR_INVALID_INPUT.identifier);

        let sparse = SparseTensor::zeros(2, 2);
        let err = erf_builtin(Value::SparseTensor(sparse)).expect_err("sparse should error");
        assert_eq!(err.identifier(), ERF_ERROR_INVALID_INPUT.identifier);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn erf_gpu_provider_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![-1.0, 0.0, 0.5, 1.0], vec![1, 4]).unwrap();
            let view = HostTensorView {
                data: tensor.as_f64_slice().expect("double input"),
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = erf_builtin(Value::GpuTensor(handle)).expect("erf");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![1, 4]);
            for (got, input) in gathered
                .materialize_f64()
                .iter()
                .zip(tensor.as_f64_slice().expect("double input"))
            {
                approx_eq(*got, erf_real_scalar(*input), 1e-15);
            }
        });
    }

    #[test]
    fn erf_gpu_rejects_integer_and_logical_storage_before_provider_dispatch() {
        test_support::with_test_provider(|provider| {
            let integer = Tensor::new_integer(IntegerStorage::U8(vec![0, 1]), vec![1, 2]).unwrap();
            let integer_handle =
                gpu_helpers::upload_tensor(provider, &integer).expect("integer upload");
            let integer_error =
                erf_builtin(Value::GpuTensor(integer_handle)).expect_err("integer GPU input");
            assert_eq!(
                integer_error.identifier(),
                ERF_ERROR_INVALID_INPUT.identifier
            );

            let logical_source = Tensor::new(vec![0.0, 1.0], vec![1, 2]).unwrap();
            let logical_handle = provider
                .upload(&HostTensorView {
                    data: logical_source
                        .as_f64_slice()
                        .expect("double logical source"),
                    shape: &logical_source.shape,
                })
                .expect("logical upload");
            runmat_accelerate_api::set_handle_logical(&logical_handle, true);
            let logical_error =
                erf_builtin(Value::GpuTensor(logical_handle)).expect_err("logical GPU input");
            assert_eq!(
                logical_error.identifier(),
                ERF_ERROR_INVALID_INPUT.identifier
            );
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn erf_wgpu_matches_cpu_elementwise() {
        if runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        )
        .is_err()
        {
            return;
        }
        let tensor = Tensor::new(vec![-3.0, -1.0, -0.25, 0.0, 0.25, 1.0, 3.0], vec![1, 7]).unwrap();
        let cpu = erf_tensor(tensor.clone()).expect("cpu erf");
        let Some(provider) = runmat_accelerate_api::provider() else {
            return;
        };
        let view = HostTensorView {
            data: tensor.as_f64_slice().expect("double input"),
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload");
        let gpu_value = block_on(erf_gpu(handle)).expect("gpu erf");
        let gathered = test_support::gather(gpu_value).expect("gather");
        assert_eq!(gathered.shape, cpu.shape);
        let tol = match provider.precision() {
            runmat_accelerate_api::ProviderPrecision::F64 => 1e-10,
            runmat_accelerate_api::ProviderPrecision::F32 => 3e-5,
        };
        for (got, expected) in gathered
            .materialize_f64()
            .iter()
            .zip(cpu.as_f64_slice().expect("double cpu result"))
        {
            approx_eq(*got, *expected, tol);
        }
        assert_eq!(gathered.materialize_f64()[3], 0.0);
    }
}
