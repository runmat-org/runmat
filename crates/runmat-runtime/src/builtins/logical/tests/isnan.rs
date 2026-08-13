//! MATLAB-compatible `isnan` builtin with GPU-aware semantics for RunMat.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinIntegerBackendRule,
    BuiltinIntegerCapabilityDescriptor, BuiltinIntegerComputationDomain,
    BuiltinIntegerInputAvailability, BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule,
    BuiltinIntegerOverflowRule, BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule,
    BuiltinOutputMode, BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType,
    BuiltinSignatureDescriptor, CharArray, ComplexTensor, LogicalArray, NumericScalar, StringArray,
    Tensor, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, FusionError,
    FusionExprContext, FusionKernelTemplate, GpuOpKind, ProviderHook, ReductionNaN,
    ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::builtins::logical::type_resolvers::logical_unary_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::logical::tests::isnan")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "isnan",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Unary {
        name: "logical_isnan",
    }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "Dispatches to the provider `logical_isnan` hook when available; otherwise the runtime gathers to host and builds the logical mask on the CPU.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::logical::tests::isnan")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "isnan",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: |ctx: &FusionExprContext| {
            let input = ctx.inputs.first().ok_or(FusionError::MissingInput(0))?;
            let (zero, one) = match ctx.scalar_ty {
                ScalarType::F32 => ("0.0", "1.0"),
                ScalarType::F64 => ("f64(0.0)", "f64(1.0)"),
                other => return Err(FusionError::UnsupportedPrecision(other)),
            };
            Ok(format!(
                "select({zero}, {one}, isNan({input}))"
            ))
        },
    }),
    reduction: None,
    emits_nan: false,
    notes: "Fused kernels emit 0/1 masks; providers can override with native logical-isnan implementations.",
};

const BUILTIN_NAME: &str = "isnan";

const ISNAN_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "tf",
    ty: BuiltinParamType::LogicalArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Logical mask for NaN elements.",
}];

const ISNAN_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input value to test for NaNs.",
}];

const ISNAN_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "tf = isnan(A)",
    inputs: &ISNAN_INPUTS,
    outputs: &ISNAN_OUTPUT,
}];

const ISNAN_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ISNAN.INVALID_INPUT",
    identifier: Some("RunMat:isnan:InvalidInput"),
    when: "Input is not numeric, logical, char, or string.",
    message: "isnan: expected numeric, logical, char, or string input",
};

const ISNAN_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ISNAN.INTERNAL",
    identifier: Some("RunMat:isnan:InternalError"),
    when: "Internal mask-construction or gather path fails.",
    message: "isnan: internal error",
};

const ISNAN_ERRORS: [BuiltinErrorDescriptor; 2] = [ISNAN_ERROR_INVALID_INPUT, ISNAN_ERROR_INTERNAL];

pub const ISNAN_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ISNAN_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ISNAN_ERRORS,
};
const ISNAN_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "A",
    classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Fixed-width integer elements cannot represent NaN.",
}];
pub const ISNAN_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "tf = isnan(integer_A)",
        inputs: &ISNAN_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Predicate,
        output_class: BuiltinIntegerOutputClassRule::Logical,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
        notes: "Returns a same-shaped all-false logical mask directly from integer class and shape. Resident integer masks are created on the exact owning provider without downloading the payload.",
    }];

fn isnan_error(name: &str, error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    isnan_error_with_message(name, error.message, error)
}

fn isnan_error_with_message(
    name: &str,
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(name);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runtime_builtin(
    name = "isnan",
    category = "logical/tests",
    summary = "Return a logical mask indicating which elements of the input are NaN.",
    keywords = "isnan,nan,logical,gpu",
    accel = "elementwise",
    type_resolver(logical_unary_type),
    descriptor(crate::builtins::logical::tests::isnan::ISNAN_DESCRIPTOR),
    integer_capabilities(crate::builtins::logical::tests::isnan::ISNAN_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::logical::tests::isnan"
)]
async fn isnan_builtin(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::GpuTensor(handle) => {
            if runmat_accelerate_api::handle_integer_type(&handle).is_some() {
                return resident_integer_mask(&handle);
            }
            let tensor = gpu_helpers::gather_tensor_async(&handle)
                .await
                .map_err(|err| {
                    isnan_error_with_message(
                        BUILTIN_NAME,
                        format!("{BUILTIN_NAME}: {err}"),
                        &ISNAN_ERROR_INTERNAL,
                    )
                })?;
            isnan_tensor(BUILTIN_NAME, tensor)
        }
        other => isnan_host(other),
    }
}

fn resident_integer_mask(handle: &runmat_accelerate_api::GpuTensorHandle) -> BuiltinResult<Value> {
    let integer = runmat_accelerate_api::handle_integer_type(handle)
        .expect("resident integer mask requires integer metadata");
    let storage = runmat_accelerate_api::handle_storage(handle);
    if gpu_helpers::exact_provider_for_handle(handle).is_none()
        || storage != runmat_accelerate_api::GpuTensorStorage::Real
        || runmat_accelerate_api::handle_precision(handle).is_some()
        || runmat_accelerate_api::handle_is_logical(handle)
        || !gpu_helpers::gpu_class_metadata_matches(handle, None, Some(integer), false)
    {
        return Err(isnan_error_with_message(
            BUILTIN_NAME,
            "isnan: resident integer metadata is contradictory",
            &ISNAN_ERROR_INTERNAL,
        ));
    }
    let mask = LogicalArray::new(
        vec![0; tensor::element_count(&handle.shape)],
        handle.shape.clone(),
    )
    .map_err(|error| internal_error(BUILTIN_NAME, format!("isnan: {error}")))?;
    gpu_helpers::restore_class_preserving_value(handle, Value::LogicalArray(mask), BUILTIN_NAME)
}

fn isnan_host(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::Num(x) => Ok(Value::Bool(x.is_nan())),
        Value::Int(_) | Value::Bool(_) => Ok(Value::Bool(false)),
        Value::Complex(re, im) => Ok(Value::Bool(re.is_nan() || im.is_nan())),
        Value::Tensor(tensor) => isnan_tensor(BUILTIN_NAME, tensor),
        Value::ComplexTensor(tensor) => isnan_complex_tensor(BUILTIN_NAME, tensor),
        Value::LogicalArray(array) => {
            let LogicalArray { shape, .. } = array;
            logical_zeros(BUILTIN_NAME, shape)
        }
        Value::CharArray(array) => {
            let CharArray { rows, cols, .. } = array;
            logical_zeros(BUILTIN_NAME, vec![rows, cols])
        }
        Value::String(_) => Ok(Value::Bool(false)),
        Value::StringArray(array) => {
            let StringArray { shape, .. } = array;
            logical_zeros(BUILTIN_NAME, shape)
        }
        _ => Err(isnan_error(BUILTIN_NAME, &ISNAN_ERROR_INVALID_INPUT)),
    }
}

fn isnan_tensor(name: &str, tensor: Tensor) -> BuiltinResult<Value> {
    let shape = tensor.shape.clone();
    let mut data = Vec::with_capacity(tensor::element_count(&shape));
    for index in 0..tensor::element_count(&shape) {
        let value = tensor
            .numeric_value_at(index)
            .ok_or_else(|| internal_error(name, format!("{name}: invalid numeric storage")))?;
        data.push(u8::from(numeric_scalar_is_nan(value)));
    }
    logical_result(name, data, shape)
}

fn isnan_complex_tensor(name: &str, tensor: ComplexTensor) -> BuiltinResult<Value> {
    let shape = tensor.shape.clone();
    let mut data = Vec::with_capacity(tensor::element_count(&shape));
    for index in 0..tensor::element_count(&shape) {
        let (real, imag) = tensor
            .numeric_value_at(index)
            .ok_or_else(|| internal_error(name, format!("{name}: invalid complex storage")))?;
        data.push(u8::from(
            numeric_scalar_is_nan(real) || numeric_scalar_is_nan(imag),
        ));
    }
    logical_result(name, data, shape)
}

fn numeric_scalar_is_nan(value: NumericScalar) -> bool {
    match value {
        NumericScalar::F64(value) => value.is_nan(),
        NumericScalar::F32(value) => value.is_nan(),
        _ => false,
    }
}

fn logical_zeros(name: &str, shape: Vec<usize>) -> BuiltinResult<Value> {
    let total = tensor::element_count(&shape);
    if total == 0 {
        return LogicalArray::new(Vec::new(), shape)
            .map(Value::LogicalArray)
            .map_err(|e| logical_array_error(name, e));
    }
    let data = vec![0u8; total];
    logical_result(name, data, shape)
}

fn logical_result(name: &str, bits: Vec<u8>, shape: Vec<usize>) -> BuiltinResult<Value> {
    let total = tensor::element_count(&shape);
    if total != bits.len() {
        return Err(internal_error(
            name,
            format!(
                "{name}: internal error, mask length {} does not match shape {:?}",
                bits.len(),
                shape
            ),
        ));
    }
    if total == 1 {
        Ok(Value::Bool(bits[0] != 0))
    } else {
        LogicalArray::new(bits, shape)
            .map(Value::LogicalArray)
            .map_err(|e| logical_array_error(name, e))
    }
}

fn logical_array_error(name: &str, err: impl std::fmt::Display) -> RuntimeError {
    internal_error(name, format!("{name}: {err}"))
}

fn internal_error(name: &str, message: impl Into<String>) -> RuntimeError {
    isnan_error_with_message(name, message, &ISNAN_ERROR_INTERNAL)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_builtins::{IntegerComplexStorage, IntegerStorage, ResolveContext, Type};

    #[test]
    fn isnan_type_returns_logical() {
        let out = logical_unary_type(
            &[Type::Tensor { shape: None }],
            &ResolveContext::new(Vec::new()),
        );
        assert_eq!(out, Type::logical());
    }

    fn run_isnan(value: Value) -> BuiltinResult<Value> {
        block_on(super::isnan_builtin(value))
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isnan_scalar_nan() {
        let result = run_isnan(Value::Num(f64::NAN)).expect("isnan");
        assert_eq!(result, Value::Bool(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isnan_scalar_finite() {
        let result = run_isnan(Value::Num(5.0)).expect("isnan");
        assert_eq!(result, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isnan_scalar_bool_false() {
        let result = run_isnan(Value::Bool(true)).expect("isnan");
        assert_eq!(result, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isnan_tensor_mask() {
        let tensor = Tensor::new(vec![1.0, f64::NAN, 3.0, f64::NAN], vec![2, 2]).unwrap();
        let result = run_isnan(Value::Tensor(tensor)).expect("isnan");
        match result {
            Value::LogicalArray(mask) => {
                assert_eq!(mask.shape, vec![2, 2]);
                assert_eq!(mask.data, vec![0, 1, 0, 1]);
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isnan_typed_integer_tensor_is_always_false() {
        let tensor =
            Tensor::new_integer(IntegerStorage::I16(vec![1, -2, 3, -4]), vec![2, 2]).unwrap();
        let result = run_isnan(Value::Tensor(tensor)).expect("isnan");
        match result {
            Value::LogicalArray(mask) => {
                assert_eq!(mask.shape, vec![2, 2]);
                assert_eq!(mask.data, vec![0, 0, 0, 0]);
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[test]
    fn isnan_returns_same_shaped_false_for_all_integer_classes() {
        let storages = [
            IntegerStorage::I8(vec![i8::MIN, i8::MAX]),
            IntegerStorage::I16(vec![i16::MIN, i16::MAX]),
            IntegerStorage::I32(vec![i32::MIN, i32::MAX]),
            IntegerStorage::I64(vec![i64::MIN, i64::MAX]),
            IntegerStorage::U8(vec![u8::MIN, u8::MAX]),
            IntegerStorage::U16(vec![u16::MIN, u16::MAX]),
            IntegerStorage::U32(vec![u32::MIN, u32::MAX]),
            IntegerStorage::U64(vec![u64::MIN, u64::MAX]),
        ];
        for storage in storages {
            let tensor = Tensor::new_integer(storage, vec![2, 1]).expect("integer tensor");
            let result = run_isnan(Value::Tensor(tensor)).expect("isnan");
            assert!(matches!(
                result,
                Value::LogicalArray(mask)
                    if mask.shape == vec![2, 1] && mask.data == vec![0, 0]
            ));
        }
    }

    #[test]
    fn isnan_resident_integer_creates_resident_logical_without_downloading_source() {
        test_support::with_test_provider(|provider| {
            let tensor =
                Tensor::new_integer(IntegerStorage::I64(vec![i64::MIN, i64::MAX]), vec![1, 2])
                    .expect("integer tensor");
            let source = gpu_helpers::upload_tensor(provider, &tensor).expect("upload integer");
            let result = run_isnan(Value::GpuTensor(source.clone())).expect("resident isnan");
            let Value::GpuTensor(mask_handle) = &result else {
                panic!("resident isnan must preserve NewHandle residency policy")
            };
            assert_ne!(mask_handle.buffer_id, source.buffer_id);
            assert!(runmat_accelerate_api::handle_is_logical(mask_handle));
            assert!(runmat_accelerate_api::handle_integer_type(mask_handle).is_none());
            assert!(gpu_helpers::exact_provider_for_handle(&source).is_some());
            let gathered = test_support::gather(result).expect("gather logical mask");
            assert_eq!(gathered.shape, vec![1, 2]);
            assert_eq!(
                gathered
                    .into_numeric_storage()
                    .expect("mask storage")
                    .materialize_f64(),
                vec![0.0, 0.0]
            );
            provider.free(&source).ok();
        });
    }

    #[test]
    fn isnan_rejects_contradictory_resident_integer_metadata() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new_integer(IntegerStorage::U16(vec![1]), vec![1, 1])
                .expect("integer tensor");
            let handle = gpu_helpers::upload_tensor(provider, &tensor).expect("upload integer");
            runmat_accelerate_api::set_handle_precision(&handle, provider.precision());
            let error = run_isnan(Value::GpuTensor(handle.clone()))
                .expect_err("integer/float metadata contradiction must reject");
            assert!(error.message().contains("metadata is contradictory"));
            provider.free(&handle).ok();
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isnan_logical_array_returns_zeros() {
        let logical = LogicalArray::new(vec![1, 0, 1], vec![3, 1]).unwrap();
        let result = run_isnan(Value::LogicalArray(logical)).expect("isnan");
        match result {
            Value::LogicalArray(mask) => {
                assert_eq!(mask.shape, vec![3, 1]);
                assert!(mask.data.iter().all(|&bit| bit == 0));
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isnan_complex_tensor_mask() {
        let tensor = ComplexTensor::new(
            vec![(0.0, 0.0), (f64::NAN, 0.0), (0.0, f64::NAN)],
            vec![3, 1],
        )
        .unwrap();
        let result = run_isnan(Value::ComplexTensor(tensor)).expect("isnan");
        match result {
            Value::LogicalArray(mask) => {
                assert_eq!(mask.shape, vec![3, 1]);
                assert_eq!(mask.data, vec![0, 1, 1]);
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isnan_typed_complex_integer_storage_is_always_false() {
        let storage = IntegerComplexStorage::new(
            IntegerStorage::I64(vec![i64::MIN, i64::MAX]),
            IntegerStorage::I64(vec![0, -7]),
        )
        .unwrap();
        let tensor = ComplexTensor::new_integer(storage, vec![1, 2]).unwrap();
        let result = run_isnan(Value::ComplexTensor(tensor)).expect("isnan");
        match result {
            Value::LogicalArray(mask) => {
                assert_eq!(mask.shape, vec![1, 2]);
                assert_eq!(mask.data, vec![0, 0]);
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isnan_string_scalar_false() {
        let result = run_isnan(Value::String("NaN".to_string())).expect("isnan");
        assert_eq!(result, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isnan_string_array_returns_all_false() {
        let strings = StringArray::new(vec!["foo".into(), "bar".into()], vec![1, 2]).unwrap();
        let result = run_isnan(Value::StringArray(strings)).expect("isnan");
        match result {
            Value::LogicalArray(mask) => {
                assert_eq!(mask.shape, vec![1, 2]);
                assert_eq!(mask.data, vec![0, 0]);
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isnan_empty_tensor_preserves_shape() {
        let tensor = Tensor::new(Vec::new(), vec![0, 3]).unwrap();
        let result = run_isnan(Value::Tensor(tensor)).expect("isnan");
        match result {
            Value::LogicalArray(mask) => {
                assert_eq!(mask.shape, vec![0, 3]);
                assert!(mask.data.is_empty());
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isnan_rejects_unsupported_types() {
        let err = run_isnan(Value::FunctionHandle("foo".to_string()))
            .expect_err("isnan should reject function handles");
        assert!(
            err.message()
                .contains("expected numeric, logical, char, or string input"),
            "unexpected error message: {err:?}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isnan_char_array_returns_zeros() {
        let array = CharArray::new("NaN".chars().collect(), 1, 3).unwrap();
        let result = run_isnan(Value::CharArray(array)).expect("isnan");
        match result {
            Value::LogicalArray(mask) => {
                assert_eq!(mask.shape, vec![1, 3]);
                assert_eq!(mask.data, vec![0, 0, 0]);
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isnan_gpu_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, f64::NAN, 2.0], vec![3, 1]).unwrap();
            let handle = gpu_helpers::upload_tensor(provider, &tensor).expect("upload");
            let result = run_isnan(Value::GpuTensor(handle)).expect("isnan");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![3, 1]);
            assert_eq!(
                gathered
                    .into_numeric_storage()
                    .expect("gathered storage")
                    .materialize_f64(),
                vec![0.0, 1.0, 0.0]
            );
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn isnan_wgpu_matches_host_path() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let tensor = Tensor::new(vec![1.0, f64::NAN, 0.0], vec![3, 1]).unwrap();
        let cpu = isnan_tensor("isnan", tensor.clone()).expect("cpu path");
        let provider = runmat_accelerate_api::provider().unwrap();
        let handle = gpu_helpers::upload_tensor(provider, &tensor).expect("upload");
        let gpu = run_isnan(Value::GpuTensor(handle)).expect("gpu path");
        let gathered = test_support::gather(gpu).expect("gather");
        let shape = gathered.shape.clone();
        let data = gathered
            .into_numeric_storage()
            .expect("gathered storage")
            .materialize_f64();
        match cpu {
            Value::LogicalArray(expected) => {
                assert_eq!(shape, expected.shape);
                let expected_f64: Vec<f64> = expected
                    .data
                    .iter()
                    .map(|&b| if b != 0 { 1.0 } else { 0.0 })
                    .collect();
                assert_eq!(data, expected_f64);
            }
            Value::Bool(flag) => {
                assert_eq!(data, vec![if flag { 1.0 } else { 0.0 }]);
            }
            other => panic!("unexpected results {other:?}"),
        }
    }
}
