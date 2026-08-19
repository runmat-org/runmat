//! MATLAB-compatible `isinf` builtin with GPU-aware semantics for RunMat.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};
use runmat_builtins::{
    BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor, BuiltinIntegerComputationDomain,
    BuiltinIntegerInputAvailability, BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule,
    BuiltinIntegerOverflowRule, BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule,
};
use runmat_macros::runtime_builtin;
use runmat_value::{
    CharArray, ComplexTensor, LogicalArray, NumericScalar, StringArray, Tensor, Value,
};

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, FusionError,
    FusionExprContext, FusionKernelTemplate, GpuOpKind, ProviderHook, ReductionNaN,
    ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::builtins::logical::type_resolvers::logical_unary_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::logical::tests::isinf")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "isinf",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Unary {
        name: "logical_isinf",
    }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "Dispatches to the provider `logical_isinf` hook when available; otherwise the runtime gathers to host and builds the logical mask on the CPU.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::logical::tests::isinf")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "isinf",
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
            Ok(format!("select({zero}, {one}, isInf({input}))"))
        },
    }),
    reduction: None,
    emits_nan: false,
    notes: "Fused kernels emit 0/1 masks; providers can override with native logical-isinf implementations.",
};

const BUILTIN_NAME: &str = "isinf";

const ISINF_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "tf",
    ty: BuiltinParamType::LogicalArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Logical mask for infinite elements.",
}];

const ISINF_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input value to test for infinities.",
}];

const ISINF_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "tf = isinf(A)",
    inputs: &ISINF_INPUTS,
    outputs: &ISINF_OUTPUT,
}];

const ISINF_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ISINF.INVALID_INPUT",
    identifier: Some("RunMat:isinf:InvalidInput"),
    when: "Input is not numeric, logical, char, or string.",
    message: "isinf: expected numeric, logical, char, or string input",
};

const ISINF_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ISINF.INTERNAL",
    identifier: Some("RunMat:isinf:InternalError"),
    when: "Internal mask-construction or gather path fails.",
    message: "isinf: internal error",
};

const ISINF_ERRORS: [BuiltinErrorDescriptor; 2] = [ISINF_ERROR_INVALID_INPUT, ISINF_ERROR_INTERNAL];

pub const ISINF_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ISINF_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ISINF_ERRORS,
};
const ISINF_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "A",
    classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "No fixed-width integer value is infinite.",
}];
pub const ISINF_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [BuiltinIntegerCapabilityDescriptor { form: "tf = isinf(integer_A)", inputs: &ISINF_INTEGER_INPUTS, computation_domain: BuiltinIntegerComputationDomain::Predicate, output_class: BuiltinIntegerOutputClassRule::Logical, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::HostAndGpu, overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving, notes: "Returns a same-shaped all-false logical mask directly from integer class and shape; resident execution or exact fallback preserves typed storage." }];

fn isinf_error(name: &str, error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    isinf_error_with_message(name, error.message, error)
}

fn isinf_error_with_message(
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
    name = "isinf",
    category = "logical/tests",
    summary = "Return a logical mask indicating which elements of the input are ±Inf.",
    keywords = "isinf,infinity,logical,gpu",
    accel = "elementwise",
    type_resolver(logical_unary_type),
    descriptor(crate::builtins::logical::tests::isinf::ISINF_DESCRIPTOR),
    integer_capabilities(crate::builtins::logical::tests::isinf::ISINF_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::logical::tests::isinf"
)]
async fn isinf_builtin(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::GpuTensor(handle) => {
            if runmat_accelerate_api::handle_integer_type(&handle).is_some() {
                return resident_integer_mask(&handle, false);
            }
            let provider = gpu_helpers::exact_provider_for_handle(&handle).ok_or_else(|| {
                isinf_error_with_message(
                    BUILTIN_NAME,
                    "isinf: no acceleration provider owns the input handle",
                    &ISINF_ERROR_INTERNAL,
                )
            })?;
            let metadata = gpu_helpers::snapshot_handle_metadata(&handle);
            {
                if let Ok(mask) = provider.logical_isinf(&handle) {
                    gpu_helpers::restore_handle_metadata(&handle, &metadata);
                    if valid_logical_mask(&handle, &mask, provider) {
                        return Ok(gpu_helpers::logical_gpu_value(mask));
                    }
                    gpu_helpers::free_unprotected_exact_owner(&mask, &[&handle]);
                    return Err(isinf_error_with_message(
                        BUILTIN_NAME,
                        "isinf: provider returned an invalid logical mask",
                        &ISINF_ERROR_INTERNAL,
                    ));
                }
            }
            gpu_helpers::restore_handle_metadata(&handle, &metadata);
            let host = gpu_helpers::download_value_preserving_residency_async(provider, &handle)
                .await
                .map_err(|err| {
                    isinf_error_with_message(
                        BUILTIN_NAME,
                        format!("{BUILTIN_NAME}: {err}"),
                        &ISINF_ERROR_INTERNAL,
                    )
                })?;
            let mask = isinf_host(host)?;
            gpu_helpers::restore_class_preserving_value(&handle, mask, BUILTIN_NAME)
        }
        other => isinf_host(other),
    }
}

fn valid_logical_mask(
    input: &runmat_accelerate_api::GpuTensorHandle,
    output: &runmat_accelerate_api::GpuTensorHandle,
    provider: &dyn runmat_accelerate_api::AccelProvider,
) -> bool {
    output.shape == input.shape
        && output.device_id == provider.device_id()
        && gpu_helpers::exact_provider_for_handle(output)
            .is_some_and(|owner| std::ptr::eq(owner, provider))
        && !gpu_helpers::same_gpu_handle(input, output)
        && runmat_accelerate_api::handle_storage(output)
            == runmat_accelerate_api::GpuTensorStorage::Real
        && runmat_accelerate_api::handle_precision(output) == Some(provider.precision())
        && runmat_accelerate_api::handle_integer_type(output).is_none()
        && runmat_accelerate_api::handle_class_name(output)
            .as_deref()
            .is_none_or(|class| matches!(class, "logical" | "single" | "double"))
}

fn resident_integer_mask(
    handle: &runmat_accelerate_api::GpuTensorHandle,
    value: bool,
) -> BuiltinResult<Value> {
    let integer = runmat_accelerate_api::handle_integer_type(handle)
        .expect("resident integer mask requires integer metadata");
    if gpu_helpers::exact_provider_for_handle(handle).is_none()
        || runmat_accelerate_api::handle_storage(handle)
            != runmat_accelerate_api::GpuTensorStorage::Real
        || runmat_accelerate_api::handle_precision(handle).is_some()
        || runmat_accelerate_api::handle_is_logical(handle)
        || !gpu_helpers::gpu_class_metadata_matches(handle, None, Some(integer), false)
    {
        return Err(isinf_error_with_message(
            BUILTIN_NAME,
            format!("{BUILTIN_NAME}: resident integer metadata is contradictory"),
            &ISINF_ERROR_INTERNAL,
        ));
    }
    let mask = LogicalArray::new(
        vec![u8::from(value); tensor::element_count(&handle.shape)],
        handle.shape.clone(),
    )
    .map_err(|error| internal_error(BUILTIN_NAME, format!("{BUILTIN_NAME}: {error}")))?;
    gpu_helpers::restore_class_preserving_value(handle, Value::LogicalArray(mask), BUILTIN_NAME)
}

fn isinf_host(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::Num(x) => Ok(Value::Bool(x.is_infinite())),
        Value::Int(_) | Value::Bool(_) => Ok(Value::Bool(false)),
        Value::Complex(re, im) => Ok(Value::Bool(re.is_infinite() || im.is_infinite())),
        Value::Tensor(tensor) => isinf_tensor(BUILTIN_NAME, tensor),
        Value::ComplexTensor(tensor) => isinf_complex_tensor(BUILTIN_NAME, tensor),
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
        _ => Err(isinf_error(BUILTIN_NAME, &ISINF_ERROR_INVALID_INPUT)),
    }
}

fn isinf_tensor(name: &str, tensor: Tensor) -> BuiltinResult<Value> {
    let shape = tensor.shape.clone();
    let mut data = Vec::with_capacity(tensor::element_count(&shape));
    for index in 0..tensor::element_count(&shape) {
        let value = tensor
            .numeric_value_at(index)
            .ok_or_else(|| internal_error(name, format!("{name}: invalid numeric storage")))?;
        data.push(u8::from(numeric_scalar_is_infinite(value)));
    }
    logical_result(name, data, shape)
}

fn isinf_complex_tensor(name: &str, tensor: ComplexTensor) -> BuiltinResult<Value> {
    let shape = tensor.shape.clone();
    let mut data = Vec::with_capacity(tensor::element_count(&shape));
    for index in 0..tensor::element_count(&shape) {
        let (real, imag) = tensor
            .numeric_value_at(index)
            .ok_or_else(|| internal_error(name, format!("{name}: invalid complex storage")))?;
        data.push(u8::from(
            numeric_scalar_is_infinite(real) || numeric_scalar_is_infinite(imag),
        ));
    }
    logical_result(name, data, shape)
}

fn numeric_scalar_is_infinite(value: NumericScalar) -> bool {
    match value {
        NumericScalar::F64(value) => value.is_infinite(),
        NumericScalar::F32(value) => value.is_infinite(),
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
    isinf_error_with_message(name, message, &ISINF_ERROR_INTERNAL)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_builtins::{ResolveContext, Type};
    use runmat_value::{IntegerComplexStorage, IntegerStorage};

    #[test]
    fn isinf_type_returns_logical() {
        let out = logical_unary_type(
            &[Type::Tensor { shape: None }],
            &ResolveContext::new(Vec::new()),
        );
        assert_eq!(out, Type::logical());
    }
    use runmat_value::IntValue;

    fn run_isinf(value: Value) -> BuiltinResult<Value> {
        block_on(super::isinf_builtin(value))
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isinf_scalar_positive() {
        let result = run_isinf(Value::Num(f64::INFINITY)).expect("isinf");
        assert_eq!(result, Value::Bool(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isinf_scalar_negative() {
        let result = run_isinf(Value::Num(f64::NEG_INFINITY)).expect("isinf");
        assert_eq!(result, Value::Bool(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isinf_scalar_finite() {
        let result = run_isinf(Value::Num(42.0)).expect("isinf");
        assert_eq!(result, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isinf_scalar_nan_false() {
        let result = run_isinf(Value::Num(f64::NAN)).expect("isinf");
        assert_eq!(result, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isinf_scalar_bool_false() {
        let result = run_isinf(Value::Bool(true)).expect("isinf");
        assert_eq!(result, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isinf_scalar_int_false() {
        let result = run_isinf(Value::Int(IntValue::I32(7))).expect("isinf");
        assert_eq!(result, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isinf_complex_scalar_detects_infinite_components() {
        let finite = run_isinf(Value::Complex(1.0, 2.0)).expect("isinf");
        assert_eq!(finite, Value::Bool(false));

        let inf_real = run_isinf(Value::Complex(f64::INFINITY, 0.0)).expect("isinf");
        assert_eq!(inf_real, Value::Bool(true));

        let inf_imag = run_isinf(Value::Complex(0.0, f64::NEG_INFINITY)).expect("isinf");
        assert_eq!(inf_imag, Value::Bool(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isinf_tensor_mask() {
        let tensor =
            Tensor::new(vec![1.0, f64::INFINITY, -f64::INFINITY, 0.0], vec![2, 2]).unwrap();
        let result = run_isinf(Value::Tensor(tensor)).expect("isinf");
        match result {
            Value::LogicalArray(mask) => {
                assert_eq!(mask.shape, vec![2, 2]);
                assert_eq!(mask.data, vec![0, 1, 1, 0]);
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isinf_typed_integer_tensor_is_always_false() {
        let tensor =
            Tensor::new_integer(IntegerStorage::U64(vec![1, 2, 3, 4]), vec![2, 2]).unwrap();
        let result = run_isinf(Value::Tensor(tensor)).expect("isinf");
        match result {
            Value::LogicalArray(mask) => {
                assert_eq!(mask.shape, vec![2, 2]);
                assert_eq!(mask.data, vec![0, 0, 0, 0]);
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isinf_logical_array_returns_zeros() {
        let logical = LogicalArray::new(vec![1, 0, 1], vec![3, 1]).unwrap();
        let result = run_isinf(Value::LogicalArray(logical)).expect("isinf");
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
    fn isinf_complex_tensor_mask() {
        let tensor = ComplexTensor::new(
            vec![(0.0, 0.0), (f64::INFINITY, 1.0), (2.0, f64::NEG_INFINITY)],
            vec![3, 1],
        )
        .unwrap();
        let result = run_isinf(Value::ComplexTensor(tensor)).expect("isinf");
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
    fn isinf_typed_complex_integer_storage_is_always_false() {
        let storage = IntegerComplexStorage::new(
            IntegerStorage::U64(vec![u64::MAX, 9_007_199_254_740_993]),
            IntegerStorage::U64(vec![0, 7]),
        )
        .unwrap();
        let tensor = ComplexTensor::new_integer(storage, vec![1, 2]).unwrap();
        let result = run_isinf(Value::ComplexTensor(tensor)).expect("isinf");
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
    fn isinf_string_scalar_false() {
        let result = run_isinf(Value::String("Inf".to_string())).expect("isinf");
        assert_eq!(result, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isinf_string_array_returns_all_false() {
        let strings = StringArray::new(vec!["foo".into(), "bar".into()], vec![1, 2]).unwrap();
        let result = run_isinf(Value::StringArray(strings)).expect("isinf");
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
    fn isinf_empty_tensor_preserves_shape() {
        let tensor = Tensor::new(Vec::new(), vec![0, 3]).unwrap();
        let result = run_isinf(Value::Tensor(tensor)).expect("isinf");
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
    fn isinf_singleton_tensor_returns_scalar_bool() {
        let tensor = Tensor::new(vec![f64::INFINITY], vec![1, 1]).unwrap();
        let result = run_isinf(Value::Tensor(tensor)).expect("isinf");
        assert_eq!(result, Value::Bool(true));

        let finite = Tensor::new(vec![0.0], vec![1, 1]).unwrap();
        let result = run_isinf(Value::Tensor(finite)).expect("isinf");
        assert_eq!(result, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isinf_rejects_unsupported_types() {
        let err = run_isinf(Value::FunctionHandle("foo".to_string()))
            .expect_err("isinf should reject function handles");
        assert!(
            err.message()
                .contains("expected numeric, logical, char, or string input"),
            "unexpected error message: {err:?}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isinf_char_array_returns_zeros() {
        let array = CharArray::new("Inf".chars().collect(), 1, 3).unwrap();
        let result = run_isinf(Value::CharArray(array)).expect("isinf");
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
    fn isinf_gpu_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, f64::INFINITY, -f64::INFINITY], vec![3, 1]).unwrap();
            let handle = gpu_helpers::upload_tensor(provider, &tensor).expect("upload");
            let result = run_isinf(Value::GpuTensor(handle)).expect("isinf");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![3, 1]);
            assert_eq!(
                gathered
                    .into_numeric_storage()
                    .expect("gathered storage")
                    .materialize_f64(),
                vec![0.0, 1.0, 1.0]
            );
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn isinf_wgpu_matches_host_path() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let tensor =
            Tensor::new(vec![1.0, f64::INFINITY, -f64::INFINITY, 0.0], vec![2, 2]).unwrap();
        let cpu = isinf_tensor("isinf", tensor.clone()).expect("cpu path");
        let provider = runmat_accelerate_api::provider().unwrap();
        let handle = gpu_helpers::upload_tensor(provider, &tensor).expect("upload");
        let gpu = run_isinf(Value::GpuTensor(handle)).expect("gpu path");
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
