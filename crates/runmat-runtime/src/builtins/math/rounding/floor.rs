//! MATLAB-compatible `floor` builtin with GPU-aware semantics for RunMat.

use runmat_accelerate_api::{AccelProvider, GpuTensorHandle};
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinIntegerBackendRule,
    BuiltinIntegerCapabilityDescriptor, BuiltinIntegerComputationDomain,
    BuiltinIntegerInputAvailability, BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule,
    BuiltinIntegerOverflowRule, BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule,
    BuiltinOutputMode, BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType,
    BuiltinSignatureDescriptor, CharArray, ComplexStorage, ComplexTensor, NumericStorage,
    ObjectInstance, StructValue, Tensor, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, FusionError,
    FusionExprContext, FusionKernelTemplate, GpuOpKind, ProviderHook, ReductionNaN,
    ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::builtins::math::type_resolvers::numeric_unary_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::rounding::floor")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "floor",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Unary { name: "unary_floor" }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "Providers may execute floor directly on the device; the runtime gathers to the host when unary_floor is unavailable.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::rounding::floor")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "floor",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: |ctx: &FusionExprContext| {
            let input = ctx
                .inputs
                .first()
                .ok_or(FusionError::MissingInput(0))?;
            Ok(format!("floor({input})"))
        },
    }),
    reduction: None,
    emits_nan: false,
    notes: "Fusion planner emits WGSL `floor` calls; providers can substitute custom kernels when available.",
};

const BUILTIN_NAME: &str = "floor";

const FLOOR_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Rounded output values.",
}];
const FLOOR_INPUTS_X: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Numeric, logical, char, or complex input.",
}];
const FLOOR_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "Y = floor(X)",
    inputs: &FLOOR_INPUTS_X,
    outputs: &FLOOR_OUTPUT,
}];
const FLOOR_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "X",
    classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Every real integer class is already integral, so floor preserves its exact class, shape, and values without floating conversion, including inside table and timetable variables.",
}];
pub const FLOOR_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "Y = floor(X) with real integer X, including integer table or timetable variables",
        inputs: &FLOOR_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
        notes: "Host integer storage is returned unchanged; resident integer storage is an exact identity operation that retains the original owning-provider handle.",
    }];
const FLOOR_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FLOOR.INVALID_INPUT",
    identifier: Some("RunMat:floor:InvalidInput"),
    when: "Input cannot be interpreted as numeric, logical, char, or complex data.",
    message: "floor: invalid input",
};
const FLOOR_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FLOOR.INVALID_ARGUMENT",
    identifier: Some("RunMat:floor:InvalidArgument"),
    when: "Argument count does not match supported floor invocation forms.",
    message: "floor: invalid argument",
};
const FLOOR_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FLOOR.INTERNAL",
    identifier: Some("RunMat:floor:Internal"),
    when: "Internal tensor conversion/allocation/provider interaction failed.",
    message: "floor: internal error",
};
const FLOOR_ERRORS: [BuiltinErrorDescriptor; 3] = [
    FLOOR_ERROR_INVALID_INPUT,
    FLOOR_ERROR_INVALID_ARGUMENT,
    FLOOR_ERROR_INTERNAL,
];
pub const FLOOR_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &FLOOR_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &FLOOR_ERRORS,
};

fn builtin_error_with_detail(
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
    name = "floor",
    category = "math/rounding",
    summary = "Round values toward negative infinity.",
    keywords = "floor,rounding,integers,gpu",
    accel = "unary",
    type_resolver(numeric_unary_type),
    descriptor(crate::builtins::math::rounding::floor::FLOOR_DESCRIPTOR),
    integer_capabilities(crate::builtins::math::rounding::floor::FLOOR_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::math::rounding::floor"
)]
async fn floor_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    if !rest.is_empty() {
        return Err(builtin_error_with_detail(
            &FLOOR_ERROR_INVALID_ARGUMENT,
            "floor accepts exactly one input",
        ));
    }
    crate::builtins::common::validation::reject_typed_complex_integer(&value, BUILTIN_NAME)?;
    match value {
        Value::GpuTensor(handle) => floor_gpu(handle).await,
        Value::Object(object) if crate::builtins::table::is_tabular_object(&object) => {
            floor_table(object).await
        }
        Value::Complex(re, im) => Ok(Value::Complex(
            apply_floor_scalar(re),
            apply_floor_scalar(im),
        )),
        Value::ComplexTensor(ct) => floor_complex_tensor(ct),
        Value::CharArray(ca) => floor_char_array(ca),
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(&logical)
                .map_err(|err| builtin_error_with_detail(&FLOOR_ERROR_INVALID_INPUT, err))?;
            Ok(tensor::tensor_into_value(floor_tensor(tensor)?))
        }
        Value::String(_) | Value::StringArray(_) => Err(builtin_error_with_detail(
            &FLOOR_ERROR_INVALID_INPUT,
            "expected numeric or logical input",
        )),
        other => floor_numeric(other),
    }
}

async fn floor_table(object: ObjectInstance) -> BuiltinResult<Value> {
    let variables = crate::builtins::table::table_variables(&object)
        .map_err(|err| builtin_error_with_detail(&FLOOR_ERROR_INVALID_INPUT, err.message))?;
    let mut rounded = StructValue::new();
    for (name, value) in variables.fields {
        crate::builtins::common::validation::reject_typed_complex_integer(&value, BUILTIN_NAME)?;
        let value = match value {
            Value::GpuTensor(handle) => floor_gpu(handle).await?,
            Value::Object(_) => {
                return Err(builtin_error_with_detail(
                    &FLOOR_ERROR_INVALID_INPUT,
                    format!("table variable {name} does not support floor"),
                ))
            }
            other => floor_host_value(other)?,
        };
        rounded.insert(name, value);
    }
    crate::builtins::table::table_replace_variables_like(&object, rounded)
        .map_err(|err| builtin_error_with_detail(&FLOOR_ERROR_INTERNAL, err.message))
}

fn floor_host_value(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::Complex(re, im) => Ok(Value::Complex(
            apply_floor_scalar(re),
            apply_floor_scalar(im),
        )),
        Value::ComplexTensor(ct) => floor_complex_tensor(ct),
        Value::CharArray(ca) => floor_char_array(ca),
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(&logical)
                .map_err(|err| builtin_error_with_detail(&FLOOR_ERROR_INVALID_INPUT, err))?;
            Ok(tensor::tensor_into_value(floor_tensor(tensor)?))
        }
        Value::String(_) | Value::StringArray(_) => Err(builtin_error_with_detail(
            &FLOOR_ERROR_INVALID_INPUT,
            "expected numeric or logical input",
        )),
        other => floor_numeric(other),
    }
}

fn floor_numeric(value: Value) -> BuiltinResult<Value> {
    let tensor = tensor::value_into_tensor_for("floor", value)
        .map_err(|err| builtin_error_with_detail(&FLOOR_ERROR_INVALID_INPUT, err))?;
    let floored = floor_tensor(tensor)?;
    Ok(tensor::tensor_into_value(floored))
}

fn floor_tensor(tensor: Tensor) -> BuiltinResult<Tensor> {
    let shape = tensor.shape.clone();
    let storage = tensor
        .into_numeric_storage()
        .map_err(|err| builtin_error_with_detail(&FLOOR_ERROR_INTERNAL, err))?;
    let output = match storage {
        NumericStorage::F64(values) => {
            NumericStorage::F64(values.into_iter().map(apply_floor_scalar).collect())
        }
        NumericStorage::F32(values) => NumericStorage::F32(
            values
                .into_iter()
                .map(|value| apply_floor_scalar(f64::from(value)) as f32)
                .collect(),
        ),
        integer => integer,
    };
    Tensor::from_numeric_storage(output, shape)
        .map_err(|err| builtin_error_with_detail(&FLOOR_ERROR_INTERNAL, err))
}

fn floor_complex_tensor(ct: ComplexTensor) -> BuiltinResult<Value> {
    let shape = ct.shape.clone();
    let storage = match ct.into_complex_storage() {
        ComplexStorage::F64(values) => ComplexStorage::F64(
            values
                .into_iter()
                .map(|(re, im)| (apply_floor_scalar(re), apply_floor_scalar(im)))
                .collect(),
        ),
        ComplexStorage::F32(values) => ComplexStorage::F32(
            values
                .into_iter()
                .map(|(re, im)| (re.floor(), im.floor()))
                .collect(),
        ),
        ComplexStorage::Integer(_) => {
            return Err(builtin_error_with_detail(
                &FLOOR_ERROR_INVALID_INPUT,
                "operations involving complex numbers with integer types are not supported",
            ))
        }
    };
    let tensor = ComplexTensor::from_complex_storage(storage, shape)
        .map_err(|e| builtin_error_with_detail(&FLOOR_ERROR_INTERNAL, e))?;
    Ok(Value::ComplexTensor(tensor))
}

fn floor_char_array(ca: CharArray) -> BuiltinResult<Value> {
    let mut data = Vec::with_capacity(ca.data.len());
    for ch in ca.data {
        data.push(apply_floor_scalar(ch as u32 as f64));
    }
    let tensor = Tensor::new(data, vec![ca.rows, ca.cols])
        .map_err(|e| builtin_error_with_detail(&FLOOR_ERROR_INTERNAL, e))?;
    Ok(Value::Tensor(tensor))
}

async fn floor_gpu(handle: GpuTensorHandle) -> BuiltinResult<Value> {
    if runmat_accelerate_api::handle_integer_type(&handle).is_some() {
        return Ok(gpu_helpers::resident_gpu_value(handle));
    }
    let provider = runmat_accelerate_api::provider_for_handle(&handle);
    if !runmat_accelerate_api::handle_is_logical(&handle) {
        if let Some(provider) = provider {
            if let Ok(out) = provider.unary_floor(&handle).await {
                if rounding_native_output_matches(&handle, &out, provider) {
                    return Ok(gpu_helpers::resident_gpu_value(out));
                }
                free_rejected_rounding_output(&out, &handle, provider);
            }
        }
    }
    let tensor = gpu_helpers::gather_tensor_async(&handle).await?;
    let floored = floor_tensor(tensor)?;
    if let Some(provider) = provider {
        let uploaded = gpu_helpers::upload_tensor(provider, &floored)
            .map_err(|err| builtin_error_with_detail(&FLOOR_ERROR_INTERNAL, err))?;
        return Ok(gpu_helpers::resident_gpu_value(uploaded));
    }
    Ok(tensor::tensor_into_value(floored))
}

fn rounding_native_output_matches(
    input: &GpuTensorHandle,
    output: &GpuTensorHandle,
    provider: &dyn AccelProvider,
) -> bool {
    output.shape == input.shape
        && output.device_id == input.device_id
        && !gpu_handles_alias(output, input)
        && runmat_accelerate_api::handle_storage(output)
            == runmat_accelerate_api::handle_storage(input)
        && runmat_accelerate_api::handle_integer_type(output).is_none()
        && !runmat_accelerate_api::handle_is_logical(output)
        && runmat_accelerate_api::handle_precision(output)
            == runmat_accelerate_api::handle_precision(input)
        && runmat_accelerate_api::provider_for_handle(output)
            .is_some_and(|owner| std::ptr::eq(owner, provider))
}

fn gpu_handles_alias(lhs: &GpuTensorHandle, rhs: &GpuTensorHandle) -> bool {
    lhs.device_id == rhs.device_id && lhs.buffer_id == rhs.buffer_id
}

fn free_rejected_rounding_output(
    output: &GpuTensorHandle,
    input: &GpuTensorHandle,
    provider: &dyn AccelProvider,
) {
    if !gpu_handles_alias(output, input) {
        let owner = runmat_accelerate_api::provider_for_handle(output).unwrap_or(provider);
        let _ = owner.free(output);
    }
}

fn apply_floor_scalar(value: f64) -> f64 {
    if !value.is_finite() {
        return value;
    }
    value.floor()
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use crate::RuntimeError;
    use futures::executor::block_on;
    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::{
        ComplexStorage, ComplexTensor, IntValue, IntegerStorage, LogicalArray, ResolveContext,
        Tensor, Type, Value,
    };

    fn floor_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        block_on(super::floor_builtin(value, rest))
    }

    fn assert_error_contains(error: RuntimeError, needle: &str) {
        assert!(
            error.message().contains(needle),
            "unexpected error: {}",
            error.message()
        );
    }

    #[test]
    fn floor_tensor_preserves_native_single_storage() {
        let input = Tensor::from_f32(vec![1.75, -1.25], vec![1, 2]).unwrap();
        let output = floor_tensor(input).unwrap();

        assert_eq!(
            output.into_numeric_storage().unwrap(),
            NumericStorage::F32(vec![1.0, -2.0])
        );
    }

    #[test]
    fn floor_descriptor_exposes_matlab_form() {
        let labels: Vec<&str> = FLOOR_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert_eq!(labels, vec!["Y = floor(X)"]);
        assert_eq!(FLOOR_INTEGER_CAPABILITIES.len(), 1);
        let capability = &FLOOR_INTEGER_CAPABILITIES[0];
        assert_eq!(capability.inputs[0].classes.len(), 8);
        assert_eq!(
            capability.inputs[0].availability,
            BuiltinIntegerInputAvailability::Documented
        );
        assert_eq!(
            capability.computation_domain,
            BuiltinIntegerComputationDomain::ExactInteger
        );
        assert_eq!(
            capability.output_class,
            BuiltinIntegerOutputClassRule::PreserveInput
        );
        assert_eq!(capability.backend, BuiltinIntegerBackendRule::HostAndGpu);
    }

    #[test]
    fn floor_type_preserves_tensor_shape() {
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
    fn floor_type_scalar_tensor_returns_num() {
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
    fn floor_scalar_positive_and_negative() {
        let value = Value::Num(-2.7);
        let result = floor_builtin(value, Vec::new()).expect("floor");
        match result {
            Value::Num(v) => assert_eq!(v, -3.0),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn floor_integer_tensor() {
        let tensor = Tensor::new(vec![1.2, 4.7, -3.4, 5.0], vec![2, 2]).unwrap();
        let result = floor_builtin(Value::Tensor(tensor), Vec::new()).expect("floor");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                assert_eq!(t.materialize_f64(), vec![1.0, 4.0, -4.0, 5.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn floor_complex_value() {
        let result = floor_builtin(Value::Complex(1.7, -2.3), Vec::new()).expect("floor");
        match result {
            Value::Complex(re, im) => {
                assert_eq!(re, 1.0);
                assert_eq!(im, -3.0);
            }
            other => panic!("expected complex result, got {other:?}"),
        }
    }

    #[test]
    fn floor_complex_tensor_preserves_native_single_storage() {
        let input = ComplexTensor::from_complex_storage(
            ComplexStorage::F32(vec![(1.7, -2.3), (-0.1, 4.0)]),
            vec![1, 2],
        )
        .unwrap();
        let Value::ComplexTensor(output) = floor_complex_tensor(input).unwrap() else {
            panic!("expected complex tensor");
        };
        assert_eq!(
            output.into_complex_storage(),
            ComplexStorage::F32(vec![(1.0, -3.0), (-1.0, 4.0)])
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn floor_char_array_to_tensor() {
        let chars = CharArray::new("AB".chars().collect(), 1, 2).unwrap();
        let result = floor_builtin(Value::CharArray(chars), Vec::new()).expect("floor");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 2]);
                assert_eq!(t.materialize_f64(), vec![65.0, 66.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn floor_logical_array_remains_same() {
        let logical = LogicalArray::new(vec![1, 0, 1, 1], vec![2, 2]).unwrap();
        let result = floor_builtin(Value::LogicalArray(logical), Vec::new()).expect("floor");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                assert_eq!(t.materialize_f64(), vec![1.0, 0.0, 1.0, 1.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn floor_int_value_passthrough() {
        let result = floor_builtin(Value::Int(IntValue::I32(-4)), Vec::new()).expect("floor");
        match result {
            Value::Int(IntValue::I32(v)) => assert_eq!(v, -4),
            other => panic!("expected int32 scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn floor_read_typed_integer_storage_exactly() {
        let scalar =
            Tensor::new_integer(IntegerStorage::I64(vec![i64::MAX]), vec![1, 1]).expect("integer");
        assert_eq!(
            floor_builtin(Value::Tensor(scalar), Vec::new()).expect("floor"),
            Value::Int(IntValue::I64(i64::MAX))
        );

        let tensor = Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX, 3]), vec![1, 2])
            .expect("integer");
        match floor_builtin(Value::Tensor(tensor), Vec::new()).expect("floor") {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![1, 2]);
                assert_eq!(
                    out.integer_storage(),
                    Some(&IntegerStorage::U64(vec![u64::MAX, 3]))
                );
            }
            other => panic!("expected typed integer tensor, got {other:?}"),
        }
    }

    #[test]
    fn floor_table_preserves_every_integer_variable_class() {
        let columns = vec![
            Tensor::new_integer(IntegerStorage::I8(vec![-3, 4]), vec![2, 1]).unwrap(),
            Tensor::new_integer(IntegerStorage::I16(vec![-3, 4]), vec![2, 1]).unwrap(),
            Tensor::new_integer(IntegerStorage::I32(vec![-3, 4]), vec![2, 1]).unwrap(),
            Tensor::new_integer(IntegerStorage::I64(vec![i64::MIN, i64::MAX]), vec![2, 1]).unwrap(),
            Tensor::new_integer(IntegerStorage::U8(vec![3, 4]), vec![2, 1]).unwrap(),
            Tensor::new_integer(IntegerStorage::U16(vec![3, 4]), vec![2, 1]).unwrap(),
            Tensor::new_integer(IntegerStorage::U32(vec![3, 4]), vec![2, 1]).unwrap(),
            Tensor::new_integer(IntegerStorage::U64(vec![0, u64::MAX]), vec![2, 1]).unwrap(),
        ];
        let names = (0..columns.len())
            .map(|index| format!("V{index}"))
            .collect::<Vec<_>>();
        let expected = columns.clone();
        let input = crate::builtins::table::table_from_columns(
            names,
            columns.into_iter().map(Value::Tensor).collect(),
        )
        .unwrap();
        let output = block_on(super::floor_builtin(input, Vec::new())).unwrap();
        let Value::Object(output) = output else {
            panic!("expected table output");
        };
        let variables = crate::builtins::table::table_variables(&output).unwrap();
        for (value, expected) in variables.fields.values().zip(expected) {
            assert_eq!(value, &Value::Tensor(expected));
        }
    }

    #[test]
    fn floor_resident_integer_is_an_exact_identity() {
        test_support::with_test_provider(|provider| {
            let input =
                Tensor::new_integer(IntegerStorage::U64(vec![0, u64::MAX]), vec![1, 2]).unwrap();
            let handle = gpu_helpers::upload_tensor(provider, &input).expect("integer upload");
            let buffer_id = handle.buffer_id;
            let Value::GpuTensor(output) = block_on(super::floor_gpu(handle)).unwrap() else {
                panic!("expected resident integer output");
            };
            assert_eq!(output.buffer_id, buffer_id);
            assert_eq!(
                test_support::gather(Value::GpuTensor(output)).unwrap(),
                input
            );
        });
    }

    #[test]
    fn floor_rejects_aliased_or_mistyped_native_outputs_without_freeing_input() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.25], vec![1, 1]).unwrap();
            let input = gpu_helpers::upload_tensor(provider, &tensor).expect("input upload");
            assert!(!rounding_native_output_matches(&input, &input, provider));
            free_rejected_rounding_output(&input, &input, provider);
            assert_eq!(
                test_support::gather(Value::GpuTensor(input.clone()))
                    .expect("input remains live")
                    .materialize_f64(),
                vec![1.25]
            );

            let logical = gpu_helpers::upload_tensor(provider, &tensor).expect("logical upload");
            runmat_accelerate_api::set_handle_logical(&logical, true);
            assert!(!rounding_native_output_matches(&input, &logical, provider));
            free_rejected_rounding_output(&logical, &input, provider);

            let mut integer =
                gpu_helpers::upload_tensor(provider, &tensor).expect("integer upload");
            integer.descriptor.element_type = Some(runmat_accelerate_api::NumericElementType::U8);
            assert!(!rounding_native_output_matches(&input, &integer, provider));
            free_rejected_rounding_output(&integer, &input, provider);
            let _ = provider.free(&input);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn floor_gpu_provider_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![0.2, 1.9, -0.1, -3.8], vec![2, 2]).unwrap();
            let view = HostTensorView {
                data: &tensor.materialize_f64(),
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = floor_builtin(Value::GpuTensor(handle), Vec::new()).expect("floor");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![2, 2]);
            assert_eq!(gathered.materialize_f64(), vec![0.0, 1.0, -1.0, -4.0]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn floor_string_input_errors() {
        let err = floor_builtin(Value::from("hello"), Vec::new()).unwrap_err();
        assert_error_contains(err, "numeric");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn floor_rejects_non_matlab_extra_forms() {
        let digits = floor_builtin(Value::Num(1.2), vec![Value::Num(2.0)]).unwrap_err();
        assert_error_contains(digits, "exactly one input");
        let like =
            floor_builtin(Value::Num(1.2), vec![Value::from("like"), Value::Num(0.0)]).unwrap_err();
        assert_error_contains(like, "exactly one input");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn floor_bool_value() {
        let result = floor_builtin(Value::Bool(true), Vec::new()).expect("floor");
        match result {
            Value::Num(v) => assert_eq!(v, 1.0),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn floor_wgpu_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let t = Tensor::new(vec![0.3, 1.1, -0.2, -1.7], vec![2, 2]).unwrap();
        let cpu = floor_numeric(Value::Tensor(t.clone())).unwrap();
        let view = HostTensorView {
            data: &t.materialize_f64(),
            shape: &t.shape,
        };
        let h = runmat_accelerate_api::provider()
            .unwrap()
            .upload(&view)
            .unwrap();
        let gpu = block_on(floor_gpu(h)).unwrap();
        let gathered = test_support::gather(gpu).expect("gather");
        match (cpu, gathered) {
            (Value::Tensor(ct), gt) => {
                assert_eq!(gt.shape, ct.shape);
                assert_eq!(gt.materialize_f64(), ct.materialize_f64());
            }
            (Value::Num(c), gt) => {
                assert_eq!(gt.materialize_f64(), vec![c]);
            }
            other => panic!("unexpected comparison {other:?}"),
        }
    }
}
