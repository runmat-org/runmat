//! MATLAB-compatible `floor` builtin with GPU-aware semantics for RunMat.

use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    CharArray, ComplexTensor, NumericStorage, Tensor, Value,
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
    let data: Vec<(f64, f64)> = ct
        .data
        .iter()
        .map(|&(re, im)| (apply_floor_scalar(re), apply_floor_scalar(im)))
        .collect();
    let tensor = ComplexTensor::new(data, ct.shape.clone())
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
    if let Some(provider) = runmat_accelerate_api::provider_for_handle(&handle) {
        if let Ok(out) = provider.unary_floor(&handle).await {
            return Ok(Value::GpuTensor(out));
        }
    }
    let tensor = gpu_helpers::gather_tensor_async(&handle).await?;
    let floored = floor_tensor(tensor)?;
    Ok(tensor::tensor_into_value(floored))
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
        IntValue, IntegerStorage, LogicalArray, ResolveContext, Tensor, Type, Value,
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
