//! MATLAB-compatible `conj` builtin with GPU-aware semantics for RunMat.

use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    CharArray, ComplexStorage, ComplexTensor, IntegerComplexStorage, IntegerStorage, Tensor, Value,
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

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::elementwise::conj")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "conj",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Unary { name: "unary_conj" }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "Providers may execute conj via unary_conj for real tensors and complex-interleaved GPU tensors, preserving complex GPU residency when supported.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::elementwise::conj")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "conj",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: |ctx: &FusionExprContext| {
            let input = ctx
                .inputs
                .first()
                .ok_or(FusionError::MissingInput(0))?;
            Ok(format!("({input})"))
        },
    }),
    reduction: None,
    emits_nan: false,
    notes:
        "Fusion kernels treat conj as an identity for real tensors; complex tensors fall back to the CPU path until native complex fusion is available.",
};

const BUILTIN_NAME: &str = "conj";

const CONJ_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Complex conjugate of X.",
}];
const CONJ_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Numeric, logical, char, or complex input.",
}];
const CONJ_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "Y = conj(X)",
    inputs: &CONJ_INPUTS,
    outputs: &CONJ_OUTPUT,
}];
const CONJ_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CONJ.INVALID_INPUT",
    identifier: Some("RunMat:conj:InvalidInput"),
    when: "Input cannot be interpreted as numeric, logical, char, or complex data.",
    message: "conj: invalid input",
};
const CONJ_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CONJ.INTERNAL",
    identifier: Some("RunMat:conj:Internal"),
    when: "Internal tensor conversion/allocation/provider interaction failed.",
    message: "conj: internal error",
};
const CONJ_ERRORS: [BuiltinErrorDescriptor; 2] = [CONJ_ERROR_INVALID_INPUT, CONJ_ERROR_INTERNAL];
pub const CONJ_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &CONJ_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &CONJ_ERRORS,
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
    name = "conj",
    category = "math/elementwise",
    summary = "Compute complex conjugates element-wise.",
    keywords = "conj,complex conjugate,complex,elementwise,gpu",
    accel = "unary",
    type_resolver(numeric_unary_type),
    descriptor(crate::builtins::math::elementwise::conj::CONJ_DESCRIPTOR),
    builtin_path = "crate::builtins::math::elementwise::conj"
)]
async fn conj_builtin(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::GpuTensor(handle) => conj_gpu(handle).await,
        Value::Complex(re, im) => conj_complex_scalar(re, im),
        Value::ComplexTensor(ct) => conj_complex_tensor(ct),
        Value::CharArray(ca) => conj_char_array(ca),
        Value::String(_) | Value::StringArray(_) => Err(builtin_error_with_detail(
            &CONJ_ERROR_INVALID_INPUT,
            "expected numeric input",
        )),
        x @ (Value::Tensor(_)
        | Value::LogicalArray(_)
        | Value::Num(_)
        | Value::Int(_)
        | Value::Bool(_)) => conj_real(x),
        other => Err(builtin_error_with_detail(
            &CONJ_ERROR_INVALID_INPUT,
            format!(
                "unsupported input type {:?}; expected numeric, logical, or char data",
                other
            ),
        )),
    }
}

async fn conj_gpu(handle: GpuTensorHandle) -> BuiltinResult<Value> {
    if let Some(provider) = runmat_accelerate_api::provider_for_handle(&handle) {
        if let Ok(out) = provider.unary_conj(&handle).await {
            return Ok(Value::GpuTensor(out));
        }
    }
    let tensor = gpu_helpers::gather_tensor_async(&handle)
        .await
        .map_err(|err| builtin_error_with_detail(&CONJ_ERROR_INTERNAL, err.to_string()))?;
    Ok(tensor::tensor_into_value(conj_tensor(tensor)?))
}

fn conj_real(value: Value) -> BuiltinResult<Value> {
    let tensor = tensor::value_into_tensor_for("conj", value)
        .map_err(|e| builtin_error_with_detail(&CONJ_ERROR_INVALID_INPUT, e))?;
    Ok(tensor::tensor_into_value(conj_tensor(tensor)?))
}

fn conj_tensor(tensor: Tensor) -> BuiltinResult<Tensor> {
    Ok(tensor)
}

fn conj_complex_scalar(re: f64, im: f64) -> BuiltinResult<Value> {
    Ok(Value::Complex(re, -im))
}

fn conj_complex_tensor(ct: ComplexTensor) -> BuiltinResult<Value> {
    let shape = ct.shape.clone();
    let storage = match ct.into_complex_storage() {
        ComplexStorage::F64(values) => ComplexStorage::F64(
            values
                .into_iter()
                .map(|(real, imag)| (real, -imag))
                .collect(),
        ),
        ComplexStorage::F32(values) => ComplexStorage::F32(
            values
                .into_iter()
                .map(|(real, imag)| (real, -imag))
                .collect(),
        ),
        ComplexStorage::Integer(storage) => ComplexStorage::Integer(
            IntegerComplexStorage::new(
                storage.real,
                conjugate_integer_imaginary_storage(storage.imag),
            )
            .map_err(|e| builtin_error_with_detail(&CONJ_ERROR_INTERNAL, e))?,
        ),
    };
    let tensor = ComplexTensor::from_complex_storage(storage, shape)
        .map_err(|e| builtin_error_with_detail(&CONJ_ERROR_INTERNAL, e))?;
    Ok(Value::ComplexTensor(tensor))
}

pub(crate) fn conjugate_integer_imaginary_storage(storage: IntegerStorage) -> IntegerStorage {
    match storage {
        IntegerStorage::I8(values) => {
            IntegerStorage::I8(values.into_iter().map(i8::saturating_neg).collect())
        }
        IntegerStorage::I16(values) => {
            IntegerStorage::I16(values.into_iter().map(i16::saturating_neg).collect())
        }
        IntegerStorage::I32(values) => {
            IntegerStorage::I32(values.into_iter().map(i32::saturating_neg).collect())
        }
        IntegerStorage::I64(values) => {
            IntegerStorage::I64(values.into_iter().map(i64::saturating_neg).collect())
        }
        IntegerStorage::U8(values) => IntegerStorage::U8(vec![0; values.len()]),
        IntegerStorage::U16(values) => IntegerStorage::U16(vec![0; values.len()]),
        IntegerStorage::U32(values) => IntegerStorage::U32(vec![0; values.len()]),
        IntegerStorage::U64(values) => IntegerStorage::U64(vec![0; values.len()]),
    }
}

fn conj_char_array(ca: CharArray) -> BuiltinResult<Value> {
    let data = ca
        .data
        .iter()
        .map(|&ch| ch as u32 as f64)
        .collect::<Vec<_>>();
    let tensor = Tensor::new(data, vec![ca.rows, ca.cols])
        .map_err(|e| builtin_error_with_detail(&CONJ_ERROR_INTERNAL, e))?;
    Ok(tensor::tensor_into_value(tensor))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;

    #[cfg(feature = "wgpu")]
    fn register_wgpu_provider_available() -> bool {
        runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        )
        .is_ok()
            && runmat_accelerate_api::provider().is_some()
    }
    use runmat_builtins::{IntValue, LogicalArray, ResolveContext, Type};

    fn conj_builtin(value: Value) -> BuiltinResult<Value> {
        block_on(super::conj_builtin(value))
    }

    #[test]
    fn conj_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = CONJ_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"Y = conj(X)"));
    }

    #[test]
    fn conj_type_preserves_tensor_shape() {
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
    fn conj_type_scalar_tensor_returns_num() {
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
    fn conj_scalar_real() {
        let result = conj_builtin(Value::Num(-2.5)).expect("conj");
        match result {
            Value::Num(n) => assert!((n + 2.5).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn conj_complex_scalar() {
        let result = conj_builtin(Value::Complex(3.0, 4.0)).expect("conj");
        match result {
            Value::Complex(re, im) => {
                assert!((re - 3.0).abs() < 1e-12);
                assert!((im + 4.0).abs() < 1e-12);
            }
            other => panic!("expected complex result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn conj_complex_scalar_zero_imag_remains_complex() {
        let result = conj_builtin(Value::Complex(5.0, 0.0)).expect("conj");
        match result {
            Value::Complex(re, im) => {
                assert!((re - 5.0).abs() < 1e-12);
                assert_eq!(im, -0.0);
            }
            other => panic!("expected complex scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn conj_promotes_logical_to_double() {
        let logical =
            LogicalArray::new(vec![0, 1, 1, 0], vec![2, 2]).expect("logical array construction");
        let result = conj_builtin(Value::LogicalArray(logical)).expect("conj");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                assert_eq!(t.materialize_f64(), vec![0.0, 1.0, 1.0, 0.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn conj_int_preserves_integer_class() {
        let result = conj_builtin(Value::Int(IntValue::I32(7))).expect("conj");
        match result {
            Value::Int(IntValue::I32(n)) => assert_eq!(n, 7),
            other => panic!("expected int32 scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn conj_complex_tensor_to_complex_tensor() {
        let tensor =
            ComplexTensor::new(vec![(1.0, 2.0), (-3.0, -4.0)], vec![2, 1]).expect("complex tensor");
        let result = conj_builtin(Value::ComplexTensor(tensor)).expect("conj");
        match result {
            Value::ComplexTensor(ct) => {
                assert_eq!(ct.shape, vec![2, 1]);
                assert_eq!(ct.materialize_f64()[0], (1.0, -2.0));
                assert_eq!(ct.materialize_f64()[1], (-3.0, 4.0));
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn conj_complex_tensor_zero_imag_remains_complex() {
        let tensor =
            ComplexTensor::new(vec![(1.0, 0.0), (2.0, -0.0)], vec![2, 1]).expect("complex tensor");
        let result = conj_builtin(Value::ComplexTensor(tensor)).expect("conj");
        match result {
            Value::ComplexTensor(t) => {
                assert_eq!(t.shape, vec![2, 1]);
                assert_eq!(t.materialize_f64(), vec![(1.0, -0.0), (2.0, 0.0)]);
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[test]
    fn conj_complex_single_preserves_native_class_shape_and_empty_storage() {
        let tensor = ComplexTensor::from_f32(vec![(1.25, 2.5), (-3.0, -4.0)], vec![1, 2]).unwrap();
        let Value::ComplexTensor(output) =
            conj_builtin(Value::ComplexTensor(tensor)).expect("conj")
        else {
            panic!("expected complex single tensor");
        };
        assert_eq!(output.shape, vec![1, 2]);
        assert_eq!(
            output.as_f32_slice(),
            Some(&[(1.25, -2.5), (-3.0, 4.0)][..])
        );

        let empty = ComplexTensor::from_f32(Vec::new(), vec![0, 2]).unwrap();
        let Value::ComplexTensor(output) = conj_builtin(Value::ComplexTensor(empty)).expect("conj")
        else {
            panic!("expected empty complex single tensor");
        };
        assert_eq!(output.shape, vec![0, 2]);
        assert_eq!(output.as_f32_slice(), Some(&[][..]));
    }

    #[test]
    fn conj_typed_integer_imaginary_storage_preserves_class_and_saturates() {
        let cases = [
            (
                IntegerStorage::I8(vec![3, i8::MIN]),
                IntegerStorage::I8(vec![-3, i8::MAX]),
            ),
            (
                IntegerStorage::I16(vec![3, i16::MIN]),
                IntegerStorage::I16(vec![-3, i16::MAX]),
            ),
            (
                IntegerStorage::I32(vec![3, i32::MIN]),
                IntegerStorage::I32(vec![-3, i32::MAX]),
            ),
            (
                IntegerStorage::I64(vec![3, i64::MIN]),
                IntegerStorage::I64(vec![-3, i64::MAX]),
            ),
            (
                IntegerStorage::U8(vec![3, u8::MAX]),
                IntegerStorage::U8(vec![0, 0]),
            ),
            (
                IntegerStorage::U16(vec![3, u16::MAX]),
                IntegerStorage::U16(vec![0, 0]),
            ),
            (
                IntegerStorage::U32(vec![3, u32::MAX]),
                IntegerStorage::U32(vec![0, 0]),
            ),
            (
                IntegerStorage::U64(vec![3, u64::MAX]),
                IntegerStorage::U64(vec![0, 0]),
            ),
        ];
        for (input, expected) in cases {
            assert_eq!(conjugate_integer_imaginary_storage(input), expected);
        }
    }

    #[test]
    fn conj_complex_integer_tensor_reads_storage_without_mirror() {
        let complex = ComplexTensor::new_integer(
            IntegerComplexStorage::new(
                IntegerStorage::I16(vec![-10, 20]),
                IntegerStorage::I16(vec![3, i16::MIN]),
            )
            .unwrap(),
            vec![1, 2],
        )
        .unwrap();

        let result = conj_builtin(Value::ComplexTensor(complex)).expect("conj");
        let Value::ComplexTensor(output) = result else {
            panic!("expected typed complex integer tensor");
        };
        assert_eq!(output.shape, vec![1, 2]);
        assert_eq!(
            output.integer_storage().cloned(),
            Some(
                IntegerComplexStorage::new(
                    IntegerStorage::I16(vec![-10, 20]),
                    IntegerStorage::I16(vec![-3, i16::MAX]),
                )
                .unwrap()
            )
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn conj_char_array_returns_double_codes() {
        let chars = CharArray::new("Hi".chars().collect(), 1, 2).expect("char array");
        let result = conj_builtin(Value::CharArray(chars)).expect("conj");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 2]);
                assert_eq!(t.materialize_f64(), vec![72.0, 105.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn conj_errors_on_string_input() {
        let err = conj_builtin(Value::from("hello")).unwrap_err();
        let identifier = err.identifier().map(str::to_string);
        assert!(err.message().contains("expected numeric input"));
        assert_eq!(identifier.as_deref(), CONJ_ERROR_INVALID_INPUT.identifier);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn conj_gpu_provider_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![0.0, 1.0, -3.0, 4.0], vec![4, 1]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.materialize_f64(),
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = conj_builtin(Value::GpuTensor(handle)).expect("conj");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![4, 1]);
            assert_eq!(gathered.materialize_f64(), tensor.materialize_f64());
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn conj_complex_gpu_provider_stays_resident() {
        test_support::with_test_provider(|provider| {
            let complex = ComplexTensor::new(vec![(1.0, 2.0), (-3.0, -4.0)], vec![2, 1]).unwrap();
            let handle = gpu_helpers::upload_complex_tensor(provider, &complex).expect("upload");
            let result = conj_builtin(Value::GpuTensor(handle)).expect("conj");
            let Value::GpuTensor(out) = result else {
                panic!("expected gpu tensor");
            };
            assert_eq!(
                runmat_accelerate_api::handle_storage(&out),
                runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
            );
            let gathered =
                block_on(gpu_helpers::gather_value_async(&Value::GpuTensor(out))).expect("gather");
            let Value::ComplexTensor(ct) = gathered else {
                panic!("expected complex tensor");
            };
            assert_eq!(ct.shape, vec![2, 1]);
            assert_eq!(ct.materialize_f64(), vec![(1.0, -2.0), (-3.0, 4.0)]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn conj_wgpu_matches_cpu_for_real() {
        let _guard = test_support::accel_test_lock();
        if !register_wgpu_provider_available() {
            return;
        }
        let tensor = Tensor::new(vec![1.0, -2.0, 3.5, 0.0], vec![4, 1]).unwrap();
        let cpu = conj_real(Value::Tensor(tensor.clone())).unwrap();
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.materialize_f64(),
            shape: &tensor.shape,
        };
        let handle = runmat_accelerate_api::provider()
            .unwrap()
            .upload(&view)
            .unwrap();
        let gpu = block_on(conj_gpu(handle)).unwrap();
        let gathered = test_support::gather(gpu).expect("gather");
        match (cpu, gathered) {
            (Value::Tensor(ct), gt) => {
                assert_eq!(ct.shape, gt.shape);
                assert_eq!(ct.materialize_f64(), gt.materialize_f64());
            }
            _ => panic!("unexpected shapes"),
        }
    }

    #[cfg(feature = "wgpu")]
    #[test]
    fn conj_wgpu_complex_matches_cpu() {
        let _guard = test_support::accel_test_lock();
        if !register_wgpu_provider_available() {
            return;
        }
        let provider = runmat_accelerate_api::provider().expect("wgpu provider");
        let complex = ComplexTensor::new(vec![(1.0, 2.0), (-3.0, -4.0)], vec![2, 1]).unwrap();
        let handle = gpu_helpers::upload_complex_tensor(provider, &complex).expect("upload");
        let gpu = block_on(conj_gpu(handle)).unwrap();
        let Value::GpuTensor(out) = gpu else {
            panic!("expected gpu tensor");
        };
        assert_eq!(
            runmat_accelerate_api::handle_storage(&out),
            runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
        );
        let gathered =
            block_on(gpu_helpers::gather_value_async(&Value::GpuTensor(out))).expect("gather");
        let Value::ComplexTensor(ct) = gathered else {
            panic!("expected complex tensor");
        };
        assert_eq!(ct.materialize_f64(), vec![(1.0, -2.0), (-3.0, 4.0)]);
    }
}
