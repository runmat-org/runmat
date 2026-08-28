//! MATLAB-compatible `angle` builtin with GPU-aware semantics for RunMat.

use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinIntegerBackendRule,
    BuiltinIntegerCapabilityDescriptor, BuiltinIntegerComputationDomain,
    BuiltinIntegerInputAvailability, BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule,
    BuiltinIntegerOverflowRule, BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule,
    BuiltinOutputMode, BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType,
    BuiltinSignatureDescriptor, ResolveContext, Type,
};
use runmat_macros::runtime_builtin;
use runmat_value::{ComplexStorage, ComplexTensor, NumericStorage, Tensor, Value};

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, FusionError,
    FusionExprContext, FusionKernelTemplate, GpuOpKind, ProviderHook, ReductionNaN,
    ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::builtins::math::type_resolvers::numeric_unary_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::elementwise::angle")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "angle",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Unary { name: "unary_angle" }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Providers implement unary_angle to evaluate atan2(imag(x), real(x)) on device for real and complex-interleaved gpuArrays; the runtime gathers to host only when the hook is unavailable or host-only conversion is required.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::elementwise::angle")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "angle",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: |ctx: &FusionExprContext| {
            let input = ctx
                .inputs
                .first()
                .ok_or(FusionError::MissingInput(0))?;
            let zero = match ctx.scalar_ty {
                ScalarType::F32 => "0.0".to_string(),
                ScalarType::F64 => "f64(0.0)".to_string(),
                other => return Err(FusionError::UnsupportedPrecision(other)),
            };
            Ok(format!("atan2({zero}, {input})"))
        },
    }),
    reduction: None,
    emits_nan: false,
    notes: "Fusion assumes real-valued inputs (imaginary part zero). Complex-interleaved gpuArrays use the provider unary_angle hook outside generic real-valued fusion.",
};

const BUILTIN_NAME: &str = "angle";

const ANGLE_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "theta",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Phase angle in radians.",
}];
const ANGLE_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Real or complex single- or double-precision input.",
}];
const ANGLE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "theta = angle(X)",
    inputs: &ANGLE_INPUTS,
    outputs: &ANGLE_OUTPUT,
}];
const ANGLE_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ANGLE.INVALID_INPUT",
    identifier: Some("RunMat:angle:InvalidInput"),
    when: "Input is not real or complex single- or double-precision data.",
    message: "angle: invalid input",
};
const ANGLE_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ANGLE.INTERNAL",
    identifier: Some("RunMat:angle:Internal"),
    when: "Internal tensor conversion/allocation/provider interaction failed.",
    message: "angle: internal error",
};
const ANGLE_ERRORS: [BuiltinErrorDescriptor; 2] = [ANGLE_ERROR_INVALID_INPUT, ANGLE_ERROR_INTERNAL];

const ANGLE_REJECTED_INTEGER_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "X",
        classes: &[],
        availability: BuiltinIntegerInputAvailability::Rejected,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The documented input domain is real or complex single/double; every real or componentwise-complex typed-integer class is rejected before host or provider computation.",
    }];

pub const INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "theta = angle(integer_X)",
        inputs: &ANGLE_REJECTED_INTEGER_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
        notes: "angle has no integer overload. Host scalars, dense arrays, typed complex integer arrays, and resident integer handles reject with the same public invalid-input category without floating materialization.",
    }];

pub const ANGLE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ANGLE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ANGLE_ERRORS,
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
    name = "angle",
    category = "math/elementwise",
    summary = "Phase angle (argument) of real and complex values.",
    keywords = "angle,phase,argument,complex,gpu",
    accel = "unary",
    type_resolver(angle_type),
    descriptor(crate::builtins::math::elementwise::angle::ANGLE_DESCRIPTOR),
    integer_capabilities(crate::builtins::math::elementwise::angle::INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::math::elementwise::angle"
)]
async fn angle_builtin(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::GpuTensor(handle) => angle_gpu(handle).await,
        Value::Complex(re, im) => Ok(Value::Num(angle_scalar(re, im))),
        Value::ComplexTensor(ct) => {
            if ct.integer_storage().is_some() {
                return Err(builtin_error_with_detail(
                    &ANGLE_ERROR_INVALID_INPUT,
                    "expected single or double input",
                ));
            }
            angle_complex_tensor(ct)
        }
        Value::Int(_)
        | Value::Bool(_)
        | Value::LogicalArray(_)
        | Value::CharArray(_)
        | Value::String(_)
        | Value::StringArray(_) => Err(builtin_error_with_detail(
            &ANGLE_ERROR_INVALID_INPUT,
            "expected single or double input",
        )),
        other => angle_real(other),
    }
}

fn angle_type(args: &[Type], context: &ResolveContext) -> Type {
    match args.first() {
        Some(Type::Int | Type::Bool | Type::Logical { .. }) => Type::Unknown,
        _ => numeric_unary_type(args, context),
    }
}

async fn angle_gpu(handle: GpuTensorHandle) -> BuiltinResult<Value> {
    if runmat_accelerate_api::handle_integer_type(&handle).is_some() {
        return Err(builtin_error_with_detail(
            &ANGLE_ERROR_INVALID_INPUT,
            "integer gpuArray input is not supported",
        ));
    }
    if let Some(provider) =
        runmat_accelerate_api::provider_for_handle(&handle).or_else(runmat_accelerate_api::provider)
    {
        if let Ok(device_result) = provider.unary_angle(&handle).await {
            return Ok(Value::GpuTensor(device_result));
        }
    }
    let gathered = gpu_helpers::gather_value_async(&Value::GpuTensor(handle))
        .await
        .map_err(|err| builtin_error_with_detail(&ANGLE_ERROR_INTERNAL, err.to_string()))?;
    match gathered {
        Value::Complex(re, im) => Ok(Value::Num(angle_scalar(re, im))),
        Value::ComplexTensor(ct) => angle_complex_tensor(ct),
        other => angle_real(other),
    }
}

fn angle_real(value: Value) -> BuiltinResult<Value> {
    let tensor = tensor::value_into_tensor_for("angle", value)
        .map_err(|e| builtin_error_with_detail(&ANGLE_ERROR_INVALID_INPUT, e))?;
    Ok(tensor::tensor_into_value(angle_tensor(tensor)?))
}

fn angle_tensor(tensor: Tensor) -> BuiltinResult<Tensor> {
    let shape = tensor.shape.clone();
    let storage = tensor
        .into_numeric_storage()
        .map_err(|error| builtin_error_with_detail(&ANGLE_ERROR_INTERNAL, error))?;
    let mapped = match storage {
        NumericStorage::F64(values) => {
            NumericStorage::F64(values.into_iter().map(|re| angle_scalar(re, 0.0)).collect())
        }
        NumericStorage::F32(values) => {
            NumericStorage::F32(values.into_iter().map(|re| 0.0_f32.atan2(re)).collect())
        }
        _ => {
            return Err(builtin_error_with_detail(
                &ANGLE_ERROR_INVALID_INPUT,
                "expected single or double input",
            ))
        }
    };
    Tensor::from_numeric_storage(mapped, shape)
        .map_err(|e| builtin_error_with_detail(&ANGLE_ERROR_INTERNAL, e))
}

fn angle_complex_tensor(ct: ComplexTensor) -> BuiltinResult<Value> {
    let shape = ct.shape.clone();
    let storage = match ct.into_complex_storage() {
        ComplexStorage::F64(values) => NumericStorage::F64(
            values
                .into_iter()
                .map(|(real, imag)| angle_scalar(real, imag))
                .collect(),
        ),
        ComplexStorage::F32(values) => NumericStorage::F32(
            values
                .into_iter()
                .map(|(real, imag)| imag.atan2(real))
                .collect(),
        ),
        ComplexStorage::Integer(_) => {
            return Err(builtin_error_with_detail(
                &ANGLE_ERROR_INVALID_INPUT,
                "expected single or double input",
            ))
        }
    };
    let tensor = Tensor::from_numeric_storage(storage, shape)
        .map_err(|e| builtin_error_with_detail(&ANGLE_ERROR_INTERNAL, e))?;
    Ok(tensor::tensor_into_value(tensor))
}

#[inline]
fn angle_scalar(re: f64, im: f64) -> f64 {
    im.atan2(re)
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
    use runmat_value::{
        CharArray, IntegerComplexStorage, IntegerStorage, LogicalArray, StringArray,
    };
    use std::f64::consts::PI;

    fn angle_builtin(value: Value) -> BuiltinResult<Value> {
        block_on(super::angle_builtin(value))
    }

    fn all_integer_storages() -> [IntegerStorage; 8] {
        [
            IntegerStorage::I8(vec![i8::MIN, i8::MAX]),
            IntegerStorage::I16(vec![i16::MIN, i16::MAX]),
            IntegerStorage::I32(vec![i32::MIN, i32::MAX]),
            IntegerStorage::I64(vec![i64::MIN, i64::MAX]),
            IntegerStorage::U8(vec![0, u8::MAX]),
            IntegerStorage::U16(vec![0, u16::MAX]),
            IntegerStorage::U32(vec![0, u32::MAX]),
            IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX]),
        ]
    }

    #[test]
    fn angle_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = ANGLE_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"theta = angle(X)"));
        assert_eq!(INTEGER_CAPABILITIES.len(), 1);
        assert_eq!(
            INTEGER_CAPABILITIES[0].inputs[0].availability,
            BuiltinIntegerInputAvailability::Rejected
        );
        assert!(INTEGER_CAPABILITIES[0].inputs[0].classes.is_empty());
    }

    #[test]
    fn angle_type_preserves_tensor_shape() {
        let out = angle_type(
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
    fn angle_type_scalar_tensor_returns_num() {
        let out = angle_type(
            &[Type::Tensor {
                shape: Some(vec![Some(1), Some(1)]),
            }],
            &ResolveContext::new(Vec::new()),
        );
        assert_eq!(out, Type::Num);
    }

    #[test]
    fn angle_type_rejects_known_integer_and_logical_inputs() {
        let context = ResolveContext::new(Vec::new());
        assert_eq!(angle_type(&[Type::Int], &context), Type::Unknown);
        assert_eq!(angle_type(&[Type::Bool], &context), Type::Unknown);
        assert_eq!(
            angle_type(
                &[Type::Logical {
                    shape: Some(vec![Some(2), Some(3)]),
                }],
                &context,
            ),
            Type::Unknown
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn angle_real_positive_negative() {
        let pos = angle_builtin(Value::Num(5.0)).expect("angle");
        assert_eq!(pos, Value::Num(0.0));

        let neg = angle_builtin(Value::Num(-3.0)).expect("angle");
        if let Value::Num(val) = neg {
            assert!((val - PI).abs() < 1e-12);
        } else {
            panic!("expected numeric result, got {neg:?}");
        }

        let zero = angle_builtin(Value::Num(0.0)).expect("angle");
        assert_eq!(zero, Value::Num(0.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn angle_complex_scalar_matches_atan2() {
        let value = Value::Complex(3.0, -4.0);
        let result = angle_builtin(value).expect("angle");
        if let Value::Num(angle) = result {
            assert!((angle - (-4.0f64).atan2(3.0)).abs() < 1e-12);
        } else {
            panic!("expected numeric result");
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn angle_tensor_values() {
        let tensor = Tensor::new(vec![1.0, -1.0, 0.0, 2.0], vec![2, 2]).unwrap();
        let result = angle_builtin(Value::Tensor(tensor)).expect("angle");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![2, 2]);
                assert!((out.materialize_f64()[0] - 0.0).abs() < 1e-12);
                assert!((out.materialize_f64()[1] - PI).abs() < 1e-12);
                assert_eq!(out.materialize_f64()[2], 0.0);
                assert_eq!(out.materialize_f64()[3], 0.0);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn angle_preserves_native_single_storage() {
        let tensor = Tensor::from_f32(vec![1.0, -1.0, 0.0], vec![1, 3]).unwrap();
        let result = angle_builtin(Value::Tensor(tensor)).expect("angle");
        match result {
            Value::Tensor(out) => match out.into_numeric_storage().expect("single storage") {
                NumericStorage::F32(values) => {
                    assert_eq!(values[0], 0.0);
                    assert!((values[1] - std::f32::consts::PI).abs() <= 2.0 * f32::EPSILON);
                    assert_eq!(values[2], 0.0);
                }
                storage => panic!("expected single storage, got {storage:?}"),
            },
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn angle_rejects_every_real_integer_scalar_and_tensor_class() {
        for storage in all_integer_storages() {
            let class = storage.class_name();
            let scalar = storage.value_at(1).expect("integer scalar");
            let scalar_error =
                angle_builtin(Value::Int(scalar)).expect_err("integer scalar must reject");
            assert_eq!(
                scalar_error.identifier(),
                ANGLE_ERROR_INVALID_INPUT.identifier,
                "{class} scalar"
            );
            let tensor = Tensor::new_integer(storage, vec![1, 2]).expect("integer tensor");
            let tensor_error =
                angle_builtin(Value::Tensor(tensor)).expect_err("integer tensor must reject");
            assert_eq!(
                tensor_error.identifier(),
                ANGLE_ERROR_INVALID_INPUT.identifier,
                "{class} tensor"
            );
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn angle_rejects_integer_logical_and_char_inputs() {
        let integer = angle_builtin(Value::Int(runmat_value::IntValue::I32(-1))).unwrap_err();
        assert_eq!(integer.identifier(), ANGLE_ERROR_INVALID_INPUT.identifier);

        let logical = LogicalArray::new(vec![0, 1, 0, 1], vec![2, 2]).unwrap();
        let logical = angle_builtin(Value::LogicalArray(logical)).unwrap_err();
        assert_eq!(logical.identifier(), ANGLE_ERROR_INVALID_INPUT.identifier);

        let chars = CharArray::new("AB".chars().collect(), 1, 2).unwrap();
        let chars = angle_builtin(Value::CharArray(chars)).unwrap_err();
        assert_eq!(chars.identifier(), ANGLE_ERROR_INVALID_INPUT.identifier);
    }

    #[test]
    fn angle_rejects_every_typed_complex_integer_class() {
        for real in all_integer_storages() {
            let class = real.class_name();
            let imaginary = real.ones_like(real.len());
            let storage = IntegerComplexStorage::new(real, imaginary).expect("complex storage");
            let tensor = ComplexTensor::new_integer(storage, vec![1, 2]).expect("complex tensor");
            let error = angle_builtin(Value::ComplexTensor(tensor))
                .expect_err("complex integer must reject");
            assert_eq!(
                error.identifier(),
                ANGLE_ERROR_INVALID_INPUT.identifier,
                "{class} complex tensor"
            );
            assert!(error.message().contains("expected single or double"));
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn angle_complex_tensor() {
        let data = vec![(1.0, 1.0), (-1.0, 1.0), (-1.0, -1.0), (1.0, -1.0)];
        let tensor = ComplexTensor::new(data, vec![2, 2]).unwrap();
        let result = angle_builtin(Value::ComplexTensor(tensor)).expect("angle");
        match result {
            Value::Tensor(out) => {
                let expected = [
                    (1.0f64).atan2(1.0),
                    (1.0f64).atan2(-1.0),
                    (-1.0f64).atan2(-1.0),
                    (-1.0f64).atan2(1.0),
                ];
                for (actual, target) in out.materialize_f64().iter().zip(expected.iter()) {
                    assert!((actual - target).abs() < 1e-12);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn angle_gpu_provider_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, -1.0, 0.5, -0.5], vec![2, 2]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.materialize_f64(),
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = angle_builtin(Value::GpuTensor(handle)).expect("angle");
            let gathered = test_support::gather(result).expect("gather");
            let expected: Vec<f64> = tensor
                .materialize_f64()
                .iter()
                .map(|&v| angle_scalar(v, 0.0))
                .collect();
            assert_eq!(gathered.shape, vec![2, 2]);
            for (actual, target) in gathered.materialize_f64().iter().zip(expected.iter()) {
                assert!((actual - target).abs() < 1e-12);
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn angle_complex_gpu_provider_roundtrip() {
        test_support::with_test_provider(|provider| {
            let complex = ComplexTensor::new(
                vec![(1.0, 1.0), (-1.0, 1.0), (-1.0, -1.0), (1.0, -1.0)],
                vec![2, 2],
            )
            .unwrap();
            let handle =
                gpu_helpers::upload_complex_tensor(provider, &complex).expect("upload complex");
            let result = angle_builtin(Value::GpuTensor(handle)).expect("angle");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, complex.shape);
            for (actual, (re, im)) in gathered
                .materialize_f64()
                .iter()
                .zip(complex.materialize_f64().iter())
            {
                assert!((actual - im.atan2(*re)).abs() < 1e-12);
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn angle_rejects_all_native_integer_gpu_classes() {
        test_support::with_test_provider(|provider| {
            for storage in all_integer_storages() {
                let class = storage.class_name();
                let tensor = Tensor::new_integer(storage, vec![1, 2]).expect("integer tensor");
                let handle =
                    gpu_helpers::upload_tensor(provider, &tensor).expect("upload integer tensor");
                let error = angle_builtin(Value::GpuTensor(handle))
                    .expect_err("integer gpuArray must reject");
                assert_eq!(
                    error.identifier(),
                    ANGLE_ERROR_INVALID_INPUT.identifier,
                    "{class} gpuArray"
                );
                assert!(error.message().contains("integer gpuArray"));
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn angle_wgpu_rejects_all_native_integer_classes_before_float_dispatch() {
        let _guard = test_support::accel_test_lock();
        if !register_wgpu_provider_available() {
            return;
        }
        let provider = runmat_accelerate_api::provider().expect("wgpu provider");
        for storage in all_integer_storages() {
            let class = storage.class_name();
            let tensor = Tensor::new_integer(storage, vec![1, 2]).expect("integer tensor");
            let handle =
                gpu_helpers::upload_tensor(provider, &tensor).expect("upload integer tensor");
            let error =
                angle_builtin(Value::GpuTensor(handle)).expect_err("integer WGPU input rejects");
            assert_eq!(
                error.identifier(),
                ANGLE_ERROR_INVALID_INPUT.identifier,
                "{class} WGPU"
            );
            assert!(error.message().contains("integer gpuArray"));
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn angle_nan_propagates() {
        let result = angle_builtin(Value::Num(f64::NAN)).expect("angle");
        match result {
            Value::Num(v) => assert!(v.is_nan()),
            other => panic!("expected numeric result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn angle_rejects_strings() {
        let err = angle_builtin(Value::from("hello")).unwrap_err();
        let identifier = err.identifier().map(str::to_string);
        assert!(err.message().contains("expected single or double input"));
        assert_eq!(identifier.as_deref(), ANGLE_ERROR_INVALID_INPUT.identifier);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn angle_rejects_string_arrays() {
        let array = StringArray::new(vec!["a".to_string(), "b".to_string()], vec![1, 2]).unwrap();
        let err = angle_builtin(Value::StringArray(array)).unwrap_err();
        let identifier = err.identifier().map(str::to_string);
        assert!(err.message().contains("expected single or double input"));
        assert_eq!(identifier.as_deref(), ANGLE_ERROR_INVALID_INPUT.identifier);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn angle_wgpu_matches_cpu() {
        let _guard = test_support::accel_test_lock();
        if !register_wgpu_provider_available() {
            return;
        }
        let tensor = Tensor::new(vec![1.0, -1.0, 0.5, -0.5], vec![2, 2]).unwrap();
        let cpu = angle_tensor(tensor.clone()).unwrap();
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.materialize_f64(),
            shape: &tensor.shape,
        };
        let handle = runmat_accelerate_api::provider()
            .unwrap()
            .upload(&view)
            .unwrap();
        let gpu = block_on(angle_gpu(handle)).unwrap();
        let gathered = test_support::gather(gpu).expect("gather");
        match (Value::Tensor(cpu), gathered) {
            (Value::Tensor(ct), gt) => {
                assert_eq!(gt.shape, ct.shape);
                let tol = match runmat_accelerate_api::provider().unwrap().precision() {
                    runmat_accelerate_api::ProviderPrecision::F64 => 1e-12,
                    runmat_accelerate_api::ProviderPrecision::F32 => 1e-5,
                };
                for (a, b) in gt.materialize_f64().iter().zip(ct.materialize_f64().iter()) {
                    assert!((a - b).abs() < tol);
                }
            }
            _ => panic!("unexpected shapes"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn angle_wgpu_complex_matches_cpu() {
        let _guard = test_support::accel_test_lock();
        if !register_wgpu_provider_available() {
            return;
        }
        let provider = runmat_accelerate_api::provider().expect("wgpu provider");
        let complex = ComplexTensor::new(
            vec![(3.0, 4.0), (-2.0, 5.0), (-1.5, -0.5), (2.5, -6.0)],
            vec![2, 2],
        )
        .unwrap();
        let handle = gpu_helpers::upload_complex_tensor(provider, &complex).expect("upload");
        let result = block_on(angle_gpu(handle)).unwrap();
        let gathered = test_support::gather(result).expect("gather");
        assert_eq!(gathered.shape, complex.shape);
        let tol = match provider.precision() {
            runmat_accelerate_api::ProviderPrecision::F64 => 1e-12,
            runmat_accelerate_api::ProviderPrecision::F32 => 1e-5,
        };
        for (actual, (re, im)) in gathered
            .materialize_f64()
            .iter()
            .zip(complex.materialize_f64().iter())
        {
            assert!((actual - im.atan2(*re)).abs() < tol);
        }
    }
}
