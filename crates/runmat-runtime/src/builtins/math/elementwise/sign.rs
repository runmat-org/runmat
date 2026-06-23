//! MATLAB-compatible `sign` builtin with GPU-aware semantics for RunMat.

use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    CharArray, ComplexTensor, Tensor, Value,
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

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::elementwise::sign")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "sign",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Unary { name: "unary_sign" }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "Providers may execute sign on-device via unary_sign for real and complex-interleaved gpuArrays; the runtime gathers to the host when the hook is unavailable.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::elementwise::sign")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "sign",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: |ctx: &FusionExprContext| {
            let input = ctx
                .inputs
                .first()
                .ok_or(FusionError::MissingInput(0))?;
            Ok(format!("sign({input})"))
        },
    }),
    reduction: None,
    emits_nan: true,
    notes: "Fusion kernels emit WGSL `sign` ops; providers can override via fused pipelines when advantageous.",
};

const BUILTIN_NAME: &str = "sign";

const SIGN_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Elementwise sign result.",
}];

const SIGN_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Real or complex numeric input.",
}];

const SIGN_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "Y = sign(X)",
    inputs: &SIGN_INPUTS,
    outputs: &SIGN_OUTPUT,
}];

const SIGN_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SIGN.INVALID_INPUT",
    identifier: Some("RunMat:sign:InvalidInput"),
    when: "Input is not a supported numeric/logical/character value.",
    message: "sign: invalid input",
};

const SIGN_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SIGN.INTERNAL",
    identifier: Some("RunMat:sign:Internal"),
    when: "Internal gather/provider/tensor construction failed.",
    message: "sign: internal error",
};

const SIGN_ERRORS: [BuiltinErrorDescriptor; 2] = [SIGN_ERROR_INVALID_INPUT, SIGN_ERROR_INTERNAL];

pub const SIGN_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGN_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &SIGN_ERRORS,
};

fn sign_error_with_detail(
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

#[runtime_builtin(
    name = "sign",
    category = "math/elementwise",
    summary = "Compute element-wise sign values for real or complex inputs.",
    keywords = "sign,signum,elementwise,complex,gpu",
    accel = "unary",
    type_resolver(numeric_unary_type),
    descriptor(crate::builtins::math::elementwise::sign::SIGN_DESCRIPTOR),
    builtin_path = "crate::builtins::math::elementwise::sign"
)]
async fn sign_builtin(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::GpuTensor(handle) => sign_gpu(handle).await,
        Value::Complex(re, im) => {
            let (re_out, im_out) = sign_complex(re, im);
            Ok(Value::Complex(re_out, im_out))
        }
        Value::ComplexTensor(ct) => sign_complex_tensor(ct),
        Value::CharArray(ca) => sign_char_array(ca),
        Value::String(_) | Value::StringArray(_) => Err(sign_error_with_detail(
            &SIGN_ERROR_INVALID_INPUT,
            "expected numeric, logical, or character input",
        )),
        other => sign_real(other),
    }
}

async fn sign_gpu(handle: GpuTensorHandle) -> BuiltinResult<Value> {
    if runmat_accelerate_api::handle_storage(&handle)
        == runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
    {
        let Some(provider) = runmat_accelerate_api::provider_for_handle(&handle) else {
            return Err(sign_error_with_detail(
                &SIGN_ERROR_INTERNAL,
                "GPU provider unavailable for complex input",
            ));
        };
        let gathered = gpu_helpers::gather_value_async(&Value::GpuTensor(handle))
            .await
            .map_err(|err| sign_error_with_detail(&SIGN_ERROR_INTERNAL, err))?;
        let Value::ComplexTensor(tensor) = sign_complex_tensor(match gathered {
            Value::ComplexTensor(tensor) => tensor,
            other => {
                return Err(sign_error_with_detail(
                    &SIGN_ERROR_INTERNAL,
                    format!("expected complex GPU input, got {other:?}"),
                ));
            }
        })?
        else {
            return Err(sign_error_with_detail(
                &SIGN_ERROR_INTERNAL,
                "complex sign returned non-complex output",
            ));
        };
        let out = gpu_helpers::upload_complex_tensor(provider, &tensor)
            .map_err(|err| sign_error_with_detail(&SIGN_ERROR_INTERNAL, err))?;
        return Ok(gpu_helpers::complex_gpu_value(out));
    }
    if let Some(provider) = runmat_accelerate_api::provider_for_handle(&handle) {
        if let Ok(out) = provider.unary_sign(&handle).await {
            if runmat_accelerate_api::handle_storage(&out)
                == runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
            {
                return Ok(gpu_helpers::complex_gpu_value(out));
            }
            return Ok(gpu_helpers::resident_gpu_value(out));
        }
    }
    let gathered = gpu_helpers::gather_value_async(&Value::GpuTensor(handle))
        .await
        .map_err(|err| sign_error_with_detail(&SIGN_ERROR_INTERNAL, err))?;
    match gathered {
        Value::Complex(re, im) => {
            let (re_out, im_out) = sign_complex(re, im);
            Ok(Value::Complex(re_out, im_out))
        }
        Value::ComplexTensor(ct) => sign_complex_tensor(ct),
        other => sign_real(other),
    }
}

fn sign_real(value: Value) -> BuiltinResult<Value> {
    let tensor = tensor::value_into_tensor_for("sign", value)
        .map_err(|e| sign_error_with_detail(&SIGN_ERROR_INVALID_INPUT, e))?;
    Ok(tensor::tensor_into_value(sign_tensor(tensor)?))
}

fn sign_tensor(tensor: Tensor) -> BuiltinResult<Tensor> {
    let data = tensor.data.iter().map(|&x| sign_real_scalar(x)).collect();
    Tensor::new(data, tensor.shape.clone())
        .map_err(|e| sign_error_with_detail(&SIGN_ERROR_INTERNAL, e))
}

fn sign_char_array(ca: CharArray) -> BuiltinResult<Value> {
    let data = ca
        .data
        .iter()
        .map(|&ch| sign_real_scalar(ch as u32 as f64))
        .collect::<Vec<_>>();
    let tensor = Tensor::new(data, vec![ca.rows, ca.cols])
        .map_err(|e| sign_error_with_detail(&SIGN_ERROR_INTERNAL, e))?;
    Ok(Value::Tensor(tensor))
}

fn sign_complex_tensor(ct: ComplexTensor) -> BuiltinResult<Value> {
    let mapped = ct
        .data
        .iter()
        .map(|&(re, im)| sign_complex(re, im))
        .collect::<Vec<_>>();
    let tensor = ComplexTensor::new(mapped, ct.shape.clone())
        .map_err(|e| sign_error_with_detail(&SIGN_ERROR_INTERNAL, e))?;
    Ok(Value::ComplexTensor(tensor))
}

#[inline]
fn sign_real_scalar(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else if x == 0.0 {
        0.0
    } else {
        x
    }
}

fn sign_complex(re: f64, im: f64) -> (f64, f64) {
    if re == 0.0 && im == 0.0 {
        return (0.0, 0.0);
    }
    if re.is_nan() || im.is_nan() {
        return (f64::NAN, f64::NAN);
    }
    let re_inf = re.is_infinite();
    let im_inf = im.is_infinite();
    if re_inf || im_inf {
        let real = if re_inf { re.signum() } else { 0.0 };
        let imag = if im_inf { im.signum() } else { 0.0 };
        let norm = (real * real + imag * imag).sqrt();
        if norm == 0.0 {
            return (real, imag);
        }
        return (real / norm, imag / norm);
    }
    let scale = re.abs().max(im.abs());
    if scale == 0.0 {
        return (0.0, 0.0);
    }
    let nr = re / scale;
    let ni = im / scale;
    let magnitude = (nr * nr + ni * ni).sqrt();
    if magnitude == 0.0 {
        (0.0, 0.0)
    } else {
        (nr / magnitude, ni / magnitude)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_builtins::{IntValue, LogicalArray, ResolveContext, Type};

    fn sign_builtin(value: Value) -> BuiltinResult<Value> {
        block_on(super::sign_builtin(value))
    }

    fn assert_complex_close(got: (f64, f64), want: (f64, f64), tol: f64) {
        if got.0.is_nan() && got.1.is_nan() && want.0.is_nan() && want.1.is_nan() {
            return;
        }
        assert!(
            (got.0 - want.0).abs() <= tol && (got.1 - want.1).abs() <= tol,
            "got {got:?}, expected {want:?}, tol {tol}"
        );
    }

    #[test]
    fn sign_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = SIGN_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"Y = sign(X)"));
    }

    #[test]
    fn sign_type_preserves_tensor_shape() {
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
    fn sign_type_scalar_tensor_returns_num() {
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
    fn sign_scalar_positive_negative_zero() {
        assert_eq!(sign_builtin(Value::Num(3.5)).unwrap(), Value::Num(1.0));
        assert_eq!(sign_builtin(Value::Num(-2.0)).unwrap(), Value::Num(-1.0));
        assert_eq!(sign_builtin(Value::Num(0.0)).unwrap(), Value::Num(0.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sign_scalar_nan_propagates() {
        let result = sign_builtin(Value::Num(f64::NAN)).unwrap();
        match result {
            Value::Num(v) => assert!(v.is_nan()),
            other => panic!("expected scalar NaN, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sign_tensor_mixed_values() {
        let tensor = Tensor::new(vec![-2.0, -0.0, 0.0, 5.0], vec![2, 2]).unwrap();
        let result = sign_builtin(Value::Tensor(tensor)).unwrap();
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![2, 2]);
                assert_eq!(out.data, vec![-1.0, 0.0, 0.0, 1.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sign_complex_scalar_normalises() {
        let result = sign_builtin(Value::Complex(3.0, 4.0)).unwrap();
        match result {
            Value::Complex(re, im) => {
                assert!((re - 0.6).abs() < 1e-12);
                assert!((im - 0.8).abs() < 1e-12);
            }
            other => panic!("expected complex value, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sign_complex_tensor_handles_zero() {
        let tensor = ComplexTensor::new(vec![(0.0, 0.0), (1.0, -1.0)], vec![2, 1]).unwrap();
        let result = sign_builtin(Value::ComplexTensor(tensor)).unwrap();
        match result {
            Value::ComplexTensor(out) => {
                assert_eq!(out.shape, vec![2, 1]);
                assert_eq!(out.data[0], (0.0, 0.0));
                let (re, im) = out.data[1];
                assert!((re - 0.7071067811865475).abs() < 1e-12);
                assert!((im + 0.7071067811865475).abs() < 1e-12);
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sign_character_array() {
        let ca = CharArray::new("RunMat".chars().collect(), 1, 6).unwrap();
        let result = sign_builtin(Value::CharArray(ca)).unwrap();
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![1, 6]);
                assert!(out.data.iter().all(|&v| (v - 1.0).abs() < 1e-12));
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sign_logical_array() {
        let logical = LogicalArray::new(vec![0, 1, 0, 1], vec![2, 2]).unwrap();
        let result = sign_builtin(Value::LogicalArray(logical)).unwrap();
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![2, 2]);
                assert_eq!(out.data, vec![0.0, 1.0, 0.0, 1.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sign_int_values() {
        let value = Value::Int(IntValue::I32(-7));
        let result = sign_builtin(value).unwrap();
        match result {
            Value::Num(v) => assert_eq!(v, -1.0),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sign_bool_values() {
        let t = sign_builtin(Value::Bool(true)).unwrap();
        let f = sign_builtin(Value::Bool(false)).unwrap();
        assert_eq!(t, Value::Num(1.0));
        assert_eq!(f, Value::Num(0.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sign_infinite_values() {
        let tensor = Tensor::new(
            vec![f64::INFINITY, f64::NEG_INFINITY, 0.0, f64::NAN],
            vec![2, 2],
        )
        .unwrap();
        let result = sign_builtin(Value::Tensor(tensor)).unwrap();
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.data[0], 1.0);
                assert_eq!(out.data[1], -1.0);
                assert_eq!(out.data[2], 0.0);
                assert!(out.data[3].is_nan());
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sign_string_input_errors() {
        let err = sign_builtin(Value::String("runmat".to_string())).unwrap_err();
        assert!(
            err.message()
                .contains("expected numeric, logical, or character input"),
            "unexpected error message: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sign_complex_with_nan() {
        let result = sign_builtin(Value::Complex(f64::NAN, 1.0)).unwrap();
        match result {
            Value::Complex(re, im) => {
                assert!(re.is_nan());
                assert!(im.is_nan());
            }
            other => panic!("expected complex result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sign_gpu_provider_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![-3.0, -0.5, 0.0, 2.5], vec![2, 2]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = sign_builtin(Value::GpuTensor(handle)).expect("sign");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![2, 2]);
            assert_eq!(gathered.data, vec![-1.0, -1.0, 0.0, 1.0]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sign_gpu_complex_input_stays_resident_when_provider_supports_sign() {
        test_support::with_test_provider(|provider| {
            let tensor =
                ComplexTensor::new(vec![(3.0, 4.0), (0.0, 0.0), (-1.0, 1.0)], vec![3, 1]).unwrap();
            let handle = gpu_helpers::upload_complex_tensor(provider, &tensor).expect("upload");
            let result = sign_builtin(Value::GpuTensor(handle)).expect("sign");
            let Value::GpuTensor(out_handle) = result else {
                panic!("expected resident gpu output");
            };
            assert_eq!(
                runmat_accelerate_api::handle_storage(&out_handle),
                runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
            );
            let gathered = block_on(gpu_helpers::gather_value_async(&Value::GpuTensor(
                out_handle,
            )))
            .expect("gather");
            let Value::ComplexTensor(actual) = gathered else {
                panic!("expected complex tensor");
            };
            let expected = match sign_complex_tensor(tensor).expect("cpu sign") {
                Value::ComplexTensor(tensor) => tensor,
                other => panic!("expected complex tensor, got {other:?}"),
            };
            assert_eq!(actual.shape, expected.shape);
            for (got, want) in actual.data.iter().zip(expected.data.iter()) {
                assert_complex_close(*got, *want, 1e-12);
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sign_gpu_complex_interleaved_rejects_odd_buffer_length() {
        test_support::with_test_provider(|provider| {
            let raw = vec![1.0, 2.0, 3.0];
            let shape = vec![2, 1];
            let view = runmat_accelerate_api::HostTensorView {
                data: &raw,
                shape: &shape,
            };
            let handle = provider.upload(&view).expect("upload");
            runmat_accelerate_api::set_handle_storage(
                &handle,
                runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved,
            );
            let err = sign_builtin(Value::GpuTensor(handle))
                .expect_err("odd complex buffer should reject");
            assert!(
                err.message().contains("sign: internal error"),
                "unexpected error: {err}"
            );
            assert!(
                err.message()
                    .contains("complex-interleaved buffer has odd length")
                    || err.message().contains("TensorShapeError")
                    || err.message().contains("shape"),
                "unexpected error detail: {err}"
            );
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn sign_wgpu_matches_cpu() {
        let _guard = test_support::accel_test_lock();
        if runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        )
        .is_err()
        {
            return;
        }
        if runmat_accelerate_api::provider().is_none() {
            return;
        }
        let tensor = Tensor::new(vec![-3.0, 0.0, 4.0, f64::NAN], vec![2, 2]).unwrap();
        let cpu = sign_real(Value::Tensor(tensor.clone())).unwrap();
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = runmat_accelerate_api::provider()
            .unwrap()
            .upload(&view)
            .unwrap();
        let gpu = block_on(sign_gpu(handle)).unwrap();
        let gathered = test_support::gather(gpu).expect("gather");
        match (cpu, gathered) {
            (Value::Tensor(ct), gt) => {
                assert_eq!(gt.shape, ct.shape);
                for (a, b) in gt.data.iter().zip(ct.data.iter()) {
                    if a.is_nan() && b.is_nan() {
                        continue;
                    }
                    assert_eq!(a, b);
                }
            }
            _ => panic!("unexpected shapes"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn sign_wgpu_complex_matches_cpu_and_stays_resident() {
        let _guard = test_support::accel_test_lock();
        if runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        )
        .is_err()
        {
            return;
        }
        let provider = runmat_accelerate_api::provider().expect("wgpu provider");
        let finite_max = match provider.precision() {
            runmat_accelerate_api::ProviderPrecision::F64 => f64::MAX,
            runmat_accelerate_api::ProviderPrecision::F32 => f32::MAX as f64,
        };
        let tensor = ComplexTensor::new(
            vec![
                (3.0, 4.0),
                (0.0, 0.0),
                (-1.0, 1.0),
                (finite_max, finite_max / 2.0),
                (f64::INFINITY, 2.0),
                (f64::INFINITY, f64::NEG_INFINITY),
                (f64::NAN, 1.0),
            ],
            vec![7, 1],
        )
        .unwrap();
        let handle = gpu_helpers::upload_complex_tensor(provider, &tensor).expect("upload");
        let gpu = block_on(sign_gpu(handle)).expect("gpu sign");
        let Value::GpuTensor(out_handle) = gpu else {
            panic!("expected resident gpu output");
        };
        assert_eq!(
            runmat_accelerate_api::handle_storage(&out_handle),
            runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
        );
        let gathered = block_on(gpu_helpers::gather_value_async(&Value::GpuTensor(
            out_handle,
        )))
        .expect("gather");
        let Value::ComplexTensor(actual) = gathered else {
            panic!("expected complex tensor");
        };
        let expected = match sign_complex_tensor(tensor).expect("cpu sign") {
            Value::ComplexTensor(tensor) => tensor,
            other => panic!("expected complex tensor, got {other:?}"),
        };
        assert_eq!(actual.shape, expected.shape);
        let tol = match provider.precision() {
            runmat_accelerate_api::ProviderPrecision::F64 => 1e-12,
            runmat_accelerate_api::ProviderPrecision::F32 => 1e-5,
        };
        for (got, want) in actual.data.iter().zip(expected.data.iter()) {
            assert_complex_close(*got, *want, tol);
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sign_rejects_string_with_stable_identifier() {
        let err = sign_builtin(Value::from("bad")).expect_err("expected error");
        assert_eq!(err.identifier(), SIGN_ERROR_INVALID_INPUT.identifier);
    }
}
