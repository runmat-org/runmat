//! MATLAB-compatible `atan` builtin with GPU-aware semantics for RunMat.
//!
//! Provides element-wise inverse tangent for scalars, tensors, and complex data while mirroring
//! MATLAB behavior. GPU execution uses provider hooks when available and falls back to the host
//! path if kernels are missing or outputs must become host-resident.

use num_complex::Complex64;
use runmat_accelerate_api::{GpuTensorHandle, HostTensorView};
use runmat_builtins::{CharArray, ComplexTensor, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::random_args::{complex_tensor_into_value, keyword_of};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, FusionError,
    FusionExprContext, FusionKernelTemplate, GpuOpKind, ProviderHook, ReductionNaN,
    ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::builtins::math::type_resolvers::numeric_unary_type;
use crate::dispatcher;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "atan";

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::trigonometry::atan")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "atan",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Unary { name: "unary_atan" }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Providers execute atan on-device via unary_atan; runtimes gather to host when the hook is unavailable.",
};

fn runtime_error_for(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .build()
}

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::trigonometry::atan")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "atan",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: |ctx: &FusionExprContext| {
            let input = ctx.inputs.first().ok_or(FusionError::MissingInput(0))?;
            Ok(format!("atan({input})"))
        },
    }),
    reduction: None,
    emits_nan: false,
    notes: "Fusion planner emits WGSL atan calls; providers may override with specialised fused kernels.",
};

#[runtime_builtin(
    name = "atan",
    category = "math/trigonometry",
    summary = "Arctangent of scalars, vectors, matrices, or N-D tensors (element-wise).",
    keywords = "atan,arctangent,inverse tangent,trigonometry,gpu",
    accel = "unary",
    type_resolver(numeric_unary_type),
    builtin_path = "crate::builtins::math::trigonometry::atan"
)]
async fn atan_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let template = parse_output_template(&rest)?;
    let base = match value {
        Value::GpuTensor(handle) => atan_gpu(handle).await?,
        Value::Complex(re, im) => {
            let (out_re, out_im) = atan_complex_components(re, im);
            Value::Complex(out_re, out_im)
        }
        Value::ComplexTensor(ct) => atan_complex_tensor(ct)?,
        Value::CharArray(ca) => atan_char_array(ca)?,
        Value::String(_) | Value::StringArray(_) => {
            return Err(runtime_error_for("atan: expected numeric input"))
        }
        other => atan_real(other)?,
    };
    apply_output_template(base, &template).await
}

async fn atan_gpu(handle: GpuTensorHandle) -> BuiltinResult<Value> {
    if let Some(provider) = runmat_accelerate_api::provider_for_handle(&handle) {
        if let Ok(out) = provider.unary_atan(&handle).await {
            return Ok(Value::GpuTensor(out));
        }
    }
    let tensor = gpu_helpers::gather_tensor_async(&handle).await?;
    atan_tensor(tensor).map(tensor::tensor_into_value)
}

fn atan_real(value: Value) -> BuiltinResult<Value> {
    let tensor = tensor::value_into_tensor_for("atan", value).map_err(runtime_error_for)?;
    atan_tensor(tensor).map(tensor::tensor_into_value)
}

fn atan_tensor(tensor: Tensor) -> BuiltinResult<Tensor> {
    let data = tensor.data.iter().map(|&v| v.atan()).collect::<Vec<_>>();
    Tensor::new(data, tensor.shape.clone()).map_err(|e| runtime_error_for(format!("atan: {e}")))
}

fn atan_complex_tensor(ct: ComplexTensor) -> BuiltinResult<Value> {
    let mapped = ct
        .data
        .iter()
        .map(|&(re, im)| atan_complex_components(re, im))
        .collect::<Vec<_>>();
    let tensor = ComplexTensor::new(mapped, ct.shape.clone())
        .map_err(|e| runtime_error_for(format!("atan: {e}")))?;
    Ok(complex_tensor_into_value(tensor))
}

fn atan_char_array(array: CharArray) -> BuiltinResult<Value> {
    let data = array
        .data
        .iter()
        .map(|&ch| (ch as u32 as f64).atan())
        .collect::<Vec<_>>();
    let tensor = Tensor::new(data, vec![array.rows, array.cols])
        .map_err(|e| runtime_error_for(format!("atan: {e}")))?;
    Ok(tensor::tensor_into_value(tensor))
}

#[inline]
fn atan_complex_components(re: f64, im: f64) -> (f64, f64) {
    let value = Complex64::new(re, im).atan();
    (value.re, value.im)
}

#[derive(Clone)]
enum OutputTemplate {
    Default,
    Like(Value),
}

#[derive(Clone, Copy)]
enum DevicePreference {
    Host,
    Gpu,
}

#[derive(Clone, Copy)]
enum PrototypeClass {
    Real,
    Complex,
}

struct LikeAnalysis {
    device: DevicePreference,
    class: PrototypeClass,
}

fn parse_output_template(args: &[Value]) -> BuiltinResult<OutputTemplate> {
    match args.len() {
        0 => Ok(OutputTemplate::Default),
        1 => {
            if let Some(keyword) = keyword_of(&args[0]) {
                if keyword.trim() == "like" {
                    return Err(runtime_error_for("atan: expected prototype after 'like'"));
                }
            }
            Err(runtime_error_for("atan: unrecognised argument for atan"))
        }
        len if len >= 2 => {
            if let Some(keyword) = keyword_of(&args[0]) {
                if keyword.trim() == "like" {
                    if len == 2 {
                        return Ok(OutputTemplate::Like(args[1].clone()));
                    }
                    return Err(runtime_error_for("atan: too many input arguments"));
                }
            }
            Err(runtime_error_for(
                "atan: unsupported option; only 'like' is accepted",
            ))
        }
        _ => unreachable!(),
    }
}

async fn apply_output_template(value: Value, template: &OutputTemplate) -> BuiltinResult<Value> {
    match template {
        OutputTemplate::Default => Ok(value),
        OutputTemplate::Like(proto) => apply_like_template(value, proto).await,
    }
}

async fn apply_like_template(value: Value, prototype: &Value) -> BuiltinResult<Value> {
    let analysis = analyse_like_prototype(prototype).await?;
    match (analysis.class, analysis.device) {
        (PrototypeClass::Real, DevicePreference::Host) => ensure_host_real(value).await,
        (PrototypeClass::Real, DevicePreference::Gpu) => ensure_gpu_real(value),
        (PrototypeClass::Complex, DevicePreference::Host) => ensure_host_complex(value).await,
        (PrototypeClass::Complex, DevicePreference::Gpu) => Err(runtime_error_for(
            "atan: GPU 'like' prototypes with complex outputs are not supported",
        )),
    }
}

#[async_recursion::async_recursion(?Send)]
async fn analyse_like_prototype(prototype: &Value) -> BuiltinResult<LikeAnalysis> {
    match prototype {
        Value::GpuTensor(_) => Ok(LikeAnalysis {
            device: DevicePreference::Gpu,
            class: PrototypeClass::Real,
        }),
        Value::Tensor(_)
        | Value::Num(_)
        | Value::Int(_)
        | Value::Bool(_)
        | Value::LogicalArray(_) => Ok(LikeAnalysis {
            device: DevicePreference::Host,
            class: PrototypeClass::Real,
        }),
        Value::Complex(_, _) | Value::ComplexTensor(_) => Ok(LikeAnalysis {
            device: DevicePreference::Host,
            class: PrototypeClass::Complex,
        }),
        Value::CharArray(_) | Value::String(_) | Value::StringArray(_) => {
            Err(runtime_error_for("atan: 'like' prototype must be numeric"))
        }
        other => {
            let gathered = dispatcher::gather_if_needed_async(other).await?;
            if &gathered == other {
                Err(runtime_error_for(format!(
                    "atan: unsupported 'like' prototype {other:?}"
                )))
            } else {
                analyse_like_prototype(&gathered).await
            }
        }
    }
}

async fn ensure_host_value(value: Value) -> BuiltinResult<Value> {
    if let Value::GpuTensor(_) = &value {
        gpu_helpers::gather_value_async(&value).await
    } else {
        Ok(value)
    }
}

async fn ensure_host_real(value: Value) -> BuiltinResult<Value> {
    let host_value = ensure_host_value(value).await?;
    if is_complex_value(&host_value) {
        return Err(runtime_error_for(
            "atan: result is complex but 'like' prototype is real",
        ));
    }
    Ok(host_value)
}

async fn ensure_host_complex(value: Value) -> BuiltinResult<Value> {
    let host_value = ensure_host_value(value).await?;
    if is_complex_value(&host_value) {
        Ok(host_value)
    } else {
        convert_real_to_complex(host_value)
    }
}

fn ensure_gpu_real(value: Value) -> BuiltinResult<Value> {
    if is_complex_value(&value) {
        return Err(runtime_error_for(
            "atan: GPU 'like' prototypes do not support complex outputs",
        ));
    }
    match value {
        Value::GpuTensor(_) => Ok(value),
        other => convert_real_value_to_gpu(other),
    }
}

fn is_complex_value(value: &Value) -> bool {
    matches!(value, Value::Complex(_, _) | Value::ComplexTensor(_))
}

fn convert_real_to_complex(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::Complex(_, _) | Value::ComplexTensor(_) => Ok(value),
        Value::Num(n) => Ok(Value::Complex(n, 0.0)),
        Value::Tensor(tensor) => {
            let data: Vec<(f64, f64)> = tensor.data.iter().map(|&v| (v, 0.0)).collect();
            let tensor = ComplexTensor::new(data, tensor.shape.clone())
                .map_err(|e| runtime_error_for(format!("atan: {e}")))?;
            Ok(complex_tensor_into_value(tensor))
        }
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(&logical).map_err(runtime_error_for)?;
            convert_real_to_complex(Value::Tensor(tensor))
        }
        Value::Int(i) => convert_real_to_complex(Value::Num(i.to_f64())),
        Value::Bool(b) => convert_real_to_complex(Value::Num(if b { 1.0 } else { 0.0 })),
        Value::String(_) | Value::StringArray(_) | Value::CharArray(_) => {
            Err(runtime_error_for("atan: 'like' prototype must be numeric"))
        }
        Value::GpuTensor(_) => Err(runtime_error_for(
            "atan: internal error converting GPU value to complex output",
        )),
        other => Err(runtime_error_for(format!(
            "atan: cannot convert value {other:?} into a complex result for 'like'"
        ))),
    }
}

fn convert_real_value_to_gpu(value: Value) -> BuiltinResult<Value> {
    let provider = runmat_accelerate_api::provider().ok_or_else(|| {
        runtime_error_for(
            "atan: GPU output requested via 'like' but no acceleration provider is active",
        )
    })?;
    match value {
        Value::Tensor(tensor) => {
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).map_err(|e| {
                runtime_error_for(format!("atan: failed to upload GPU result: {e}"))
            })?;
            Ok(Value::GpuTensor(handle))
        }
        Value::Num(n) => {
            let tensor = Tensor::new(vec![n], vec![1, 1])
                .map_err(|e| runtime_error_for(format!("atan: {e}")))?;
            convert_real_value_to_gpu(Value::Tensor(tensor))
        }
        Value::Int(i) => convert_real_value_to_gpu(Value::Num(i.to_f64())),
        Value::Bool(b) => convert_real_value_to_gpu(Value::Num(if b { 1.0 } else { 0.0 })),
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(&logical).map_err(runtime_error_for)?;
            convert_real_value_to_gpu(Value::Tensor(tensor))
        }
        Value::GpuTensor(_) => Ok(value),
        Value::Complex(_, _) | Value::ComplexTensor(_) => Err(runtime_error_for(
            "atan: GPU 'like' prototypes do not support complex outputs",
        )),
        Value::String(_) | Value::StringArray(_) | Value::CharArray(_) => {
            Err(runtime_error_for("atan: 'like' prototype must be numeric"))
        }
        other => Err(runtime_error_for(format!(
            "atan: unsupported result type {other:?} for GPU output via 'like'"
        ))),
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_builtins::{IntValue, Tensor, Type};

    fn atan_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        block_on(super::atan_builtin(value, rest))
    }

    #[test]
    fn atan_type_preserves_tensor_shape() {
        let out = numeric_unary_type(&[Type::Tensor {
            shape: Some(vec![Some(2), Some(3)]),
        }]);
        assert_eq!(
            out,
            Type::Tensor {
                shape: Some(vec![Some(2), Some(3)])
            }
        );
    }

    #[test]
    fn atan_type_scalar_tensor_returns_num() {
        let out = numeric_unary_type(&[Type::Tensor {
            shape: Some(vec![Some(1), Some(1)]),
        }]);
        assert_eq!(out, Type::Num);
    }

    fn error_message(err: RuntimeError) -> String {
        err.message().to_string()
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn atan_scalar() {
        let result = atan_builtin(Value::Num(1.0), Vec::new()).expect("atan");
        match result {
            Value::Num(v) => assert!((v - std::f64::consts::FRAC_PI_4).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn atan_tensor_elements() {
        let tensor = Tensor::new(vec![-1.0, 0.0, 1.0], vec![3, 1]).unwrap();
        let result = atan_builtin(Value::Tensor(tensor.clone()), Vec::new()).expect("atan");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![3, 1]);
                for (value, expected) in out.data.iter().zip(tensor.data.iter().map(|v| v.atan())) {
                    assert!((value - expected).abs() < 1e-12);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn atan_int_value_promotes() {
        let result = atan_builtin(Value::Int(IntValue::I32(-1)), Vec::new()).expect("atan");
        match result {
            Value::Num(v) => assert!((v + std::f64::consts::FRAC_PI_4).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn atan_complex_scalar() {
        let result = atan_builtin(Value::Complex(1.0, 2.0), Vec::new()).expect("atan");
        match result {
            Value::Complex(re, im) => {
                let expected = Complex64::new(1.0, 2.0).atan();
                assert!((re - expected.re).abs() < 1e-12);
                assert!((im - expected.im).abs() < 1e-12);
            }
            other => panic!("expected complex result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn atan_complex_tensor_elements() {
        let tensor = ComplexTensor::new(vec![(1.0, 0.5), (-0.5, 1.0)], vec![2, 1]).unwrap();
        let result = atan_builtin(Value::ComplexTensor(tensor.clone()), Vec::new()).expect("atan");
        match result {
            Value::ComplexTensor(out) => {
                assert_eq!(out.shape, vec![2, 1]);
                for (value, expected) in out.data.iter().zip(
                    tensor
                        .data
                        .iter()
                        .map(|&(r, i)| atan_complex_components(r, i)),
                ) {
                    assert!((value.0 - expected.0).abs() < 1e-12);
                    assert!((value.1 - expected.1).abs() < 1e-12);
                }
            }
            Value::Complex(re, im) => {
                panic!("expected tensor result, got scalar {re}+{im}i");
            }
            other => panic!("unexpected result {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn atan_char_array_roundtrip() {
        let chars = CharArray::new_row("RU");
        let result = atan_builtin(Value::CharArray(chars.clone()), Vec::new()).expect("atan");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, chars.cols]);
                for (value, ch) in t.data.iter().zip(chars.data.iter()) {
                    let expected = (*ch as u32 as f64).atan();
                    assert!((value - expected).abs() < 1e-12);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn atan_string_errors() {
        let err = atan_builtin(Value::from("runmat"), Vec::new()).expect_err("expected error");
        let message = error_message(err);
        assert!(message.contains("expected numeric input"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn atan_like_missing_prototype_errors() {
        let err =
            atan_builtin(Value::Num(0.0), vec![Value::from("like")]).expect_err("expected error");
        let message = error_message(err);
        assert!(message.contains("expected prototype"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn atan_like_host_prototype() {
        let tensor = Tensor::new(vec![0.0, 1.0], vec![2, 1]).unwrap();
        let result = atan_builtin(
            Value::Tensor(tensor.clone()),
            vec![Value::from("like"), Value::Num(0.0)],
        )
        .expect("atan");
        match result {
            Value::Tensor(out) => {
                let expected: Vec<f64> = tensor.data.iter().map(|&v| v.atan()).collect();
                assert_eq!(out.data, expected);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn atan_like_complex_prototype_promotes() {
        let result = atan_builtin(
            Value::Num(0.5),
            vec![Value::from("like"), Value::Complex(0.0, 1.0)],
        )
        .expect("atan");
        match result {
            Value::Complex(re, im) => {
                assert!((re - 0.5f64.atan()).abs() < 1e-12);
                assert!(im.abs() < 1e-12);
            }
            other => panic!("expected complex result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn atan_gpu_provider_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![-1.0, 0.0, 1.0], vec![3, 1]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = atan_builtin(Value::GpuTensor(handle), Vec::new()).expect("atan");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![3, 1]);
            let expected: Vec<f64> = tensor.data.iter().map(|&v| v.atan()).collect();
            assert_eq!(gathered.data, expected);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn atan_like_gpu_prototype_keeps_residency() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![0.0, 1.0], vec![2, 1]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let input = provider.upload(&view).expect("upload");
            let proto = provider.upload(&view).expect("proto upload");
            let result = atan_builtin(
                Value::GpuTensor(input),
                vec![Value::from("like"), Value::GpuTensor(proto)],
            )
            .expect("atan");
            match result {
                Value::GpuTensor(handle) => {
                    let gathered = test_support::gather(Value::GpuTensor(handle)).expect("gather");
                    let expected: Vec<f64> = tensor.data.iter().map(|&v| v.atan()).collect();
                    assert_eq!(gathered.data, expected);
                }
                other => panic!("expected GPU tensor, got {other:?}"),
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn atan_like_real_with_complex_output_errors() {
        let err = atan_builtin(
            Value::Complex(1.0, 1.0),
            vec![Value::from("like"), Value::Num(0.0)],
        )
        .expect_err("expected error");
        let message = error_message(err);
        assert!(message.contains("complex"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn atan_like_gpu_with_complex_output_errors() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![0.0], vec![1, 1]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let proto = provider.upload(&view).expect("upload");
            let err = atan_builtin(
                Value::Complex(1.0, 1.0),
                vec![Value::from("like"), Value::GpuTensor(proto)],
            )
            .expect_err("expected error");
            let message = error_message(err);
            assert!(message.contains("complex"));
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn atan_like_non_numeric_prototype_errors() {
        let err = atan_builtin(
            Value::Num(0.0),
            vec![Value::from("like"), Value::from("not-a-proto")],
        )
        .expect_err("expected error");
        let message = error_message(err);
        assert!(message.contains("prototype must be numeric"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn atan_like_rejects_extra_arguments() {
        let err = atan_builtin(
            Value::Num(0.0),
            vec![Value::from("like"), Value::Num(0.0), Value::Num(1.0)],
        )
        .expect_err("expected error");
        let message = error_message(err);
        assert!(message.contains("too many input arguments"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn atan_like_keyword_flexible_case() {
        let result = atan_builtin(Value::Num(1.0), vec![Value::from("LIKE"), Value::Num(0.0)])
            .expect("atan");
        match result {
            Value::Num(v) => assert!((v - std::f64::consts::FRAC_PI_4).abs() < 1e-12),
            other => panic!("unexpected result {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn atan_like_char_array_keyword() {
        let keyword = CharArray::new_row("like");
        let result = atan_builtin(
            Value::Num(0.0),
            vec![Value::CharArray(keyword), Value::Num(0.0)],
        )
        .expect("atan");
        match result {
            Value::Num(v) => assert!(v.abs() < 1e-12),
            other => panic!("unexpected result {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn atan_unrecognised_argument_errors() {
        let err = atan_builtin(Value::Num(0.0), vec![Value::from("invalid")])
            .expect_err("expected error");
        let message = error_message(err);
        assert!(message.contains("unrecognised argument"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn atan_wgpu_matches_cpu_elementwise() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let tensor = Tensor::new(vec![-2.0, -0.5, 0.0, 0.5, 2.0], vec![5, 1]).unwrap();
        let cpu = atan_real(Value::Tensor(tensor.clone())).unwrap();
        let view = HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = runmat_accelerate_api::provider()
            .unwrap()
            .upload(&view)
            .unwrap();
        let gpu = block_on(atan_gpu(handle)).unwrap();
        let gathered = test_support::gather(gpu).expect("gather");
        match (cpu, gathered) {
            (Value::Tensor(ct), gt) => {
                assert_eq!(gt.shape, ct.shape);
                let tol = match runmat_accelerate_api::provider().unwrap().precision() {
                    runmat_accelerate_api::ProviderPrecision::F64 => 1e-12,
                    runmat_accelerate_api::ProviderPrecision::F32 => 1e-5,
                };
                for (a, b) in gt.data.iter().zip(ct.data.iter()) {
                    assert!((a - b).abs() < tol, "|{a} - {b}| >= {tol}");
                }
            }
            _ => panic!("unexpected comparison result"),
        }
    }
}
