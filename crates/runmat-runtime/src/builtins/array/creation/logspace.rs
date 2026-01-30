//! MATLAB-compatible `logspace` builtin with GPU-aware semantics for RunMat.

use runmat_accelerate_api::HostTensorView;
use runmat_builtins::{ComplexTensor, Tensor, Type, Value};
use runmat_macros::runtime_builtin;

use crate::build_runtime_error;
use crate::builtins::array::type_resolvers::row_vector_type;
use crate::builtins::common::residency::{sequence_gpu_preference, SequenceIntent};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};

const LN_10: f64 = std::f64::consts::LN_10;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::array::creation::logspace")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "logspace",
    op_kind: GpuOpKind::Custom("generator"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[
        ProviderHook::Custom("linspace"),
        ProviderHook::Custom("scalar_mul"),
        ProviderHook::Unary {
            name: "unary_exp",
        },
    ],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Providers may implement a dedicated logspace path or compose it from linspace + scalar multiply + unary_exp. The runtime uploads host-generated data when hooks are unavailable.",
};

fn builtin_error(message: impl Into<String>) -> crate::RuntimeError {
    build_runtime_error(message)
        .with_builtin("logspace")
        .build()
}

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::array::creation::logspace")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "logspace",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Sequence generation is treated as a sink and is not fused with other operations.",
};

fn logspace_type(_args: &[Type]) -> Type {
    row_vector_type()
}

#[runtime_builtin(
    name = "logspace",
    category = "array/creation",
    summary = "Logarithmically spaced vector.",
    keywords = "logspace,logarithmic,vector,gpu",
    examples = "x = logspace(1, 3, 3)  % [10 100 1000]",
    accel = "array_construct",
    type_resolver(logspace_type),
    builtin_path = "crate::builtins::array::creation::logspace"
)]
async fn logspace_builtin(
    start: Value,
    stop: Value,
    rest: Vec<Value>,
) -> crate::BuiltinResult<Value> {
    if rest.len() > 1 {
        return Err(builtin_error(
            "logspace: expected two or three input arguments",
        ));
    }

    let (start_scalar, start_gpu) = parse_scalar("logspace", start).await?;
    let (stop_scalar, stop_gpu) = parse_scalar("logspace", stop).await?;
    let count = if rest.is_empty() {
        50usize
    } else {
        parse_count(&rest[0]).await?
    };

    let prefer_gpu =
        sequence_gpu_preference(count, SequenceIntent::Logspace, start_gpu || stop_gpu).prefer_gpu;
    build_sequence(start_scalar, stop_scalar, count, prefer_gpu).await
}

#[derive(Clone, Copy)]
enum Scalar {
    Real(f64),
    Complex { re: f64, im: f64 },
}

impl Scalar {
    fn parts(&self) -> (f64, f64) {
        match *self {
            Scalar::Real(r) => (r, 0.0),
            Scalar::Complex { re, im } => (re, im),
        }
    }
}

async fn parse_scalar(name: &str, value: Value) -> crate::BuiltinResult<(Scalar, bool)> {
    match value {
        Value::Num(n) => Ok((Scalar::Real(n), false)),
        Value::Int(i) => Ok((Scalar::Real(i.to_f64()), false)),
        Value::Bool(b) => Ok((Scalar::Real(if b { 1.0 } else { 0.0 }), false)),
        Value::Complex(re, im) => Ok((Scalar::Complex { re, im }, false)),
        Value::Tensor(t) => tensor_scalar(name, &t).map(|scalar| (scalar, false)),
        Value::ComplexTensor(t) => complex_tensor_scalar(name, &t).map(|scalar| (scalar, false)),
        Value::GpuTensor(handle) => {
            let tensor = gpu_helpers::gather_tensor_async(&handle).await?;
            tensor_scalar(name, &tensor).map(|scalar| (scalar, true))
        }
        Value::String(_) | Value::StringArray(_) | Value::CharArray(_) => Err(builtin_error(
            format!("{name}: endpoints must be numeric scalars; received a string-like value"),
        )),
        other => Err(builtin_error(format!(
            "{name}: endpoints must be numeric scalars; received {other:?}"
        ))),
    }
}

fn tensor_scalar(name: &str, tensor: &Tensor) -> crate::BuiltinResult<Scalar> {
    if !tensor::is_scalar_tensor(tensor) {
        return Err(builtin_error(format!("{name}: expected scalar input")));
    }
    Ok(Scalar::Real(tensor.data[0]))
}

fn complex_tensor_scalar(name: &str, tensor: &ComplexTensor) -> crate::BuiltinResult<Scalar> {
    if tensor.data.len() != 1 {
        return Err(builtin_error(format!("{name}: expected scalar input")));
    }
    let (re, im) = tensor.data[0];
    Ok(Scalar::Complex { re, im })
}

async fn parse_count(value: &Value) -> crate::BuiltinResult<usize> {
    match value {
        Value::Int(i) => {
            let raw = i.to_i64();
            if raw < 0 {
                return Err(builtin_error("logspace: number of points must be >= 0"));
            }
            usize::try_from(raw).map_err(|_| {
                builtin_error("logspace: number of points is too large for this platform")
            })
        }
        Value::Num(n) => parse_numeric_count(*n),
        Value::Bool(b) => Ok(if *b { 1 } else { 0 }),
        Value::Tensor(t) => {
            if !tensor::is_scalar_tensor(t) {
                return Err(builtin_error("logspace: number of points must be a scalar"));
            }
            parse_numeric_count(t.data[0])
        }
        Value::GpuTensor(handle) => {
            let tensor = gpu_helpers::gather_tensor_async(handle).await?;
            if !tensor::is_scalar_tensor(&tensor) {
                return Err(builtin_error("logspace: number of points must be a scalar"));
            }
            parse_numeric_count(tensor.data[0])
        }
        other => Err(builtin_error(format!(
            "logspace: number of points must be numeric, got {other:?}"
        ))),
    }
}

fn parse_numeric_count(raw: f64) -> crate::BuiltinResult<usize> {
    if !raw.is_finite() {
        return Err(builtin_error("logspace: number of points must be finite"));
    }
    let rounded = raw.round();
    if (rounded - raw).abs() > f64::EPSILON {
        return Err(builtin_error(
            "logspace: number of points must be an integer",
        ));
    }
    if rounded < 0.0 {
        return Err(builtin_error("logspace: number of points must be >= 0"));
    }
    if rounded > usize::MAX as f64 {
        return Err(builtin_error(
            "logspace: number of points is too large for this platform",
        ));
    }
    Ok(rounded as usize)
}

async fn build_sequence(
    start: Scalar,
    stop: Scalar,
    count: usize,
    prefer_gpu: bool,
) -> crate::BuiltinResult<Value> {
    let (start_re, start_im) = start.parts();
    let (stop_re, stop_im) = stop.parts();
    let complex = start_im != 0.0 || stop_im != 0.0;

    if complex {
        let data = generate_complex_log_sequence(start_re, start_im, stop_re, stop_im, count);
        let tensor = ComplexTensor::new(data, vec![1, count])
            .map_err(|e| builtin_error(format!("logspace: {e}")))?;
        return Ok(Value::ComplexTensor(tensor));
    }

    if prefer_gpu {
        if let Some(value) = try_gpu_logspace(start_re, stop_re, count).await {
            return Ok(value);
        }
    }

    let data = generate_real_log_sequence(start_re, stop_re, count);
    if prefer_gpu {
        #[cfg(all(test, feature = "wgpu"))]
        {
            if runmat_accelerate_api::provider().is_none() {
                let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
                    runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
                );
            }
        }
        if let Some(provider) = runmat_accelerate_api::provider() {
            let shape = [1usize, count];
            let view = HostTensorView {
                data: &data,
                shape: &shape,
            };
            if let Ok(handle) = provider.upload(&view) {
                return Ok(Value::GpuTensor(handle));
            }
        }
    }

    let tensor =
        Tensor::new(data, vec![1, count]).map_err(|e| builtin_error(format!("logspace: {e}")))?;
    Ok(Value::Tensor(tensor))
}

async fn try_gpu_logspace(start: f64, stop: f64, count: usize) -> Option<Value> {
    #[cfg(all(test, feature = "wgpu"))]
    {
        if runmat_accelerate_api::provider().is_none() {
            let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
                runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
            );
        }
    }
    let provider = runmat_accelerate_api::provider()?;
    let exponents = provider.linspace(start, stop, count).ok()?;

    let scaled = match provider.scalar_mul(&exponents, LN_10) {
        Ok(handle) => handle,
        Err(_) => {
            provider.free(&exponents).ok();
            return None;
        }
    };
    provider.free(&exponents).ok();

    let result = match provider.unary_exp(&scaled).await {
        Ok(handle) => handle,
        Err(_) => {
            provider.free(&scaled).ok();
            return None;
        }
    };
    provider.free(&scaled).ok();

    Some(Value::GpuTensor(result))
}

fn generate_real_log_sequence(start: f64, stop: f64, count: usize) -> Vec<f64> {
    if count == 0 {
        return Vec::new();
    }
    if count == 1 {
        return vec![10f64.powf(stop)];
    }
    let mut data = Vec::with_capacity(count);
    let step = (stop - start) / ((count - 1) as f64);
    for idx in 0..count {
        let exponent = start + (idx as f64) * step;
        data.push(10f64.powf(exponent));
    }
    if let Some(first) = data.first_mut() {
        *first = 10f64.powf(start);
    }
    if let Some(last) = data.last_mut() {
        *last = 10f64.powf(stop);
    }
    data
}

fn generate_complex_log_sequence(
    start_re: f64,
    start_im: f64,
    stop_re: f64,
    stop_im: f64,
    count: usize,
) -> Vec<(f64, f64)> {
    if count == 0 {
        return Vec::new();
    }
    let steps = generate_complex_sequence(start_re, start_im, stop_re, stop_im, count);
    steps
        .into_iter()
        .map(|(re, im)| complex_pow10(re, im))
        .collect()
}

fn generate_complex_sequence(
    start_re: f64,
    start_im: f64,
    stop_re: f64,
    stop_im: f64,
    count: usize,
) -> Vec<(f64, f64)> {
    if count == 0 {
        return Vec::new();
    }
    if count == 1 {
        return vec![(stop_re, stop_im)];
    }
    let mut data = Vec::with_capacity(count);
    let step_re = (stop_re - start_re) / ((count - 1) as f64);
    let step_im = (stop_im - start_im) / ((count - 1) as f64);
    for idx in 0..count {
        let re = start_re + (idx as f64) * step_re;
        let im = start_im + (idx as f64) * step_im;
        data.push((re, im));
    }
    if let Some(first) = data.first_mut() {
        *first = (start_re, start_im);
    }
    if let Some(last) = data.last_mut() {
        *last = (stop_re, stop_im);
    }
    data
}

fn complex_pow10(re: f64, im: f64) -> (f64, f64) {
    // 10^(re + i*im) = exp((re + i*im) * ln(10))
    let scaled_re = re * LN_10;
    let scaled_im = im * LN_10;
    let mag = scaled_re.exp();
    let cos = scaled_im.cos();
    let sin = scaled_im.sin();
    (mag * cos, mag * sin)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_builtins::IntValue;

    fn logspace_builtin(
        start: Value,
        stop: Value,
        rest: Vec<Value>,
    ) -> crate::BuiltinResult<Value> {
        block_on(super::logspace_builtin(start, stop, rest))
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn logspace_default_points() {
        let result =
            logspace_builtin(Value::Num(1.0), Value::Num(3.0), Vec::new()).expect("logspace");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 50]);
                assert!((t.data[0] - 10.0).abs() < 1e-12);
                assert!((t.data[49] - 1000.0).abs() < 1e-9);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn logspace_type_is_row_vector() {
        assert_eq!(
            logspace_type(&[Type::Num, Type::Num]),
            Type::Tensor {
                shape: Some(vec![Some(1), None])
            }
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn logspace_custom_points() {
        let result = logspace_builtin(
            Value::Num(0.0),
            Value::Num(2.0),
            vec![Value::Int(IntValue::I32(5))],
        )
        .expect("logspace");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 5]);
                let expected = [1.0, 3.1622776601683795, 10.0, 31.622776601683793, 100.0];
                for (a, b) in t.data.iter().zip(expected.iter()) {
                    assert!((a - b).abs() < 1e-12);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn logspace_zero_points() {
        let result = logspace_builtin(
            Value::Num(1.0),
            Value::Num(3.0),
            vec![Value::Int(IntValue::I32(0))],
        )
        .expect("logspace");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 0]);
                assert!(t.data.is_empty());
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn logspace_zero_points_bool_count() {
        let result = logspace_builtin(Value::Num(0.0), Value::Num(1.0), vec![Value::Bool(false)])
            .expect("logspace");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 0]);
                assert!(t.data.is_empty());
            }
            other => panic!("expected empty tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn logspace_single_point() {
        let result = logspace_builtin(
            Value::Num(-2.0),
            Value::Num(0.0),
            vec![Value::Int(IntValue::I32(1))],
        )
        .expect("logspace");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 1]);
                assert!((t.data[0] - 1.0).abs() < 1e-12);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn logspace_complex_points() {
        let result = logspace_builtin(
            Value::Complex(0.0, 1.0),
            Value::Complex(0.0, 2.0),
            vec![Value::Int(IntValue::I32(4))],
        )
        .expect("logspace");
        match result {
            Value::ComplexTensor(t) => {
                assert_eq!(t.shape, vec![1, 4]);
                assert_eq!(t.data.len(), 4);
                let expected = generate_complex_log_sequence(0.0, 1.0, 0.0, 2.0, 4);
                for (actual, exp) in t.data.iter().zip(expected.iter()) {
                    assert!((actual.0 - exp.0).abs() < 1e-12);
                    assert!((actual.1 - exp.1).abs() < 1e-12);
                }
            }
            other => panic!("expected complex tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn logspace_tensor_scalar_inputs() {
        let start = Tensor::new(vec![2.0], vec![1, 1]).unwrap();
        let stop = Tensor::new(vec![3.0], vec![1, 1]).unwrap();
        let result = logspace_builtin(
            Value::Tensor(start),
            Value::Tensor(stop),
            vec![Value::Int(IntValue::I32(3))],
        )
        .expect("logspace");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 3]);
                let expected = generate_real_log_sequence(2.0, 3.0, 3);
                for (actual, exp) in t.data.iter().zip(expected.iter()) {
                    assert!((actual - exp).abs() < 1e-12);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn logspace_rejects_non_integer_count() {
        let err = logspace_builtin(Value::Num(1.0), Value::Num(2.0), vec![Value::Num(3.5)])
            .expect_err("expected error");
        assert!(err.message().contains("integer"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn logspace_rejects_negative_count() {
        let err = logspace_builtin(
            Value::Num(0.0),
            Value::Num(1.0),
            vec![Value::Int(IntValue::I32(-1))],
        )
        .expect_err("expected error");
        assert!(err.message().contains(">="));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn logspace_rejects_infinite_count() {
        let err = logspace_builtin(
            Value::Num(0.0),
            Value::Num(1.0),
            vec![Value::Num(f64::INFINITY)],
        )
        .expect_err("expected error");
        assert!(err.message().contains("finite"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn logspace_rejects_non_scalar_inputs() {
        let start = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
        let err =
            logspace_builtin(Value::Tensor(start), Value::Num(1.0), Vec::new()).expect_err("error");
        assert!(err.message().contains("scalar"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn logspace_gpu_roundtrip() {
        test_support::with_test_provider(|provider| {
            let lo = Value::GpuTensor(
                provider
                    .upload(&HostTensorView {
                        data: &[1.0],
                        shape: &[1, 1],
                    })
                    .expect("upload"),
            );
            let hi = Value::GpuTensor(
                provider
                    .upload(&HostTensorView {
                        data: &[3.0],
                        shape: &[1, 1],
                    })
                    .expect("upload"),
            );
            let result =
                logspace_builtin(lo, hi, vec![Value::Int(IntValue::I32(3))]).expect("logspace");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![1, 3]);
            let expected = [10.0, 100.0, 1000.0];
            for (a, b) in gathered.data.iter().zip(expected.iter()) {
                assert!((a - b).abs() < 1e-9);
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn logspace_wgpu_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let cpu = logspace_builtin(
            Value::Num(1.0),
            Value::Num(2.0),
            vec![Value::Int(IntValue::I32(5))],
        )
        .expect("cpu");
        let gpu = {
            let view = HostTensorView {
                data: &[1.0],
                shape: &[1, 1],
            };
            let lo = Value::GpuTensor(
                runmat_accelerate_api::provider()
                    .unwrap()
                    .upload(&view)
                    .expect("upload lo"),
            );
            let hi = Value::GpuTensor(
                runmat_accelerate_api::provider()
                    .unwrap()
                    .upload(&HostTensorView {
                        data: &[2.0],
                        shape: &[1, 1],
                    })
                    .expect("upload hi"),
            );
            logspace_builtin(lo, hi, vec![Value::Int(IntValue::I32(5))]).expect("gpu")
        };
        let gathered = test_support::gather(gpu).expect("gather");
        match (cpu, gathered) {
            (Value::Tensor(ct), gt) => {
                assert_eq!(ct.shape, gt.shape);
                let tol = match runmat_accelerate_api::provider().unwrap().precision() {
                    runmat_accelerate_api::ProviderPrecision::F64 => 1e-12,
                    runmat_accelerate_api::ProviderPrecision::F32 => 1e-5,
                };
                for (a, b) in ct.data.iter().zip(gt.data.iter()) {
                    assert!((a - b).abs() < tol);
                }
            }
            _ => panic!("unexpected value variants"),
        }
    }
}
