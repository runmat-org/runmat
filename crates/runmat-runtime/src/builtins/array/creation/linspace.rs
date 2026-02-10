//! MATLAB-compatible `linspace` builtin with GPU-aware semantics for RunMat.

use log::trace;
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
use runmat_builtins::ResolveContext;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::array::creation::linspace")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "linspace",
    op_kind: GpuOpKind::Custom("generator"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("linspace")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Providers may generate sequences directly; the runtime uploads host-generated data when hooks are absent.",
};

fn builtin_error(message: impl Into<String>) -> crate::RuntimeError {
    build_runtime_error(message)
        .with_builtin("linspace")
        .build()
}

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::array::creation::linspace")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "linspace",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Sequence generation is treated as a sink and is not fused with other operations.",
};

fn linspace_type(_args: &[Type], ctx: &ResolveContext) -> Type {
    row_vector_type(ctx)
}

#[runtime_builtin(
    name = "linspace",
    category = "array/creation",
    summary = "Linearly spaced vector.",
    keywords = "linspace,range,vector,gpu",
    examples = "x = linspace(0, 1, 5)  % [0 0.25 0.5 0.75 1]",
    accel = "array_construct",
    type_resolver(linspace_type),
    builtin_path = "crate::builtins::array::creation::linspace"
)]
async fn linspace_builtin(
    start: Value,
    stop: Value,
    rest: Vec<Value>,
) -> crate::BuiltinResult<Value> {
    if rest.len() > 1 {
        return Err(builtin_error(
            "linspace: expected at most three input arguments",
        ));
    }

    let (start_scalar, start_gpu) = parse_scalar("linspace", start).await?;
    let (stop_scalar, stop_gpu) = parse_scalar("linspace", stop).await?;

    let count = if rest.is_empty() {
        100usize
    } else {
        parse_count(&rest[0]).await?
    };

    let residency = sequence_gpu_preference(count, SequenceIntent::Linspace, start_gpu || stop_gpu);
    if log::log_enabled!(log::Level::Trace) {
        trace!(
            "linspace: len={} prefer_gpu={} reason={:?}",
            count,
            residency.prefer_gpu,
            residency.reason
        );
    }
    let prefer_gpu = residency.prefer_gpu;
    build_sequence(start_scalar, stop_scalar, count, prefer_gpu)
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
        Value::GpuTensor(handle) => {
            let tensor = gpu_helpers::gather_tensor_async(&handle).await?;
            let scalar = tensor_scalar(name, &tensor)?;
            Ok((scalar, true))
        }
        other => parse_scalar_host(name, other),
    }
}

fn parse_scalar_host(name: &str, value: Value) -> crate::BuiltinResult<(Scalar, bool)> {
    match value {
        Value::Num(n) => Ok((Scalar::Real(n), false)),
        Value::Int(i) => Ok((Scalar::Real(i.to_f64()), false)),
        Value::Bool(b) => Ok((Scalar::Real(if b { 1.0 } else { 0.0 }), false)),
        Value::Complex(re, im) => Ok((Scalar::Complex { re, im }, false)),
        Value::Tensor(t) => tensor_scalar(name, &t).map(|scalar| (scalar, false)),
        Value::ComplexTensor(t) => complex_tensor_scalar(name, &t).map(|scalar| (scalar, false)),
        Value::GpuTensor(_) => unreachable!("GpuTensor handled by parse_scalar"),
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
        Value::GpuTensor(handle) => {
            let tensor = gpu_helpers::gather_tensor_async(handle).await?;
            if !tensor::is_scalar_tensor(&tensor) {
                return Err(builtin_error("linspace: number of points must be a scalar"));
            }
            parse_numeric_count(tensor.data[0])
        }
        other => parse_count_host(other),
    }
}

fn parse_count_host(value: &Value) -> crate::BuiltinResult<usize> {
    match value {
        Value::Int(i) => {
            let raw = i.to_i64();
            if raw < 0 {
                return Err(builtin_error("linspace: number of points must be >= 0"));
            }
            usize::try_from(raw).map_err(|_| {
                builtin_error("linspace: number of points is too large for this platform")
            })
        }
        Value::Num(n) => parse_numeric_count(*n),
        Value::Bool(b) => Ok(if *b { 1 } else { 0 }),
        Value::Tensor(t) => {
            if !tensor::is_scalar_tensor(t) {
                return Err(builtin_error("linspace: number of points must be a scalar"));
            }
            parse_numeric_count(t.data[0])
        }
        Value::GpuTensor(_) => unreachable!("GpuTensor handled by parse_count"),
        other => Err(builtin_error(format!(
            "linspace: number of points must be numeric, got {other:?}"
        ))),
    }
}

fn parse_numeric_count(raw: f64) -> crate::BuiltinResult<usize> {
    if !raw.is_finite() {
        return Err(builtin_error("linspace: number of points must be finite"));
    }
    let rounded = raw.round();
    if (rounded - raw).abs() > f64::EPSILON {
        return Err(builtin_error(
            "linspace: number of points must be an integer",
        ));
    }
    if rounded < 0.0 {
        return Err(builtin_error("linspace: number of points must be >= 0"));
    }
    if rounded > usize::MAX as f64 {
        return Err(builtin_error(
            "linspace: number of points is too large for this platform",
        ));
    }
    Ok(rounded as usize)
}

fn build_sequence(
    start: Scalar,
    stop: Scalar,
    count: usize,
    prefer_gpu: bool,
) -> crate::BuiltinResult<Value> {
    let (start_re, start_im) = start.parts();
    let (stop_re, stop_im) = stop.parts();
    let complex = start_im != 0.0 || stop_im != 0.0;

    if complex {
        let data = generate_complex_sequence(start_re, start_im, stop_re, stop_im, count);
        let tensor = ComplexTensor::new(data, vec![1, count])
            .map_err(|e| builtin_error(format!("linspace: {e}")))?;
        return Ok(Value::ComplexTensor(tensor));
    }

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
            if count > 0 {
                if log::log_enabled!(log::Level::Trace) {
                    trace!(
                        "linspace: attempting provider.linspace start={} stop={} count={}",
                        start_re,
                        stop_re,
                        count
                    );
                }
                match provider.linspace(start_re, stop_re, count) {
                    Ok(handle) => {
                        trace!("linspace: provider.linspace succeeded");
                        return Ok(Value::GpuTensor(handle));
                    }
                    Err(err) => {
                        trace!("linspace: provider.linspace failed: {err}");
                    }
                }
            }
        }
    }

    let data = generate_real_sequence(start_re, stop_re, count);
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
        Tensor::new(data, vec![1, count]).map_err(|e| builtin_error(format!("linspace: {e}")))?;
    Ok(Value::Tensor(tensor))
}

fn generate_real_sequence(start: f64, stop: f64, count: usize) -> Vec<f64> {
    if count == 0 {
        return Vec::new();
    }
    if count == 1 {
        return vec![stop];
    }
    let mut data = Vec::with_capacity(count);
    let step = (stop - start) / ((count - 1) as f64);
    for idx in 0..count {
        data.push(start + (idx as f64) * step);
    }
    if let Some(last) = data.last_mut() {
        *last = stop;
    }
    data
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
    if let Some(last) = data.last_mut() {
        *last = (stop_re, stop_im);
    }
    data
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_builtins::{IntValue, Tensor};

    fn linspace_builtin(
        start: Value,
        stop: Value,
        rest: Vec<Value>,
    ) -> crate::BuiltinResult<Value> {
        block_on(super::linspace_builtin(start, stop, rest))
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn linspace_basic() {
        let result = linspace_builtin(
            Value::Num(0.0),
            Value::Num(1.0),
            vec![Value::Int(IntValue::I32(5))],
        )
        .expect("linspace");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 5]);
                let expected = [0.0, 0.25, 0.5, 0.75, 1.0];
                for (idx, expected_val) in expected.iter().enumerate() {
                    assert!((t.data[idx] - expected_val).abs() < 1e-12);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn linspace_type_is_row_vector() {
        assert_eq!(
            linspace_type(&[Type::Num, Type::Num], &ResolveContext::new(Vec::new())),
            Type::Tensor {
                shape: Some(vec![Some(1), None])
            }
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn linspace_default_count() {
        let result =
            linspace_builtin(Value::Num(-1.0), Value::Num(1.0), Vec::new()).expect("linspace");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 100]);
                assert!((t.data.first().copied().unwrap() + 1.0).abs() < 1e-12);
                assert!((t.data.last().copied().unwrap() - 1.0).abs() < 1e-12);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn linspace_zero_count() {
        let result = linspace_builtin(
            Value::Num(0.0),
            Value::Num(10.0),
            vec![Value::Int(IntValue::I32(0))],
        )
        .expect("linspace");
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
    fn linspace_single_point() {
        let result = linspace_builtin(
            Value::Num(5.0),
            Value::Num(9.0),
            vec![Value::Int(IntValue::I32(1))],
        )
        .expect("linspace");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 1]);
                assert!((t.data[0] - 9.0).abs() < 1e-12);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn linspace_non_integer_count_errors() {
        let err = linspace_builtin(Value::Num(0.0), Value::Num(1.0), vec![Value::Num(3.5)])
            .expect_err("expected error");
        assert!(err.message().contains("integer"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn linspace_negative_count_errors() {
        let err = linspace_builtin(
            Value::Num(0.0),
            Value::Num(1.0),
            vec![Value::Int(IntValue::I32(-2))],
        )
        .expect_err("expected error");
        assert!(err.message().contains(">= 0"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn linspace_infinite_count_errors() {
        let err = linspace_builtin(
            Value::Num(0.0),
            Value::Num(1.0),
            vec![Value::Num(f64::INFINITY)],
        )
        .expect_err("expected error");
        assert!(err.message().contains("finite"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn linspace_nan_count_errors() {
        let err = linspace_builtin(Value::Num(0.0), Value::Num(1.0), vec![Value::Num(f64::NAN)])
            .expect_err("expected error");
        assert!(err.message().contains("finite"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn linspace_non_scalar_count_errors() {
        let sz = Tensor::new(vec![2.0, 3.0], vec![2, 1]).unwrap();
        let err = linspace_builtin(Value::Num(0.0), Value::Num(1.0), vec![Value::Tensor(sz)])
            .expect_err("expected error");
        assert!(err.message().contains("scalar"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn linspace_complex_sequence() {
        let result = linspace_builtin(
            Value::Complex(1.0, 1.0),
            Value::Complex(-3.0, 2.0),
            vec![Value::Int(IntValue::I32(4))],
        )
        .expect("linspace");
        match result {
            Value::ComplexTensor(t) => {
                assert_eq!(t.shape, vec![1, 4]);
                let expected = [
                    (1.0, 1.0),
                    (-0.3333333333333333, 1.3333333333333333),
                    (-1.6666666666666667, 1.6666666666666667),
                    (-3.0, 2.0),
                ];
                for (idx, &(re, im)) in expected.iter().enumerate() {
                    let (r, i) = t.data[idx];
                    assert!((r - re).abs() < 1e-9);
                    assert!((i - im).abs() < 1e-9);
                }
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn linspace_boolean_arguments_are_promoted() {
        let result = linspace_builtin(
            Value::Bool(true),
            Value::Bool(false),
            vec![Value::Int(IntValue::I32(3))],
        )
        .expect("linspace");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 3]);
                let expected = [1.0, 0.5, 0.0];
                for (idx, expected_val) in expected.iter().enumerate() {
                    assert!((t.data[idx] - expected_val).abs() < 1e-12);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn linspace_boolean_count_supported() {
        let result = linspace_builtin(Value::Num(3.0), Value::Num(7.0), vec![Value::Bool(true)])
            .expect("linspace");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 1]);
                assert!((t.data[0] - 7.0).abs() < 1e-12);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn linspace_tensor_scalar_arguments() {
        let start = Tensor::new(vec![2.0], vec![1, 1]).unwrap();
        let stop = Tensor::new(vec![4.0], vec![1, 1]).unwrap();
        let result = linspace_builtin(
            Value::Tensor(start),
            Value::Tensor(stop),
            vec![Value::Int(IntValue::I32(3))],
        )
        .expect("linspace");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 3]);
                let expected = [2.0, 3.0, 4.0];
                for (idx, expected_val) in expected.iter().enumerate() {
                    assert!((t.data[idx] - expected_val).abs() < 1e-12);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn linspace_equal_endpoints_fill_with_endpoint() {
        let result = linspace_builtin(
            Value::Num(5.0),
            Value::Num(5.0),
            vec![Value::Int(IntValue::I32(4))],
        )
        .expect("linspace");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 4]);
                assert!(t.data.iter().all(|v| (*v - 5.0).abs() < 1e-12));
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn linspace_gpu_roundtrip() {
        test_support::with_test_provider(|provider| {
            let start = Tensor::new(vec![0.0], vec![1, 1]).unwrap();
            let stop = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
            let start_view = runmat_accelerate_api::HostTensorView {
                data: &start.data,
                shape: &start.shape,
            };
            let stop_view = runmat_accelerate_api::HostTensorView {
                data: &stop.data,
                shape: &stop.shape,
            };
            let start_handle = provider.upload(&start_view).expect("upload start");
            let stop_handle = provider.upload(&stop_view).expect("upload stop");
            let result = linspace_builtin(
                Value::GpuTensor(start_handle),
                Value::GpuTensor(stop_handle),
                vec![Value::Int(IntValue::I32(5))],
            )
            .expect("linspace");
            match result {
                Value::GpuTensor(handle) => {
                    let gathered = test_support::gather(Value::GpuTensor(handle)).expect("gather");
                    let expected = [0.0, 0.25, 0.5, 0.75, 1.0];
                    assert_eq!(gathered.shape, vec![1, 5]);
                    for (idx, expected_val) in expected.iter().enumerate() {
                        assert!((gathered.data[idx] - expected_val).abs() < 1e-12);
                    }
                }
                other => panic!("expected gpu tensor, got {other:?}"),
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn linspace_gpu_zero_count_produces_gpu_empty_vector() {
        test_support::with_test_provider(|provider| {
            let start = Tensor::new(vec![0.0], vec![1, 1]).unwrap();
            let stop = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
            let start_view = runmat_accelerate_api::HostTensorView {
                data: &start.data,
                shape: &start.shape,
            };
            let stop_view = runmat_accelerate_api::HostTensorView {
                data: &stop.data,
                shape: &stop.shape,
            };
            let start_handle = provider.upload(&start_view).expect("upload start");
            let stop_handle = provider.upload(&stop_view).expect("upload stop");
            let result = linspace_builtin(
                Value::GpuTensor(start_handle),
                Value::GpuTensor(stop_handle),
                vec![Value::Int(IntValue::I32(0))],
            )
            .expect("linspace");
            match result {
                Value::GpuTensor(handle) => {
                    let gathered = test_support::gather(Value::GpuTensor(handle)).expect("gather");
                    assert_eq!(gathered.shape, vec![1, 0]);
                    assert!(gathered.data.is_empty());
                }
                other => panic!("expected gpu tensor, got {other:?}"),
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn linspace_wgpu_matches_cpu() {
        use runmat_accelerate::backend::wgpu::provider::{
            register_wgpu_provider, WgpuProviderOptions,
        };
        use runmat_accelerate_api::{AccelProvider, HostTensorView, ProviderPrecision};

        let provider =
            register_wgpu_provider(WgpuProviderOptions::default()).expect("wgpu provider");

        let start = Tensor::new(vec![0.0], vec![1, 1]).unwrap();
        let stop = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
        let start_view = HostTensorView {
            data: &start.data,
            shape: &start.shape,
        };
        let stop_view = HostTensorView {
            data: &stop.data,
            shape: &stop.shape,
        };
        let start_handle = provider.upload(&start_view).expect("upload start");
        let stop_handle = provider.upload(&stop_view).expect("upload stop");

        let result = linspace_builtin(
            Value::GpuTensor(start_handle),
            Value::GpuTensor(stop_handle),
            vec![Value::Int(IntValue::I32(9))],
        )
        .expect("linspace");
        let gathered = test_support::gather(result).expect("gather");
        let expected = generate_real_sequence(0.0, 1.0, 9);

        let precision = runmat_accelerate_api::provider()
            .expect("provider")
            .precision();
        let tol = match precision {
            ProviderPrecision::F64 => 1e-12,
            ProviderPrecision::F32 => 1e-5,
        };

        assert_eq!(gathered.shape, vec![1, 9]);
        for (idx, expected_value) in expected.iter().enumerate() {
            let actual = gathered.data[idx];
            assert!(
                (actual - expected_value).abs() <= tol,
                "mismatch at {idx}: gpu={} expected={}",
                actual,
                expected_value
            );
        }
    }
}
