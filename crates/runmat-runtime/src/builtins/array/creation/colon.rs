//! MATLAB-compatible `colon` builtin with GPU-aware semantics for RunMat.

use runmat_accelerate_api::HostTensorView;
use runmat_builtins::{CharArray, ComplexTensor, LogicalArray, Tensor, Type, Value};
use runmat_macros::runtime_builtin;

use crate::build_runtime_error;
use crate::builtins::array::type_resolvers::row_vector_type;
use runmat_builtins::ResolveContext;
use crate::builtins::common::residency::{sequence_gpu_preference, SequenceIntent};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};

const MIN_RATIO_TOL: f64 = f64::EPSILON * 8.0;
const MAX_RATIO_TOL: f64 = 1e-9;
const ZERO_IM_TOL: f64 = f64::EPSILON * 32.0;
const CHAR_TOL: f64 = 1e-6;

#[derive(Clone, Copy, PartialEq, Eq)]
enum ScalarOrigin {
    Numeric,
    Char,
}

#[derive(Clone, Copy)]
struct ParsedScalar {
    value: f64,
    prefer_gpu: bool,
    origin: ScalarOrigin,
}

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::array::creation::colon")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "colon",
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
    notes: "Falls back to uploading the host-generated vector when provider linspace kernels are unavailable.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::array::creation::colon")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "colon",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Sequence generation is treated as a sink; it does not participate in fusion.",
};

fn colon_type(_args: &[Type], ctx: &ResolveContext) -> Type {
    row_vector_type(ctx)
}

fn builtin_error(message: impl Into<String>) -> crate::RuntimeError {
    build_runtime_error(message).with_builtin("colon").build()
}

#[runtime_builtin(
    name = "colon",
    category = "array/creation",
    summary = "Arithmetic progression that mirrors MATLAB's colon operator.",
    keywords = "colon,sequence,range,step,gpu",
    accel = "array_construct",
    type_resolver(colon_type),
    type_resolver_context = true,
    builtin_path = "crate::builtins::array::creation::colon"
)]
async fn colon_builtin(
    start: Value,
    step_or_end: Value,
    rest: Vec<Value>,
) -> crate::BuiltinResult<Value> {
    if rest.len() > 1 {
        return Err(builtin_error(
            "colon: expected two or three input arguments",
        ));
    }

    let start_scalar = parse_real_scalar("colon", start).await?;

    if rest.is_empty() {
        let stop_scalar = parse_real_scalar("colon", step_or_end).await?;
        let step = default_step(start_scalar.value, stop_scalar.value);
        let char_mode =
            start_scalar.origin == ScalarOrigin::Char && stop_scalar.origin == ScalarOrigin::Char;
        let explicit_gpu = if char_mode {
            false
        } else {
            start_scalar.prefer_gpu || stop_scalar.prefer_gpu
        };
        build_sequence(
            start_scalar.value,
            step,
            stop_scalar.value,
            explicit_gpu,
            char_mode,
        )
    } else {
        let step_scalar = parse_real_scalar("colon", step_or_end).await?;
        if step_scalar.value == 0.0 {
            return Err(builtin_error("colon: increment must be nonzero"));
        }
        let stop_scalar = parse_real_scalar("colon", rest[0].clone()).await?;
        let char_mode =
            start_scalar.origin == ScalarOrigin::Char && stop_scalar.origin == ScalarOrigin::Char;
        let explicit_gpu = if char_mode {
            false
        } else {
            start_scalar.prefer_gpu || step_scalar.prefer_gpu || stop_scalar.prefer_gpu
        };
        build_sequence(
            start_scalar.value,
            step_scalar.value,
            stop_scalar.value,
            explicit_gpu,
            char_mode,
        )
    }
}

fn build_sequence(
    start: f64,
    step: f64,
    stop: f64,
    explicit_gpu: bool,
    char_mode: bool,
) -> crate::BuiltinResult<Value> {
    if !start.is_finite() || !step.is_finite() || !stop.is_finite() {
        return Err(builtin_error(
            "colon: inputs must be finite numeric scalars",
        ));
    }
    if step == 0.0 {
        return Err(builtin_error("colon: increment must be nonzero"));
    }

    let plan = plan_progression(start, step, stop)?;

    if char_mode {
        let data = materialize_progression(&plan, start, step);
        return build_char_sequence(data);
    }

    if plan.count == 0 {
        return finalize_numeric_sequence(Vec::new(), explicit_gpu);
    }

    let prefer_gpu =
        sequence_gpu_preference(plan.count, SequenceIntent::Colon, explicit_gpu).prefer_gpu;

    if prefer_gpu {
        #[cfg(all(test, feature = "wgpu"))]
        {
            let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
                runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
            );
        }
        if let Some(provider) = runmat_accelerate_api::provider() {
            if let Ok(handle) = provider.linspace(start, plan.final_end, plan.count) {
                return Ok(Value::GpuTensor(handle));
            }
        }
    }

    let data = materialize_progression(&plan, start, step);
    finalize_numeric_sequence(data, prefer_gpu)
}

fn finalize_numeric_sequence(data: Vec<f64>, prefer_gpu: bool) -> crate::BuiltinResult<Value> {
    let len = data.len();
    let shape = vec![1usize, len];

    if prefer_gpu {
        #[cfg(all(test, feature = "wgpu"))]
        {
            let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
                runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
            );
        }
        if let Some(provider) = runmat_accelerate_api::provider() {
            let view = HostTensorView {
                data: &data,
                shape: &shape,
            };
            if let Ok(handle) = provider.upload(&view) {
                return Ok(Value::GpuTensor(handle));
            }
        }
    }

    Tensor::new(data, shape)
        .map(tensor::tensor_into_value)
        .map_err(|e| builtin_error(format!("colon: {e}")))
}

struct ProgressionPlan {
    count: usize,
    final_end: f64,
}

fn plan_progression(start: f64, step: f64, stop: f64) -> crate::BuiltinResult<ProgressionPlan> {
    let tol = tolerance(start, step, stop);
    let step_abs = step.abs();

    if step > 0.0 && start > stop + tol {
        return Ok(ProgressionPlan {
            count: 0,
            final_end: start,
        });
    }
    if step < 0.0 && start < stop - tol {
        return Ok(ProgressionPlan {
            count: 0,
            final_end: start,
        });
    }

    let diff = (stop - start) / step;
    if !diff.is_finite() {
        return Err(builtin_error(
            "colon: sequence length exceeds representable range",
        ));
    }

    let ratio_raw = (tol / step_abs).abs();
    let ratio_tol = ratio_raw
        .max(MIN_RATIO_TOL)
        .clamp(f64::EPSILON, MAX_RATIO_TOL);
    let mut approx = diff + ratio_tol;

    if approx < 0.0 {
        if approx.abs() <= ratio_tol {
            approx = 0.0;
        } else {
            return Ok(ProgressionPlan {
                count: 0,
                final_end: start,
            });
        }
    }

    if approx.is_infinite() || approx > usize::MAX as f64 {
        return Err(builtin_error(
            "colon: sequence length exceeds platform limits",
        ));
    }

    let floor = approx.floor();
    let count = floor as usize;
    let count = count
        .checked_add(1)
        .ok_or_else(|| builtin_error("colon: sequence length exceeds platform limits"))?;

    if count == 0 {
        return Ok(ProgressionPlan {
            count: 0,
            final_end: start,
        });
    }

    let computed_end = start + step * ((count - 1) as f64);
    let final_end = if (computed_end - stop).abs() <= tol {
        stop
    } else {
        computed_end
    };

    Ok(ProgressionPlan { count, final_end })
}

fn materialize_progression(plan: &ProgressionPlan, start: f64, step: f64) -> Vec<f64> {
    if plan.count == 0 {
        return Vec::new();
    }
    let mut data = Vec::with_capacity(plan.count);
    for idx in 0..plan.count {
        data.push(start + step * (idx as f64));
    }
    if let Some(last) = data.last_mut() {
        *last = plan.final_end;
    }
    data
}

fn default_step(start: f64, stop: f64) -> f64 {
    if stop >= start {
        1.0
    } else {
        -1.0
    }
}

fn tolerance(start: f64, step: f64, stop: f64) -> f64 {
    let span = (stop - start).abs();
    let base = start.abs().max(stop.abs()).max(span).max(1.0);
    let step_term = step.abs().max(1.0);
    let tol = base * f64::EPSILON * 32.0 + step_term * f64::EPSILON * 16.0;
    tol.max(f64::EPSILON)
}

async fn parse_real_scalar(name: &str, value: Value) -> crate::BuiltinResult<ParsedScalar> {
    match value {
        Value::GpuTensor(handle) => {
            let tensor = gpu_helpers::gather_tensor_async(&handle).await?;
            let scalar = tensor_scalar(name, &tensor)?;
            Ok(ParsedScalar {
                value: scalar,
                prefer_gpu: true,
                origin: ScalarOrigin::Numeric,
            })
        }
        other => parse_real_scalar_host(name, other),
    }
}

fn parse_real_scalar_host(name: &str, value: Value) -> crate::BuiltinResult<ParsedScalar> {
    match value {
        Value::Num(n) => ensure_finite(name, n).map(|v| ParsedScalar {
            value: v,
            prefer_gpu: false,
            origin: ScalarOrigin::Numeric,
        }),
        Value::Int(i) => Ok(ParsedScalar {
            value: i.to_f64(),
            prefer_gpu: false,
            origin: ScalarOrigin::Numeric,
        }),
        Value::Bool(b) => Ok(ParsedScalar {
            value: if b { 1.0 } else { 0.0 },
            prefer_gpu: false,
            origin: ScalarOrigin::Numeric,
        }),
        Value::Tensor(t) => tensor_scalar(name, &t).map(|v| ParsedScalar {
            value: v,
            prefer_gpu: false,
            origin: ScalarOrigin::Numeric,
        }),
        Value::LogicalArray(logical) => logical_scalar(name, &logical).map(|v| ParsedScalar {
            value: v,
            prefer_gpu: false,
            origin: ScalarOrigin::Numeric,
        }),
        Value::Complex(re, im) => complex_to_real(name, re, im).map(|v| ParsedScalar {
            value: v,
            prefer_gpu: false,
            origin: ScalarOrigin::Numeric,
        }),
        Value::ComplexTensor(t) => complex_tensor_scalar(name, &t).map(|v| ParsedScalar {
            value: v,
            prefer_gpu: false,
            origin: ScalarOrigin::Numeric,
        }),
        Value::CharArray(chars) => char_scalar(name, &chars).map(|v| ParsedScalar {
            value: v,
            prefer_gpu: false,
            origin: ScalarOrigin::Char,
        }),
        Value::String(_) | Value::StringArray(_) => Err(builtin_error(format!(
            "{name}: inputs must be real scalar values; received a string-like argument"
        ))),
        Value::GpuTensor(_) => unreachable!("GpuTensor handled by parse_real_scalar"),
        other => Err(builtin_error(format!(
            "{name}: inputs must be real scalar values; received {other:?}"
        ))),
    }
}

fn ensure_finite(name: &str, value: f64) -> crate::BuiltinResult<f64> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(builtin_error(format!(
            "{name}: inputs must be finite numeric scalars"
        )))
    }
}

fn tensor_scalar(name: &str, tensor: &Tensor) -> crate::BuiltinResult<f64> {
    if !tensor::is_scalar_tensor(tensor) {
        return Err(builtin_error(format!("{name}: expected scalar input")));
    }
    ensure_finite(name, tensor.data[0])
}

fn logical_scalar(name: &str, logical: &LogicalArray) -> crate::BuiltinResult<f64> {
    if logical.len() != 1 {
        return Err(builtin_error(format!("{name}: expected scalar input")));
    }
    Ok(if logical.data[0] != 0 { 1.0 } else { 0.0 })
}

fn complex_to_real(name: &str, re: f64, im: f64) -> crate::BuiltinResult<f64> {
    if im.abs() > ZERO_IM_TOL * re.abs().max(1.0) {
        return Err(builtin_error(format!(
            "{name}: complex inputs must have zero imaginary part"
        )));
    }
    ensure_finite(name, re)
}

fn complex_tensor_scalar(name: &str, tensor: &ComplexTensor) -> crate::BuiltinResult<f64> {
    if tensor.data.len() != 1 {
        return Err(builtin_error(format!("{name}: expected scalar input")));
    }
    let (re, im) = tensor.data[0];
    complex_to_real(name, re, im)
}

fn char_scalar(name: &str, array: &CharArray) -> crate::BuiltinResult<f64> {
    if array.rows * array.cols != 1 {
        return Err(builtin_error(format!("{name}: expected scalar input")));
    }
    let ch = array.data[0];
    Ok(ch as u32 as f64)
}

fn build_char_sequence(data: Vec<f64>) -> crate::BuiltinResult<Value> {
    let len = data.len();
    let mut chars = Vec::with_capacity(len);
    for value in data {
        let rounded = value.round();
        if (value - rounded).abs() > CHAR_TOL {
            return Err(builtin_error(
                "colon: character sequence requires integer code points",
            ));
        }
        if !(0.0..=(u32::MAX as f64)).contains(&rounded) {
            return Err(builtin_error("colon: character code point out of range"));
        }
        let code = rounded as u32;
        let ch = std::char::from_u32(code)
            .ok_or_else(|| builtin_error("colon: character code point out of range"))?;
        chars.push(ch);
    }

    let array = CharArray::new(chars, 1, len).map_err(|e| builtin_error(format!("colon: {e}")))?;
    Ok(Value::CharArray(array))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_builtins::{CharArray, Tensor};

    fn colon_builtin(start: Value, stop: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
        block_on(super::colon_builtin(start, stop, rest))
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn colon_basic_increasing() {
        let result = colon_builtin(Value::Num(1.0), Value::Num(5.0), Vec::new()).expect("colon");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 5]);
                assert_eq!(t.data, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn colon_type_is_row_vector() {
        assert_eq!(
            colon_type(&[Type::Num, Type::Num], &ResolveContext::empty()),
            Type::Tensor {
                shape: Some(vec![Some(1), None])
            }
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn colon_basic_descending() {
        let result = colon_builtin(Value::Num(5.0), Value::Num(1.0), Vec::new()).expect("colon");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 5]);
                assert_eq!(t.data, vec![5.0, 4.0, 3.0, 2.0, 1.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn colon_custom_step_reaches_stop() {
        let result =
            colon_builtin(Value::Num(0.0), Value::Num(0.5), vec![Value::Num(2.0)]).expect("colon");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 5]);
                assert_eq!(t.data, vec![0.0, 0.5, 1.0, 1.5, 2.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn colon_custom_step_stops_before_bound() {
        let result =
            colon_builtin(Value::Num(0.0), Value::Num(2.0), vec![Value::Num(5.0)]).expect("colon");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 3]);
                assert_eq!(t.data, vec![0.0, 2.0, 4.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn colon_sign_mismatch_returns_empty() {
        let result =
            colon_builtin(Value::Num(0.0), Value::Num(1.0), vec![Value::Num(-1.0)]).expect("colon");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 0]);
                assert!(t.data.is_empty());
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn colon_zero_increment_errors() {
        let err = colon_builtin(Value::Num(0.0), Value::Num(0.0), vec![Value::Num(1.0)])
            .expect_err("colon should error");
        assert!(err.message().contains("increment must be nonzero"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn colon_accepts_scalar_tensors() {
        let start = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
        let stop = Tensor::new(vec![3.0], vec![1, 1]).unwrap();
        let result =
            colon_builtin(Value::Tensor(start), Value::Tensor(stop), Vec::new()).expect("colon");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.data, vec![1.0, 2.0, 3.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn colon_gpu_roundtrip() {
        test_support::with_test_provider(|provider| {
            let start = Tensor::new(vec![0.0], vec![1, 1]).unwrap();
            let start_view = HostTensorView {
                data: &start.data,
                shape: &start.shape,
            };
            let start_handle = provider.upload(&start_view).expect("upload start");

            let result = colon_builtin(
                Value::GpuTensor(start_handle),
                Value::Num(0.5),
                vec![Value::Num(2.0)],
            )
            .expect("colon");

            match result {
                Value::GpuTensor(handle) => {
                    let gathered = test_support::gather(Value::GpuTensor(handle)).expect("gather");
                    assert_eq!(gathered.shape, vec![1, 5]);
                    assert_eq!(gathered.data, vec![0.0, 0.5, 1.0, 1.5, 2.0]);
                }
                other => panic!("expected GPU tensor, got {other:?}"),
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn colon_wgpu_matches_cpu() {
        use runmat_accelerate::backend::wgpu::provider::{
            register_wgpu_provider, WgpuProviderOptions,
        };

        let _ = register_wgpu_provider(WgpuProviderOptions::default());

        let cpu = colon_builtin(Value::Num(-2.0), Value::Num(0.5), vec![Value::Num(1.0)])
            .expect("colon host");

        let provider = runmat_accelerate_api::provider().expect("wgpu provider");
        let start = Tensor::new(vec![-2.0], vec![1, 1]).unwrap();
        let start_view = HostTensorView {
            data: &start.data,
            shape: &start.shape,
        };
        let start_handle = provider.upload(&start_view).expect("upload start");
        let gpu = colon_builtin(
            Value::GpuTensor(start_handle),
            Value::Num(0.5),
            vec![Value::Num(1.0)],
        )
        .expect("colon gpu");

        let gathered = match gpu {
            Value::GpuTensor(handle) => {
                test_support::gather(Value::GpuTensor(handle)).expect("gather gpu")
            }
            other => panic!("expected GPU tensor, got {other:?}"),
        };

        let expected = match cpu {
            Value::Tensor(t) => t,
            Value::Num(n) => Tensor::new(vec![n], vec![1, 1]).unwrap(),
            other => panic!("unexpected CPU result {other:?}"),
        };

        assert_eq!(gathered.shape, expected.shape);
        assert_eq!(gathered.data, expected.data);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn colon_bool_inputs_promote() {
        let result =
            colon_builtin(Value::Bool(false), Value::Bool(true), Vec::new()).expect("colon");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.data, vec![0.0, 1.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn colon_char_increasing() {
        let start = Value::CharArray(CharArray::new_row("a"));
        let stop = Value::CharArray(CharArray::new_row("e"));
        let result = colon_builtin(start, stop, Vec::new()).expect("colon");
        match result {
            Value::CharArray(arr) => {
                assert_eq!(arr.rows, 1);
                assert_eq!(arr.cols, 5);
                let expected: Vec<char> = "abcde".chars().collect();
                assert_eq!(arr.data, expected);
            }
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn colon_char_with_step() {
        let start = Value::CharArray(CharArray::new_row("a"));
        let step = Value::Num(2.0);
        let stop = Value::CharArray(CharArray::new_row("g"));
        let result = colon_builtin(start, step, vec![stop]).expect("colon");
        match result {
            Value::CharArray(arr) => {
                assert_eq!(arr.rows, 1);
                assert_eq!(arr.cols, 4);
                let expected: Vec<char> = "aceg".chars().collect();
                assert_eq!(arr.data, expected);
            }
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn colon_equal_endpoints_singleton() {
        let result = colon_builtin(Value::Num(3.0), Value::Num(3.0), Vec::new()).expect("colon");
        match result {
            Value::Num(v) => assert!((v - 3.0).abs() < 1e-12),
            Value::Tensor(t) => {
                assert_eq!(t.data, vec![3.0]);
            }
            other => panic!("expected scalar-compatible result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn colon_complex_imaginary_errors() {
        let err = colon_builtin(Value::Complex(1.0, 1e-2), Value::Num(2.0), Vec::new())
            .expect_err("colon should reject complex inputs");
        assert!(
            err.message().contains("zero imaginary part"),
            "unexpected error message: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn colon_string_input_errors() {
        let err = colon_builtin(Value::from("hello"), Value::Num(2.0), Vec::new())
            .expect_err("colon should reject string inputs");
        assert!(
            err.message().contains("string-like"),
            "unexpected error message: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn colon_char_descending() {
        let start = Value::CharArray(CharArray::new_row("f"));
        let stop = Value::CharArray(CharArray::new_row("b"));
        let result = colon_builtin(start, stop, Vec::new()).expect("colon");
        match result {
            Value::CharArray(arr) => {
                assert_eq!(arr.rows, 1);
                assert_eq!(arr.cols, 5);
                let expected: Vec<char> = "fedcb".chars().collect();
                assert_eq!(arr.data, expected);
            }
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn colon_char_fractional_step_errors() {
        let start = Value::CharArray(CharArray::new_row("a"));
        let stop = Value::CharArray(CharArray::new_row("d"));
        let err = colon_builtin(start, Value::Num(1.5), vec![stop])
            .expect_err("colon should reject fractional char steps");
        assert!(
            err.message()
                .contains("character sequence requires integer"),
            "unexpected error message: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn colon_gpu_step_scalar_residency() {
        test_support::with_test_provider(|provider| {
            let step = Tensor::new(vec![0.5], vec![1, 1]).unwrap();
            let view = HostTensorView {
                data: &step.data,
                shape: &step.shape,
            };
            let step_handle = provider.upload(&view).expect("upload step");
            let result = colon_builtin(
                Value::Num(0.0),
                Value::GpuTensor(step_handle),
                vec![Value::Num(2.0)],
            )
            .expect("colon");
            match result {
                Value::GpuTensor(handle) => {
                    let gathered = test_support::gather(Value::GpuTensor(handle)).expect("gather");
                    assert_eq!(gathered.data, vec![0.0, 0.5, 1.0, 1.5, 2.0]);
                }
                other => panic!("expected GPU tensor, got {other:?}"),
            }
        });
    }
}
