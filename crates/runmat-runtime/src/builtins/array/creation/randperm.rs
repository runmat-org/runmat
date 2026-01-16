//! MATLAB-compatible `randperm` builtin with GPU-aware semantics for RunMat.

use runmat_accelerate_api::{GpuTensorHandle, HostTensorView};
use runmat_builtins::{IntValue, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::random;
use crate::builtins::common::random_args::keyword_of;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::tensor;

const MAX_SAFE_INTEGER: u64 = 1 << 53;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::array::creation::randperm")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "randperm",
    op_kind: GpuOpKind::Custom("permutation"),
    supported_precisions: &[ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[
        ProviderHook::Custom("random_permutation"),
        ProviderHook::Custom("random_permutation_like"),
    ],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Uses provider random_permutation(_like) hooks (WGPU implements a native kernel); falls back to host generation + upload when unavailable.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::array::creation::randperm")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "randperm",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Random permutation generation is treated as a sink and is not eligible for fusion.",
};

#[runtime_builtin(
    name = "randperm",
    category = "array/creation",
    summary = "Random permutations of 1:n.",
    keywords = "randperm,permutation,random,indices,gpu,like",
    accel = "array_construct",
    builtin_path = "crate::builtins::array::creation::randperm"
)]
fn randperm_builtin(args: Vec<Value>) -> Result<Value, String> {
    let parsed = ParsedRandPerm::parse(args)?;
    build_output(parsed)
}

struct ParsedRandPerm {
    n: usize,
    k: usize,
    template: OutputTemplate,
}

#[derive(Clone)]
enum OutputTemplate {
    Double,
    Like(Value),
}

impl ParsedRandPerm {
    fn parse(args: Vec<Value>) -> Result<Self, String> {
        if args.is_empty() {
            return Err("randperm: requires at least one input argument".to_string());
        }

        let n = parse_size_argument(
            &args[0],
            true,
            "randperm: N must be a non-negative integer (and <= 2^53)",
        )?;
        if n == 0 && args.len() == 1 {
            return Ok(Self {
                n,
                k: 0,
                template: OutputTemplate::Double,
            });
        }

        let mut k: Option<usize> = None;
        let mut template: OutputTemplate = OutputTemplate::Double;

        let mut idx = 1;
        while idx < args.len() {
            let arg = args[idx].clone();
            if let Some(keyword) = keyword_of(&arg) {
                match keyword.as_str() {
                    "like" => {
                        if matches!(template, OutputTemplate::Like(_)) {
                            return Err(
                                "randperm: duplicate 'like' prototype specified".to_string()
                            );
                        }
                        let Some(proto) = args.get(idx + 1).cloned() else {
                            return Err("randperm: expected prototype after 'like'".to_string());
                        };
                        template = OutputTemplate::Like(proto);
                        idx += 2;
                        continue;
                    }
                    "double" => {
                        if matches!(template, OutputTemplate::Like(_)) {
                            return Err(
                                "randperm: cannot combine 'double' with a 'like' prototype"
                                    .to_string(),
                            );
                        }
                        idx += 1;
                        continue;
                    }
                    "single" => {
                        return Err(
                            "randperm: single precision output is not implemented yet".to_string()
                        );
                    }
                    other => {
                        return Err(format!("randperm: unrecognised option '{other}'"));
                    }
                }
            }

            if k.is_none() {
                k = Some(parse_size_argument(
                    &arg,
                    true,
                    "randperm: K must be a non-negative integer (and <= N)",
                )?);
                idx += 1;
                continue;
            }

            return Err("randperm: too many input arguments".to_string());
        }

        let k = k.unwrap_or(n);

        if k > n {
            return Err("randperm: K must satisfy 0 <= K <= N".to_string());
        }

        Ok(Self { n, k, template })
    }
}

fn build_output(parsed: ParsedRandPerm) -> Result<Value, String> {
    match parsed.template {
        OutputTemplate::Double => randperm_double(parsed.n, parsed.k),
        OutputTemplate::Like(proto) => randperm_like(&proto, parsed.n, parsed.k),
    }
}

fn randperm_double(n: usize, k: usize) -> Result<Value, String> {
    let tensor = randperm_tensor(n, k)?;
    Ok(tensor::tensor_into_value(tensor))
}

fn randperm_like(proto: &Value, n: usize, k: usize) -> Result<Value, String> {
    match proto {
        Value::GpuTensor(handle) => randperm_gpu(handle, n, k),
        Value::Tensor(_) | Value::Num(_) | Value::Int(_) => randperm_double(n, k),
        Value::LogicalArray(_) => Err(
            "randperm: logical prototypes cannot represent permutation values (requires numeric output)"
                .to_string(),
        ),
        Value::Complex(_, _) | Value::ComplexTensor(_) => {
            Err("randperm: complex prototypes are not supported".to_string())
        }
        Value::Bool(_) => Err("randperm: prototypes must be numeric".to_string()),
        Value::CharArray(_) | Value::String(_) | Value::StringArray(_) => {
            Err("randperm: prototypes must be numeric".to_string())
        }
        Value::Cell(_) => Err("randperm: cell prototypes are not supported".to_string()),
        other => Err(format!("randperm: unsupported prototype {other:?}")),
    }
}

fn randperm_gpu(handle: &GpuTensorHandle, n: usize, k: usize) -> Result<Value, String> {
    if let Some(provider) = runmat_accelerate_api::provider() {
        if let Ok(device) = provider.random_permutation_like(handle, n, k) {
            return Ok(Value::GpuTensor(device));
        }
    }

    let tensor = randperm_tensor(n, k)?;
    if let Some(provider) = runmat_accelerate_api::provider() {
        let view = HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        if let Ok(device) = provider.upload(&view) {
            return Ok(Value::GpuTensor(device));
        }
    }
    Ok(tensor::tensor_into_value(tensor))
}

fn randperm_tensor(n: usize, k: usize) -> Result<Tensor, String> {
    let mut values: Vec<f64> = if n == 0 {
        Vec::new()
    } else {
        (1..=n).map(|v| v as f64).collect()
    };

    if k > 0 {
        let uniforms = random::generate_uniform(k, "randperm")?;
        for (i, u) in uniforms.into_iter().enumerate() {
            if i >= k || i >= n {
                break;
            }
            let span = n - i;
            if span == 0 {
                break;
            }
            let mut offset = (u * span as f64).floor() as usize;
            if offset >= span {
                offset = span - 1;
            }
            let j = i + offset;
            values.swap(i, j);
        }
    }

    if values.len() > k {
        values.truncate(k);
    }

    Tensor::new(values, vec![1, k]).map_err(|e| format!("randperm: {e}"))
}

fn parse_size_argument(value: &Value, allow_zero: bool, message: &str) -> Result<usize, String> {
    match value {
        Value::Int(i) => parse_intvalue(i, allow_zero, message),
        Value::Num(n) => parse_numeric(*n, allow_zero, message),
        Value::Tensor(t) => {
            if t.data.len() != 1 {
                return Err("randperm: size arguments must be scalar".to_string());
            }
            parse_numeric(t.data[0], allow_zero, message)
        }
        other => Err(format!(
            "randperm: size arguments must be numeric scalars, got {other:?}"
        )),
    }
}

fn parse_intvalue(value: &IntValue, allow_zero: bool, message: &str) -> Result<usize, String> {
    let raw = match value {
        IntValue::I8(v) => *v as i128,
        IntValue::I16(v) => *v as i128,
        IntValue::I32(v) => *v as i128,
        IntValue::I64(v) => *v as i128,
        IntValue::U8(v) => *v as i128,
        IntValue::U16(v) => *v as i128,
        IntValue::U32(v) => *v as i128,
        IntValue::U64(v) => *v as i128,
    };
    if raw < 0 {
        return Err(message.to_string());
    }
    if !allow_zero && raw == 0 {
        return Err(message.to_string());
    }
    if raw as u128 > MAX_SAFE_INTEGER as u128 {
        return Err("randperm: values larger than 2^53 are not supported".to_string());
    }
    if raw > usize::MAX as i128 {
        return Err("randperm: input exceeds platform limits".to_string());
    }
    Ok(raw as usize)
}

fn parse_numeric(value: f64, allow_zero: bool, message: &str) -> Result<usize, String> {
    if !value.is_finite() {
        return Err(message.to_string());
    }
    let rounded = value.round();
    if (rounded - value).abs() > f64::EPSILON {
        return Err(message.to_string());
    }
    if rounded < 0.0 {
        return Err(message.to_string());
    }
    if !allow_zero && rounded == 0.0 {
        return Err(message.to_string());
    }
    if rounded > MAX_SAFE_INTEGER as f64 {
        return Err("randperm: values larger than 2^53 are not supported".to_string());
    }
    if rounded > usize::MAX as f64 {
        return Err("randperm: input exceeds platform limits".to_string());
    }
    Ok(rounded as usize)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::{random, test_support};

    fn reset_rng_clean() {
        runmat_accelerate_api::clear_provider();
        random::reset_rng();
    }

    fn expected_randperm(n: usize, k: usize) -> Vec<f64> {
        let mut values: Vec<f64> = if n == 0 {
            Vec::new()
        } else {
            (1..=n).map(|v| v as f64).collect()
        };
        if k > 0 {
            let uniforms = random::expected_uniform_sequence(k);
            for (i, u) in uniforms.iter().copied().enumerate() {
                if i >= k || i >= n {
                    break;
                }
                let span = n - i;
                if span == 0 {
                    break;
                }
                let mut offset = (u * span as f64).floor() as usize;
                if offset >= span {
                    offset = span - 1;
                }
                let j = i + offset;
                values.swap(i, j);
            }
        }
        if values.len() > k {
            values.truncate(k);
        }
        values
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn randperm_full_permutation_matches_expected_sequence() {
        let _guard = random::test_lock().lock().unwrap();
        reset_rng_clean();
        let args = vec![Value::from(5)];
        let result = randperm_builtin(args).expect("randperm");
        let gathered = test_support::gather(result).expect("gather");
        assert_eq!(gathered.shape, vec![1, 5]);
        let expected = expected_randperm(5, 5);
        assert_eq!(gathered.data, expected);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn randperm_partial_selection_is_unique_and_sorted() {
        let _guard = random::test_lock().lock().unwrap();
        reset_rng_clean();
        let args = vec![Value::from(10), Value::from(4)];
        let result = randperm_builtin(args).expect("randperm");
        let gathered = test_support::gather(result).expect("gather");
        assert_eq!(gathered.shape, vec![1, 4]);
        let data = gathered.data;
        let expected = expected_randperm(10, 4);
        assert_eq!(data, expected);
        let mut sorted = data.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        sorted.dedup();
        assert_eq!(sorted.len(), 4);
        for value in expected {
            assert!((1.0..=10.0).contains(&value));
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn randperm_zero_length_returns_empty() {
        let args = vec![Value::from(0)];
        let result = randperm_builtin(args).expect("randperm");
        let gathered = test_support::gather(result).expect("gather");
        assert_eq!(gathered.shape, vec![1, 0]);
        assert!(gathered.data.is_empty());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn randperm_errors_when_k_exceeds_n() {
        let args = vec![Value::from(3), Value::from(4)];
        let err = randperm_builtin(args).unwrap_err();
        assert!(err.contains("K must satisfy 0 <= K <= N"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn randperm_errors_for_negative_input() {
        let args = vec![Value::Num(-1.0)];
        let err = randperm_builtin(args).unwrap_err();
        assert!(err.contains("N must be a non-negative integer"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn randperm_rejects_single_precision_request() {
        let args = vec![Value::from(5), Value::from("single")];
        let err = randperm_builtin(args).unwrap_err();
        assert!(err.contains("single precision"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn randperm_accepts_double_keyword() {
        let _guard = random::test_lock().lock().unwrap();
        reset_rng_clean();
        let args = vec![Value::from(5), Value::from("double")];
        let result = randperm_builtin(args).expect("randperm");
        let gathered = test_support::gather(result).expect("gather");
        assert_eq!(gathered.shape, vec![1, 5]);
        let expected = expected_randperm(5, 5);
        assert_eq!(gathered.data, expected);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn randperm_like_tensor_matches_host_output() {
        let _guard = random::test_lock().lock().unwrap();
        reset_rng_clean();
        let proto_tensor = Tensor::new(vec![0.0, 0.0], vec![1, 2]).unwrap();
        let args = vec![
            Value::from(4),
            Value::from("like"),
            Value::Tensor(proto_tensor),
        ];
        let result = randperm_builtin(args).expect("randperm");
        let gathered = test_support::gather(result).expect("gather");
        assert_eq!(gathered.shape, vec![1, 4]);
        let expected = expected_randperm(4, 4);
        assert_eq!(gathered.data, expected);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn randperm_gpu_like_roundtrip() {
        let _guard = random::test_lock().lock().unwrap();
        random::reset_rng();
        test_support::with_test_provider(|provider| {
            let proto_tensor = Tensor::new(vec![0.0, 0.0], vec![1, 2]).unwrap();
            let view = HostTensorView {
                data: &proto_tensor.data,
                shape: &proto_tensor.shape,
            };
            let proto_handle = provider.upload(&view).expect("upload prototype");
            let args = vec![
                Value::from(6),
                Value::from(3),
                Value::from("like"),
                Value::GpuTensor(proto_handle.clone()),
            ];
            let result = randperm_builtin(args).expect("randperm");
            match &result {
                Value::GpuTensor(_) => {}
                other => panic!("expected GPU tensor, got {other:?}"),
            }
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![1, 3]);
            let expected = expected_randperm(6, 3);
            assert_eq!(gathered.data, expected);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn randperm_like_requires_prototype() {
        let args = vec![Value::from(4), Value::from("like")];
        let err = randperm_builtin(args).unwrap_err();
        assert!(err.contains("prototype after 'like'"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn randperm_wgpu_produces_unique_indices() {
        let _guard = random::test_lock().lock().unwrap();
        random::reset_rng();
        use runmat_accelerate::backend::wgpu::provider::{
            register_wgpu_provider, WgpuProviderOptions,
        };
        use runmat_accelerate_api::HostTensorView;

        let registration =
            std::panic::catch_unwind(|| register_wgpu_provider(WgpuProviderOptions::default()));
        let provider = match registration {
            Ok(Ok(_)) => runmat_accelerate_api::provider().expect("wgpu provider registered"),
            Ok(Err(err)) => {
                tracing::warn!("skipping wgpu randperm test: {err}");
                return;
            }
            Err(_) => {
                tracing::warn!("skipping wgpu randperm test: provider initialization panicked");
                return;
            }
        };

        let proto_data = [0.0];
        let proto_shape = [1usize, 1];
        let proto_view = HostTensorView {
            data: &proto_data,
            shape: &proto_shape,
        };
        let proto_handle = provider.upload(&proto_view).expect("upload prototype");

        let args = vec![
            Value::from(12),
            Value::from(7),
            Value::from("like"),
            Value::GpuTensor(proto_handle),
        ];
        let result = randperm_builtin(args).expect("randperm");
        let gpu_handle = match result {
            Value::GpuTensor(ref h) => h.clone(),
            other => panic!("expected GPU tensor result, got {other:?}"),
        };

        let host = provider
            .download(&gpu_handle)
            .expect("download permutation");
        assert_eq!(host.shape, vec![1, 7]);
        assert_eq!(host.data.len(), 7);

        let mut sorted = host.data.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        for window in sorted.windows(2) {
            assert_ne!(
                window[0], window[1],
                "duplicate value detected in permutation"
            );
        }
        for value in host.data {
            assert!(
                (1.0..=12.0).contains(&value),
                "value {value} outside expected range 1..12"
            );
        }
    }
}
