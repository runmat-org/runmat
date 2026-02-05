//! MATLAB-compatible `rand` builtin with GPU-aware semantics for RunMat.

use runmat_accelerate_api::{GpuTensorHandle, HostTensorView, ProviderPrecision};
use runmat_builtins::{ComplexTensor, NumericDType, Tensor, Value};
use runmat_macros::runtime_builtin;
use std::sync::OnceLock;

use crate::build_runtime_error;
use crate::builtins::common::random;
use crate::builtins::common::random_args::{
    complex_tensor_into_value, extract_dims, keyword_of, shape_from_value,
};
use crate::builtins::array::type_resolvers::tensor_type_from_rank;
use runmat_builtins::ResolveContext;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::tensor;
use runmat_builtins::Type;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::array::creation::rand")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "rand",
    op_kind: GpuOpKind::Custom("generator"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[
        ProviderHook::Custom("random_uniform"),
        ProviderHook::Custom("random_uniform_like"),
    ],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Delegates to provider random_uniform hooks; falls back to host sampling + upload when hooks are unavailable.",
};

fn builtin_error(message: impl Into<String>) -> crate::RuntimeError {
    build_runtime_error(message).with_builtin("rand").build()
}

fn rand_type(args: &[Type], ctx: &ResolveContext) -> Type {
    if args.is_empty() {
        return Type::Num;
    }
    if args.iter().any(|arg| matches!(arg, Type::String)) {
        return Type::Unknown;
    }
    tensor_type_from_rank(args, ctx)
}

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::array::creation::rand")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "rand",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Random generation is treated as a sink and is not eligible for fusion.",
};

#[runtime_builtin(
    name = "rand",
    category = "array/creation",
    summary = "Uniform random numbers on (0, 1).",
    keywords = "rand,random,uniform,gpu,like",
    accel = "array_construct",
    type_resolver(rand_type),
    type_resolver_context = true,
    builtin_path = "crate::builtins::array::creation::rand"
)]
async fn rand_builtin(rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    let parsed = ParsedRand::parse(rest).await?;
    build_output(parsed).await
}

struct ParsedRand {
    shape: Vec<usize>,
    template: RandTemplate,
}

#[derive(Clone)]
enum RandTemplate {
    Double,
    Single,
    Like(Value),
}

impl ParsedRand {
    async fn parse(args: Vec<Value>) -> crate::BuiltinResult<Self> {
        let mut dims: Vec<usize> = Vec::new();
        let mut saw_dims_arg = false;
        let mut shape_source: Option<Vec<usize>> = None;
        let mut template: Option<RandTemplate> = None;

        let mut idx = 0;
        while idx < args.len() {
            let arg = args[idx].clone();

            if let Some(keyword) = keyword_of(&arg) {
                match keyword.as_str() {
                    "like" => {
                        let Some(proto) = args.get(idx + 1).cloned() else {
                            return Err(builtin_error("rand: expected prototype after 'like'"));
                        };
                        template = Some(RandTemplate::Like(proto.clone()));
                        shape_source = Some(shape_from_value(&proto, "rand")?);
                        idx += 2;
                        continue;
                    }
                    "double" => {
                        template = Some(RandTemplate::Double);
                        idx += 1;
                        continue;
                    }
                    "single" => {
                        template = Some(RandTemplate::Single);
                        idx += 1;
                        continue;
                    }
                    other => {
                        return Err(builtin_error(format!(
                            "rand: unrecognised option '{other}'"
                        )));
                    }
                }
            }

            if let Some(parsed_dims) = extract_dims(&arg, "rand").await? {
                saw_dims_arg = true;
                if dims.is_empty() {
                    dims = parsed_dims;
                } else {
                    dims.extend(parsed_dims);
                }
                idx += 1;
                continue;
            }

            if shape_source.is_none() {
                shape_source = Some(shape_from_value(&arg, "rand")?);
            }
            if template.is_none() {
                template = Some(RandTemplate::Like(arg.clone()));
            }
            idx += 1;
        }

        let shape = if saw_dims_arg {
            if dims.is_empty() {
                vec![0, 0]
            } else if dims.len() == 1 {
                vec![dims[0], dims[0]]
            } else {
                dims
            }
        } else if let Some(shape) = shape_source {
            shape
        } else {
            vec![1, 1]
        };

        let template = template.unwrap_or(RandTemplate::Double);

        Ok(Self { shape, template })
    }
}

async fn build_output(parsed: ParsedRand) -> crate::BuiltinResult<Value> {
    match parsed.template {
        RandTemplate::Double => rand_double(&parsed.shape),
        RandTemplate::Single => rand_single(&parsed.shape),
        RandTemplate::Like(proto) => rand_like(&proto, &parsed.shape).await,
    }
}

fn rand_double(shape: &[usize]) -> crate::BuiltinResult<Value> {
    if let Some(value) = try_gpu_uniform(shape, NumericDType::F64)? {
        return Ok(value);
    }
    let len = tensor::element_count(shape);
    let data = random::generate_uniform(len, "rand")?;
    let tensor =
        Tensor::new(data, shape.to_vec()).map_err(|e| builtin_error(format!("rand: {e}")))?;
    Ok(tensor::tensor_into_value(tensor))
}

#[async_recursion::async_recursion(?Send)]
async fn rand_like(proto: &Value, shape: &[usize]) -> crate::BuiltinResult<Value> {
    match proto {
        Value::GpuTensor(handle) => rand_like_gpu(handle, shape).await,
        Value::ComplexTensor(_) | Value::Complex(_, _) => rand_complex(shape),
        Value::Tensor(t) => match t.dtype {
            NumericDType::F32 => rand_single(shape),
            NumericDType::F64 => rand_double(shape),
        },
        Value::Num(_) | Value::Int(_) | Value::Bool(_) | Value::LogicalArray(_) => {
            rand_double(shape)
        }
        Value::CharArray(_) | Value::Cell(_) => rand_double(shape),
        other => Err(builtin_error(format!(
            "rand: unsupported prototype {other:?}"
        ))),
    }
}

fn rand_single(shape: &[usize]) -> crate::BuiltinResult<Value> {
    if let Some(value) = try_gpu_uniform(shape, NumericDType::F32)? {
        return Ok(value);
    }
    let len = tensor::element_count(shape);
    let data = random::generate_uniform_single(len, "rand")?;
    let tensor = Tensor::new_with_dtype(data, shape.to_vec(), NumericDType::F32)
        .map_err(|e| builtin_error(format!("rand: {e}")))?;
    Ok(tensor::tensor_into_value(tensor))
}

fn rand_complex(shape: &[usize]) -> crate::BuiltinResult<Value> {
    let len = tensor::element_count(shape);
    let data = random::generate_complex(len, "rand")?;
    let tensor = ComplexTensor::new(data, shape.to_vec())
        .map_err(|e| builtin_error(format!("rand: {e}")))?;
    Ok(complex_tensor_into_value(tensor))
}

#[async_recursion::async_recursion(?Send)]
async fn rand_like_gpu(handle: &GpuTensorHandle, shape: &[usize]) -> crate::BuiltinResult<Value> {
    if let Some(provider) = runmat_accelerate_api::provider() {
        let precision =
            runmat_accelerate_api::handle_precision(handle).unwrap_or_else(|| provider.precision());
        let dtype = dtype_from_precision(precision);
        let attempt = if handle.shape == shape {
            provider.random_uniform_like(handle)
        } else {
            provider.random_uniform(shape)
        };
        if let Ok(gpu) = attempt {
            runmat_accelerate_api::set_handle_precision(&gpu, precision);
            let len = tensor::element_count(shape);
            random::skip_uniform(len, "rand")?;
            return Ok(Value::GpuTensor(gpu));
        } else {
            log_rand_fallback(shape, dtype, "provider-like-error");
        }

        let len = tensor::element_count(shape);
        let data = random::generate_uniform(len, "rand")?;

        let tensor =
            Tensor::new(data, shape.to_vec()).map_err(|e| builtin_error(format!("rand: {e}")))?;
        let view = HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        if let Ok(gpu) = provider.upload(&view) {
            runmat_accelerate_api::set_handle_precision(&gpu, precision);
            return Ok(Value::GpuTensor(gpu));
        } else {
            log_rand_fallback(shape, dtype, "upload-error");
        }
    } else {
        log_rand_fallback(shape, NumericDType::F32, "no-provider-like");
    }

    let gathered = crate::dispatcher::gather_if_needed_async(&Value::GpuTensor(handle.clone()))
        .await
        .map_err(|e| builtin_error(format!("rand: {e}")))?;
    log_rand_fallback(shape, NumericDType::F32, "gather-fallback");
    rand_like(&gathered, shape).await
}

fn try_gpu_uniform(shape: &[usize], dtype: NumericDType) -> crate::BuiltinResult<Option<Value>> {
    let Some(provider) = runmat_accelerate_api::provider() else {
        log_rand_fallback(shape, dtype, "no-provider");
        return Ok(None);
    };
    let precision = match dtype {
        NumericDType::F32 => ProviderPrecision::F32,
        NumericDType::F64 => ProviderPrecision::F64,
    };
    if provider.precision() != precision {
        log_rand_fallback(shape, dtype, "precision-mismatch");
        return Ok(None);
    }
    match provider.random_uniform(shape) {
        Ok(handle) => {
            runmat_accelerate_api::set_handle_precision(&handle, precision);
            let len = tensor::element_count(shape);
            random::skip_uniform(len, "rand")?;
            Ok(Some(Value::GpuTensor(handle)))
        }
        Err(err) => {
            log::warn!(
                "rand: provider random_uniform failed ({err}); falling back to host tensor path"
            );
            log_rand_fallback(shape, dtype, "provider-error");
            Ok(None)
        }
    }
}

fn rand_fallback_debug_enabled() -> bool {
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| {
        matches!(
            std::env::var("RUNMAT_DEBUG_RAND_FALLBACK"),
            Ok(value) if value == "1"
                || value.eq_ignore_ascii_case("true")
                || value.eq_ignore_ascii_case("yes")
        )
    })
}

fn log_rand_fallback(shape: &[usize], dtype: NumericDType, reason: &str) {
    if !rand_fallback_debug_enabled() {
        return;
    }
    let elems = tensor::element_count(shape);
    tracing::debug!(
        dtype = ?dtype,
        elems,
        shape = ?shape,
        reason,
        "[rand_debug] fallback"
    );
}

fn dtype_from_precision(precision: ProviderPrecision) -> NumericDType {
    match precision {
        ProviderPrecision::F32 => NumericDType::F32,
        ProviderPrecision::F64 => NumericDType::F64,
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::{random, test_support};
    use futures::executor::block_on;

    fn reset_rng_clean() {
        runmat_accelerate_api::clear_provider();
        random::reset_rng();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rand_default_scalar() {
        let _guard = random::test_lock().lock().unwrap();
        reset_rng_clean();
        let result = block_on(rand_builtin(Vec::new())).expect("rand");
        let expected = random::expected_uniform_sequence(1)[0];
        match result {
            Value::Num(v) => {
                assert!((0.0..1.0).contains(&v));
                assert!((v - expected).abs() < 1e-12);
            }
            other => panic!("expected scalar double, got {other:?}"),
        }
    }

    #[test]
    fn rand_type_defaults_to_num() {
        assert_eq!(rand_type(&[], &ResolveContext::new(Vec::new())), Type::Num);
    }

    #[test]
    fn rand_type_infers_rank_from_scalar_dim() {
        assert_eq!(
            rand_type(&[Type::Num], &ResolveContext::new(Vec::new())),
            Type::Tensor {
                shape: Some(vec![None, None])
            }
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rand_square_from_single_dimension() {
        let _guard = random::test_lock().lock().unwrap();
        reset_rng_clean();
        let args = vec![Value::Num(3.0)];
        let result = block_on(rand_builtin(args)).expect("rand");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![3, 3]);
                let expected = random::expected_uniform_sequence(9);
                assert_eq!(t.data.len(), expected.len());
                for (observed, exp) in t.data.iter().zip(expected.iter()) {
                    assert!((*observed - exp).abs() < 1e-12);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rand_like_tensor_infers_shape() {
        let _guard = random::test_lock().lock().unwrap();
        reset_rng_clean();
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let args = vec![Value::Tensor(tensor)];
        let result = block_on(rand_builtin(args)).expect("rand");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                let expected = random::expected_uniform_sequence(4);
                for (observed, exp) in t.data.iter().zip(expected.iter()) {
                    assert!((*observed - exp).abs() < 1e-12);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rand_single_matrix_has_f32_dtype() {
        let _guard = random::test_lock().lock().unwrap();
        reset_rng_clean();
        let args = vec![Value::Num(2.0), Value::Num(2.0), Value::from("single")];
        let result = block_on(rand_builtin(args)).expect("rand single");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                assert_eq!(t.dtype, NumericDType::F32);
                let expected = random::expected_uniform_sequence(4)
                    .into_iter()
                    .map(|v| {
                        let val = v as f32;
                        val as f64
                    })
                    .collect::<Vec<f64>>();
                for (observed, exp) in t.data.iter().zip(expected.iter()) {
                    assert!((*observed - *exp).abs() < 1e-7);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rand_like_complex_produces_complex_tensor() {
        let _guard = random::test_lock().lock().unwrap();
        reset_rng_clean();
        let args = vec![
            Value::Num(2.0),
            Value::Num(2.0),
            Value::from("like"),
            Value::Complex(0.0, 1.0),
        ];
        let result = block_on(rand_builtin(args)).expect("rand");
        match result {
            Value::ComplexTensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                let expected = random::expected_complex_sequence(4);
                for ((re, im), (eref, eim)) in t.data.iter().zip(expected.iter()) {
                    assert!((*re - *eref).abs() < 1e-12);
                    assert!((*im - *eim).abs() < 1e-12);
                }
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rand_gpu_like_uniform() {
        let _guard = random::test_lock().lock().unwrap();
        random::reset_rng();
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![0.0, 0.0, 0.0, 0.0], vec![2, 2]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let args = vec![
                Value::Num(2.0),
                Value::Num(2.0),
                Value::from("like"),
                Value::GpuTensor(handle),
            ];
            let result = block_on(rand_builtin(args)).expect("rand");
            match result {
                Value::GpuTensor(gpu) => {
                    assert_eq!(gpu.shape, vec![2, 2]);
                    let gathered =
                        test_support::gather(Value::GpuTensor(gpu)).expect("gather to host");
                    assert_eq!(gathered.shape, vec![2, 2]);
                    for value in gathered.data {
                        assert!((0.0..1.0).contains(&value));
                    }
                }
                other => panic!("expected GPU tensor, got {other:?}"),
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn rand_wgpu_like_uniform_and_gather() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        // Create a GPU prototype and request rand like it
        let tensor = Tensor::new(vec![0.0; 4], vec![2, 2]).unwrap();
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let provider = runmat_accelerate_api::provider().unwrap();
        let handle = provider.upload(&view).expect("upload");
        let result =
            block_on(rand_like(&Value::GpuTensor(handle), &[2, 2])).expect("rand like gpu");
        match result {
            Value::GpuTensor(h) => {
                let gathered = test_support::gather(Value::GpuTensor(h)).expect("gather to host");
                assert_eq!(gathered.shape, vec![2, 2]);
                for v in gathered.data {
                    assert!((0.0..1.0).contains(&v));
                }
            }
            other => panic!("expected gpu tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn rand_wgpu_fusion_then_sin_then_sum() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let r = rand_double(&[2, 2]).expect("rand");
        let s = block_on(crate::call_builtin_async("sin", &[r])).expect("sin");
        let summed =
            block_on(crate::call_builtin_async("sum", &[s, Value::Num(1.0)])).expect("sum");
        let gathered = test_support::gather(summed).expect("gather");
        assert_eq!(gathered.shape, vec![1, 2]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn rand_wgpu_single_allocates_gpu_without_like() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let value = rand_single(&[2, 2]).expect("rand single");
        match value {
            Value::GpuTensor(handle) => {
                let gathered =
                    test_support::gather(Value::GpuTensor(handle)).expect("gather to host");
                assert_eq!(gathered.shape, vec![2, 2]);
            }
            other => panic!("expected gpu tensor, got {other:?}"),
        }
    }
}
