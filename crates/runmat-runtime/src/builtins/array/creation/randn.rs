//! MATLAB-compatible `randn` builtin with GPU-aware semantics for RunMat.

use runmat_accelerate_api::{GpuTensorHandle, HostTensorView};
use runmat_builtins::{ComplexTensor, NumericDType, Tensor, Value};
use runmat_macros::runtime_builtin;

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

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::array::creation::randn")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "randn",
    op_kind: GpuOpKind::Custom("generator"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[
        ProviderHook::Custom("random_normal"),
        ProviderHook::Custom("random_normal_like"),
    ],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Leverages provider normal RNG hooks when available; otherwise falls back to host sampling followed by a single upload to preserve GPU residency.",
};

fn builtin_error(message: impl Into<String>) -> crate::RuntimeError {
    build_runtime_error(message).with_builtin("randn").build()
}

fn randn_type(args: &[Type], ctx: &ResolveContext) -> Type {
    if args.is_empty() {
        return Type::Num;
    }
    if args.iter().any(|arg| matches!(arg, Type::String)) {
        return Type::Unknown;
    }
    tensor_type_from_rank(args, ctx)
}

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::array::creation::randn")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "randn",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Random generation is treated as a sink and excluded from fusion planning.",
};

#[runtime_builtin(
    name = "randn",
    category = "array/creation",
    summary = "Standard normal random numbers.",
    keywords = "randn,random,normal,gaussian,gpu,like",
    accel = "array_construct",
    type_resolver(randn_type),
    type_resolver_context = true,
    builtin_path = "crate::builtins::array::creation::randn"
)]
async fn randn_builtin(rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    let parsed = ParsedRandn::parse(rest).await?;
    build_output(parsed).await
}

struct ParsedRandn {
    shape: Vec<usize>,
    template: RandnTemplate,
    dtype: NumericDType,
}

#[derive(Clone)]
enum RandnTemplate {
    Double,
    Like(Value),
}

impl ParsedRandn {
    async fn parse(args: Vec<Value>) -> crate::BuiltinResult<Self> {
        let mut dims: Vec<usize> = Vec::new();
        let mut saw_dims_arg = false;
        let mut shape_source: Option<Vec<usize>> = None;
        let mut template: Option<RandnTemplate> = None;
        let mut dtype: NumericDType = NumericDType::F64;

        let mut idx = 0;
        while idx < args.len() {
            let arg = args[idx].clone();

            if let Some(keyword) = keyword_of(&arg) {
                match keyword.as_str() {
                    "like" => {
                        let Some(proto) = args.get(idx + 1).cloned() else {
                            return Err(builtin_error("randn: expected prototype after 'like'"));
                        };
                        template = Some(RandnTemplate::Like(proto.clone()));
                        shape_source = Some(shape_from_value(&proto, "randn")?);
                        idx += 2;
                        continue;
                    }
                    "double" => {
                        template = Some(RandnTemplate::Double);
                        dtype = NumericDType::F64;
                        idx += 1;
                        continue;
                    }
                    "single" => {
                        template = Some(RandnTemplate::Double);
                        dtype = NumericDType::F32;
                        idx += 1;
                        continue;
                    }
                    other => {
                        return Err(builtin_error(format!(
                            "randn: unrecognised option '{other}'"
                        )));
                    }
                }
            }

            if let Some(parsed_dims) = extract_dims(&arg, "randn").await? {
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
                shape_source = Some(shape_from_value(&arg, "randn")?);
            }
            if template.is_none() {
                template = Some(RandnTemplate::Like(arg.clone()));
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

        let template = template.unwrap_or(RandnTemplate::Double);

        Ok(Self {
            shape,
            template,
            dtype,
        })
    }
}

async fn build_output(parsed: ParsedRandn) -> crate::BuiltinResult<Value> {
    match parsed.template {
        RandnTemplate::Double => match parsed.dtype {
            NumericDType::F64 => randn_double(&parsed.shape),
            NumericDType::F32 => randn_single(&parsed.shape),
        },
        RandnTemplate::Like(proto) => randn_like(&proto, &parsed.shape).await,
    }
}

fn randn_double(shape: &[usize]) -> crate::BuiltinResult<Value> {
    let len = tensor::element_count(shape);
    let data = random::generate_normal(len, "randn")?;
    let tensor =
        Tensor::new(data, shape.to_vec()).map_err(|e| builtin_error(format!("randn: {e}")))?;
    Ok(tensor::tensor_into_value(tensor))
}

fn randn_single(shape: &[usize]) -> crate::BuiltinResult<Value> {
    let len = tensor::element_count(shape);
    let data = random::generate_normal(len, "randn")?
        .into_iter()
        .map(|v| {
            let f32_val = v as f32;
            f32_val as f64
        })
        .collect::<Vec<f64>>();
    let tensor = Tensor::new_with_dtype(data, shape.to_vec(), NumericDType::F32)
        .map_err(|e| builtin_error(format!("randn: {e}")))?;
    Ok(Value::Tensor(tensor))
}

#[async_recursion::async_recursion(?Send)]
async fn randn_like(proto: &Value, shape: &[usize]) -> crate::BuiltinResult<Value> {
    match proto {
        Value::GpuTensor(handle) => randn_like_gpu(handle, shape).await,
        Value::ComplexTensor(_) | Value::Complex(_, _) => randn_complex(shape),
        Value::Tensor(t) => match t.dtype {
            NumericDType::F32 => randn_single(shape),
            NumericDType::F64 => randn_double(shape),
        },
        Value::Num(_) | Value::Int(_) | Value::Bool(_) | Value::LogicalArray(_) => {
            randn_double(shape)
        }
        Value::CharArray(_) | Value::Cell(_) => randn_double(shape),
        other => Err(builtin_error(format!(
            "randn: unsupported prototype {other:?}"
        ))),
    }
}

fn randn_complex(shape: &[usize]) -> crate::BuiltinResult<Value> {
    let len = tensor::element_count(shape);
    let data = random::generate_normal_complex(len, "randn")?;
    let tensor = ComplexTensor::new(data, shape.to_vec())
        .map_err(|e| builtin_error(format!("randn: {e}")))?;
    Ok(complex_tensor_into_value(tensor))
}

#[async_recursion::async_recursion(?Send)]
async fn randn_like_gpu(handle: &GpuTensorHandle, shape: &[usize]) -> crate::BuiltinResult<Value> {
    if let Some(provider) = runmat_accelerate_api::provider() {
        let attempt = if handle.shape == shape {
            provider.random_normal_like(handle)
        } else {
            provider.random_normal(shape)
        };
        if let Ok(gpu) = attempt {
            return Ok(Value::GpuTensor(gpu));
        }

        let len = tensor::element_count(shape);
        let data = random::generate_normal(len, "randn")?;

        let tensor =
            Tensor::new(data, shape.to_vec()).map_err(|e| builtin_error(format!("randn: {e}")))?;
        let view = HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        if let Ok(gpu) = provider.upload(&view) {
            return Ok(Value::GpuTensor(gpu));
        }
    }

    let gathered = crate::dispatcher::gather_if_needed_async(&Value::GpuTensor(handle.clone()))
        .await
        .map_err(|e| builtin_error(format!("randn: {e}")))?;
    randn_like(&gathered, shape).await
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
    fn randn_default_scalar() {
        let _guard = random::test_lock().lock().unwrap();
        reset_rng_clean();
        let result = block_on(randn_builtin(Vec::new())).expect("randn");
        let expected = random::expected_normal_sequence(1)[0];
        match result {
            Value::Num(v) => assert!((v - expected).abs() < 1e-12),
            other => panic!("expected scalar double, got {other:?}"),
        }
    }

    #[test]
    fn randn_type_defaults_to_num() {
        assert_eq!(randn_type(&[], &ResolveContext::empty()), Type::Num);
    }

    #[test]
    fn randn_type_infers_rank_from_scalar_dim() {
        assert_eq!(
            randn_type(&[Type::Num], &ResolveContext::empty()),
            Type::Tensor {
                shape: Some(vec![None, None])
            }
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn randn_square_from_single_dimension() {
        let _guard = random::test_lock().lock().unwrap();
        reset_rng_clean();
        let args = vec![Value::Num(2.0)];
        let result = block_on(randn_builtin(args)).expect("randn");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                let expected = random::expected_normal_sequence(4);
                for (observed, exp) in t.data.iter().zip(expected.iter()) {
                    assert!((*observed - *exp).abs() < 1e-12);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn randn_size_vector_argument() {
        let _guard = random::test_lock().lock().unwrap();
        reset_rng_clean();
        let size_vec = Tensor::new(vec![2.0, 3.0, 4.0], vec![1, 3]).unwrap();
        let result = block_on(randn_builtin(vec![Value::Tensor(size_vec)])).expect("randn");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 3, 4]);
                let expected = random::expected_normal_sequence(24);
                assert_eq!(t.data.len(), expected.len());
                for (observed, exp) in t.data.iter().zip(expected.iter()) {
                    assert!((*observed - *exp).abs() < 1e-12);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn randn_zero_dimension_returns_empty() {
        let _guard = random::test_lock().lock().unwrap();
        reset_rng_clean();
        let result = block_on(randn_builtin(vec![Value::Num(0.0)])).expect("randn");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![0, 0]);
                assert!(t.data.is_empty());
            }
            other => panic!("expected empty tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn randn_single_precision_produces_f32() {
        let _guard = random::test_lock().lock().unwrap();
        reset_rng_clean();
        let result = block_on(randn_builtin(vec![Value::from("single")])).expect("randn single");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.dtype, NumericDType::F32);
                assert_eq!(t.shape, vec![1, 1]);
            }
            Value::Num(_) => {
                // Fallback scalar path remains F64-backed but should definitely not allocate on GPU.
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn randn_like_tensor_infers_shape() {
        let _guard = random::test_lock().lock().unwrap();
        reset_rng_clean();
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let args = vec![Value::Tensor(tensor)];
        let result = block_on(randn_builtin(args)).expect("randn");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                let expected = random::expected_normal_sequence(4);
                for (observed, exp) in t.data.iter().zip(expected.iter()) {
                    assert!((*observed - *exp).abs() < 1e-12);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn randn_like_complex_produces_complex_tensor() {
        let _guard = random::test_lock().lock().unwrap();
        reset_rng_clean();
        let args = vec![
            Value::Num(2.0),
            Value::Num(1.0),
            Value::from("like"),
            Value::Complex(0.0, 1.0),
        ];
        let result = block_on(randn_builtin(args)).expect("randn");
        match result {
            Value::ComplexTensor(t) => {
                assert_eq!(t.shape, vec![2, 1]);
                let expected = random::expected_complex_normal_sequence(2);
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
    fn randn_gpu_like_roundtrip() {
        let _guard = random::test_lock().lock().unwrap();
        random::reset_rng();
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![0.0; 4], vec![2, 2]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let args = vec![Value::from("like"), Value::GpuTensor(handle)];
            let result = block_on(randn_builtin(args)).expect("randn");
            match result {
                Value::GpuTensor(gpu) => {
                    assert_eq!(gpu.shape, vec![2, 2]);
                    let gathered = test_support::gather(Value::GpuTensor(gpu)).expect("gather");
                    assert_eq!(gathered.shape, vec![2, 2]);
                    for value in gathered.data {
                        assert!(value.is_finite());
                    }
                }
                other => panic!("expected GPU tensor, got {other:?}"),
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn randn_wgpu_like_and_gather() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let tensor = Tensor::new(vec![0.0; 4], vec![2, 2]).unwrap();
        let view = HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let provider = runmat_accelerate_api::provider().unwrap();
        let handle = provider.upload(&view).expect("upload");
        let result =
            block_on(randn_like(&Value::GpuTensor(handle), &[2, 2])).expect("randn like gpu");
        match result {
            Value::GpuTensor(h) => {
                let gathered = test_support::gather(Value::GpuTensor(h)).expect("gather to host");
                assert_eq!(gathered.shape, vec![2, 2]);
                for v in gathered.data {
                    assert!(v.is_finite());
                }
            }
            other => panic!("expected gpu tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn randn_wgpu_provider_random_normal() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let provider = runmat_accelerate_api::provider().expect("wgpu provider registered");
        let handle = provider
            .random_normal(&[4, 4])
            .expect("wgpu random_normal hook");
        let gathered =
            test_support::gather(Value::GpuTensor(handle)).expect("gather random_normal output");
        assert_eq!(gathered.shape, vec![4, 4]);
        assert!(
            gathered
                .data
                .iter()
                .any(|value| value.is_finite() && value.abs() > 1.0e-6),
            "expected at least one non-trivial normal sample"
        );
    }
}
