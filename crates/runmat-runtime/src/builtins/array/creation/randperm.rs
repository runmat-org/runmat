//! MATLAB-compatible `randperm` builtin with GPU-aware semantics for RunMat.
use runmat_builtins::{RANDPERM_EXPLICIT_DOUBLE_EXTENSION, RANDPERM_LIKE_EXTENSION};

use runmat_macros::runtime_builtin;
use runmat_value::{Tensor, Value};

mod gpu_child;
mod host_output;

#[cfg(test)]
#[path = "randperm/owning_tests.rs"]
mod owning_tests;

use crate::build_runtime_error;
use crate::builtins::common::random;
use crate::builtins::common::random_args::keyword_of;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::tensor;

const MAX_SAFE_INTEGER: u64 = 1 << 53;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::array::creation::randperm")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "randperm",
    op_kind: GpuOpKind::Custom("permutation"),
    supported_precisions: &[ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "The GPU Like route uses an independent request-bound child stream, unbiased reservoir/shuffle indices and an exact-owner Double upload. The older process-global permutation hooks are not semantic authority.",
};

fn builtin_error(message: impl Into<String>) -> crate::RuntimeError {
    build_runtime_error(message)
        .with_builtin("randperm")
        .build()
}

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
    binding_variant = "default",
    builtin_path = "crate::builtins::array::creation::randperm"
)]
async fn randperm_builtin(args: crate::call::BuiltinArguments) -> crate::BuiltinResult<Value> {
    let parsed = ParsedRandPerm::parse(args).await?;
    if let OutputTemplate::Like(Value::GpuTensor(handle)) = &parsed.template {
        return match random::gpu_like::intent("randperm", handle)? {
            random::gpu_like::GpuLikeIntent::ExplicitGpuStream => {
                gpu_child::like(handle, parsed.n, parsed.k).await
            }
            random::gpu_like::GpuLikeIntent::AutomaticHostStream => {
                match random::gpu_like::class(handle) {
                    Some(random::gpu_like::GpuLikeClass::Floating { complex: false, .. })
                    | Some(random::gpu_like::GpuLikeClass::Integer { complex: false, .. }) => {
                        let shape = [1, parsed.k];
                        let output_class = random::gpu_like::GpuLikeClass::Floating {
                            precision: runmat_accelerate_api::ProviderPrecision::F64,
                            complex: false,
                        };
                        let host_mode = random::gpu_like::host_random_mode(
                            crate::context::legacy::active().as_ref(),
                        );
                        let result_bytes = random::gpu_like::host_result_bytes(
                            "randperm", &shape, output_class, host_mode,
                        )?;
                        let scratch = random::gpu_like::exact_host_workspace(
                            "randperm",
                            random::gpu_like::ExactHostRandomWorkspace::Permutation {
                                population: parsed.n,
                                count: parsed.k,
                            },
                            host_mode,
                        )?;
                        let presentation = random::gpu_like::select_presentation(
                            "randperm",
                            handle,
                            &shape,
                            output_class,
                            result_bytes,
                            scratch,
                            scratch,
                            0,
                            host_mode,
                        )?;
                        let mut turn = random::acquire_turn(presentation.runtime(), "randperm").await?;
                        presentation.checkpoint_for_turn(&turn)?;
                        let result = host_output::double(
                            &mut turn,
                            parsed.n,
                            parsed.k,
                            presentation.mode(),
                        );
                        presentation.finish_result(
                            "randperm",
                            result,
                            &shape,
                        )
                    }
                    Some(random::gpu_like::GpuLikeClass::Floating { complex: true, .. })
                    | Some(random::gpu_like::GpuLikeClass::Integer { complex: true, .. }) => {
                        Err(builtin_error("randperm: complex prototypes are not supported"))
                    }
                    Some(random::gpu_like::GpuLikeClass::Logical) => Err(builtin_error(
                        "randperm: logical prototypes cannot represent permutation values (requires numeric output)",
                    )),
                    None => Err(builtin_error("randperm: unsupported gpuArray prototype class")),
                }
            }
        };
    }
    let mut turn = random::acquire_turn(None, "randperm").await?;
    build_output(&mut turn, parsed)
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
    async fn parse(args: crate::call::BuiltinArguments) -> crate::BuiltinResult<Self> {
        if args.is_empty() {
            return Err(builtin_error(
                "randperm: requires at least one input argument",
            ));
        }

        let n = parse_size_argument(
            &args[0],
            true,
            "randperm: N must be a non-negative integer (and <= 2^53)",
        )
        .await?;
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
            if let Some(keyword) = keyword_of(&arg)? {
                match keyword.as_str() {
                    "like" => {
                        crate::compatibility::ensure_builtin_extension_enabled(
                            &RANDPERM_LIKE_EXTENSION,
                            "randperm",
                        )?;
                        if matches!(template, OutputTemplate::Like(_)) {
                            return Err(builtin_error(
                                "randperm: duplicate 'like' prototype specified",
                            ));
                        }
                        let Some(proto) = args.get(idx + 1).cloned() else {
                            return Err(builtin_error("randperm: expected prototype after 'like'"));
                        };
                        template = OutputTemplate::Like(proto);
                        idx += 2;
                        continue;
                    }
                    "double" => {
                        crate::compatibility::ensure_builtin_extension_enabled(
                            &RANDPERM_EXPLICIT_DOUBLE_EXTENSION,
                            "randperm",
                        )?;
                        if matches!(template, OutputTemplate::Like(_)) {
                            return Err(builtin_error(
                                "randperm: cannot combine 'double' with a 'like' prototype",
                            ));
                        }
                        idx += 1;
                        continue;
                    }
                    "single" => {
                        return Err(builtin_error(
                            "randperm: single precision output is not implemented yet",
                        ));
                    }
                    other => {
                        return Err(builtin_error(format!(
                            "randperm: unrecognised option '{other}'"
                        )));
                    }
                }
            }

            if k.is_none() {
                k = Some(
                    parse_size_argument(
                        &arg,
                        true,
                        "randperm: K must be a non-negative integer (and <= N)",
                    )
                    .await?,
                );
                idx += 1;
                continue;
            }

            return Err(builtin_error("randperm: too many input arguments"));
        }

        let k = k.unwrap_or(n);

        if k > n {
            return Err(builtin_error("randperm: K must satisfy 0 <= K <= N"));
        }

        Ok(Self { n, k, template })
    }
}

fn build_output(
    turn: &mut random::RandomTurn,
    parsed: ParsedRandPerm,
) -> crate::BuiltinResult<Value> {
    match parsed.template {
        OutputTemplate::Double => randperm_double(turn, parsed.n, parsed.k),
        OutputTemplate::Like(proto) => randperm_like(turn, &proto, parsed.n, parsed.k),
    }
}

fn randperm_double(
    turn: &mut random::RandomTurn,
    n: usize,
    k: usize,
) -> crate::BuiltinResult<Value> {
    let tensor = randperm_tensor(turn, n, k)?;
    Ok(tensor::tensor_into_value(tensor))
}

fn randperm_like(
    turn: &mut random::RandomTurn,
    proto: &Value,
    n: usize,
    k: usize,
) -> crate::BuiltinResult<Value> {
    match proto {
        Value::GpuTensor(_) => unreachable!("GPU prototype handled before host turn"),
        Value::Tensor(_) | Value::Num(_) | Value::Int(_) => randperm_double(turn, n, k),
        Value::LogicalArray(_) => Err(builtin_error(
            "randperm: logical prototypes cannot represent permutation values (requires numeric output)",
        )),
        Value::Complex(_, _) | Value::ComplexTensor(_) => {
            Err(builtin_error("randperm: complex prototypes are not supported"))
        }
        Value::Bool(_) => Err(builtin_error("randperm: prototypes must be numeric")),
        Value::CharArray(_) | Value::String(_) | Value::StringArray(_) => {
            Err(builtin_error("randperm: prototypes must be numeric"))
        }
        Value::Cell(_) => Err(builtin_error("randperm: cell prototypes are not supported")),
        other => Err(builtin_error(format!("randperm: unsupported prototype {other:?}"))),
    }
}

fn randperm_tensor(
    turn: &mut random::RandomTurn,
    n: usize,
    k: usize,
) -> crate::BuiltinResult<Tensor> {
    let mut values: Vec<f64> = if n == 0 {
        Vec::new()
    } else {
        (1..=n).map(|v| v as f64).collect()
    };

    if k > 0 {
        let offsets = random::generate_permutation_offsets(turn, n, k)?;
        for (i, offset) in offsets.into_iter().enumerate() {
            let j = i + offset;
            values.swap(i, j);
        }
    }

    if values.len() > k {
        values.truncate(k);
    }

    Tensor::new(values, vec![1, k]).map_err(|e| builtin_error(format!("randperm: {e}")))
}

async fn parse_size_argument(
    value: &Value,
    allow_zero: bool,
    message: &str,
) -> crate::BuiltinResult<usize> {
    let is_vector = match value {
        Value::Tensor(t) => tensor::tensor_element_len(t) != 1,
        Value::GpuTensor(handle) => tensor::element_count(handle.shape()) != 1,
        _ => false,
    };

    if let Ok(Some(dim)) = tensor::dimension_from_value_async(value, "randperm", allow_zero).await {
        return validate_size_argument(dim, allow_zero, message);
    }

    match tensor::dims_from_value_async(value).await {
        Ok(Some(dims)) => {
            if dims.len() != 1 {
                return Err(builtin_error("randperm: size arguments must be scalar"));
            }
            validate_size_argument(dims[0], allow_zero, message)
        }
        Ok(None) => {
            if is_vector {
                Err(builtin_error("randperm: size arguments must be scalar"))
            } else {
                Err(builtin_error(format!(
                    "randperm: size arguments must be numeric scalars, got {value:?}"
                )))
            }
        }
        Err(_) => {
            if is_vector {
                Err(builtin_error("randperm: size arguments must be scalar"))
            } else {
                Err(builtin_error(message))
            }
        }
    }
}

fn validate_size_argument(
    value: usize,
    allow_zero: bool,
    message: &str,
) -> crate::BuiltinResult<usize> {
    if !allow_zero && value == 0 {
        return Err(builtin_error(message));
    }
    if value as u64 > MAX_SAFE_INTEGER {
        return Err(builtin_error(
            "randperm: values larger than 2^53 are not supported",
        ));
    }
    Ok(value)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::{random, test_support};
    use futures::executor::block_on;
    use runmat_accelerate_api::HostTensorView;
    use runmat_value::IntegerStorage;

    fn randperm_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
        block_on(super::randperm_builtin(
            crate::builtins::common::test_support::builtin_arguments(args),
        ))
    }

    fn reset_rng_clean() -> impl Drop {
        let guard = random::test_guard();
        runmat_accelerate_api::clear_provider();
        random::reset_rng();
        guard
    }

    fn expected_randperm(n: usize, k: usize) -> Vec<f64> {
        let mut values: Vec<f64> = if n == 0 {
            Vec::new()
        } else {
            (1..=n).map(|v| v as f64).collect()
        };
        if k > 0 {
            for (i, offset) in random::expected_permutation_offsets(n, k)
                .into_iter()
                .enumerate()
            {
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
        let _guard = random::test_guard();
        let _guard = reset_rng_clean();
        let args = vec![Value::from(5)];
        let result = randperm_builtin(args).expect("randperm");
        let gathered = test_support::gather(result).expect("gather");
        assert_eq!(gathered.shape(), vec![1, 5]);
        let expected = expected_randperm(5, 5);
        assert_eq!(gathered.materialize_f64(), expected);
    }

    #[test]
    fn randperm_size_arguments_read_typed_integer_storage_exactly() {
        let n = Tensor::new_integer(IntegerStorage::U64(vec![5]), vec![1, 1]).expect("n");
        assert_eq!(
            block_on(parse_size_argument(
                &Value::Tensor(n),
                false,
                "randperm: N must be a non-negative integer (and <= 2^53)",
            ))
            .expect("N"),
            5
        );

        let vector =
            Tensor::new_integer(IntegerStorage::U16(vec![1, 2]), vec![1, 2]).expect("vector");
        assert!(block_on(parse_size_argument(
            &Value::Tensor(vector),
            false,
            "randperm: N must be a non-negative integer (and <= 2^53)",
        ))
        .is_err());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn randperm_partial_selection_is_unique_and_sorted() {
        let _guard = random::test_guard();
        let _guard = reset_rng_clean();
        let args = vec![Value::from(10), Value::from(4)];
        let result = randperm_builtin(args).expect("randperm");
        let gathered = test_support::gather(result).expect("gather");
        assert_eq!(gathered.shape(), vec![1, 4]);
        let data = gathered.materialize_f64();
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
        assert_eq!(gathered.shape(), vec![1, 0]);
        assert!(gathered.materialize_f64().is_empty());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn randperm_errors_when_k_exceeds_n() {
        let args = vec![Value::from(3), Value::from(4)];
        let err = randperm_builtin(args).unwrap_err();
        assert!(err.message().contains("K must satisfy 0 <= K <= N"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn randperm_errors_for_negative_input() {
        let args = vec![Value::Num(-1.0)];
        let err = randperm_builtin(args).unwrap_err();
        assert!(err.message().contains("N must be a non-negative integer"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn randperm_rejects_single_precision_request() {
        let args = vec![Value::from(5), Value::from("single")];
        let err = randperm_builtin(args).unwrap_err();
        assert!(err.message().contains("single precision"));
    }

    #[test]
    fn randperm_extension_selectors_follow_compatibility_mode() {
        {
            let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
            let explicit = randperm_builtin(vec![Value::from(3), Value::from("double")])
                .expect_err("MATLAB mode rejects explicit double selector");
            assert_eq!(
                explicit.identifier(),
                Some("RunMat:compatibility:RandpermExplicitDoubleExtension")
            );
            let like = randperm_builtin(vec![Value::from(3), Value::from("like"), Value::Num(0.0)])
                .expect_err("MATLAB mode rejects like selector");
            assert_eq!(
                like.identifier(),
                Some("RunMat:compatibility:RandpermLikeExtension")
            );
        }
        {
            let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
            assert!(
                randperm_builtin(vec![Value::from(0), Value::from("double")]).is_ok(),
                "RunMat mode accepts explicit double selector"
            );
            assert!(
                randperm_builtin(vec![Value::from(0), Value::from("like"), Value::Num(0.0),])
                    .is_ok(),
                "RunMat mode accepts like selector"
            );
        }
    }

    #[test]
    fn randperm_integer_counts_always_return_double() {
        let _guard = random::test_guard();
        for storage in [
            IntegerStorage::I8(vec![0]),
            IntegerStorage::I16(vec![0]),
            IntegerStorage::I32(vec![0]),
            IntegerStorage::I64(vec![0]),
            IntegerStorage::U8(vec![0]),
            IntegerStorage::U16(vec![0]),
            IntegerStorage::U32(vec![0]),
            IntegerStorage::U64(vec![0]),
        ] {
            reset_rng_clean();
            let count = Tensor::new_integer(storage, vec![1, 1]).expect("integer count");
            let output =
                randperm_builtin(vec![Value::Tensor(count)]).expect("integer count randperm");
            let Value::Tensor(output) = output else {
                panic!("zero-count randperm must return a tensor");
            };
            assert_eq!(output.numeric_dtype(), runmat_value::NumericDType::F64);
            assert_eq!(output.shape(), vec![1, 0]);
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn randperm_accepts_double_keyword() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let _guard = random::test_guard();
        let _guard = reset_rng_clean();
        let args = vec![Value::from(5), Value::from("double")];
        let result = randperm_builtin(args).expect("randperm");
        let gathered = test_support::gather(result).expect("gather");
        assert_eq!(gathered.shape(), vec![1, 5]);
        let expected = expected_randperm(5, 5);
        assert_eq!(gathered.materialize_f64(), expected);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn randperm_like_tensor_matches_host_output() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let _guard = random::test_guard();
        let _guard = reset_rng_clean();
        let proto_tensor = Tensor::new(vec![0.0, 0.0], vec![1, 2]).unwrap();
        let args = vec![
            Value::from(4),
            Value::from("like"),
            Value::Tensor(proto_tensor),
        ];
        let result = randperm_builtin(args).expect("randperm");
        let gathered = test_support::gather(result).expect("gather");
        assert_eq!(gathered.shape(), vec![1, 4]);
        let expected = expected_randperm(4, 4);
        assert_eq!(gathered.materialize_f64(), expected);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn randperm_gpu_like_roundtrip() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let _guard = random::test_guard();
        random::reset_rng();
        random::gpu::replace(random::gpu::GpuRandomSnapshot::default())
            .expect("reset independent GPU child");
        test_support::with_test_provider(|provider| {
            let cpu_before = random::snapshot().expect("CPU stream before GPU permutation");
            let proto_tensor = Tensor::new(vec![0.0, 0.0], vec![1, 2]).unwrap();
            let view = HostTensorView {
                data: &proto_tensor.materialize_f64(),
                shape: proto_tensor.shape(),
            };
            let proto_handle = provider
                .upload(&view)
                .expect("upload prototype")
                .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
                .expect("explicit gpuArray prototype");
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
            assert_eq!(gathered.shape(), vec![1, 3]);
            let values = gathered.materialize_f64();
            assert!(values.iter().all(|value| (1.0..=6.0).contains(value)));
            let mut unique = values.clone();
            unique.sort_by(f64::total_cmp);
            unique.dedup();
            assert_eq!(unique.len(), 3);
            assert_eq!(
                random::snapshot().expect("CPU after GPU permutation").state,
                cpu_before.state
            );
            assert_eq!(
                random::gpu::snapshot()
                    .expect("GPU child after permutation")
                    .position,
                6
            );
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn automatic_like_permutation_uses_ordered_cpu_stream() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let _guard = random::test_guard();
        random::reset_rng();
        test_support::with_test_provider(|provider| {
            let values = [0.0_f64; 2];
            let source = provider
                .upload(&HostTensorView {
                    data: &values,
                    shape: &[1, 2],
                })
                .expect("upload prototype")
                .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Automatic)
                .expect("automatic prototype");
            let child_before = random::gpu::snapshot().expect("child before");
            let expected =
                randperm_builtin(vec![Value::from(6), Value::from(3)]).expect("host permutation");
            random::reset_rng();
            let actual = randperm_builtin(vec![
                Value::from(6),
                Value::from(3),
                Value::from("like"),
                Value::GpuTensor(source),
            ])
            .expect("automatic permutation");
            assert_eq!(actual, expected);
            assert_eq!(random::gpu::snapshot().expect("child after"), child_before);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn randperm_like_requires_prototype() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let args = vec![Value::from(4), Value::from("like")];
        let err = randperm_builtin(args).unwrap_err();
        assert!(err.message().contains("prototype after 'like'"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn randperm_wgpu_produces_unique_indices() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let _guard = random::test_guard();
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
        let proto_handle = provider
            .upload(&proto_view)
            .expect("upload prototype")
            .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
            .expect("explicit gpuArray prototype");

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

        let host =
            test_support::gather(Value::GpuTensor(gpu_handle.clone())).expect("gather permutation");
        assert_eq!(host.shape(), vec![1, 7]);
        assert_eq!(host.len(), 7);

        let mut sorted = host.materialize_f64();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        for window in sorted.windows(2) {
            assert_ne!(
                window[0], window[1],
                "duplicate value detected in permutation"
            );
        }
        for value in host.materialize_f64() {
            assert!(
                (1.0..=12.0).contains(&value),
                "value {value} outside expected range 1..12"
            );
        }
        drop(gpu_handle);
    }
}
