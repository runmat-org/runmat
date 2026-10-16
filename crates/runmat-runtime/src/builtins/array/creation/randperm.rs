//! MATLAB-compatible `randperm` builtin with GPU-aware semantics for RunMat.
use runmat_builtins::{RANDPERM_EXPLICIT_DOUBLE_EXTENSION, RANDPERM_LIKE_EXTENSION};

use runmat_accelerate_api::{GpuTensorHandle, GpuTensorStorage, ProviderPrecision};
use runmat_macros::runtime_builtin;
use runmat_value::{Tensor, Value};

#[cfg(test)]
#[path = "randperm/owning_tests.rs"]
mod owning_tests;

use crate::build_runtime_error;
#[cfg(test)]
use crate::builtins::array::type_resolvers::row_vector_type;
use crate::builtins::common::random;
use crate::builtins::common::random_args::keyword_of;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
#[cfg(test)]
use runmat_builtins::ResolveContext;
#[cfg(test)]
use runmat_builtins::Type;

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
    notes: "Uses the prototype's exact owner for a validated Double permutation; only certified pre-effect unsupported hooks permit host generation and class-preserving restoration.",
};

fn builtin_error(message: impl Into<String>) -> crate::RuntimeError {
    build_runtime_error(message)
        .with_builtin("randperm")
        .build()
}

#[cfg(test)]
fn randperm_type(args: &[Type], ctx: &ResolveContext) -> Type {
    if args.is_empty() {
        return Type::Unknown;
    }
    if args.iter().any(|arg| matches!(arg, Type::String)) {
        return Type::Unknown;
    }
    row_vector_type(ctx)
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
async fn randperm_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    let parsed = ParsedRandPerm::parse(args).await?;
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
    async fn parse(args: Vec<Value>) -> crate::BuiltinResult<Self> {
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
            if let Some(keyword) = keyword_of(&arg) {
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

fn build_output(parsed: ParsedRandPerm) -> crate::BuiltinResult<Value> {
    match parsed.template {
        OutputTemplate::Double => randperm_double(parsed.n, parsed.k),
        OutputTemplate::Like(proto) => randperm_like(&proto, parsed.n, parsed.k),
    }
}

fn randperm_double(n: usize, k: usize) -> crate::BuiltinResult<Value> {
    let tensor = randperm_tensor(n, k)?;
    Ok(tensor::tensor_into_value(tensor))
}

fn randperm_like(proto: &Value, n: usize, k: usize) -> crate::BuiltinResult<Value> {
    match proto {
        Value::GpuTensor(handle) => randperm_gpu(handle, n, k),
        Value::Tensor(_) | Value::Num(_) | Value::Int(_) => randperm_double(n, k),
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

fn randperm_gpu(handle: &GpuTensorHandle, n: usize, k: usize) -> crate::BuiltinResult<Value> {
    use gpu_helpers::{GpuOutputAliasPolicy, UnaryGpuOutputContract};

    let terminal = |detail: String| {
        let mut error = builtin_error(format!("randperm: {detail}"));
        error.gpu_gather_retry = crate::GpuGatherRetry::Never;
        error
    };
    handle.allocation_lease().validate_live().map_err(|cause| {
        crate::runtime_error::with_provider_execution_source(
            terminal(format!("prototype allocation is not live: {cause}")),
            runmat_accelerate_api::ProviderExecutionError::from_source(cause)
                .with_unproven_effects(),
        )
    })?;
    if runmat_accelerate_api::handle_is_logical(handle) {
        return Err(terminal(
            "logical prototypes cannot represent permutation values".into(),
        ));
    }
    if runmat_accelerate_api::handle_storage(handle) != GpuTensorStorage::Real {
        return Err(terminal("complex prototypes are not supported".into()));
    }
    let provider = gpu_helpers::exact_provider_for_handle(handle)
        .ok_or_else(|| terminal("no provider owns the resident prototype".into()))?;
    let output = match provider.random_permutation_like(handle, n, k) {
        Ok(output) => Some(output),
        Err(cause) if cause.is_unsupported() => None,
        Err(cause) => {
            return Err(crate::runtime_error::with_provider_execution_source(
                terminal(format!("provider permutation failed: {cause}")),
                cause,
            ))
        }
    };
    if let Some(mut output) = output {
        let contract = UnaryGpuOutputContract {
            storage: GpuTensorStorage::Real,
            precision: Some(ProviderPrecision::F64),
            integer: None,
            logical: false,
            alias: GpuOutputAliasPolicy::RequireDistinctAllocation,
        };
        if !gpu_helpers::unary_gpu_output_matches_shape(
            &output,
            handle,
            provider,
            &[1, k],
            contract,
        ) {
            return Err(gpu_helpers::reject_gpu_output(
                output,
                terminal("provider returned a malformed, aliased, or foreign result".into()),
            ));
        }
        if let Err(cause) = handle.allocation_lease().validate_live() {
            return Err(gpu_helpers::reject_gpu_output(
                output,
                terminal(format!(
                    "prototype allocation retired during permutation: {cause}"
                )),
            ));
        }
        if let Err(cause) = gpu_helpers::propagate_output_provenance(&mut output, [handle]) {
            return Err(gpu_helpers::reject_gpu_output(output, cause));
        }
        return gpu_helpers::resident_gpu_value(output).map_err(|mut cause| {
            cause.gpu_gather_retry = crate::GpuGatherRetry::Never;
            cause
        });
    }

    let tensor = randperm_tensor(n, k)?;
    gpu_helpers::restore_class_preserving_value_protecting(
        handle,
        Value::Tensor(tensor),
        "randperm",
        &[],
    )
}

fn randperm_tensor(n: usize, k: usize) -> crate::BuiltinResult<Tensor> {
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
        block_on(super::randperm_builtin(args))
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
    fn randperm_type_is_row_vector() {
        assert_eq!(
            randperm_type(&[Type::Num], &ResolveContext::new(Vec::new())),
            Type::Tensor {
                shape: Some(vec![Some(1), None])
            }
        );
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
        test_support::with_test_provider(|provider| {
            let proto_tensor = Tensor::new(vec![0.0, 0.0], vec![1, 2]).unwrap();
            let view = HostTensorView {
                data: &proto_tensor.materialize_f64(),
                shape: proto_tensor.shape(),
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
            assert_eq!(gathered.shape(), vec![1, 3]);
            let expected = expected_randperm(6, 3);
            assert_eq!(gathered.materialize_f64(), expected);
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
