//! MATLAB-compatible `randn` builtin with GPU-aware semantics for RunMat.
use runmat_builtins::{RANDN_COLUMN_SIZE_VECTOR_EXTENSION, RANDN_RESIDENT_SIZE_EXTENSION};

#[cfg(test)]
use runmat_accelerate_api::HostTensorView;
use runmat_accelerate_api::{GpuTensorHandle, ProviderPrecision};
use runmat_macros::runtime_builtin;
use runmat_value::{NumericDType, Tensor, Value};

use crate::build_runtime_error;
use crate::builtins::common::random;
use crate::builtins::common::random::gpu_like::ExactHostRandomMode;
use crate::builtins::common::random_args::{
    complex_tensor_into_value, extract_constructor_dimensions, keyword_of,
};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::tensor;
#[path = "randn/host_output.rs"]
mod host_output;
pub(crate) mod source_controls;
use source_controls::{normalized_shape, RandnKeyword};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::array::creation::randn")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "randn",
    op_kind: GpuOpKind::Custom("generator"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("random.gpu_child")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Ordinary output consumes the CPU session stream; explicit resident output uses the independent request-bound GPU child and never replays an unproven provider attempt on host.",
};

fn builtin_error(message: impl Into<String>) -> crate::RuntimeError {
    build_runtime_error(message).with_builtin("randn").build()
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
    binding_variant = "default",
    builtin_path = "crate::builtins::array::creation::randn"
)]
async fn randn_builtin(rest: crate::call::BuiltinArguments) -> crate::BuiltinResult<Value> {
    let parsed = ParsedRandn::parse(rest.view()).await?;
    build_output(parsed).await
}

/// Draws one already-described real source block from the caller's ordered
/// host turn. The caller owns the turn across every logical stochastic step;
/// this entry neither reparses controls nor acquires another random stream.
pub(crate) async fn draw_source_normal(
    turn: &mut random::RandomTurn,
    shape: &[usize],
    dtype: NumericDType,
    mode: ExactHostRandomMode,
) -> crate::BuiltinResult<Value> {
    let count = shape.iter().try_fold(1usize, |total, dimension| {
        total
            .checked_mul(*dimension)
            .ok_or_else(|| builtin_error("randn: output dimensions overflow"))
    })?;
    if dtype == NumericDType::F64 && count == 1 {
        let mut value = 0.0;
        if let Some(runtime) = turn.owner().cloned() {
            let mut budget = crate::execution::CpuCooperationBudget::new(
                runtime.execution().clone(),
                runtime.cancellation(),
            );
            random::fill_normal_cooperative(turn, 1, "randn", &runtime, &mut budget, |_, drawn| {
                value = drawn;
                Ok(())
            })
            .await?;
        } else {
            random::fill_normal(turn, 1, "randn", |_, drawn| {
                value = drawn;
                Ok(())
            })?;
        }
        return Ok(Value::Num(value));
    }
    let tensor = host_output::real(turn, shape, dtype, mode).await?;
    Ok(tensor::tensor_into_value(tensor))
}

struct ParsedRandn {
    shape: Vec<usize>,
    template: RandnTemplate,
    dtype: NumericDType,
}

#[derive(Clone)]
enum RandnTemplate {
    Double,
    GpuArray,
    Like(Value),
}

impl ParsedRandn {
    async fn parse(args: crate::call::FunctionArgumentView<'_>) -> crate::BuiltinResult<Self> {
        let mut dims: Vec<usize> = Vec::new();
        let mut saw_dims_arg = false;
        let mut template: Option<RandnTemplate> = None;
        let mut dtype: NumericDType = NumericDType::F64;
        let mut saw_size_vector = false;

        let mut idx = 0;
        while idx < args.len() {
            let arg = &args[idx];

            if let Some(keyword) = keyword_of(arg)? {
                match RandnKeyword::from_text(&keyword) {
                    RandnKeyword::Like => {
                        let Some(proto) = args.get(idx + 1).cloned() else {
                            return Err(builtin_error("randn: expected prototype after 'like'"));
                        };
                        template = Some(RandnTemplate::Like(proto));
                        idx += 2;
                        continue;
                    }
                    RandnKeyword::Double => {
                        template = Some(RandnTemplate::Double);
                        dtype = NumericDType::F64;
                        idx += 1;
                        continue;
                    }
                    RandnKeyword::Single => {
                        template = Some(RandnTemplate::Double);
                        dtype = NumericDType::F32;
                        idx += 1;
                        continue;
                    }
                    RandnKeyword::GpuArray => {
                        // MATLAB class-specification syntax: randn(m,n,"gpuArray") or
                        // gpuArray.randn(m,n). Keep explicit residency in the parsed
                        // template so the producer, rather than the dispatcher, owns it.
                        template = Some(RandnTemplate::GpuArray);
                        dtype = NumericDType::F64;
                        idx += 1;
                        continue;
                    }
                    RandnKeyword::Unknown => {
                        return Err(builtin_error(format!(
                            "randn: unrecognised option '{keyword}'"
                        )));
                    }
                }
            }

            if matches!(arg, Value::GpuTensor(_)) {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &RANDN_RESIDENT_SIZE_EXTENSION,
                    "randn",
                )?;
            }
            if let Some(parsed_dims) = extract_constructor_dimensions(arg, "randn")
                .await
                .map_err(builtin_error)?
            {
                if parsed_dims.is_column_vector {
                    crate::compatibility::ensure_builtin_extension_enabled(
                        &RANDN_COLUMN_SIZE_VECTOR_EXTENSION,
                        "randn",
                    )?;
                }
                if parsed_dims.values.len() > 1 {
                    if saw_size_vector || saw_dims_arg {
                        return Err(builtin_error(
                            "randn: a size vector must be the only dimension argument",
                        ));
                    }
                    saw_size_vector = true;
                } else if saw_size_vector {
                    return Err(builtin_error(
                        "randn: a size vector must be the only dimension argument",
                    ));
                }
                saw_dims_arg = true;
                if dims.is_empty() {
                    dims = parsed_dims.values;
                } else {
                    dims.extend(parsed_dims.values);
                }
                idx += 1;
                continue;
            }

            return Err(builtin_error(format!(
                "randn: unsupported dimension or option {arg:?}"
            )));
        }

        let shape = normalized_shape(dims, saw_dims_arg);

        let template = template.unwrap_or(RandnTemplate::Double);

        Ok(Self {
            shape,
            template,
            dtype,
        })
    }
}

async fn build_output(parsed: ParsedRandn) -> crate::BuiltinResult<Value> {
    match &parsed.template {
        RandnTemplate::GpuArray => return randn_gpu(&parsed.shape).await,
        RandnTemplate::Like(Value::GpuTensor(handle)) => {
            return match random::gpu_like::intent("randn", handle)? {
                random::gpu_like::GpuLikeIntent::ExplicitGpuStream => {
                    randn_like_gpu(handle, &parsed.shape).await
                }
                random::gpu_like::GpuLikeIntent::AutomaticHostStream => {
                    let class = random::gpu_like::class(handle).ok_or_else(|| {
                        builtin_error("randn: unsupported gpuArray prototype class")
                    })?;
                    if !matches!(class, random::gpu_like::GpuLikeClass::Floating { .. }) {
                        return Err(builtin_error(
                            "randn: 'like' prototype must have single or double underlying type",
                        ));
                    }
                    let host_mode = random::gpu_like::host_random_mode(
                        crate::context::legacy::active().as_ref(),
                    );
                    let scratch = random::gpu_like::exact_host_workspace(
                        "randn",
                        random::gpu_like::ExactHostRandomWorkspace::DirectOutput,
                        host_mode,
                    )?;
                    let upload_scratch = match class {
                        random::gpu_like::GpuLikeClass::Floating {
                            precision,
                            complex: true,
                        } => random::gpu_like::complex_upload_workspace_bytes(
                            "randn",
                            &parsed.shape,
                            precision,
                        )?,
                        _ => scratch,
                    };
                    let presentation = random::gpu_like::select_presentation(
                        "randn",
                        handle,
                        &parsed.shape,
                        class,
                        random::gpu_like::host_result_bytes("randn", &parsed.shape, class, host_mode)?,
                        scratch,
                        upload_scratch,
                        match upload_scratch {
                            runmat_execution::ResourceByteBound::Known(bytes)
                                if class.native_storage()
                                    == runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved => bytes,
                            _ => 0,
                        },
                        host_mode,
                    )?;
                    let mut turn = random::acquire_turn(presentation.runtime(), "randn").await?;
                    presentation.checkpoint_for_turn(&turn)?;
                    let result = match class {
                        random::gpu_like::GpuLikeClass::Floating { precision, complex } => {
                            let dtype = match precision {
                                ProviderPrecision::F32 => NumericDType::F32,
                                ProviderPrecision::F64 => NumericDType::F64,
                            };
                            if complex {
                                host_output::complex(
                                    &mut turn,
                                    &parsed.shape,
                                    dtype,
                                    presentation.mode(),
                                )
                                .await
                                .map(complex_tensor_into_value)
                            } else {
                                host_output::real(
                                    &mut turn,
                                    &parsed.shape,
                                    dtype,
                                    presentation.mode(),
                                )
                                .await
                                .map(tensor::tensor_into_value)
                            }
                        }
                        random::gpu_like::GpuLikeClass::Integer { .. }
                        | random::gpu_like::GpuLikeClass::Logical => Err(builtin_error(
                            "randn: 'like' prototype must have single or double underlying type",
                        )),
                    };
                    presentation.finish_result("randn", result, &parsed.shape)
                }
            };
        }
        _ => {}
    }
    let mut turn = random::acquire_turn(None, "randn").await?;
    match parsed.template {
        RandnTemplate::Double => match parsed.dtype {
            NumericDType::F64 => randn_double(&mut turn, &parsed.shape).await,
            NumericDType::F32 => randn_single(&mut turn, &parsed.shape).await,
            NumericDType::I8
            | NumericDType::I16
            | NumericDType::I32
            | NumericDType::I64
            | NumericDType::U8
            | NumericDType::U16
            | NumericDType::U32
            | NumericDType::U64 => randn_double(&mut turn, &parsed.shape).await,
        },
        RandnTemplate::GpuArray => unreachable!("GPU output handled before host turn"),
        RandnTemplate::Like(proto) => randn_like(&mut turn, &proto, &parsed.shape).await,
    }
}

async fn randn_gpu(shape: &[usize]) -> crate::BuiltinResult<Value> {
    let Some(provider) = runmat_accelerate_api::provider() else {
        return Err(builtin_error(
            "randn: gpuArray output requires an active double-precision provider",
        ));
    };
    if provider.precision() != ProviderPrecision::F64 {
        return Err(builtin_error(
            "randn: gpuArray output requires an active double-precision provider",
        ));
    }
    crate::builtins::common::random::gpu_execution::explicit_real(
        "randn",
        provider,
        shape,
        ProviderPrecision::F64,
        runmat_accelerate_api::random::GpuRandomDistribution::StandardNormal,
        None,
    )
    .map(Value::GpuTensor)
}

async fn randn_double(
    turn: &mut random::RandomTurn,
    shape: &[usize],
) -> crate::BuiltinResult<Value> {
    let mode = random::gpu_like::host_random_mode(turn.owner());
    draw_source_normal(turn, shape, NumericDType::F64, mode).await
}

async fn host_normal_tensor(
    turn: &mut random::RandomTurn,
    shape: &[usize],
    dtype: NumericDType,
) -> crate::BuiltinResult<Tensor> {
    let mode = random::gpu_like::host_random_mode(turn.owner());
    host_output::real(turn, shape, dtype, mode).await
}

async fn randn_single(
    turn: &mut random::RandomTurn,
    shape: &[usize],
) -> crate::BuiltinResult<Value> {
    host_normal_tensor(turn, shape, NumericDType::F32)
        .await
        .map(Value::Tensor)
}

#[async_recursion::async_recursion(?Send)]
async fn randn_like(
    turn: &mut random::RandomTurn,
    proto: &Value,
    shape: &[usize],
) -> crate::BuiltinResult<Value> {
    match proto {
        Value::GpuTensor(_) => unreachable!("GPU prototype handled before host turn"),
        Value::ComplexTensor(tensor) => {
            crate::builtins::common::validation::reject_typed_complex_integer_tensor(
                tensor, "randn",
            )?;
            randn_complex(turn, shape, tensor.numeric_dtype()).await
        }
        Value::Complex(_, _) => randn_complex(turn, shape, NumericDType::F64).await,
        Value::Tensor(tensor) => match tensor.numeric_dtype() {
            NumericDType::F32 => randn_single(turn, shape).await,
            NumericDType::F64 => randn_double(turn, shape).await,
            _ => Err(builtin_error(
                "randn: 'like' prototype must be single or double",
            )),
        },
        Value::Num(_) => randn_double(turn, shape).await,
        Value::Int(_)
        | Value::Bool(_)
        | Value::LogicalArray(_)
        | Value::CharArray(_)
        | Value::Cell(_) => Err(builtin_error(
            "randn: 'like' prototype must be single or double",
        )),
        other => Err(builtin_error(format!(
            "randn: unsupported prototype {other:?}"
        ))),
    }
}

async fn randn_complex(
    turn: &mut random::RandomTurn,
    shape: &[usize],
    dtype: NumericDType,
) -> crate::BuiltinResult<Value> {
    let mode = random::gpu_like::host_random_mode(turn.owner());
    host_output::complex(turn, shape, dtype, mode)
        .await
        .map(complex_tensor_into_value)
}

async fn randn_like_gpu(handle: &GpuTensorHandle, shape: &[usize]) -> crate::BuiltinResult<Value> {
    if runmat_accelerate_api::handle_integer_type(handle).is_some()
        || runmat_accelerate_api::handle_is_logical(handle)
    {
        return Err(builtin_error(
            "randn: 'like' prototype must have single or double underlying type",
        ));
    }
    let provider = crate::builtins::common::gpu_helpers::exact_provider_for_handle(handle)
        .ok_or_else(|| builtin_error("randn: no live provider owns the gpuArray prototype"))?;
    let precision = runmat_accelerate_api::handle_precision(handle)
        .ok_or_else(|| builtin_error("randn: unsupported gpuArray prototype precision"))?;
    if runmat_accelerate_api::handle_storage(handle)
        == runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
    {
        return crate::builtins::common::random::gpu_execution::explicit_complex(
            "randn",
            provider,
            shape,
            precision,
            runmat_accelerate_api::random::GpuRandomDistribution::StandardNormal,
            handle,
        )
        .map(Value::GpuTensor);
    }
    crate::builtins::common::random::gpu_execution::explicit_real(
        "randn",
        provider,
        shape,
        precision,
        runmat_accelerate_api::random::GpuRandomDistribution::StandardNormal,
        Some(handle),
    )
    .map(Value::GpuTensor)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::{gpu_helpers, random, test_support};
    use futures::executor::block_on;
    use runmat_accelerate_api::test_support::{RetirementProbe, TestAllocationOwner};
    use runmat_accelerate_api::AccelProvider;
    use runmat_value::{ComplexTensor, IntValue, IntegerComplexStorage, IntegerStorage};
    use std::sync::Mutex;

    struct InvalidExplicitRandnProvider {
        owner: TestAllocationOwner,
        result: Mutex<Option<RetirementProbe>>,
    }

    impl AccelProvider for InvalidExplicitRandnProvider {
        fn upload(
            &self,
            _host: &runmat_accelerate_api::HostTensorView,
        ) -> anyhow::Result<GpuTensorHandle> {
            unreachable!("no upload for invalid constructor result")
        }

        fn download<'a>(
            &'a self,
            _handle: &'a GpuTensorHandle,
        ) -> runmat_accelerate_api::AccelDownloadFuture<'a> {
            unreachable!("no download for invalid constructor result")
        }

        fn device_info(&self) -> String {
            "invalid owned randn result".into()
        }

        fn device_id(&self) -> u32 {
            self.owner.device_id()
        }

        fn propose_gpu_random(
            &self,
            request: &runmat_accelerate_api::random::GpuRandomRequest<'_>,
        ) -> runmat_accelerate_api::ProviderExecutionResult<
            Option<runmat_accelerate_api::random::GpuRandomPhysicalProposal>,
        > {
            let count = request.element_count().expect("test request geometry");
            Ok(Some(
                runmat_accelerate_api::random::GpuRandomPhysicalProposal {
                    output_buffer_bytes: u64::try_from(count).unwrap() * 8,
                    uniform_buffer_bytes: 0,
                    uniform_allocations: 0,
                    dispatches: 1,
                    queue_submissions: 0,
                },
            ))
        }

        fn execute_gpu_random(
            &self,
            _request: &runmat_accelerate_api::random::GpuRandomRequest<'_>,
            _metadata: Option<&dyn runmat_execution::resource::ProviderHostMetadataAdmission>,
        ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
            let allocation = self
                .owner
                .allocate_zeroed(
                    vec![1, 1],
                    runmat_accelerate_api::NumericElementType::F64,
                    runmat_accelerate_api::GpuTensorStorage::Real,
                )
                .map_err(runmat_accelerate_api::ProviderExecutionError::from)?;
            *self.result.lock().unwrap() = Some(allocation.probe);
            Ok(allocation.handle)
        }
    }

    fn reset_rng_clean() -> impl Drop {
        let guard = random::test_guard();
        runmat_accelerate_api::clear_provider();
        random::reset_rng();
        guard
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn randn_default_scalar() {
        let _guard = random::test_guard();
        let _guard = reset_rng_clean();
        let result = block_on(randn_builtin(
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .expect("randn");
        let expected = random::expected_normal_sequence(1)[0];
        match result {
            Value::Num(v) => assert!((v - expected).abs() < 1e-12),
            other => panic!("expected scalar double, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn randn_square_from_single_dimension() {
        let _guard = random::test_guard();
        let _guard = reset_rng_clean();
        let args = vec![Value::Num(2.0)];
        let result = block_on(randn_builtin(
            crate::builtins::common::test_support::builtin_arguments(args),
        ))
        .expect("randn");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape(), &[2, 2]);
                let expected = random::expected_normal_sequence(4);
                for (observed, exp) in t.materialize_f64().iter().zip(expected.iter()) {
                    assert!((*observed - *exp).abs() < 1e-12);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn randn_size_vector_argument() {
        let _guard = random::test_guard();
        let _guard = reset_rng_clean();
        let size_vec = Tensor::new(vec![2.0, 3.0, 4.0], vec![1, 3]).unwrap();
        let result = block_on(randn_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Tensor(size_vec)]),
        ))
        .expect("randn");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape(), &[2, 3, 4]);
                let expected = random::expected_normal_sequence(24);
                assert_eq!(t.materialize_f64().len(), expected.len());
                for (observed, exp) in t.materialize_f64().iter().zip(expected.iter()) {
                    assert!((*observed - *exp).abs() < 1e-12);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn randn_column_and_resident_size_controls_follow_compatibility_mode() {
        let _guard = random::test_guard();
        reset_rng_clean();
        let column = || {
            Value::Tensor(
                Tensor::new_integer(IntegerStorage::U8(vec![2, 3]), vec![2, 1])
                    .expect("column size vector"),
            )
        };
        let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
        let Value::Tensor(output) = block_on(randn_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![column()]),
        ))
        .expect("RunMat column size") else {
            panic!("expected tensor");
        };
        assert_eq!(output.shape(), &[2, 3]);
        drop(_runmat);

        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = block_on(randn_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![column()]),
        ))
        .expect_err("strict column size");
        assert_eq!(
            error.identifier(),
            RANDN_COLUMN_SIZE_VECTOR_EXTENSION.error_identifier
        );

        let handle = runmat_accelerate_api::test_support::TestAllocationOwner::new()
            .allocate_zeroed(
                vec![1, 1],
                runmat_accelerate_api::NumericElementType::U8,
                runmat_accelerate_api::GpuTensorStorage::Real,
            )
            .expect("genuine unregistered allocation")
            .handle;
        let error = block_on(randn_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![Value::GpuTensor(
                handle.clone(),
            )]),
        ))
        .expect_err("strict resident size");
        assert_eq!(
            error.identifier(),
            RANDN_RESIDENT_SIZE_EXTENSION.error_identifier
        );
        drop(handle);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn randn_zero_dimension_returns_empty() {
        let _guard = random::test_guard();
        let _guard = reset_rng_clean();
        let result = block_on(randn_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(0.0)]),
        ))
        .expect("randn");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape(), &[0, 0]);
                assert!(t.materialize_f64().is_empty());
            }
            other => panic!("expected empty tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn randn_single_precision_produces_f32() {
        let _guard = random::test_guard();
        let _guard = reset_rng_clean();
        let result = block_on(randn_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![Value::from("single")]),
        ))
        .expect("randn single");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.numeric_dtype(), NumericDType::F32);
                assert_eq!(t.shape(), &[1, 1]);
            }
            Value::Num(_) => {
                // Fallback scalar path remains F64-backed but should definitely not allocate on GPU.
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn randn_like_without_dims_is_scalar_and_implicit_prototype_is_rejected() {
        let _guard = random::test_guard();
        let _guard = reset_rng_clean();
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let args = vec![Value::from("like"), Value::Tensor(tensor.clone())];
        let result = block_on(randn_builtin(
            crate::builtins::common::test_support::builtin_arguments(args),
        ))
        .expect("randn");
        assert!(matches!(result, Value::Num(value) if value.is_finite()));

        let error = block_on(randn_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Tensor(tensor)]),
        ))
        .unwrap_err();
        assert!(error.message().contains("unsupported dimension or option"));
    }

    #[test]
    fn randn_like_rejects_every_integer_class_and_nonfloating_prototypes() {
        let _guard = random::test_guard();
        let storages = vec![
            IntegerStorage::I8(vec![i8::MIN]),
            IntegerStorage::I16(vec![i16::MIN]),
            IntegerStorage::I32(vec![i32::MIN]),
            IntegerStorage::I64(vec![i64::MIN]),
            IntegerStorage::U8(vec![u8::MAX]),
            IntegerStorage::U16(vec![u16::MAX]),
            IntegerStorage::U32(vec![u32::MAX]),
            IntegerStorage::U64(vec![u64::MAX]),
        ];

        for storage in storages {
            let prototype = Tensor::new_integer(storage, vec![1, 1]).expect("prototype");
            let error = block_on(randn_builtin(
                crate::builtins::common::test_support::builtin_arguments(vec![
                    Value::Num(1.0),
                    Value::Num(2.0),
                    Value::from("like"),
                    Value::Tensor(prototype),
                ]),
            ))
            .unwrap_err();
            assert!(error.message().contains("must be single or double"));
        }

        let scalar = block_on(randn_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Num(2.0),
                Value::from("like"),
                Value::Int(IntValue::I64(i64::MAX)),
            ]),
        ))
        .unwrap_err();
        assert!(scalar.message().contains("must be single or double"));

        let logical = block_on(randn_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Num(2.0),
                Value::from("like"),
                Value::Bool(true),
            ]),
        ))
        .unwrap_err();
        assert!(logical.message().contains("must be single or double"));

        let complex_integer = ComplexTensor::new_integer(
            IntegerComplexStorage::new(IntegerStorage::I16(vec![1]), IntegerStorage::I16(vec![2]))
                .unwrap(),
            vec![1, 1],
        )
        .unwrap();
        let complex_integer = block_on(randn_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Num(2.0),
                Value::from("like"),
                Value::ComplexTensor(complex_integer),
            ]),
        ))
        .unwrap_err();
        assert!(complex_integer
            .message()
            .contains("complex numbers with integer types"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn randn_like_complex_produces_complex_tensor() {
        let _guard = random::test_guard();
        for (prototype, dtype) in [
            (Value::Complex(0.0, 1.0), NumericDType::F64),
            (
                Value::ComplexTensor(ComplexTensor::new(vec![(0.0, 1.0)], vec![1, 1]).unwrap()),
                NumericDType::F64,
            ),
            (
                Value::ComplexTensor(
                    ComplexTensor::from_f32(vec![(0.0, 1.0)], vec![1, 1]).unwrap(),
                ),
                NumericDType::F32,
            ),
        ] {
            let _rng = reset_rng_clean();
            let result = block_on(randn_builtin(
                crate::builtins::common::test_support::builtin_arguments(vec![
                    Value::Num(2.0),
                    Value::Num(1.0),
                    Value::from("like"),
                    prototype,
                ]),
            ))
            .expect("randn");
            let Value::ComplexTensor(tensor) = result else {
                panic!("expected complex tensor, got {result:?}")
            };
            assert_eq!(tensor.shape(), &[2, 1]);
            assert_eq!(tensor.numeric_dtype(), dtype);
            let component = |normal: f64| {
                let scaled = normal * std::f64::consts::FRAC_1_SQRT_2;
                if dtype == NumericDType::F32 {
                    f64::from(scaled as f32)
                } else {
                    scaled
                }
            };
            // Complex samples consume the same real draws in pairs, with
            // variance one half per component and native output precision.
            for ((real, imaginary), expected) in tensor
                .materialize_f64()
                .iter()
                .zip(random::expected_normal_sequence(4).chunks_exact(2))
            {
                assert_eq!(*real, component(expected[0]));
                assert_eq!(*imaginary, component(expected[1]));
            }
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn randn_gpuarray_keyword_produces_valid_output() {
        let _guard = random::test_guard();
        let _guard = reset_rng_clean();
        test_support::with_test_provider(|_| {
            let args = vec![Value::Num(3.0), Value::Num(4.0), Value::from("gpuArray")];
            let result = block_on(randn_builtin(
                crate::builtins::common::test_support::builtin_arguments(args),
            ))
            .expect("randn gpuArray");
            match result {
                Value::GpuTensor(h) => {
                    assert_eq!(h.shape(), vec![3, 4]);
                    assert!(runmat_accelerate_api::handle_is_explicit(&h));
                    assert_eq!(
                        runmat_accelerate_api::handle_precision(&h),
                        Some(ProviderPrecision::F64)
                    );
                    let gathered =
                        test_support::gather(Value::GpuTensor(h.clone())).expect("gather output");
                    assert!(gathered
                        .materialize_f64()
                        .iter()
                        .all(|value| value.is_finite()));
                    let lease = h.allocation_lease().downgrade();
                    drop(h);
                    assert!(lease.acquire().is_none());
                }
                other => panic!("explicit gpuArray requires resident output, got {other:?}"),
            }
        });
    }

    #[test]
    fn randn_parser_records_explicit_residency_in_its_typed_template() {
        let parsed = block_on(ParsedRandn::parse(
            crate::call::FunctionArgumentView::from_slice(&[Value::from("gpuArray")]),
        ))
        .expect("gpuArray form");
        assert!(matches!(parsed.template, RandnTemplate::GpuArray));

        let parsed = block_on(ParsedRandn::parse(
            crate::call::FunctionArgumentView::from_slice(&[]),
        ))
        .expect("default form");
        assert!(matches!(parsed.template, RandnTemplate::Double));
    }

    #[test]
    fn randn_gpu_rejects_and_frees_invalid_provider_output() {
        let _guard = test_support::accel_test_lock();
        let provider = Box::leak(Box::new(InvalidExplicitRandnProvider {
            owner: TestAllocationOwner::new(),
            result: Mutex::new(None),
        }));
        let _thread = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));

        let error =
            block_on(randn_gpu(&[2, 2])).expect_err("invalid provider output must be terminal");
        assert!(error.message().contains("invalid constructor result"));
        assert_eq!(
            provider
                .result
                .lock()
                .unwrap()
                .as_ref()
                .unwrap()
                .retirements(),
            1
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn randn_gpu_like_roundtrip() {
        let _guard = random::test_guard();
        random::reset_rng();
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![0.0; 4], vec![2, 2]).unwrap();
            let view = HostTensorView {
                data: &tensor.materialize_f64(),
                shape: tensor.shape(),
            };
            let handle = provider
                .upload(&view)
                .expect("upload")
                .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
                .expect("explicit gpuArray prototype");
            let args = vec![
                Value::Num(2.0),
                Value::Num(2.0),
                Value::from("like"),
                Value::GpuTensor(handle),
            ];
            let result = block_on(randn_builtin(
                crate::builtins::common::test_support::builtin_arguments(args),
            ))
            .expect("randn");
            match result {
                Value::GpuTensor(gpu) => {
                    assert_eq!(gpu.shape(), vec![2, 2]);
                    let gathered = test_support::gather(Value::GpuTensor(gpu)).expect("gather");
                    assert_eq!(gathered.shape(), &[2, 2]);
                    for value in gathered.materialize_f64() {
                        assert!(value.is_finite());
                    }
                }
                other => panic!("expected GPU tensor, got {other:?}"),
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn automatic_like_normal_uses_cpu_stream_and_preserves_single_class() {
        let _guard = random::test_guard();
        random::reset_rng();
        test_support::with_test_provider(|provider| {
            let prototype = Tensor::from_f32(vec![0.0; 4], vec![2, 2]).expect("single");
            let source = gpu_helpers::upload_tensor(provider, &prototype)
                .expect("upload prototype")
                .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Automatic)
                .expect("automatic prototype");
            let child_before = random::gpu::snapshot().expect("child before");
            let expected = block_on(randn_builtin(
                crate::builtins::common::test_support::builtin_arguments(vec![
                    Value::Num(2.0),
                    Value::from("single"),
                ]),
            ))
            .expect("host single normal");
            random::reset_rng();
            let actual = block_on(randn_builtin(
                crate::builtins::common::test_support::builtin_arguments(vec![
                    Value::Num(2.0),
                    Value::from("like"),
                    Value::GpuTensor(source),
                ]),
            ))
            .expect("automatic normal");
            assert_eq!(actual, expected);
            assert_eq!(random::gpu::snapshot().expect("child after"), child_before);
            let Value::Tensor(output) = actual else {
                panic!("single storage must survive");
            };
            assert_eq!(output.numeric_dtype(), NumericDType::F32);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn automatic_complex_like_uses_cpu_stream_and_single_class() {
        let _guard = random::test_guard();
        random::reset_rng();
        test_support::with_f32_test_provider(|provider| {
            let prototype = ComplexTensor::from_f32(vec![(0.0, 0.0); 4], vec![2, 2])
                .expect("complex single prototype");
            let source = gpu_helpers::upload_complex_tensor(provider, &prototype)
                .expect("upload prototype")
                .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Automatic)
                .expect("automatic prototype");
            let child_before = random::gpu::snapshot().expect("child before");
            let expected = block_on(randn_builtin(
                crate::builtins::common::test_support::builtin_arguments(vec![
                    Value::Num(2.0),
                    Value::from("like"),
                    Value::ComplexTensor(prototype),
                ]),
            ))
            .expect("host complex normal");
            random::reset_rng();
            let actual = block_on(randn_builtin(
                crate::builtins::common::test_support::builtin_arguments(vec![
                    Value::Num(2.0),
                    Value::from("like"),
                    Value::GpuTensor(source),
                ]),
            ))
            .expect("automatic complex normal");
            assert_eq!(actual, expected);
            assert_eq!(random::gpu::snapshot().expect("child after"), child_before);
            let Value::ComplexTensor(output) = actual else {
                panic!("complex output required");
            };
            assert_eq!(output.numeric_dtype(), NumericDType::F32);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn randn_rejects_integer_gpu_like_prototype() {
        let _guard = random::test_guard();
        test_support::with_test_provider(|provider| {
            let values = [i64::MIN, 9_007_199_254_740_993];
            let shape = [1usize, 2usize];
            let handle = provider
                .upload_integer(&runmat_accelerate_api::HostIntegerTensorView {
                    data: runmat_accelerate_api::HostIntegerDataView::I64(&values),
                    shape: &shape,
                })
                .expect("upload integer gpu prototype")
                .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
                .expect("explicit gpuArray prototype");
            let error = block_on(randn_builtin(
                crate::builtins::common::test_support::builtin_arguments(vec![
                    Value::Num(2.0),
                    Value::from("like"),
                    Value::GpuTensor(handle),
                ]),
            ))
            .unwrap_err();
            assert!(error.message().contains("single or double underlying type"));
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
            data: &tensor.materialize_f64(),
            shape: tensor.shape(),
        };
        let provider = runmat_accelerate_api::provider().unwrap();
        let handle = provider.upload(&view).expect("upload");
        let result = block_on(randn_like_gpu(&handle, &[2, 2])).expect("randn like gpu");
        match result {
            Value::GpuTensor(h) => {
                let gathered = test_support::gather(Value::GpuTensor(h)).expect("gather to host");
                assert_eq!(gathered.shape(), &[2, 2]);
                for v in gathered.materialize_f64() {
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
        assert_eq!(gathered.shape(), &[4, 4]);
        assert!(
            gathered
                .materialize_f64()
                .iter()
                .any(|value| value.is_finite() && value.abs() > 1.0e-6),
            "expected at least one non-trivial normal sample"
        );
    }
}
