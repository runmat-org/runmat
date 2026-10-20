use super::*;
use crate::builtins::common::random;
use runmat_accelerate_api::{
    test_support::TestAllocationOwner, AccelDownloadFuture, AccelNumericDownloadFuture,
    AccelProvider, GpuTensorHandle, GpuTensorStorage, HostTensorView, NumericElementType,
    ThreadProviderGuard,
};
use runmat_value::IntegerStorage;
use std::sync::atomic::{AtomicUsize, Ordering};

#[test]
fn registered_tsne_preserves_pre_entry_cancellation_cause() {
    use runmat_execution::cancellation::CancellationSource;
    use runmat_execution::CancellationReason;
    use std::rc::Rc;

    let source = CancellationSource::new();
    let runtime = crate::context::RuntimeContext::with_cancellation(
        Rc::new(crate::execution::RuntimeExecutionService::new()),
        source.token(),
    );
    source.cancel(CancellationReason::User);
    let error = futures::executor::block_on(tsne_builtin_registered(
        Some(runtime),
        tensor(vec![0.0, 1.0, 2.0, 3.0], vec![2, 2]),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect_err("cancelled invocation must not prepare observations");
    assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
    assert!(matches!(
        error
            .source
            .as_deref()
            .and_then(|source| source.downcast_ref::<crate::execution::ExecutionServiceError>()),
        Some(crate::execution::ExecutionServiceError::Cancelled)
    ));
    assert_eq!(error.gpu_gather_retry(), crate::GpuGatherRetry::Never);
}

#[test]
fn observation_materialization_yields_and_cancels_before_solver() {
    use crate::execution::{CpuCooperationBudget, CpuCooperationPolicy};
    use runmat_execution::cancellation::CancellationSource;
    use runmat_execution::CancellationReason;
    use std::cell::Cell;
    use std::num::NonZeroU64;
    use std::rc::Rc;

    let observations = Tensor::new(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0], vec![3, 2]).unwrap();
    let expected = futures::executor::block_on(geometry::prepare_data_cooperative(
        &observations,
        &mut cooperation::TsneCooperation::new(None),
    ))
    .unwrap();
    let source = Rc::new(CancellationSource::new());
    let turns = Rc::new(Cell::new(0));
    let cancel_on_turn = source.clone();
    let counted = turns.clone();
    let budget = CpuCooperationBudget::with_test_turn(
        CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(1).unwrap()),
        source.token(),
        move || {
            counted.set(counted.get() + 1);
            if counted.get() == 4 {
                cancel_on_turn.cancel(CancellationReason::User);
            }
            Box::pin(async { Ok(()) })
        },
    );
    let error = futures::executor::block_on(geometry::prepare_data_cooperative(
        &observations,
        &mut cooperation::TsneCooperation::with_budget(budget),
    ))
    .expect_err("cancelled observation materialization must not reach the solver");
    assert_eq!(
        expected.rows,
        vec![vec![0.0, 3.0], vec![1.0, 4.0], vec![2.0, 5.0]]
    );
    assert_eq!(turns.get(), 4);
    assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
    assert!(matches!(
        error
            .source
            .as_deref()
            .and_then(|source| source.downcast_ref::<crate::execution::ExecutionServiceError>()),
        Some(crate::execution::ExecutionServiceError::Cancelled)
    ));
}

#[test]
fn option_materialization_cancellation_does_not_publish_partial_controls() {
    use crate::execution::{CpuCooperationBudget, CpuCooperationPolicy};
    use runmat_execution::cancellation::CancellationSource;
    use runmat_execution::CancellationReason;
    use std::num::NonZeroU64;
    use std::rc::Rc;

    let source = Rc::new(CancellationSource::new());
    let cancel = source.clone();
    let budget = CpuCooperationBudget::with_test_turn(
        CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(1).unwrap()),
        source.token(),
        move || {
            cancel.cancel(CancellationReason::User);
            Box::pin(async { Ok(()) })
        },
    );
    let options = vec![
        admission::AdmittedOption::Perplexity(Value::Num(2.0)),
        admission::AdmittedOption::LearnRate(Value::Num(100.0)),
    ];
    let Err(error) = futures::executor::block_on(materialize_options(
        options,
        &mut cooperation::TsneCooperation::with_budget(budget),
    )) else {
        panic!("a cancelled materialization must not publish a partial option list");
    };
    assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
    assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
}

#[test]
fn dropping_pending_option_materialization_does_not_resume_work() {
    use crate::execution::{CpuCooperationBudget, CpuCooperationPolicy};
    use futures::task::noop_waker;
    use runmat_execution::cancellation::CancellationSource;
    use std::cell::Cell;
    use std::future::Future;
    use std::num::NonZeroU64;
    use std::pin::Pin;
    use std::rc::Rc;
    use std::task::{Context, Poll};

    let source = CancellationSource::new();
    let turns = Rc::new(Cell::new(0));
    let observed = turns.clone();
    let budget = CpuCooperationBudget::with_test_turn(
        CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(1).unwrap()),
        source.token(),
        move || {
            observed.set(observed.get() + 1);
            Box::pin(std::future::pending())
        },
    );
    let mut cooperation = cooperation::TsneCooperation::with_budget(budget);
    let mut future = Box::pin(materialize_options(
        vec![
            admission::AdmittedOption::Perplexity(Value::Num(2.0)),
            admission::AdmittedOption::LearnRate(Value::Num(100.0)),
        ],
        &mut cooperation,
    ));
    let waker = noop_waker();
    let mut context = Context::from_waker(&waker);
    assert!(matches!(
        Pin::as_mut(&mut future).poll(&mut context),
        Poll::Pending
    ));
    assert_eq!(
        turns.get(),
        1,
        "the first control reached a real cooperation turn"
    );
    drop(future);
    assert_eq!(
        turns.get(),
        1,
        "dropping the pending future cannot resume control work"
    );
}

struct RejectingDownloadProvider {
    owner: TestAllocationOwner,
    downloads: AtomicUsize,
}

impl AccelProvider for RejectingDownloadProvider {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }
    fn device_info(&self) -> String {
        "t-SNE admission fixture".into()
    }
    fn upload(&self, _: &HostTensorView<'_>) -> anyhow::Result<GpuTensorHandle> {
        anyhow::bail!("unexpected upload")
    }
    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        self.downloads.fetch_add(1, Ordering::SeqCst);
        Box::pin(async { anyhow::bail!("unexpected download") })
    }
    fn download_numeric<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelNumericDownloadFuture<'a> {
        self.downloads.fetch_add(1, Ordering::SeqCst);
        Box::pin(async { anyhow::bail!("unexpected native download") })
    }
}

#[tokio::test]
async fn malformed_host_visible_controls_reject_before_provider_work() {
    let provider = Box::leak(Box::new(RejectingDownloadProvider {
        owner: TestAllocationOwner::new(),
        downloads: AtomicUsize::new(0),
    }));
    let _selected = ThreadProviderGuard::set(Some(provider));
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let resident = provider
        .owner
        .allocate_zeroed(vec![2, 2], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let observation = Value::GpuTensor(resident.handle.clone());
    for controls in [
        vec![Value::from("Algorithm"), Value::Num(3.0)],
        vec![Value::from("Distance"), Value::from("not-a-metric")],
        vec![Value::from("NumDimensions"), Value::Num(0.0)],
        vec![Value::from("Perplexity"), Value::Num(-1.0)],
        vec![
            Value::from("NumDimensions"),
            Value::Tensor(Tensor::new(vec![2.0, 3.0], vec![1, 2]).unwrap()),
        ],
        vec![Value::from("Options"), {
            let mut fields = runmat_value::UnboundedStructValueBuilder::new();
            fields.insert("OutputFcn", observation.clone());
            Value::Struct(fields.finish())
        }],
        vec![Value::from("Options"), {
            let mut fields = runmat_value::UnboundedStructValueBuilder::new();
            fields.insert("MaxIter", Value::Num(0.0));
            Value::Struct(fields.finish())
        }],
    ] {
        assert!(tsne_builtin(observation.clone(), controls).await.is_err());
        assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
    }
}

#[test]
fn option_spellings_and_preflight_are_typed_and_preeffect() {
    assert_eq!(
        TsneOption::from_keyword("NumDimensions"),
        Some(TsneOption::NumDimensions)
    );
    assert_eq!(
        DistanceMetric::from_keyword("fastseuclidean"),
        Some(DistanceMetric::Seuclidean)
    );
    assert_eq!(Algorithm::from_keyword("exact"), Some(Algorithm::Exact));
    assert!(
        option_headers(crate::builtins::common::test_support::builtin_arguments(
            vec![Value::from("Distance")]
        ))
        .is_err()
    );
    assert!(
        option_headers(crate::builtins::common::test_support::builtin_arguments(
            vec![Value::from("unknown"), Value::from("value")]
        ))
        .is_err()
    );
    assert!(preflight_observations(&Value::from("not observations")).is_err());
}

fn tensor(data: Vec<f64>, shape: Vec<usize>) -> Value {
    Value::Tensor(Tensor::new(data, shape).unwrap())
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
async fn tsne_preserves_rank_zero_and_rank_one_observation_geometry() {
    let error = tsne_builtin(tensor(vec![7.0], Vec::new()), Vec::new())
        .await
        .expect_err("rank-zero input has no cached observation rows");
    assert!(error
        .message()
        .contains("at least one complete observation is required"));

    let Value::Tensor(embedding) = tsne_builtin(tensor(vec![1.0, 2.0, 3.0], vec![3]), Vec::new())
        .await
        .expect("rank-one input is one observation with three features")
    else {
        panic!("expected embedding tensor");
    };
    assert_eq!(embedding.shape(), [1, 2]);
    assert_eq!(embedding.materialize_f64(), vec![0.0, 0.0]);

    let error = tsne_builtin(
        tensor(vec![0.0, 1.0, 2.0, 3.0], vec![2, 2]),
        vec![
            Value::String("InitialY".into()),
            tensor(vec![4.0, 5.0], vec![2]),
        ],
    )
    .await
    .expect_err("rank-one InitialY has one cached row, not two");
    assert!(error.message().contains("InitialY row count"));
}

fn poisoned_int_tensor(storage: IntegerStorage, shape: Vec<usize>) -> Value {
    let tensor = Tensor::new_integer(storage, shape).unwrap();
    Value::Tensor(tensor)
}

fn cleared_int_tensor(storage: IntegerStorage, shape: Vec<usize>) -> Value {
    let tensor = Tensor::new_integer(storage, shape).unwrap();
    Value::Tensor(tensor)
}

fn reset_rng() -> impl Drop {
    let guard = random::test_guard();
    random::reset_rng();
    guard
}

#[tokio::test]
async fn tsne_embeds_rows_and_returns_loss() {
    let _guard = reset_rng();
    let _out_guard = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    let value = tsne_builtin(
        tensor(vec![0.0, 0.1, 5.0, 5.1, 0.0, 0.1, 5.0, 5.1], vec![4, 2]),
        vec![
            Value::String("Algorithm".into()),
            Value::String("exact".into()),
            Value::String("Perplexity".into()),
            Value::Num(2.0),
            Value::String("Options".into()),
            {
                let mut options = runmat_value::UnboundedStructValueBuilder::new();
                options.insert("MaxIter", Value::Num(50.0));
                Value::Struct(options.finish())
            },
        ],
    )
    .await
    .unwrap();
    match value {
        Value::OutputList(outputs) => {
            match &outputs[0] {
                Value::Tensor(y) => {
                    assert_eq!(y.shape(), [4, 2]);
                    assert!(y.materialize_f64().iter().all(|value| value.is_finite()));
                }
                other => panic!("expected tensor, got {other:?}"),
            }
            assert!(matches!(&outputs[1], Value::Num(loss) if loss.is_finite()));
        }
        other => panic!("expected outputs, got {other:?}"),
    }
}

#[tokio::test]
async fn tsne_removes_nan_rows_and_honors_num_dimensions() {
    let _guard = reset_rng();
    let value = tsne_builtin(
        tensor(
            vec![
                0.0,
                1.0,
                f64::NAN,
                3.0, //
                1.0,
                2.0,
                3.0,
                4.0,
            ],
            vec![4, 2],
        ),
        vec![
            Value::String("NumDimensions".into()),
            Value::Num(3.0),
            Value::String("InitialY".into()),
            Value::Tensor(Tensor::new(vec![0.0; 12], vec![4, 3]).unwrap()),
            Value::String("Options".into()),
            {
                let mut options = runmat_value::UnboundedStructValueBuilder::new();
                options.insert("MaxIter", Value::Num(2.0));
                Value::Struct(options.finish())
            },
        ],
    )
    .await
    .unwrap();
    match value {
        Value::Tensor(y) => assert_eq!(y.shape(), [3, 3]),
        other => panic!("expected tensor, got {other:?}"),
    }
}

#[tokio::test]
async fn tsne_supports_standardize_pca_and_cosine_distance() {
    let _guard = reset_rng();
    let value = tsne_builtin(
        tensor(
            vec![0.0, 1.0, 2.0, 3.0, 1.0, 1.0, 2.0, 2.0, 4.0, 3.0, 2.0, 1.0],
            vec![4, 3],
        ),
        vec![
            Value::String("Standardize".into()),
            Value::Bool(true),
            Value::String("NumPCAComponents".into()),
            Value::Num(2.0),
            Value::String("Distance".into()),
            Value::String("cosine".into()),
            Value::String("Options".into()),
            {
                let mut options = runmat_value::UnboundedStructValueBuilder::new();
                options.insert("MaxIter", Value::Num(5.0));
                Value::Struct(options.finish())
            },
        ],
    )
    .await
    .unwrap();
    match value {
        Value::Tensor(y) => assert_eq!(y.shape(), [4, 2]),
        other => panic!("expected tensor, got {other:?}"),
    }
}

#[tokio::test]
async fn tsne_reads_typed_integer_data_initial_y_and_options_exactly() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let _guard = reset_rng();
    let value = tsne_builtin(
        poisoned_int_tensor(
            IntegerStorage::I16(vec![0, 1, 5, 6, 0, 1, 5, 6]),
            vec![4, 2],
        ),
        vec![
            Value::String("NumDimensions".into()),
            cleared_int_tensor(IntegerStorage::U8(vec![2]), vec![1, 1]),
            Value::String("NumPCAComponents".into()),
            cleared_int_tensor(IntegerStorage::U8(vec![0]), vec![1, 1]),
            Value::String("Perplexity".into()),
            cleared_int_tensor(IntegerStorage::U8(vec![2]), vec![1, 1]),
            Value::String("Standardize".into()),
            cleared_int_tensor(IntegerStorage::U8(vec![0]), vec![1, 1]),
            Value::String("InitialY".into()),
            poisoned_int_tensor(IntegerStorage::I16(vec![0; 8]), vec![4, 2]),
            Value::String("Options".into()),
            {
                let mut options = runmat_value::UnboundedStructValueBuilder::new();
                options.insert(
                    "MaxIter",
                    cleared_int_tensor(IntegerStorage::U16(vec![2]), vec![1, 1]),
                );
                Value::Struct(options.finish())
            },
        ],
    )
    .await
    .unwrap();
    match value {
        Value::Tensor(y) => assert_eq!(y.shape(), [4, 2]),
        other => panic!("expected tensor, got {other:?}"),
    }
}

#[test]
fn tsne_gates_integer_roles_and_explicit_residency_before_gather() {
    let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
    let integer = poisoned_int_tensor(IntegerStorage::U8(vec![1]), vec![1, 1]);
    let error = ensure_tsne_extensions(&integer, &[]).expect_err("integer data must gate");
    assert_eq!(
        error.identifier(),
        Some("RunMat:compatibility:TsneIntegerDataExtension")
    );
    let headers = option_headers(crate::builtins::common::test_support::builtin_arguments(
        vec![Value::from("NumDimensions"), integer],
    ))
    .unwrap();
    let error =
        ensure_tsne_extensions(&Value::Num(1.0), &headers).expect_err("integer option must gate");
    assert_eq!(
        error.identifier(),
        Some("RunMat:compatibility:TsneIntegerOptionExtension")
    );
    let explicit = runmat_accelerate_api::test_support::TestAllocationOwner::new()
        .allocate_zeroed(
            vec![1, 1],
            runmat_accelerate_api::NumericElementType::F64,
            runmat_accelerate_api::GpuTensorStorage::Real,
        )
        .expect("genuine unregistered resident fixture")
        .handle;
    let explicit = explicit
        .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
        .expect("validated provenance");
    let error = ensure_tsne_extensions(&Value::GpuTensor(explicit), &[])
        .expect_err("explicit residency must gate before provider access");
    assert_eq!(
        error.identifier(),
        Some("RunMat:compatibility:TsneResidentInputExtension")
    );
}

#[test]
fn tsne_rejects_lossy_integer_floating_boundaries() {
    let wide = poisoned_int_tensor(IntegerStorage::U64(vec![(1_u64 << 53) + 1]), vec![1, 1]);
    let error = futures::executor::block_on(ensure_exact_tsne_integer_value(
        &wide,
        "observation data",
        &mut cooperation::TsneCooperation::new(None),
    ))
    .expect_err("lossy integer must reject");
    assert!(error.message().contains("exactly representable"));

    let exact_wide = poisoned_int_tensor(IntegerStorage::U64(vec![1_u64 << 54]), vec![1, 1]);
    futures::executor::block_on(ensure_exact_tsne_integer_value(
        &exact_wide,
        "observation data",
        &mut cooperation::TsneCooperation::new(None),
    ))
    .expect("aligned wide integer remains exactly representable");
}

#[test]
fn nested_integer_boundary_checks_large_aggregates_incrementally() {
    use crate::execution::{CpuCooperationBudget, CpuCooperationPolicy};
    use runmat_execution::cancellation::CancellationSource;
    use runmat_execution::CancellationReason;
    use std::num::NonZeroU64;
    use std::rc::Rc;

    let values = Value::output_list(vec![Value::Num(1.0); 4_096]);
    let source = Rc::new(CancellationSource::new());
    let cancel = source.clone();
    let budget = CpuCooperationBudget::with_test_turn(
        CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(1).unwrap()),
        source.token(),
        move || {
            cancel.cancel(CancellationReason::User);
            Box::pin(async { Ok(()) })
        },
    );
    let error = futures::executor::block_on(ensure_exact_tsne_integer_value(
        &values,
        "InitialY",
        &mut cooperation::TsneCooperation::with_budget(budget),
    ))
    .expect_err("the first cooperative turn must cancel aggregate inspection");
    assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
}

#[tokio::test]
async fn tsne_host_materialization_preserves_owned_tensor_buffer() {
    let input = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
    let buffer = input.as_f64_slice().unwrap().as_ptr();
    let Value::Tensor(output) = gather(Value::Tensor(input)).await.unwrap() else {
        panic!("host tensor remains a tensor");
    };
    assert_eq!(output.as_f64_slice().unwrap().as_ptr(), buffer);
}

#[tokio::test]
async fn tsne_rejects_bad_initial_y_shape() {
    let err = tsne_builtin(
        tensor(vec![0.0, 1.0, 2.0, 3.0], vec![2, 2]),
        vec![
            Value::String("InitialY".into()),
            Value::Tensor(Tensor::new(vec![0.0; 3], vec![3, 1]).unwrap()),
        ],
    )
    .await
    .unwrap_err();
    assert!(err.message.contains("InitialY"));
}

#[test]
fn tsne_empty_option_helpers_use_typed_integer_storage_len() {
    let empty = Tensor::new_integer(IntegerStorage::U16(Vec::new()), vec![0, 1]).unwrap();
    let empty = Value::Tensor(empty);

    assert!(is_empty_numeric(&empty));
    assert!(is_empty_option_value(&empty));
}

#[tokio::test]
async fn tsne_rejects_unbounded_resource_options() {
    let err = tsne_builtin(
        tensor(vec![0.0, 1.0, 2.0, 3.0], vec![2, 2]),
        vec![Value::String("NumDimensions".into()), Value::Num(1.0e20)],
    )
    .await
    .unwrap_err();
    assert!(err.message.contains("NumDimensions"));

    let err = tsne_builtin(
        tensor(vec![0.0, 1.0, 2.0, 3.0], vec![2, 2]),
        vec![Value::String("Options".into()), {
            let mut options = runmat_value::UnboundedStructValueBuilder::new();
            options.insert("MaxIter", Value::Num(1.0e20));
            Value::Struct(options.finish())
        }],
    )
    .await
    .unwrap_err();
    assert!(err.message.contains("MaxIter"));
}

#[tokio::test]
async fn tsne_rejects_dense_feature_work_before_allocation() {
    let x = Tensor::new(vec![1.0; 2 * 2001], vec![2, 2001]).unwrap();
    let err = tsne_builtin(
        Value::Tensor(x),
        vec![
            Value::String("Distance".into()),
            Value::String("mahalanobis".into()),
        ],
    )
    .await
    .unwrap_err();
    assert!(err.message.contains("dense feature cells"));
}

#[tokio::test]
async fn tsne_rejects_inf_even_after_nan_in_row() {
    let err = tsne_builtin(
        tensor(vec![f64::NAN, 1.0, f64::INFINITY, 2.0], vec![2, 2]),
        Vec::new(),
    )
    .await
    .unwrap_err();
    assert!(err.message.contains("Inf"));
}

#[tokio::test]
async fn tsne_rejects_unsupported_fast_distance_names() {
    let err = tsne_builtin(
        tensor(vec![0.0, 1.0, 2.0, 3.0], vec![2, 2]),
        vec![
            Value::String("Distance".into()),
            Value::String("fastcityblock".into()),
        ],
    )
    .await
    .unwrap_err();
    assert!(err.message.contains("unsupported Distance"));
}
use super::options::is_empty_option_value;
