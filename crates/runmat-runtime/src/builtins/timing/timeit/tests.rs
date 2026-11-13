use super::*;

fn runtime() -> RuntimeContext {
    use crate::execution::RuntimeExecutionService;
    use std::rc::Rc;
    RuntimeContext::new(Rc::new(RuntimeExecutionService::new()))
}

fn rooted_sequence(
    sequence: runmat_value::ValueSequence,
) -> Result<crate::execution::RootedValueSequence, crate::RuntimeError> {
    crate::execution::RootedValueSequence::new(sequence).map_err(|error| error.into_runtime_error())
}

fn inert_handle(name: &str, function: usize) -> Value {
    Value::BoundFunctionHandle(runmat_value::BoundFunctionHandle::from_data(
        runmat_value::BoundFunctionHandleData {
            name: name.into(),
            function,
            authority: runmat_value::BoundFunctionAuthority::unavailable(function),
            source_role: runmat_types::FunctionSourceRole::Unknown,
            source_file: None,
        },
    ))
}

#[test]
fn output_count_preserves_representable_uint64() {
    assert_eq!(
        parse_non_negative_integer(&Value::Int(IntValue::U64(u64::MAX))).ok(),
        usize::try_from(u64::MAX).ok()
    );
    assert!(parse_non_negative_integer(&Value::Int(IntValue::I64(-1))).is_err());
}

#[test]
fn output_count_reads_typed_integer_scalar_storage_exactly() {
    let count =
        Tensor::new_integer(IntegerStorage::U16(vec![2]), vec![1, 1]).expect("typed output count");

    assert_eq!(
        parse_non_negative_integer(&Value::Tensor(count)).unwrap(),
        2
    );

    let negative = Tensor::new_integer(IntegerStorage::I16(vec![-1]), vec![1, 1])
        .expect("negative output count");
    assert!(parse_non_negative_integer(&Value::Tensor(negative)).is_err());
}

#[test]
fn output_count_rejects_unrepresentable_double_boundary() {
    let boundary = if usize::BITS == 64 {
        usize::MAX as f64
    } else {
        (usize::MAX as f64) + 1.0
    };
    assert!(parse_non_negative_integer(&Value::Num(boundary)).is_err());
}

#[test]
fn invalid_callable_precedes_resident_count_materialization() {
    let allocation = runmat_accelerate_api::test_support::TestAllocationOwner::new()
        .allocate_zeroed(
            vec![1, 1],
            runmat_accelerate_api::NumericElementType::F64,
            runmat_accelerate_api::GpuTensorStorage::Real,
        )
        .expect("unregistered resident owner");
    let resident = allocation
        .handle
        .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Automatic)
        .expect("automatic view");
    let error = block_on(timeit_builtin(
        Some(runtime()),
        Value::Num(7.0),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::GpuTensor(resident)]),
    ))
    .expect_err("invalid callable before provider access");
    assert_eq!(error.identifier(), TIMEIT_ERROR_FIRST_ARG_KIND.identifier);
    assert_eq!(allocation.probe.retirements(), 0);
    drop((error, allocation.handle));
    assert_eq!(allocation.probe.retirements(), 1);
}

#[test]
fn host_count_validation_keeps_count_before_callable_error_order() {
    let error = block_on(timeit_builtin(
        Some(runtime()),
        Value::Num(7.0),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::String(
            "invalid count".into(),
        )]),
    ))
    .expect_err("invalid host count first");
    assert_eq!(
        error.identifier(),
        TIMEIT_ERROR_NUM_OUTPUTS_SCALAR.identifier
    );
}

use futures::executor::block_on;
use runmat_value::{Closure, IntValue, IntegerStorage, Tensor};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

struct TestMonotonicClock(Arc<std::sync::atomic::AtomicU64>);

impl crate::context::RuntimeTimingService for TestMonotonicClock {
    fn monotonic_elapsed(&self) -> std::time::Duration {
        std::time::Duration::from_micros(self.0.load(Ordering::SeqCst))
    }

    fn wall_time_since_epoch(&self) -> std::time::Duration {
        std::time::Duration::ZERO
    }

    fn local_datetime_to_instant(
        &self,
        _: chrono::NaiveDateTime,
    ) -> Option<crate::context::RuntimeCalendarInstant> {
        panic!("timeit must not use a calendar clock")
    }

    fn local_datetime_at_instant(
        &self,
        _: crate::context::RuntimeCalendarInstant,
    ) -> Option<chrono::NaiveDateTime> {
        panic!("timeit must not use a calendar clock")
    }

    fn process_cpu_seconds(&self) -> Option<f64> {
        None
    }

    fn sleep(
        &self,
        _: std::time::Duration,
    ) -> crate::context::RuntimeServiceFuture<Result<(), crate::RuntimeError>> {
        panic!("timeit must not schedule a service sleep")
    }
}

#[test]
fn timeit_uses_injected_monotonic_service_for_every_batch() {
    let ticks = Arc::new(std::sync::atomic::AtomicU64::new(0));
    let runtime = runtime().with_service_ports(
        crate::context::RuntimeServicePorts::default()
            .with_timing(std::rc::Rc::new(TestMonotonicClock(ticks.clone()))),
    );
    let _scope = runtime.enter();
    let binding = crate::user_functions::test_support::exact_local_binding(&runtime, 991);
    let _bound = crate::user_functions::test_support::install_bound_forwarder(binding.clone());
    let callback_ticks = ticks.clone();
    let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
        move |function, args, requested_outputs| {
            assert_eq!(function, 991);
            assert!(args.is_empty());
            assert_eq!(requested_outputs, 1);
            callback_ticks.fetch_add(1000, Ordering::SeqCst);
            Box::pin(async {
                rooted_sequence(runmat_value::ValueSequence::single(Value::Num(1.0)).unwrap())
            })
        },
    )))
    .expect("call-port scope");
    let result = block_on(timeit_builtin(
        Some(runtime),
        crate::user_functions::test_support::exact_local_value("test.monotonic.callback", binding),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .unwrap();
    let Value::Num(seconds) = result else {
        panic!("expected timeit scalar")
    };
    assert!((seconds - 0.001).abs() < 1e-12, "{seconds}");
    assert!(ticks.load(Ordering::SeqCst) >= 8000);
}

#[test]
fn cancelled_pending_callback_is_dropped_without_replay() {
    use runmat_execution::cancellation::CancellationSource;
    use runmat_execution::CancellationReason;
    use std::future::Future;
    use std::pin::Pin;
    use std::task::{Context, Poll};

    struct PendingCallback(Arc<AtomicUsize>);

    impl Future for PendingCallback {
        type Output = Result<crate::execution::RootedValueSequence, crate::RuntimeError>;

        fn poll(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Self::Output> {
            Poll::Pending
        }
    }

    impl Drop for PendingCallback {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    let source = CancellationSource::new();
    let runtime = RuntimeContext::with_cancellation(
        std::rc::Rc::new(crate::execution::RuntimeExecutionService::new()),
        source.token(),
    );
    let _scope = runtime.enter();
    let binding = crate::user_functions::test_support::exact_local_binding(&runtime, 992);
    let _bound = crate::user_functions::test_support::install_bound_forwarder(binding.clone());
    let calls = Arc::new(AtomicUsize::new(0));
    let drops = Arc::new(AtomicUsize::new(0));
    let callback_calls = calls.clone();
    let callback_drops = drops.clone();
    let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
        move |function, args, requested_outputs| {
            assert_eq!(function, 992);
            assert!(args.is_empty());
            assert_eq!(requested_outputs, 1);
            callback_calls.fetch_add(1, Ordering::SeqCst);
            Box::pin(PendingCallback(callback_drops.clone()))
        },
    )))
    .expect("call-port scope");
    let unpolled = timeit_builtin(
        Some(runtime.clone()),
        crate::user_functions::test_support::exact_local_value(
            "test.pending.callback",
            binding.clone(),
        ),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    );
    drop(unpolled);
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    let mut call = Box::pin(timeit_builtin(
        Some(runtime),
        crate::user_functions::test_support::exact_local_value("test.pending.callback", binding),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ));
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    let waker = futures::task::noop_waker();
    let mut cx = Context::from_waker(&waker);
    assert!(call.as_mut().poll(&mut cx).is_pending());
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    source.cancel(CancellationReason::User);
    assert!(matches!(call.as_mut().poll(&mut cx), Poll::Ready(Err(_))));
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}

static COUNTER_DEFAULT: AtomicUsize = AtomicUsize::new(0);
static COUNTER_NUM_OUTPUTS: AtomicUsize = AtomicUsize::new(0);
static COUNTER_NUM_OUTPUTS_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
static COUNTER_INVALID: AtomicUsize = AtomicUsize::new(0);
static COUNTER_ZERO_OUTPUTS: AtomicUsize = AtomicUsize::new(0);

/// Scoped execution authority for the four retired test-only builtin names.
fn install_test_callbacks(
    runtime: &RuntimeContext,
) -> (
    crate::user_functions::FunctionResolverGuard,
    crate::user_functions::FunctionInvokerGuard,
    crate::user_functions::SourceFunctionCatalogGuard,
    crate::user_functions::BoundFunctionInvokerGuard,
) {
    let _scope = runtime.enter();
    let owners = [1001, 1002, 1003, 1004]
        .into_iter()
        .map(|function| {
            (
                function,
                Arc::new(runmat_value::FunctionDefinitionOwner::default()),
            )
        })
        .collect::<Vec<_>>();
    let catalog = crate::user_functions::install_source_function_catalog(Some(Arc::new(
        owners
            .iter()
            .map(
                |(function, owner)| crate::user_functions::SourceFunctionInfo {
                    source_id: runmat_types::SourceId(*function),
                    name: format!("timeit_fixture_{function}"),
                    function: *function,
                    role: runmat_types::FunctionSourceRole::Unknown,
                    binding_origin: None,
                    source_file: None,
                    definition_owners: Some(Arc::new(vec![(*function, Arc::clone(owner))])),
                },
            )
            .collect(),
    )))
    .expect("call-port scope");
    let bound = crate::user_functions::install_local_bound_function_invoker(std::rc::Rc::new(
        move |runtime, binding, arguments, outputs, _lexical_environment| {
            let lease = binding
                .authority
                .local_definition()
                .expect("timeit fixture exact local lease");
            assert!(owners.iter().any(|(function, owner)| {
                *function == binding.function && lease.matches(owner)
            }));
            assert_eq!(arguments.capture_count(), 0);
            let invoker = crate::user_functions::current_semantic_function_invoker()
                .expect("timeit fixture numeric callback");
            invoker(Some(runtime), binding.function, arguments, outputs)
        },
    ))
    .expect("call-port scope");
    let resolver = crate::user_functions::install_semantic_function_resolver(Some(Arc::new(
        |name| match name {
            "__timeit_helper_counter_default" => Some(1001),
            "__timeit_helper_counter_outputs" => Some(1002),
            "__timeit_helper_counter_invalid" => Some(1003),
            "__timeit_helper_zero_outputs" => Some(1004),
            _ => None,
        },
    )))
    .expect("call-port scope");
    let invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
        |function, args, requested_outputs| {
            assert!(args.is_empty());
            let result = match function {
                1001 => {
                    assert_eq!(requested_outputs, 1);
                    COUNTER_DEFAULT.fetch_add(1, Ordering::SeqCst);
                    runmat_value::ValueSequence::single(Value::Num(1.0))
                }
                1002 => {
                    assert_eq!(requested_outputs, 3);
                    COUNTER_NUM_OUTPUTS.fetch_add(1, Ordering::SeqCst);
                    runmat_value::ValueSequence::comma_separated(vec![
                        Value::Num(1.0),
                        Value::Num(2.0),
                        Value::Num(3.0),
                    ])
                }
                1003 => {
                    COUNTER_INVALID.fetch_add(1, Ordering::SeqCst);
                    runmat_value::ValueSequence::single(Value::Num(1.0))
                }
                1004 => {
                    assert_eq!(requested_outputs, 0);
                    COUNTER_ZERO_OUTPUTS.fetch_add(1, Ordering::SeqCst);
                    Ok(runmat_value::ValueSequence::empty())
                }
                _ => panic!("unexpected test callback identity {function}"),
            }
            .map_err(crate::sequence::sequence_error_to_runtime);
            Box::pin(async move { result.and_then(rooted_sequence) })
        },
    )))
    .expect("call-port scope");
    (resolver, invoker, catalog, bound)
}

fn default_handle() -> Value {
    Value::String("@__timeit_helper_counter_default".to_string().into())
}

fn assert_timeit_error_contains(err: &crate::RuntimeError, needle: &str) {
    let message = err.message().to_ascii_lowercase();
    assert!(
        message.contains(&needle.to_ascii_lowercase()),
        "unexpected error text: {}",
        err.message()
    );
}

fn assert_timeit_error_identifier(err: &crate::RuntimeError, identifier: &'static str) {
    assert_eq!(err.identifier(), Some(identifier), "{}", err.message());
}

fn outputs_handle() -> Value {
    Value::String("@__timeit_helper_counter_outputs".to_string().into())
}

fn invalid_handle() -> Value {
    Value::String("@__timeit_helper_counter_invalid".to_string().into())
}

fn zero_outputs_handle() -> Value {
    Value::String("@__timeit_helper_zero_outputs".to_string().into())
}

#[test]
fn retired_test_helpers_have_no_product_catalog_identity() {
    for name in [
        "__timeit_helper_counter_default",
        "__timeit_helper_counter_outputs",
        "__timeit_helper_counter_invalid",
        "__timeit_helper_zero_outputs",
    ] {
        assert!(runmat_builtins::builtin_catalog_entry_by_name(name).is_none());
    }
}

#[test]
fn timeit_declares_integer_output_count_compatibility_metadata() {
    let builtin = runmat_builtins::builtin_catalog_entry_by_name("timeit").unwrap();
    assert_eq!(builtin.integer_capabilities.len(), 1);
    assert_eq!(builtin.extensions.len(), 2);
    assert_eq!(builtin.integer_capabilities[0].inputs[0].classes.len(), 8);
}

#[test]
fn timeit_accepts_external_function_handle() {
    let callable = prepare_callable(
        Value::ExternalFunctionHandle("pkg.callback".into()),
        Some(2),
    )
    .expect("timeit should accept external function handle");
    assert_eq!(
        callable.handle,
        Value::ExternalFunctionHandle("pkg.callback".into())
    );
    assert_eq!(callable.num_outputs, Some(2));
}

#[test]
fn timeit_rejects_empty_function_handle_name_value() {
    let err = prepare_callable(Value::FunctionHandle("   ".into()), None)
        .expect_err("timeit should reject empty function-handle payload name");
    assert_timeit_error_contains(&err, "empty function handle");
    assert_timeit_error_identifier(&err, TIMEIT_ERROR_EMPTY_HANDLE.identifier.unwrap());
}

#[test]
fn timeit_rejects_empty_external_function_handle_name_value() {
    let err = prepare_callable(Value::ExternalFunctionHandle("   ".into()), None)
        .expect_err("timeit should reject empty external function-handle payload name");
    assert_timeit_error_contains(&err, "empty function handle");
    assert_timeit_error_identifier(&err, TIMEIT_ERROR_EMPTY_HANDLE.identifier.unwrap());
}

#[test]
fn timeit_trims_function_handle_name_for_semantic_resolution() {
    let _resolver_guard =
        crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|name| {
            (name == "__timeit_helper_counter_default").then_some(188)
        })))
        .expect("call-port scope");
    let callable = prepare_callable(
        Value::FunctionHandle("  __timeit_helper_counter_default  ".into()),
        None,
    )
    .expect("timeit should normalize function-handle payload name");
    assert_eq!(
        callable.handle,
        inert_handle("__timeit_helper_counter_default", 188)
    );
}

#[test]
fn timeit_callable_invoke_honors_multi_requested_outputs() {
    let runtime = runtime();
    let _scope = runtime.enter();
    let binding = crate::user_functions::test_support::exact_local_binding(&runtime, 612);
    let _bound = crate::user_functions::test_support::install_bound_forwarder(binding.clone());
    let _invoker_guard = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
        |function, args, requested_outputs| {
            assert_eq!(function, 612);
            assert!(args.is_empty());
            assert_eq!(requested_outputs, 3);
            Box::pin(async {
                crate::sequence::ValueSequence::comma_separated(vec![
                    Value::Num(1.0),
                    Value::Num(2.0),
                    Value::Num(3.0),
                ])
                .map_err(crate::sequence::sequence_error_to_runtime)
                .and_then(rooted_sequence)
            })
        },
    )))
    .expect("call-port scope");

    let callable = prepare_callable(
        crate::user_functions::test_support::exact_local_value("function_target", binding),
        Some(3),
    )
    .expect("timeit should accept semantic callback handles");

    let invoked =
        block_on(callable.invoke(&runtime)).expect("timeit callable invoke should succeed");
    assert_eq!(invoked, Value::Num(0.0));
}

#[test]
fn timeit_string_handle_prefers_semantic_resolver_identity() {
    let _resolver_guard =
        crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|name| {
            (name == "__timeit_helper_counter_default").then_some(87)
        })))
        .expect("call-port scope");
    let callable = prepare_callable(
        Value::String("@__timeit_helper_counter_default".to_string().into()),
        None,
    )
    .expect("timeit should accept string function handle");
    assert_eq!(
        callable.handle,
        inert_handle("__timeit_helper_counter_default", 87)
    );
}

#[test]
fn timeit_char_handle_prefers_semantic_resolver_identity() {
    let _resolver_guard =
        crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|name| {
            (name == "__timeit_helper_counter_default").then_some(88)
        })))
        .expect("call-port scope");
    let callable = prepare_callable(
        Value::CharArray(runmat_value::CharArray::new_row(
            "@__timeit_helper_counter_default",
        )),
        None,
    )
    .expect("timeit should accept char function handle");
    assert_eq!(
        callable.handle,
        inert_handle("__timeit_helper_counter_default", 88)
    );
}

#[test]
fn timeit_external_function_handle_prefers_semantic_resolver_identity() {
    let _resolver_guard =
        crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|name| {
            (name == "pkg.callback").then_some(86)
        })))
        .expect("call-port scope");
    let callable = prepare_callable(
        Value::ExternalFunctionHandle("pkg.callback".into()),
        Some(2),
    )
    .expect("timeit should accept external function handle");
    assert_eq!(callable.handle, inert_handle("pkg.callback", 86));
    assert_eq!(callable.num_outputs, Some(2));
}

#[test]
fn timeit_accepts_semantic_function_handle() {
    let callable = prepare_callable(inert_handle("function_target", 41), Some(1))
        .expect("timeit should accept semantic function handle");
    assert_eq!(callable.handle, inert_handle("function_target", 41));
    assert_eq!(callable.num_outputs, Some(1));
}

#[test]
fn timeit_name_only_closure_prefers_semantic_resolver_identity() {
    let _resolver_guard =
        crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|name| {
            (name == "__timeit_helper_counter_default").then_some(89)
        })))
        .expect("call-port scope");
    let callable = prepare_callable(
        Value::Closure(Closure::from_data(runmat_value::ClosureData {
            lexical_environment: None,
            function_name: "__timeit_helper_counter_default".into(),
            binding: None,
            source_role: runmat_types::FunctionSourceRole::Unknown,
            source_file: None,
            captures: vec![Value::Num(9.0)].into(),
        })),
        None,
    )
    .expect("timeit should accept closure callback");
    assert_eq!(
        callable.handle,
        Value::Closure(Closure::from_data(runmat_value::ClosureData {
            lexical_environment: None,
            function_name: "__timeit_helper_counter_default".into(),
            binding: Some(runmat_value::BoundFunctionBinding::unavailable(89)),
            source_role: runmat_types::FunctionSourceRole::Unknown,
            source_file: None,
            captures: vec![Value::Num(9.0)].into(),
        }))
    );
}

#[test]
fn timeit_name_only_closure_without_resolver_keeps_name_shaped_identity() {
    let callable = prepare_callable(
        Value::Closure(Closure::from_data(runmat_value::ClosureData {
            lexical_environment: None,
            function_name: "__timeit_helper_counter_default".into(),
            binding: None,
            source_role: runmat_types::FunctionSourceRole::Unknown,
            source_file: None,
            captures: vec![Value::Num(9.0)].into(),
        })),
        None,
    )
    .expect("timeit should accept closure callback");
    assert_eq!(
        callable.handle,
        Value::Closure(Closure::from_data(runmat_value::ClosureData {
            lexical_environment: None,
            function_name: "__timeit_helper_counter_default".into(),
            binding: None,
            source_role: runmat_types::FunctionSourceRole::Unknown,
            source_file: None,
            captures: vec![Value::Num(9.0)].into(),
        }))
    );
}

#[path = "tests/execution.rs"]
mod execution;
