use runmat_accelerate::{prepare_builtin_args, promote_binary, BinaryOp};
use runmat_accelerate_api::{AccelProvider, ThreadProviderGuard};
use runmat_runtime::{builtins::common::gpu_helpers, call_builtin_async_in_context};
use runmat_value::{Tensor, Value};

#[path = "../../runmat-runtime/tests/support/budgeted_placement.rs"]
mod budgeted_placement;
#[path = "support/native_auto_provider.rs"]
mod native_auto_provider;
#[path = "../../runmat-runtime/tests/support/retained_provider.rs"]
mod retained_provider;

fn provider() -> &'static native_auto_provider::Provider {
    retained_provider::retained_test_provider!(INIT: native_auto_provider::Provider = {
        // Skip ambient calibration reads; disabled calibration below also
        // prevents refreshing or writing a performance cache.
        std::env::set_var("RUNMAT_ACCEL_CALIBRATE_REFRESH", "1");
        std::env::set_var("RUNMAT_ACCEL_AUTO_OFFLOAD", "1");
        runmat_accelerate::configure_auto_offload(runmat_accelerate::AutoOffloadOptions {
            enabled: true,
            calibrate: false,
            ..Default::default()
        });
        native_auto_provider::Provider::default()
    })
}

fn runtime() -> (
    std::rc::Rc<runmat_accelerate::placement::PlacementSession>,
    runmat_runtime::context::RuntimeContext,
) {
    let placement = budgeted_placement::session();
    let runtime = runmat_accelerate::native_auto::install_preparation_service(
        budgeted_placement::context(placement.clone(), 1_000_000),
    );
    (placement, runtime)
}

fn make_tensor(len: usize) -> Tensor {
    let data: Vec<f64> = (0..len).map(|i| i as f64).collect();
    Tensor::new(data, vec![len, 1]).expect("tensor")
}

#[tokio::test]
async fn selected_canonical_plus_promotes_large_tensors_without_exporting_session_feedback() {
    let provider = provider();
    let _selected = ThreadProviderGuard::set(Some(provider));
    let value = Value::Tensor(make_tensor(8_192));
    // Generic preparation has neither a selected binding nor execution
    // authority. Thresholds alone cannot grant device execution.
    let (left, right) = promote_binary(BinaryOp::Elementwise, &value, &value)
        .await
        .expect("unselected preparation stays host");
    assert!(matches!(left, Value::Tensor(_)));
    assert!(matches!(right, Value::Tensor(_)));

    let (placement, runtime) = runtime();
    let before = provider
        .executed_adds
        .load(std::sync::atomic::Ordering::SeqCst);
    let exhausted = runmat_accelerate::native_auto::install_preparation_service(
        budgeted_placement::context(placement.clone(), 1),
    );
    let refused =
        call_builtin_async_in_context(&exhausted, "plus", &[value.clone(), value.clone()])
            .await
            .expect_err("finite root/control capacity rejects before arithmetic");
    budgeted_placement::assert_capacity_denied(&refused);
    assert_eq!(
        provider
            .executed_adds
            .load(std::sync::atomic::Ordering::SeqCst),
        before
    );
    let result = call_builtin_async_in_context(&runtime, "plus", &[value.clone(), value])
        .await
        .expect("actual admitted plus");
    assert!(matches!(result.value(), Value::GpuTensor(_)));
    let gathered = runmat_runtime::gather_if_needed_async(result.value())
        .await
        .unwrap();
    let Value::Tensor(gathered) = gathered else {
        panic!("host readback");
    };
    assert_eq!(gathered.shape(), &[8_192, 1]);
    assert_eq!(
        gathered.materialize_f64(),
        (0..8_192)
            .map(|index| 2.0 * index as f64)
            .collect::<Vec<_>>()
    );
    // Runtime-binding generations are session-local and cannot be reused by
    // another process via profile export. Actual managed observation insertion
    // and one-shot deferred delivery are covered by the placement owner tests.
    assert!(placement.profile_snapshot().feedback.is_empty());
}

#[tokio::test]
async fn selected_sink_gathers_actual_resident_input() {
    let provider = provider();
    let _selected = ThreadProviderGuard::set(Some(provider));
    let input = make_tensor(8_192);
    let gpu = gpu_helpers::upload_tensor(provider, &input).expect("actual resident input");
    let args = [Value::GpuTensor(gpu)];
    let prepared = prepare_builtin_args("disp", &args)
        .await
        .expect("legacy preparation");
    assert!(matches!(prepared.as_slice(), [Value::GpuTensor(_)]));
    let (_, runtime) = runtime();
    let capture = {
        let _scope = runtime.enter();
        runmat_runtime::console::begin_capture().unwrap()
    };
    let before = provider.telemetry_snapshot().download_bytes;
    runmat_runtime::call_builtin_sequence_async_with_outputs_in_context(&runtime, "disp", &args, 0)
        .await
        .expect("selected sink owns its real gather");
    assert!(provider.telemetry_snapshot().download_bytes > before);
    // Inspect the admitted entries without flattening their managed backing
    // into an unbudgeted String outside the language output boundary.
    assert!(capture
        .finish_entries()
        .iter()
        .any(|entry| !entry.text.is_empty()));
}

#[tokio::test]
async fn sinc_runtime_owned_host_and_explicit_resident_numeric_contracts() {
    let provider = provider();
    let _selected = ThreadProviderGuard::set(Some(provider));
    let tensor = make_tensor(8_192);
    let prepared = prepare_builtin_args("sinc", &[Value::Tensor(tensor.clone())])
        .await
        .expect("legacy preparation");
    assert!(matches!(prepared.as_slice(), [Value::Tensor(_)]));
    let (_, runtime) = runtime();
    let host = call_builtin_async_in_context(&runtime, "sinc", &[Value::Tensor(tensor.clone())])
        .await
        .expect("RuntimeOwned host sinc");
    assert!(matches!(host.value(), Value::Tensor(_)));
    let gpu = gpu_helpers::upload_tensor(provider, &tensor).unwrap();
    let resident = call_builtin_async_in_context(&runtime, "sinc", &[Value::GpuTensor(gpu)])
        .await
        .expect("explicit resident sinc");
    assert!(matches!(resident.value(), Value::GpuTensor(_)));
    let gathered = runmat_runtime::gather_if_needed_async(resident.value())
        .await
        .unwrap();
    let (Value::Tensor(actual), Value::Tensor(expected)) = (gathered, host.value()) else {
        panic!("dense sinc outputs");
    };
    assert_eq!(actual.shape(), expected.shape());
    for (actual, expected) in actual
        .materialize_f64()
        .iter()
        .zip(expected.materialize_f64())
    {
        assert!((actual - expected).abs() <= 1e-12);
    }
    // RuntimeOwned sinc does not promise automatic host-to-device promotion.
    // This fixture qualifies its numeric host and explicit-resident contracts.
}
