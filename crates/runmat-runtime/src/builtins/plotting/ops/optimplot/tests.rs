use super::*;
use crate::builtins::plotting::state;
use futures::executor::block_on;
fn setup() -> state::PlotTestLockGuard {
    let guard = crate::builtins::plotting::tests::lock_plot_registry();
    crate::builtins::plotting::tests::ensure_plot_test_env();
    guard
}
fn values(iteration: Value, name: &str, value: Value) -> Value {
    let mut fields = runmat_value::UnboundedStructValueBuilder::new();
    fields.insert("iteration", iteration);
    fields.insert(name, value);
    Value::Struct(fields.finish())
}
fn run(
    kind: OptimPlotKind,
    x: Value,
    values: Value,
    phase: &str,
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<Value> {
    block_on(evaluate(
        kind,
        runtime,
        crate::builtins::common::test_support::builtin_arguments(vec![
            x,
            values,
            Value::from(phase),
        ]),
    ))
}
#[test]
fn real_public_callbacks_return_false_and_validate_only_their_consumed_fields() {
    let _guard = setup();
    for kind in [
        OptimPlotKind::CurrentPoint,
        OptimPlotKind::FunctionCount,
        OptimPlotKind::FunctionValue,
    ] {
        let (field, value) = match kind {
            OptimPlotKind::CurrentPoint => ("unused", Value::from("ignored")),
            OptimPlotKind::FunctionCount => ("funccount", Value::Num(4.0)),
            OptimPlotKind::FunctionValue => ("fval", Value::Num(2.0)),
            _ => unreachable!("bounded fixture kinds"),
        };
        assert_eq!(
            run(
                kind,
                Value::Num(2.0),
                values(Value::Num(0.0), field, value.clone()),
                "init",
                None
            )
            .unwrap(),
            Value::Bool(false)
        );
        assert_eq!(
            run(
                kind,
                Value::Num(1.0),
                values(Value::Num(1.0), field, value.clone()),
                "iter",
                None
            )
            .unwrap(),
            Value::Bool(false)
        );
        assert_eq!(
            run(
                kind,
                Value::Num(1.0),
                values(Value::Num(1.0), field, value),
                "done",
                None
            )
            .unwrap(),
            Value::Bool(false)
        );
        assert!(run(
            kind,
            Value::Num(1.0),
            values(Value::Num(-1.0), field, Value::Num(1.0)),
            "iter",
            None
        )
        .is_err());
        assert!(run(
            kind,
            Value::Num(1.0),
            values(Value::Num(0.0), field, Value::Num(1.0)),
            "unknown",
            None
        )
        .is_err());
    }
    assert!(run(
        OptimPlotKind::FunctionCount,
        Value::Num(1.0),
        values(Value::Num(0.0), "funccount", Value::Num(0.5)),
        "init",
        None
    )
    .is_err());
    assert!(run(
        OptimPlotKind::CurrentPoint,
        Value::Num(f64::NAN),
        values(Value::Num(0.0), "unused", Value::Num(1.0)),
        "init",
        None
    )
    .is_err());
    // x is deliberately not read by scalar-history callbacks.
    assert_eq!(
        run(
            OptimPlotKind::FunctionValue,
            Value::from("unused current point"),
            values(Value::Num(0.0), "fval", Value::Num(f64::INFINITY)),
            "init",
            None
        )
        .unwrap(),
        Value::Bool(false)
    );
}

#[derive(Default)]
struct Transfers {
    owner: runmat_accelerate_api::test_support::TestAllocationOwner,
    allocations: std::sync::Mutex<Vec<runmat_accelerate_api::test_support::TestAllocation>>,
    downloads: std::sync::atomic::AtomicUsize,
}
impl Transfers {
    fn value(&self, value: f64, intent: runmat_accelerate_api::GpuHandleProvenance) -> Value {
        let allocation = self
            .owner
            .allocate_numeric(runmat_accelerate_api::HostNumericTensorOwned {
                data: runmat_accelerate_api::HostNumericDataOwned::F64(vec![value]),
                shape: vec![1, 1],
                storage: runmat_accelerate_api::GpuTensorStorage::Real,
            })
            .unwrap();
        let handle = allocation.handle.clone().with_provenance(intent).unwrap();
        self.allocations.lock().unwrap().push(allocation);
        Value::GpuTensor(handle)
    }
}
impl runmat_accelerate_api::AccelProvider for Transfers {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }
    fn device_info(&self) -> String {
        "optimization plot owned-transfer witness".into()
    }
    fn upload(
        &self,
        _: &runmat_accelerate_api::HostTensorView,
    ) -> anyhow::Result<runmat_accelerate_api::GpuTensorHandle> {
        panic!("fixture owns allocations directly")
    }
    fn download<'a>(
        &'a self,
        _: &'a runmat_accelerate_api::GpuTensorHandle,
    ) -> runmat_accelerate_api::AccelDownloadFuture<'a> {
        panic!("typed download required")
    }
    fn download_numeric<'a>(
        &'a self,
        handle: &'a runmat_accelerate_api::GpuTensorHandle,
    ) -> runmat_accelerate_api::AccelNumericDownloadFuture<'a> {
        self.downloads
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Box::pin(async move {
            let allocations = self.allocations.lock().unwrap();
            let allocation = allocations
                .iter()
                .find(|a| a.handle.allocation_identity() == handle.allocation_identity())
                .unwrap();
            let data = allocation.probe.with_data(handle, |data| {
                let runmat_accelerate_api::HostNumericDataView::F64(values) = data else {
                    panic!("double fixture")
                };
                runmat_accelerate_api::HostNumericDataOwned::F64(values.to_vec())
            })?;
            Ok(runmat_accelerate_api::HostNumericTensorOwned {
                data,
                shape: handle.shape().to_vec(),
                storage: runmat_accelerate_api::GpuTensorStorage::Real,
            })
        })
    }
}
#[test]
fn automatic_needed_operands_download_but_explicit_and_unused_operands_do_not() {
    use runmat_accelerate_api::GpuHandleProvenance::{Automatic, Explicit};
    use std::sync::atomic::Ordering;
    let _guard = setup();
    let _serial = crate::builtins::common::test_support::accel_test_lock();
    let provider = Box::leak(Box::new(Transfers::default()));
    unsafe { runmat_accelerate_api::register_provider(provider) };
    let _selected = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let mut fields = runmat_value::UnboundedStructValueBuilder::new();
    fields.insert("iteration", provider.value(0.0, Automatic));
    fields.insert("fval", provider.value(4.0, Automatic));
    fields.insert("unused", provider.value(99.0, Explicit));
    assert_eq!(
        run(
            OptimPlotKind::FunctionValue,
            provider.value(3.0, Explicit),
            Value::Struct(fields.finish()),
            "init",
            None
        )
        .unwrap(),
        Value::Bool(false)
    );
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 2);
    run(
        OptimPlotKind::CurrentPoint,
        provider.value(2.0, Automatic),
        values(Value::Num(0.0), "unused", Value::Num(0.0)),
        "init",
        None,
    )
    .unwrap();
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 3);
    for kind in [
        OptimPlotKind::CurrentPoint,
        OptimPlotKind::FunctionCount,
        OptimPlotKind::FunctionValue,
    ] {
        let (field, value, x) = match kind {
            OptimPlotKind::CurrentPoint => {
                ("unused", Value::Num(0.0), provider.value(2.0, Explicit))
            }
            OptimPlotKind::FunctionCount => {
                ("funccount", provider.value(2.0, Explicit), Value::Num(1.0))
            }
            OptimPlotKind::FunctionValue => {
                ("fval", provider.value(2.0, Explicit), Value::Num(1.0))
            }
            _ => unreachable!("bounded fixture kinds"),
        };
        let error = run(
            kind,
            x,
            values(provider.value(0.0, Automatic), field, value),
            "iter",
            None,
        )
        .unwrap_err();
        assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
        assert_eq!(provider.downloads.load(Ordering::SeqCst), 3);
    }
}

#[test]
fn finite_and_cancelled_invocations_refuse_before_any_scene_or_point_work() {
    use crate::context::{
        host_effect::ManagedHostEffectLedger, ParallelCapability, RuntimeContext,
        RuntimeParallelResources, RuntimeParallelService, RuntimeServicePorts,
    };
    use std::rc::Rc;
    struct Finite(ManagedHostEffectLedger);
    impl RuntimeParallelService for Finite {
        fn supports(&self, _: ParallelCapability) -> bool {
            false
        }
        fn placement_resources(&self) -> RuntimeParallelResources {
            let s = self.0.snapshot();
            RuntimeParallelResources {
                memory_available_bytes: Some(s.free_bytes()),
                epoch: s.epoch,
                ..Default::default()
            }
        }
        fn managed_host_effect_ledger(&self) -> Option<ManagedHostEffectLedger> {
            Some(self.0.clone())
        }
    }
    let _guard = setup();
    let before = state::current_figure_handle_if_exists();
    let ledger = ManagedHostEffectLedger::new(1024 * 1024);
    let runtime = RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
        .with_service_ports(
            RuntimeServicePorts::default().with_parallel(Rc::new(Finite(ledger.clone()))),
        );
    assert!(state::OptimizationPlotLayout::prepare(1, Some(&runtime)).is_err());
    for kind in [
        OptimPlotKind::CurrentPoint,
        OptimPlotKind::FunctionCount,
        OptimPlotKind::FunctionValue,
    ] {
        let error = run(
            kind,
            Value::Num(1.0),
            values(Value::Num(0.0), "fval", Value::Num(4.0)),
            "init",
            Some(&runtime),
        )
        .unwrap_err();
        assert_eq!(
            error.identifier(),
            Some("RunMat:Placement:InsufficientResourceEvidence")
        );
    }
    assert_eq!(before, state::current_figure_handle_if_exists());
    assert_eq!(ledger.snapshot().peak_reserved_bytes, 0);
    let source = runmat_execution::cancellation::CancellationSource::new();
    let runtime = RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
        .with_cancellation_token(source.token());
    runtime.set_error_namespace("ProgressInvocation");
    source.cancel(runmat_execution::CancellationReason::User);
    let error = run(
        OptimPlotKind::FunctionValue,
        Value::Num(1.0),
        values(Value::Num(0.0), "fval", Value::Num(4.0)),
        "init",
        Some(&runtime),
    )
    .unwrap_err();
    assert_eq!(error.identifier(), Some("ProgressInvocation:Cancelled"));
    assert!(crate::runtime_error::error_has_control_cancellation_origin(
        &error
    ));
    assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
    assert_eq!(before, state::current_figure_handle_if_exists());
}
