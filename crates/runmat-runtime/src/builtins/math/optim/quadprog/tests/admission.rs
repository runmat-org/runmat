//! Conserved original datatype/context/Clock admission controls.
use super::*;
#[derive(Default)]
struct Transfers {
    owner: runmat_accelerate_api::test_support::TestAllocationOwner,
    allocations: std::sync::Mutex<Vec<runmat_accelerate_api::test_support::TestAllocation>>,
    downloads: std::sync::atomic::AtomicUsize,
}
impl Transfers {
    fn value(
        &self,
        value: f64,
        intent: Option<runmat_accelerate_api::GpuHandleProvenance>,
    ) -> Value {
        let allocation = self
            .owner
            .allocate_numeric(runmat_accelerate_api::HostNumericTensorOwned {
                data: runmat_accelerate_api::HostNumericDataOwned::F64(vec![value]),
                shape: vec![1, 1],
                storage: runmat_accelerate_api::GpuTensorStorage::Real,
            })
            .unwrap();
        let handle = match intent {
            Some(intent) => allocation.handle.clone().with_provenance(intent).unwrap(),
            None => allocation.handle.clone(),
        };
        self.allocations.lock().unwrap().push(allocation);
        Value::GpuTensor(handle)
    }
}
impl runmat_accelerate_api::AccelProvider for Transfers {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }
    fn device_info(&self) -> String {
        "quadratic input admission witness".into()
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
fn compatibility_is_whole_call_pretransfer_and_uses_original_context_policy() {
    use runmat_accelerate_api::GpuHandleProvenance::{Automatic, Explicit};
    use std::rc::Rc;
    use std::sync::atomic::Ordering;
    let _serial = crate::builtins::common::test_support::accel_test_lock();
    let provider = Box::leak(Box::new(Transfers::default()));
    unsafe { runmat_accelerate_api::register_provider(provider) };
    let _selected = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let strict = crate::context::RuntimeContext::new(Rc::new(
        crate::execution::RuntimeExecutionService::new(),
    ));
    strict.set_runmat_extensions_enabled(false);
    let enabled = crate::context::RuntimeContext::new(Rc::new(
        crate::execution::RuntimeExecutionService::new(),
    ));
    enabled.set_runmat_extensions_enabled(true);
    let args = |h: Value, x0: Value, options: Value| {
        vec![
            h,
            Value::Num(2.0),
            empty(),
            empty(),
            empty(),
            empty(),
            empty(),
            empty(),
            x0,
            options,
        ]
    };
    let parse = |runtime: &crate::context::RuntimeContext, args| {
        let mut coop =
            ScalarCooperation::new(runmat_builtins::OptimSolver::Quadprog, Some(runtime));
        block_on(super::parse::problem(
            crate::builtins::common::test_support::builtin_arguments(args),
            None,
            &mut coop,
        ))
    };
    {
        let _foreign = enabled.enter();
        let single_tol = Value::Tensor(
            Tensor::new_with_dtype(vec![1e-8], vec![1, 1], runmat_value::NumericDType::F32)
                .unwrap(),
        );
        assert!(parse(
            &strict,
            args(Value::Num(1.0), empty(), options([("TolFun", single_tol)]))
        )
        .is_ok());
        for late in [
            provider.value(0.0, Some(Explicit)),
            provider.value(0.0, None),
            Value::Int(runmat_value::IntValue::I32(0)),
            Value::Tensor(
                Tensor::new_with_dtype(vec![0.0], vec![1, 1], runmat_value::NumericDType::F32)
                    .unwrap(),
            ),
        ] {
            assert!(parse(
                &strict,
                args(
                    provider.value(1.0, Some(Automatic)),
                    late,
                    options([("TolFun", provider.value(1e-8, Some(Automatic)))])
                )
            )
            .is_err());
            assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
        }
        for option in [
            options([("TolFun", provider.value(1e-8, Some(Explicit)))]),
            options([("MaxIter", Value::Int(runmat_value::IntValue::I32(20)))]),
        ] {
            assert!(parse(
                &strict,
                args(provider.value(1.0, Some(Automatic)), empty(), option)
            )
            .is_err());
            assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
        }
        assert!(parse(
            &strict,
            args(
                provider.value(1.0, Some(Automatic)),
                empty(),
                options([("TolFun", provider.value(1e-8, Some(Automatic)))])
            )
        )
        .is_ok());
        assert_eq!(provider.downloads.load(Ordering::SeqCst), 2);
    }
    {
        let _foreign = strict.enter();
        assert!(parse(
            &enabled,
            args(
                provider.value(1.0, Some(Explicit)),
                Value::Int(runmat_value::IntValue::I32(0)),
                options([
                    ("MaxIter", Value::Int(runmat_value::IntValue::I32(20))),
                    ("TolFun", provider.value(1e-8, Some(Explicit)))
                ])
            )
        )
        .is_ok());
        assert_eq!(provider.downloads.load(Ordering::SeqCst), 4);
        assert!(parse(
            &enabled,
            args(
                Value::Tensor(
                    Tensor::new_with_dtype(vec![1.0], vec![1, 1], runmat_value::NumericDType::F32)
                        .unwrap()
                ),
                empty(),
                empty()
            )
        )
        .is_ok());
    }
    assert_eq!(runmat_builtins::QUADPROG_CATALOG_ENTRY.extensions.len(), 12);
    assert!(runmat_builtins::QUADPROG_INTEGER_CAPABILITIES
        .iter()
        .flat_map(|cap| cap.inputs)
        .all(|input| input.availability
            == runmat_builtins::BuiltinIntegerInputAvailability::RunMatOnly));
}
#[test]
fn observable_solver_time_requires_original_clock_before_transfer_and_unobserved_calls_do_not_read_it(
) {
    use crate::context::{DeterministicTimingService, RuntimeContext, RuntimeTimingService};
    use runmat_accelerate_api::GpuHandleProvenance::Automatic;
    use std::{rc::Rc, sync::atomic::Ordering, time::Duration};
    fn clock_runtime(step: u64) -> (RuntimeContext, Rc<DeterministicTimingService>) {
        let runtime = crate::user_functions::test_support::exact_fixture_runtime();
        let clock = Rc::new(
            DeterministicTimingService::new(
                runmat_execution::host::NativeClockInputs {
                    schema_version: runmat_execution::host::NATIVE_CLOCK_INPUTS_SCHEMA_VERSION,
                    initial_monotonic_nanos: 100,
                    initial_wall_unix_seconds: 0,
                    initial_wall_nanosecond: 0,
                    read_step_nanos: step,
                    local_offset_seconds: 0,
                    process_cpu: None,
                },
                runtime.cancellation(),
            )
            .unwrap(),
        );
        let ports = runtime.service_ports().clone().with_timing(clock.clone());
        (runtime.with_service_ports(ports), clock)
    }
    let _serial = crate::builtins::common::test_support::accel_test_lock();
    let provider = Box::leak(Box::new(Transfers::default()));
    unsafe { runmat_accelerate_api::register_provider(provider) };
    let _selected = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let denied = crate::user_functions::test_support::exact_fixture_runtime();
    let denied = denied
        .clone()
        .with_service_ports(denied.service_ports().clone().without_timing());
    let (original, clock) = clock_runtime(1_000_000);
    let (foreign, foreign_clock) = clock_runtime(9_000_000);
    let args = |name: &str| {
        let matrix = provider.value(if name == "quadprog" { 2.0 } else { 1.0 }, Some(Automatic));
        if name == "quadprog" {
            vec![matrix, Value::Num(-2.0)]
        } else {
            vec![matrix, Value::Num(2.0), empty(), empty()]
        }
    };
    let _ambient = foreign.enter();
    assert_eq!(
        SolverTiming::start(None).err().unwrap().identifier(),
        Some("RunMat:Clock:ServiceUnavailable"),
        "an ambient clock cannot replace a missing original context"
    );
    for (name, threshold) in [("quadprog", 4), ("lsqlin", 5)] {
        let before = provider.downloads.load(Ordering::SeqCst);
        let error = block_on(crate::call_builtin_async_with_outputs_in_context(
            &denied,
            name,
            &args(name),
            threshold,
        ))
        .unwrap_err();
        assert_eq!(error.identifier(), Some("RunMat:Clock:ServiceUnavailable"));
        assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
        assert_eq!(
            provider.downloads.load(Ordering::SeqCst),
            before,
            "clock denial must precede input materialization"
        );
        let result = block_on(crate::call_builtin_async_with_outputs_in_context(
            &denied,
            name,
            &args(name),
            threshold - 1,
        ))
        .expect("without diagnostic output no clock is read");
        assert!(matches!(result.value(),Value::OutputList(outputs)if outputs.len()==threshold-1));
        assert_eq!(provider.downloads.load(Ordering::SeqCst), before + 1);
        let result = block_on(crate::call_builtin_async_with_outputs_in_context(
            &original,
            name,
            &args(name),
            threshold,
        ))
        .unwrap();
        let Value::OutputList(outputs) = result.value() else {
            panic!("requested diagnostics")
        };
        assert_eq!(
            field(&outputs[threshold - 1], "wallclocktime"),
            &Value::Num(Duration::from_millis(1).as_secs_f64())
        );
        assert_eq!(provider.downloads.load(Ordering::SeqCst), before + 2);
    }
    assert_eq!(
        clock.monotonic_elapsed(),
        Duration::from_nanos(100 + 4_000_000),
        "exactly two selected-clock reads per observed solve"
    );
    assert_eq!(
        foreign_clock.monotonic_elapsed(),
        Duration::from_nanos(100),
        "foreign ambient timing was never consulted"
    );
}
