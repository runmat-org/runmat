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
        "MILP whole-call admission witness".into()
    }
    fn upload(
        &self,
        _: &runmat_accelerate_api::HostTensorView,
    ) -> anyhow::Result<runmat_accelerate_api::GpuTensorHandle> {
        panic!("owned test allocations")
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
                .find(|v| v.handle.allocation_identity() == handle.allocation_identity())
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
fn data_controls_and_clock_are_admitted_before_transfer_using_original_policy() {
    use runmat_accelerate_api::GpuHandleProvenance::{Automatic, Explicit};
    use std::sync::atomic::Ordering;
    let _serial = crate::builtins::common::test_support::accel_test_lock();
    let provider = Box::leak(Box::new(Transfers::default()));
    unsafe { runmat_accelerate_api::register_provider(provider) };
    let _selected = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let strict = context();
    strict.set_runmat_extensions_enabled(false);
    let enabled = context();
    enabled.set_runmat_extensions_enabled(true);
    let args = |x0: Value, options: Value| {
        vec![
            provider.value(1.0, Some(Automatic)),
            Value::Num(1.0),
            empty(),
            empty(),
            empty(),
            empty(),
            Value::Num(0.0),
            Value::Num(1.0),
            x0,
            options,
        ]
    };
    let parse = |runtime: &crate::context::RuntimeContext, args| {
        let mut cooperation = ScalarCooperation::for_builtin(
            runmat_builtins::INTLINPROG_CATALOG_ENTRY.identity,
            Some(runtime),
        );
        block_on(super::super::parse::parse(
            crate::builtins::common::test_support::builtin_arguments(args),
            runtime,
            &mut cooperation,
        ))
    };
    {
        let _foreign = enabled.enter();
        for late in [
            provider.value(0.0, Some(Explicit)),
            provider.value(0.0, None),
            Value::Int(runmat_value::IntValue::I32(0)),
        ] {
            let error = parse(
                &strict,
                args(
                    late,
                    options([("ConstraintTolerance", provider.value(1e-6, Some(Automatic)))]),
                ),
            )
            .err()
            .unwrap();
            assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
            assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
        }
        let single = Value::Tensor(
            Tensor::new_with_dtype(vec![0.0], vec![1, 1], runmat_value::NumericDType::F32).unwrap(),
        );
        assert!(parse(
            &strict,
            args(
                single,
                options([("ConstraintTolerance", provider.value(1e-6, Some(Automatic)))])
            )
        )
        .is_err());
        assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
        assert!(parse(
            &strict,
            args(
                empty(),
                options([("MaxNodes", Value::Int(runmat_value::IntValue::U16(3)))])
            )
        )
        .is_err());
        assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
    }
    let denied = context();
    let ports = denied.service_ports().clone().without_timing();
    let denied = denied.with_service_ports(ports);
    let error = parse(
        &denied,
        args(
            empty(),
            options([("ConstraintTolerance", provider.value(1e-6, Some(Automatic)))]),
        ),
    )
    .err()
    .unwrap();
    assert_eq!(error.identifier(), Some("RunMat:Clock:ServiceUnavailable"));
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
    // Ordinary Single A/b/beq and floating controls do not inherit the
    // role-specific Double-only restriction on f/Aeq/bounds/x0.
    let single = |v| {
        Value::Tensor(
            Tensor::new_with_dtype(vec![v], vec![1, 1], runmat_value::NumericDType::F32).unwrap(),
        )
    };
    let parsed = parse(
        &strict,
        vec![
            provider.value(1.0, Some(Automatic)),
            Value::Num(1.0),
            single(1.0),
            single(1.0),
            Value::Num(1.0),
            single(0.0),
            Value::Num(0.0),
            Value::Num(1.0),
            empty(),
            options([("ConstraintTolerance", provider.value(1e-6, Some(Automatic)))]),
        ],
    )
    .unwrap();
    assert_eq!(parsed.0.f, vec![1.0]);
    assert_eq!(parsed.0.equalities, 1);
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 2);
}
#[test]
fn finite_host_policy_refuses_before_solver_workspace_or_provider_argument_work() {
    use crate::context::host_effect::ManagedHostEffectLedger;
    use crate::context::{
        ParallelCapability, RuntimeParallelResources, RuntimeParallelService, RuntimeServicePorts,
    };
    use std::rc::Rc;
    struct Managed(ManagedHostEffectLedger);
    impl RuntimeParallelService for Managed {
        fn supports(&self, _: ParallelCapability) -> bool {
            false
        }
        fn placement_resources(&self) -> RuntimeParallelResources {
            let snapshot = self.0.snapshot();
            RuntimeParallelResources {
                memory_available_bytes: Some(snapshot.free_bytes()),
                epoch: snapshot.epoch,
                ..RuntimeParallelResources::default()
            }
        }
        fn managed_host_effect_ledger(&self) -> Option<ManagedHostEffectLedger> {
            Some(self.0.clone())
        }
    }
    let ledger = ManagedHostEffectLedger::new(1_000_000);
    let runtime = context().with_service_ports(
        RuntimeServicePorts::default().with_parallel(Rc::new(Managed(ledger.clone()))),
    );
    let before = ledger.snapshot();
    let error = block_on(execute(
        Some(&runtime),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .unwrap_err();
    assert!(error.to_string().contains("finite host-memory"));
    assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
    let after = ledger.snapshot();
    assert_eq!(before.live_reserved_bytes, after.live_reserved_bytes);
}
