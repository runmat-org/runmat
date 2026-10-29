//! Selector oracles are hardware-independent. The native fixture certifies
//! Runtime ownership/control semantics only; the ignored WGPU witness below
//! is a required separate physical-backend acceptance, never a silent fallback.
use super::*;
use crate::builtins::common::test_support;
use runmat_value::{IntegerStorage, NumericStorage, Tensor};
use std::rc::Rc;

fn runtime() -> RuntimeContext {
    RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
}

#[test]
fn logical_extension_gate_keeps_runtime_precedence_over_arity_and_context() {
    let _guard = test_support::accel_test_lock();
    let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
    let error = futures::executor::block_on(gpu_device_builtin(
        None,
        test_support::builtin_arguments(vec![Value::Bool(true), Value::Num(1.0)]),
    ))
    .unwrap_err();
    assert_eq!(
        error.identifier(),
        Some("RunMat:compatibility:GpuDeviceLogicalSelectorExtension")
    );
}

#[test]
fn gpu_device_without_an_authorized_provider_has_its_own_public_error() {
    use runmat_execution::identity::{AttemptId, ExecutionScopeId, PoolId, TaskId, WorkerId};
    let _guard = test_support::accel_test_lock();
    let runtime = runtime();
    let _context = runtime.enter();
    let _assignment =
        runtime.enter_execution_assignment(Some(runmat_execution::ProgramExecutionAssignment {
            scope_id: ExecutionScopeId::derive(&[b"GPUDevice-absent"]),
            pool_id: PoolId::derive(&[b"pool"]),
            task_id: TaskId::derive(&[b"task"]),
            attempt_id: AttemptId::derive(&[b"attempt"]),
            worker_id: WorkerId::derive(&[b"worker"]),
            backend: runmat_execution::PoolBackend::LocalProcesses,
            resources: Default::default(),
        }));
    let error = futures::executor::block_on(gpu_device_builtin(
        Some(runtime.clone()),
        test_support::builtin_arguments(vec![]),
    ))
    .unwrap_err();
    assert_eq!(error.identifier(), GPU_DEVICE_ERROR_NO_PROVIDER.identifier);
    assert_eq!(error.to_string(), GPU_DEVICE_ERROR_NO_PROVIDER.message);
}

#[test]
fn gpu_device_index_parser_preserves_native_storage_and_exact_boundaries() {
    let values = [
        Value::Int(IntValue::I32(1)),
        Value::Int(IntValue::U64(1)),
        Value::Num(1.0),
        Value::Bool(true),
        Value::Tensor(Tensor::new_integer(IntegerStorage::U64(vec![1]), vec![1, 1]).unwrap()),
        Value::Tensor(
            Tensor::from_numeric_storage(NumericStorage::F32(vec![1.0]), vec![1, 1]).unwrap(),
        ),
        Value::LogicalArray(runmat_value::LogicalArray::new(vec![1], vec![1, 1]).unwrap()),
    ];
    for value in values {
        assert_eq!(parse_device_index(&value).unwrap(), Some(1));
    }
    let invalid = [
        Value::Num(0.0),
        Value::Num(-1.0),
        Value::Num(1.5),
        Value::Num(1.0 - 1e-10),
        Value::Num(1.0 + 1e-10),
        Value::Num(f64::NAN),
        Value::Num(f64::INFINITY),
        Value::Num(f64::from(u32::MAX) + 1.0),
        Value::Int(IntValue::U64(u64::MAX)),
        Value::Int(IntValue::I32(0)),
        Value::Bool(false),
        Value::Tensor(Tensor::new_integer(IntegerStorage::I16(vec![-1]), vec![1, 1]).unwrap()),
        Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap()),
        Value::LogicalArray(runmat_value::LogicalArray::new(vec![0], vec![1, 1]).unwrap()),
    ];
    for value in invalid {
        let error = parse_device_index(&value).unwrap_err();
        assert_eq!(
            error.identifier(),
            GPU_DEVICE_ERROR_INVALID_INDEX.identifier
        );
    }
    assert_eq!(
        parse_device_index(&Value::Int(IntValue::U32(2))).unwrap(),
        Some(2)
    );
    for value in [
        Value::from("status"),
        Value::from("reset"),
        Value::CharArray(runmat_value::CharArray::new_row("reset")),
    ] {
        assert_eq!(parse_device_index(&value).unwrap(), None);
    }
    assert_eq!(
        parse_device_index(&Value::Tensor(Tensor::zeros(vec![0, 0]))).unwrap(),
        None
    );
}

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use super::*;
    use crate::context::{RuntimeLocalTaskSupervisor, RuntimeServicePorts};
    use runmat_accelerate_api::{
        AccelProviderFuture, DeviceAllocationScope, ProviderDeviceControl, ProviderGpuDeviceFacts,
    };
    use std::future::Future;
    use std::sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Mutex,
    };

    struct DeviceFixture {
        id: u32,
        resets: AtomicUsize,
        waits: AtomicUsize,
        fail: AtomicBool,
        scopes: Mutex<Vec<DeviceAllocationScope>>,
        gate: Mutex<Option<futures::channel::oneshot::Receiver<()>>>,
        wait_gate: Mutex<Option<futures::channel::oneshot::Receiver<()>>>,
    }
    impl DeviceFixture {
        const fn new(id: u32) -> Self {
            Self {
                id,
                resets: AtomicUsize::new(0),
                waits: AtomicUsize::new(0),
                fail: AtomicBool::new(false),
                scopes: Mutex::new(Vec::new()),
                gate: Mutex::new(None),
                wait_gate: Mutex::new(None),
            }
        }
        fn clear(&self) {
            self.resets.store(0, Ordering::SeqCst);
            self.waits.store(0, Ordering::SeqCst);
            self.fail.store(false, Ordering::SeqCst);
            self.scopes.lock().unwrap().clear();
            self.gate.lock().unwrap().take();
            self.wait_gate.lock().unwrap().take();
        }
    }
    static DEVICE: DeviceFixture = DeviceFixture::new(0xD275);
    static OTHER: DeviceFixture = DeviceFixture::new(0xD276);
    #[derive(Debug, thiserror::Error)]
    #[error("injected physical device failure")]
    struct PhysicalFailure;
    impl AccelProvider for DeviceFixture {
        fn upload(
            &self,
            _: &runmat_accelerate_api::HostTensorView,
        ) -> anyhow::Result<runmat_accelerate_api::GpuTensorHandle> {
            anyhow::bail!("fixture does not execute arrays")
        }
        fn download<'a>(
            &'a self,
            _: &'a runmat_accelerate_api::GpuTensorHandle,
        ) -> runmat_accelerate_api::AccelDownloadFuture<'a> {
            Box::pin(async { anyhow::bail!("fixture does not execute arrays") })
        }
        fn device_info(&self) -> String {
            "device-control fixture, not physical GPU evidence".into()
        }
        fn device_id(&self) -> u32 {
            self.id
        }
        fn gpu_device_control(&self) -> Option<&dyn ProviderDeviceControl> {
            Some(self)
        }
    }
    impl ProviderDeviceControl for DeviceFixture {
        fn gpu_device_facts(&self) -> ProviderGpuDeviceFacts<'_> {
            ProviderGpuDeviceFacts {
                name: "typed fixture",
                available: true,
                total_memory_bytes: Some(1024),
                available_memory_bytes: None,
                supports_double: Some(true),
            }
        }
        fn has_device_scope(&self, scope: DeviceAllocationScope) -> Option<bool> {
            Some(
                self.scopes
                    .lock()
                    .unwrap()
                    .iter()
                    .any(|created| scope.covers(*created)),
            )
        }
        fn control_host_bytes(&self) -> Option<u64> {
            Some(0)
        }
        fn reset_host_bytes(&self, _: DeviceAllocationScope) -> Option<u64> {
            Some(0)
        }
        fn control_completion_tail_bytes(&self) -> Option<u64> {
            Some(0)
        }
        fn wait_for_device(
            &self,
            _: Option<std::sync::Arc<dyn runmat_execution::resource::RetainedHostAllocationClaim>>,
        ) -> Option<AccelProviderFuture<'_, ()>> {
            self.waits.fetch_add(1, Ordering::SeqCst);
            let gate = self.wait_gate.lock().unwrap().take();
            Some(Box::pin(async move {
                if let Some(gate) = gate {
                    gate.await?;
                }
                Ok(())
            }))
        }
        fn reset_device_scope(
            &self,
            scope: DeviceAllocationScope,
            _: Option<std::sync::Arc<dyn runmat_execution::resource::RetainedHostAllocationClaim>>,
        ) -> Option<AccelProviderFuture<'_, ()>> {
            let gate = self.gate.lock().unwrap().take();
            Some(Box::pin(async move {
                self.resets.fetch_add(1, Ordering::SeqCst);
                if let Some(gate) = gate {
                    gate.await?;
                }
                if self.fail.load(Ordering::SeqCst) {
                    return Err(PhysicalFailure.into());
                }
                self.scopes
                    .lock()
                    .unwrap()
                    .retain(|created| !scope.covers(*created));
                Ok(())
            }))
        }
    }
    fn fixture(
        test: impl FnOnce(RuntimeContext, RuntimeLocalTaskSupervisor, tokio::runtime::Runtime),
    ) {
        let _guard = test_support::accel_test_lock();
        DEVICE.clear();
        OTHER.clear();
        unsafe {
            runmat_accelerate_api::register_provider(&DEVICE);
            runmat_accelerate_api::register_device_provider(&OTHER);
        }
        let runtime = runtime();
        let supervisor = RuntimeLocalTaskSupervisor::new();
        let runtime = runtime.clone().with_service_ports(
            RuntimeServicePorts::default()
                .with_local_tasks(supervisor.submission_port(runtime.local_identity())),
        );
        let host = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        test(runtime, supervisor, host);
    }
    fn call(
        runtime: &RuntimeContext,
        supervisor: &RuntimeLocalTaskSupervisor,
        host: &tokio::runtime::Runtime,
        args: Vec<Value>,
    ) -> BuiltinResult<RootedValue> {
        host.block_on(supervisor.drive_until(runtime.scope(gpu_device_builtin(
            Some(runtime.clone()),
            test_support::builtin_arguments(args),
        ))))
        .unwrap()
    }

    #[test]
    fn gpu_device_is_a_typed_ungated_session_bound_handle_with_real_facts() {
        fixture(|runtime, supervisor, host| {
            let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
            let value = call(&runtime, &supervisor, &host, vec![]).unwrap();
            let _selected = runtime.enter();
            assert!(
                matches!(value.value(), Value::HandleObject(handle) if handle.class_name.is(runmat_builtins::GPU_DEVICE_CLASS))
            );
            assert!(matches!(
                object::read(&runtime, value.value(), GpuDeviceProperty::Name).unwrap(),
                Value::CharArray(_)
            ));
            assert_eq!(
                object::read(&runtime, value.value(), GpuDeviceProperty::SupportsDouble).unwrap(),
                Value::Bool(true)
            );
            assert_eq!(
                object::read(&runtime, value.value(), GpuDeviceProperty::TotalMemory).unwrap(),
                Value::Num(1024.0)
            );
            assert!(
                object::read(&runtime, value.value(), GpuDeviceProperty::AvailableMemory).is_err(),
                "total bytes are never free bytes"
            );
            assert_eq!(
                object::read(&runtime, value.value(), GpuDeviceProperty::DeviceSelected).unwrap(),
                Value::Bool(true)
            );
            assert!(
                object::identity(&super::runtime(), value.value()).is_err(),
                "foreign session cannot use the handle"
            );
            let class =
                crate::class_registry::get_class(&runmat_builtins::GPU_DEVICE_CLASS.owned())
                    .unwrap();
            for property in GpuDeviceProperty::ALL {
                let name: runmat_types::MemberName = property.name().into();
                assert_eq!(
                    class.properties[&name].set_access,
                    runmat_types::MemberAccess::Private
                );
            }
        });
    }

    #[test]
    fn gpu_device_public_member_getters_preserve_issued_handle_identity() {
        fixture(|runtime, supervisor, host| {
            let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
            let (selected, index) = current_device(&runtime).unwrap();
            assert!(runmat_accelerate_api::same_provider_instance(
                selected, &DEVICE
            ));
            let value = call(&runtime, &supervisor, &host, vec![]).unwrap();
            let read = |context: &RuntimeContext, receiver: Value, name: &str| {
                host.block_on(supervisor.drive_until(context.scope(
                    crate::object::resolve::load_member_with_context(
                        Some(context),
                        receiver,
                        name.into(),
                        false,
                        None,
                    ),
                )))
                .unwrap()
            };
            for (name, expected) in [
                (
                    "Name",
                    Value::CharArray(runmat_value::CharArray::new_row("typed fixture")),
                ),
                ("Index", Value::Num(f64::from(index))),
                ("TotalMemory", Value::Num(1024.0)),
                ("SupportsDouble", Value::Bool(true)),
                ("DeviceAvailable", Value::Bool(true)),
                ("DeviceSelected", Value::Bool(true)),
            ] {
                let actual = read(&runtime, value.value().clone(), name)
                    .expect("normal member getter receives issued handle");
                assert_eq!(actual.value(), &expected, "property {name}");
            }
            assert!(
                read(&runtime, value.value().clone(), "AvailableMemory").is_err(),
                "missing backend measurements remain unsupported"
            );
            assert!(
                read(
                    &runtime,
                    value.value().clone(),
                    "__runmat_gpudevice_provider"
                )
                .is_err(),
                "backing identity stays private"
            );
            assert!(
                read(&super::runtime(), value.value().clone(), "Name").is_err(),
                "public dispatch must not bypass session identity"
            );
            let Value::HandleObject(mut invalid) = value.value().clone() else {
                panic!("handle")
            };
            invalid.valid = false;
            assert!(
                read(&runtime, Value::HandleObject(invalid), "Name").is_err(),
                "invalid handle remains rejected before getter execution"
            );
        });
    }

    #[test]
    fn wait_rejects_already_stopped_before_installing_provider_work() {
        use runmat_execution::{cancellation::CancellationSource, CancellationReason};
        fixture(|runtime, supervisor, host| {
            let value = call(&runtime, &supervisor, &host, vec![]).unwrap();
            for reason in [CancellationReason::User, CancellationReason::Deadline] {
                let source = CancellationSource::new();
                source.cancel(reason);
                let stopped = runtime.clone().with_cancellation_token(source.token());
                let failure = host
                    .block_on(
                        stopped.scope(wait_device(Some(stopped.clone()), value.value().clone())),
                    )
                    .unwrap_err();
                assert!(crate::runtime_error::error_has_control_cancellation_origin(
                    &failure
                ));
                assert_eq!(stopped.cancellation().reason(), Some(reason));
                assert_eq!(DEVICE.waits.load(Ordering::SeqCst), 0);
            }
        });
    }

    #[test]
    fn pending_wait_is_cancellation_woken_without_backend_completion() {
        use runmat_execution::{cancellation::CancellationSource, CancellationReason};
        use std::{
            sync::Arc,
            task::{Context, Poll},
        };
        struct WakeCount(AtomicUsize);
        impl futures::task::ArcWake for WakeCount {
            fn wake_by_ref(this: &Arc<Self>) {
                this.0.fetch_add(1, Ordering::SeqCst);
            }
        }
        fixture(|runtime, supervisor, host| {
            let value = call(&runtime, &supervisor, &host, vec![]).unwrap();
            let source = CancellationSource::new();
            let runtime = runtime.with_cancellation_token(source.token());
            let (send, receive) = futures::channel::oneshot::channel();
            *DEVICE.wait_gate.lock().unwrap() = Some(receive);
            let wake = Arc::new(WakeCount(AtomicUsize::new(0)));
            let waker = futures::task::waker(wake.clone());
            let mut cx = Context::from_waker(&waker);
            let mut pending =
                Box::pin(runtime.scope(wait_device(Some(runtime.clone()), value.value().clone())));
            assert!(pending.as_mut().poll(&mut cx).is_pending());
            assert_eq!(DEVICE.waits.load(Ordering::SeqCst), 1);
            source.cancel(CancellationReason::Deadline);
            assert!(wake.0.load(Ordering::SeqCst) > 0);
            let Poll::Ready(Err(failure)) = pending.as_mut().poll(&mut cx) else {
                panic!("cancel must complete without backend callback")
            };
            assert!(crate::runtime_error::error_has_control_cancellation_origin(
                &failure
            ));
            assert_eq!(
                runtime.cancellation().reason(),
                Some(CancellationReason::Deadline)
            );
            assert!(
                send.send(()).is_err(),
                "logical wait receiver retired without a completion signal"
            );
        });
    }

    #[test]
    fn reset_retires_inherited_other_provider_scopes_but_preserves_peer_epochs() {
        fixture(|runtime, supervisor, host| {
            let scope = runtime.gpu_device_allocation_scope().unwrap();
            let peer = super::runtime().gpu_device_allocation_scope().unwrap();
            DEVICE.scopes.lock().unwrap().extend([scope, peer]);
            OTHER.scopes.lock().unwrap().extend([scope, peer]);
            let index = {
                let _selected = runtime.enter();
                current_device(&runtime).unwrap().1
            };
            let value = call(
                &runtime,
                &supervisor,
                &host,
                vec![Value::Int(IntValue::U32(index))],
            )
            .unwrap();
            assert!(matches!(value.value(), Value::HandleObject(_)));
            assert_eq!(&*DEVICE.scopes.lock().unwrap(), &[peer]);
            assert_eq!(&*OTHER.scopes.lock().unwrap(), &[peer]);
            assert_eq!(
                runtime.gpu_device_allocation_scope().unwrap(),
                scope.successor().unwrap()
            );
            call(
                &runtime,
                &supervisor,
                &host,
                vec![Value::Tensor(Tensor::zeros(vec![0, 0]))],
            )
            .unwrap();
            let _selected = runtime.enter();
            assert!(
                runmat_accelerate_api::provider().is_none(),
                "deselection cannot fall through to default/registry"
            );
            assert!(runmat_accelerate_api::provider_for_device(DEVICE.id).is_none());
            assert!(runmat_accelerate_api::validate_device_allocation_scope(None).is_err());
            assert_eq!(
                object::read(&runtime, value.value(), GpuDeviceProperty::DeviceSelected).unwrap(),
                Value::Bool(false)
            );
        });
    }

    #[test]
    fn dropped_awaiting_caller_does_not_abandon_reset_or_publish_old_scope() {
        use std::task::Poll;
        fixture(|runtime, supervisor, host| {
            let scope = runtime.gpu_device_allocation_scope().unwrap();
            let (send, receive) = futures::channel::oneshot::channel();
            *DEVICE.gate.lock().unwrap() = Some(receive);
            host.block_on(supervisor.drive_until(async {
                let mut caller = Box::pin(runtime.scope(runtime.reset_gpu_device(&DEVICE, false)));
                std::future::poll_fn(|cx| {
                    assert!(caller.as_mut().poll(cx).is_pending());
                    Poll::Ready(())
                })
                .await;
                let retained = runtime
                    .gpu_device_reset_lease_for_test()
                    .expect("supervisor owns reset");
                drop(caller);
                {
                    let _selected = runtime.enter();
                    assert!(runmat_accelerate_api::validate_device_allocation_scope(None).is_err());
                    assert!(
                        runmat_accelerate_api::validate_device_allocation_scope(Some(scope))
                            .is_err()
                    );
                    assert!(runmat_accelerate_api::provider().is_none());
                }
                send.send(()).unwrap();
                retained.join().await;
            }))
            .unwrap();
            let _selected = runtime.enter();
            assert!(runmat_accelerate_api::validate_device_allocation_scope(Some(scope)).is_err());
            assert!(
                runmat_accelerate_api::validate_device_allocation_scope(scope.successor()).is_ok()
            );
        });
    }

    #[test]
    fn failure_retains_typed_source_and_epoch_and_explicit_retry_keeps_authority() {
        use std::error::Error;
        fixture(|runtime, supervisor, host| {
            let scope = runtime.gpu_device_allocation_scope().unwrap();
            DEVICE.fail.store(true, Ordering::SeqCst);
            let failure = host
                .block_on(
                    supervisor.drive_until(runtime.scope(runtime.reset_gpu_device(&DEVICE, false))),
                )
                .unwrap()
                .unwrap_err();
            let mut source: &(dyn Error + 'static) = &failure;
            let mut found = false;
            loop {
                found |= source.is::<PhysicalFailure>();
                match source.source() {
                    Some(next) => source = next,
                    None => break,
                }
            }
            assert!(
                found,
                "the actual backend source survives the retained failure receipt"
            );
            let _selected = runtime.enter();
            assert!(runmat_accelerate_api::provider().is_none());
            assert!(runmat_accelerate_api::validate_device_allocation_scope(None).is_err());
            drop(_selected);
            DEVICE.fail.store(false, Ordering::SeqCst);
            host.block_on(
                supervisor.drive_until(runtime.scope(runtime.reset_gpu_device(&DEVICE, false))),
            )
            .unwrap()
            .unwrap();
            assert_eq!(runtime.gpu_device_allocation_scope(), scope.successor());
        });
    }

    #[cfg(feature = "wgpu")]
    #[test]
    #[ignore = "requires actual WGPU adapter; required native physical reset witness"]
    fn gpu_device_wgpu_reports_facts_and_retires_only_its_session() {
        use runmat_accelerate::backend::wgpu::provider as wgpu_provider;
        let _guard = test_support::accel_test_lock();
        let provider =
            wgpu_provider::register_wgpu_provider(wgpu_provider::WgpuProviderOptions::default())
                .expect("required physical WGPU adapter");
        let owner = runtime();
        let peer = runtime();
        let host = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let supervisor = RuntimeLocalTaskSupervisor::new();
        let owner = owner.clone().with_service_ports(
            RuntimeServicePorts::default()
                .with_local_tasks(supervisor.submission_port(owner.local_identity())),
        );
        let make = |runtime: &RuntimeContext| {
            let _selected = runtime.enter();
            provider
                .upload(&runmat_accelerate_api::HostTensorView {
                    data: &[1.0, 2.0],
                    shape: &[2, 1],
                })
                .unwrap()
        };
        let old = make(&owner);
        let still_live = make(&peer);
        let value = call(&owner, &supervisor, &host, vec![]).unwrap();
        {
            let _selected = owner.enter();
            assert!(object::read(&owner, value.value(), GpuDeviceProperty::TotalMemory).is_err());
            assert!(
                object::read(&owner, value.value(), GpuDeviceProperty::AvailableMemory).is_err()
            );
            assert!(
                matches!(object::read(&owner, value.value(), GpuDeviceProperty::Name).unwrap(), Value::CharArray(name) if !name.data().is_empty())
            );
        }
        host.block_on(supervisor.drive_until(owner.scope(owner.reset_gpu_device(provider, false))))
            .unwrap()
            .unwrap();
        assert!(old.allocation_lease().validate_live().is_err());
        assert!(still_live.allocation_lease().validate_live().is_ok());
    }
}
