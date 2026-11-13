use super::*;
use crate::context::host_effect::ManagedHostEffectLedger;
use crate::context::{
    ParallelCapability, RuntimeParallelResources, RuntimeParallelService, RuntimeServicePorts,
};
use futures::executor::block_on;
use runmat_accelerate_api::{
    test_support::TestAllocationOwner, AccelProvider, GpuTensorHandle, GpuTensorStorage,
    HostNumericDataOwned, HostNumericTensorOwned, NumericDownloadStageLease, NumericElementType,
    PreparedNumericDownloadStage, ProviderFeasibilityQuery,
};
use runmat_execution::{cancellation::CancellationSource, CancellationReason, ResourceByteBound};
use runmat_value::{CellArray, ClosureData, ObjectArray, ObjectInstance, StructArray, Tensor};
use std::{
    future::Future,
    rc::Rc,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Mutex,
    },
    task::{Context, Poll},
};

struct Parallel(ManagedHostEffectLedger);
impl RuntimeParallelService for Parallel {
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

fn runtime(ledger: &ManagedHostEffectLedger, stop: &CancellationSource) -> RuntimeContext {
    RuntimeContext::with_cancellation(
        Rc::new(crate::execution::RuntimeExecutionService::new()),
        stop.token(),
    )
    .with_service_ports(
        RuntimeServicePorts::default().with_parallel(Rc::new(Parallel(ledger.clone()))),
    )
}

struct Lease(Arc<()>);
impl NumericDownloadStageLease for Lease {
    fn clone_for_callback(&self) -> Box<dyn NumericDownloadStageLease> {
        Box::new(Self(self.0.clone()))
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

struct Provider {
    owner: TestAllocationOwner,
    ordinary: AtomicUsize,
    factories: AtomicUsize,
    pending: AtomicUsize,
    physical: Mutex<Option<PreparedNumericDownloadStage>>,
    marker: Arc<()>,
}
impl Provider {
    async fn staged(
        &self,
        handle: &GpuTensorHandle,
        mut stage: Option<PreparedNumericDownloadStage>,
    ) -> anyhow::Result<HostNumericTensorOwned> {
        let callback = stage.as_mut().unwrap().take_callback().unwrap();
        *self.physical.lock().unwrap() = Some(callback);
        if self.pending.load(Ordering::SeqCst) != 0 {
            return futures::future::pending().await;
        }
        drop(self.physical.lock().unwrap().take());
        drop(stage);
        Ok(HostNumericTensorOwned {
            data: HostNumericDataOwned::F64(vec![17.0; handle.shape().iter().product()]),
            shape: handle.shape().to_vec(),
            storage: GpuTensorStorage::Real,
        })
    }
}
impl AccelProvider for Provider {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }
    fn device_info(&self) -> String {
        "managed gather staging fixture".into()
    }
    fn upload(&self, _: &runmat_accelerate_api::HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        anyhow::bail!("unused upload")
    }
    fn download<'a>(
        &'a self,
        _: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::AccelDownloadFuture<'a> {
        Box::pin(async { anyhow::bail!("ordinary download must not run") })
    }
    fn download_numeric<'a>(
        &'a self,
        handle: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::AccelNumericDownloadFuture<'a> {
        self.ordinary.fetch_add(1, Ordering::SeqCst);
        Box::pin(async move {
            Ok(HostNumericTensorOwned {
                data: HostNumericDataOwned::F64(vec![17.0; handle.shape().iter().product()]),
                shape: handle.shape().to_vec(),
                storage: GpuTensorStorage::Real,
            })
        })
    }
    fn device_info_struct(&self) -> runmat_accelerate_api::ApiDeviceInfo {
        runmat_accelerate_api::ApiDeviceInfo {
            device_id: self.device_id(),
            name: self.device_info(),
            vendor: "RunMat".into(),
            memory_bytes: Some(1 << 20),
            backend: Some("test".into()),
        }
    }
    fn execution_accelerator_device(
        &self,
        epoch: u64,
    ) -> anyhow::Result<Option<runmat_execution::resource::AcceleratorDevice>> {
        runmat_accelerate_api::execution_accelerator_device(
            self,
            "runmat.test.readback",
            "1",
            runmat_accelerate_api::ExecutionProviderContract::GenericComputeV1,
            &format!("readback-{}", self.device_id()),
            &format!("readback-{}", self.device_id()),
            epoch,
        )
    }
    fn download_numeric_host_workspace_bound(&self, handle: &GpuTensorHandle) -> ResourceByteBound {
        ResourceByteBound::Known(
            (handle.shape().iter().product::<usize>() * 8 + std::mem::size_of_val(handle.shape()))
                as u64,
        )
    }
    fn download_numeric_provider_scratch_bound(&self, _: &GpuTensorHandle) -> ResourceByteBound {
        ResourceByteBound::Known(0)
    }
    fn download_numeric_completion_host_bound(
        &self,
        handle: &GpuTensorHandle,
    ) -> ResourceByteBound {
        let future = self.staged(handle, None);
        ResourceByteBound::Known(
            (std::mem::size_of_val(&future) + 2 * std::mem::size_of::<Lease>()) as u64,
        )
    }
    fn prepare_numeric_download_stage(
        &self,
        handle: &GpuTensorHandle,
        query: &ProviderFeasibilityQuery,
        claim: Arc<dyn runmat_execution::resource::RetainedHostAllocationClaim>,
    ) -> anyhow::Result<PreparedNumericDownloadStage> {
        assert!(query.matches_numeric_download(handle));
        Ok(PreparedNumericDownloadStage::issued(
            Box::new(Lease(self.marker.clone())),
            claim,
            handle,
            0,
            0,
        ))
    }
    fn download_numeric_admitted_stage<'a>(
        &'a self,
        handle: &'a GpuTensorHandle,
        stage: PreparedNumericDownloadStage,
    ) -> runmat_accelerate_api::AccelNumericDownloadFuture<'a> {
        assert!(stage.matches_source(handle));
        self.factories.fetch_add(1, Ordering::SeqCst);
        Box::pin(self.staged(handle, Some(stage)))
    }
}

fn provider() -> &'static Provider {
    let provider = Box::leak(Box::new(Provider {
        owner: TestAllocationOwner::new(),
        ordinary: AtomicUsize::new(0),
        factories: AtomicUsize::new(0),
        pending: AtomicUsize::new(0),
        physical: Mutex::new(None),
        marker: Arc::new(()),
    }));
    // A distinct durable fixture device, never the process's active provider.
    unsafe {
        runmat_accelerate_api::register_device_provider(provider);
    }
    provider
}

#[test]
fn readback_policy_matrix_preserves_source_assignment_and_denies_new_compute() {
    use crate::context::{RuntimeAccelerationPolicy, RuntimeProviderPreference};
    use runmat_execution::identity::{AttemptId, ExecutionScopeId, PoolId, TaskId, WorkerId};
    for managed in [false, true] {
        for (enabled, preference) in [
            (false, RuntimeProviderPreference::Auto),
            (true, RuntimeProviderPreference::InProcess),
            (true, RuntimeProviderPreference::Auto),
        ] {
            let provider = provider();
            let allocation = provider
                .owner
                .allocate_zeroed(vec![1, 1], NumericElementType::F64, GpuTensorStorage::Real)
                .unwrap();
            let ledger = ManagedHostEffectLedger::new(1 << 20);
            let stop = CancellationSource::new();
            let runtime = if managed {
                runtime(&ledger, &stop)
            } else {
                RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
            }
            .with_acceleration_policy(RuntimeAccelerationPolicy {
                enabled,
                provider: preference,
                automatic_offload: false,
                allow_host_fallback: false,
                calibrate: false,
            });
            for matching_assignment in [true, false] {
                let attempt_id = AttemptId::derive(&[b"readback-policy"]);
                let device = provider.execution_accelerator_device(7).unwrap().unwrap();
                let assignment = runmat_execution::ProgramExecutionAssignment {
                    scope_id: ExecutionScopeId::derive(&[b"readback"]),
                    pool_id: PoolId::derive(&[b"readback"]),
                    task_id: TaskId::derive(&[b"readback"]),
                    attempt_id,
                    worker_id: WorkerId::derive(&[b"readback"]),
                    backend: runmat_execution::PoolBackend::LocalProcesses,
                    resources: runmat_execution::resource::ResourceAssignment {
                        accelerator_leases: if matching_assignment {
                            vec![
                                runmat_execution::resource::AcceleratorDeviceLease::for_attempt(
                                    attempt_id, 11, &device,
                                )
                                .unwrap(),
                            ]
                        } else {
                            vec![]
                        },
                    },
                };
                let _assignment = runtime.enter_execution_assignment(Some(assignment));
                {
                    let _scope = runtime.enter();
                    assert_eq!(
                        gpu_helpers::exact_provider_for_handle(&allocation.handle).is_some(),
                        matching_assignment
                            && enabled
                            && preference == RuntimeProviderPreference::Auto,
                        "readback capability must never relax compute lookup"
                    );
                }
                let input = Value::GpuTensor(allocation.handle.clone());
                // Explicit context, no ambient guard: each download poll must
                // still resolve this exact assignment rather than a default.
                let output = block_on(super::super::gather_if_needed_in_context_async(
                    &input,
                    Some(&runtime),
                ));
                assert_eq!(output.is_ok(), matching_assignment);
                if let Ok(Value::Tensor(tensor)) = &output {
                    assert_eq!(tensor.materialize_f64(), vec![17.0]);
                }
                drop(output);
                drop(input);
                assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
            }
            assert_eq!(
                provider.ordinary.load(Ordering::SeqCst),
                usize::from(!managed)
            );
            assert_eq!(
                provider.factories.load(Ordering::SeqCst),
                usize::from(managed)
            );
            allocation.handle.release().unwrap();
            assert_eq!(allocation.probe.retirements(), 1);
        }
    }
}

#[test]
fn real_builtin_policy_decline_retries_only_automatic_valid_owner_inputs() {
    use crate::context::{RuntimeAccelerationPolicy, RuntimeProviderPreference};
    let provider = provider();
    for builtin in ["erf", "sum", "sin"] {
        for (explicit, fallback, registered) in [
            (false, true, true),
            (true, true, true),
            (false, false, true),
            (false, true, false),
        ] {
            let absent_owner = TestAllocationOwner::new();
            let allocation = if registered {
                &provider.owner
            } else {
                &absent_owner
            }
            .allocate_zeroed(vec![1, 1], NumericElementType::F64, GpuTensorStorage::Real)
            .unwrap();
            let handle = allocation.handle;
            let handle = if explicit {
                let explicit_handle = handle
                    .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
                    .unwrap();
                drop(handle);
                explicit_handle
            } else {
                handle
            };
            let input = Value::GpuTensor(handle);
            let runtime =
                RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
                    .with_acceleration_policy(RuntimeAccelerationPolicy {
                        enabled: false,
                        provider: RuntimeProviderPreference::Auto,
                        automatic_offload: false,
                        allow_host_fallback: fallback,
                        calibrate: false,
                    });
            let before = provider.ordinary.load(Ordering::SeqCst);
            let output = block_on(
                runtime.scope(crate::dispatcher::call_builtin_async_in_context(
                    &runtime,
                    builtin,
                    std::slice::from_ref(&input),
                )),
            );
            assert_eq!(
                output.is_ok(),
                !explicit && fallback && registered,
                "{builtin}"
            );
            assert_eq!(
                provider.ordinary.load(Ordering::SeqCst) - before,
                usize::from(!explicit && fallback && registered),
                "{builtin}: no hidden failed-attempt transfer"
            );
            if let Err(error) = &output {
                if registered {
                    assert!(
                        error.source.as_deref().is_some_and(|source| {
                            source.is::<gpu_helpers::ResidentComputePolicyDenied>()
                        }),
                        "{builtin}: retain typed policy cause"
                    );
                } else {
                    assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
                }
            }
            drop(output);
            drop(input);
            assert_eq!(allocation.probe.retirements(), 1);
        }
    }
    // Multi-input ingress must not turn a sibling ownership failure into a
    // requested retry merely because another source is policy-denied.
    let valid = provider
        .owner
        .allocate_zeroed(vec![1], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let missing_owner = TestAllocationOwner::new();
    let missing = missing_owner
        .allocate_zeroed(vec![1], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let sibling_owner = self::provider();
    let sibling = sibling_owner
        .owner
        .allocate_zeroed(vec![1], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let runtime = RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
        .with_acceleration_policy(RuntimeAccelerationPolicy {
            enabled: false,
            provider: RuntimeProviderPreference::Auto,
            automatic_offload: false,
            allow_host_fallback: true,
            calibrate: false,
        });
    let _scope = runtime.enter();
    enum ExpectedCause {
        Policy,
        InvalidSource,
        DifferentOwner,
    }
    for (sources, expected) in [
        ([&valid.handle, &valid.handle], ExpectedCause::Policy),
        (
            [&valid.handle, &missing.handle],
            ExpectedCause::InvalidSource,
        ),
        (
            [&missing.handle, &valid.handle],
            ExpectedCause::InvalidSource,
        ),
        (
            [&valid.handle, &sibling.handle],
            ExpectedCause::DifferentOwner,
        ),
    ] {
        let error = gpu_helpers::require_resident_compute_owners(&sources, true, || {
            crate::build_runtime_error("original domain diagnostic")
                .with_source(std::io::Error::other("original cause"))
                .build()
        })
        .unwrap_err();
        assert_eq!(error.message(), "original domain diagnostic");
        let source = error.source.as_deref().unwrap();
        match expected {
            ExpectedCause::Policy => {
                assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Requested);
                assert!(source.is::<gpu_helpers::ResidentComputePolicyDenied>());
                assert!(source.source().unwrap().is::<std::io::Error>());
            }
            ExpectedCause::InvalidSource => {
                assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
                assert!(source.is::<runmat_accelerate_api::ProviderExecutionError>());
            }
            ExpectedCause::DifferentOwner => {
                assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
                assert!(source.is::<std::io::Error>());
            }
        }
    }
}

#[test]
fn managed_borrowed_all_aggregate_slots_share_payloads_and_settle_exact_downloads() {
    let provider = provider();
    let allocation = provider
        .owner
        .allocate_zeroed(vec![1, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let gpu = || Value::GpuTensor(allocation.handle.clone());
    let record = || {
        let mut record = runmat_value::UnboundedStructValueBuilder::new();
        record.insert("payload", gpu());
        record.finish()
    };
    let object = || {
        let mut object = ObjectInstance::new("ManagedGather");
        object.replace_property_unbounded("payload".into(), gpu());
        object
    };
    let transformed = Closure::from_data(ClosureData {
        function_name: "gatherFixture".into(),
        binding: None,
        source_role: runmat_types::FunctionSourceRole::Unknown,
        source_file: None,
        captures: vec![gpu()].into(),
        lexical_environment: None,
    });
    let transformed_id = transformed.instance_id();
    let unchanged = Closure::from_data(ClosureData {
        function_name: "hostFixture".into(),
        binding: None,
        source_role: runmat_types::FunctionSourceRole::Unknown,
        source_file: None,
        captures: vec![Value::Num(9.0)].into(),
        lexical_environment: None,
    });
    let unchanged_id = unchanged.instance_id();
    let host = Tensor::new(vec![23.0], vec![1, 1]).unwrap();
    let host_ptr = host.as_f64_slice().unwrap().as_ptr();
    let mut chars = runmat_value::CharArray::new_row("original UTF16");
    let _chars_before_cow = chars.clone();
    chars.make_data_unique_unbounded().unwrap();
    assert!(chars.data_owner().is_shallow_clone());
    let chars_ptr = chars.data().as_ptr();
    let mut strings =
        runmat_value::StringArray::new(vec!["original UTF8".into()], vec![1, 1]).unwrap();
    let _strings_before_cow = strings.clone();
    strings.make_data_unique_unbounded().unwrap();
    assert!(strings.data_owner().is_shallow_clone());
    let strings_ptr = strings.data().as_ptr();
    let mut symbolic = runmat_value::SymbolicArray::new(
        vec![runmat_value::SymbolicExpr::Variable("x".into())],
        vec![1, 1],
    )
    .unwrap();
    let _symbolic_before_cow = symbolic.clone();
    symbolic.make_data_unique_unbounded().unwrap();
    assert!(symbolic.data_owner().is_shallow_clone());
    let symbolic_ptr = symbolic.data().as_ptr();
    let input = Value::Cell(
        CellArray::new(
            vec![
                Value::Tensor(host),
                gpu(),
                Value::Struct(record()),
                Value::StructArray(StructArray::new(vec![record(), record()], vec![1, 2]).unwrap()),
                Value::Object(object()),
                Value::ObjectArray(
                    ObjectArray::from_objects("ManagedGather", vec![object()], vec![1, 1]).unwrap(),
                ),
                Value::Closure(transformed),
                Value::Closure(unchanged),
                Value::CharArray(chars),
                Value::StringArray(strings),
                Value::SymbolicArray(symbolic),
            ],
            1,
            11,
        )
        .unwrap(),
    );
    let ledger = ManagedHostEffectLedger::new(8 * 1024 * 1024);
    let stop = CancellationSource::new();
    let runtime = runtime(&ledger, &stop);
    // No ambient scope: prove the explicit owner, including the console joint.
    let output = block_on(super::super::gather_if_needed_in_context_async(
        &input,
        Some(&runtime),
    ))
    .unwrap();
    assert_eq!(provider.ordinary.load(Ordering::SeqCst), 0);
    assert_eq!(provider.factories.load(Ordering::SeqCst), 7);
    assert!(provider.physical.lock().unwrap().is_none());
    let Value::Cell(output_cell) = &output else {
        panic!("cell output")
    };
    let Value::Tensor(host) = &output_cell.data()[0] else {
        panic!("retained host")
    };
    assert_eq!(host.as_f64_slice().unwrap().as_ptr(), host_ptr);
    assert!(output_cell.data_owner().is_exact());
    assert!(super::super::owned_gpu::has_transformable_gpu(&input));
    assert!(!super::super::owned_gpu::has_transformable_gpu(&output));
    let Value::Closure(transformed) = &output_cell.data()[6] else {
        panic!("transformed closure")
    };
    let Value::Closure(unchanged) = &output_cell.data()[7] else {
        panic!("unchanged closure")
    };
    assert_ne!(transformed.instance_id(), transformed_id);
    assert_eq!(unchanged.instance_id(), unchanged_id);
    assert!(
        matches!(&output_cell.data()[8], Value::CharArray(array) if array.data().as_ptr() == chars_ptr)
    );
    assert!(
        matches!(&output_cell.data()[9], Value::StringArray(array) if array.data().as_ptr() == strings_ptr)
    );
    assert!(
        matches!(&output_cell.data()[10], Value::SymbolicArray(array) if array.data().as_ptr() == symbolic_ptr)
    );
    assert!(ledger.snapshot().live_claimed_bytes > 0);
    assert_eq!(allocation.probe.retirements(), 0);
    drop(output);
    {
        let _scope = runtime.enter();
        let output = block_on(super::super::owned_gather::gather_if_needed_owned_async(
            input.clone(),
            None,
        ))
        .unwrap();
        assert!(!super::super::owned_gpu::has_transformable_gpu(&output));
        assert_eq!(provider.factories.load(Ordering::SeqCst), 14);
        drop(output);
    }
    drop(input);
    assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
    allocation.handle.release().unwrap();
    assert_eq!(allocation.probe.retirements(), 1);
}

#[test]
fn managed_deny_cancel_and_revocation_never_publish_or_retire_physical_callback_early() {
    for revoke in [false, true] {
        let provider = provider();
        let allocation = provider
            .owner
            .allocate_zeroed(vec![1, 1], NumericElementType::F64, GpuTensorStorage::Real)
            .unwrap();
        let input = Value::GpuTensor(allocation.handle.clone());
        let stop = CancellationSource::new();
        let denied = ManagedHostEffectLedger::new(0);
        assert_eq!(
            block_on(borrowed(&Value::Num(9.0), &runtime(&denied, &stop))).unwrap(),
            Value::Num(9.0)
        );
        assert!(block_on(borrowed(&input, &runtime(&denied, &stop))).is_err());
        assert_eq!(provider.factories.load(Ordering::SeqCst), 0);
        assert_eq!(denied.snapshot().live_reserved_bytes, 0);
        for reason in [CancellationReason::User, CancellationReason::Deadline] {
            let already = CancellationSource::new();
            already.cancel(reason);
            let ledger = ManagedHostEffectLedger::new(1024 * 1024);
            let error = block_on(super::super::gather_if_needed_in_context_async(
                &input,
                Some(&runtime(&ledger, &already)),
            ))
            .unwrap_err();
            assert!(matches!(
                error
                    .source
                    .as_deref()
                    .and_then(|source| (source as &dyn std::error::Error)
                        .downcast_ref::<crate::execution::ExecutionServiceError>()),
                Some(crate::execution::ExecutionServiceError::Cancelled)
            ));
            assert_eq!(provider.factories.load(Ordering::SeqCst), 0);
            assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
        }
        let ledger = ManagedHostEffectLedger::new(1024 * 1024);
        let runtime = runtime(&ledger, &stop);
        provider.pending.store(1, Ordering::SeqCst);
        let mut pending = Box::pin(borrowed(&input, &runtime));
        let mut cx = Context::from_waker(std::task::Waker::noop());
        assert!(pending.as_mut().poll(&mut cx).is_pending());
        assert_eq!(provider.factories.load(Ordering::SeqCst), 1);
        assert!(provider.physical.lock().unwrap().is_some());
        if revoke {
            ledger.revoke();
        } else {
            stop.cancel(CancellationReason::User);
        }
        assert!(matches!(
            pending.as_mut().poll(&mut cx),
            Poll::Ready(Err(_))
        ));
        // Drop the logical frame; the physical callback keeps its exact claim.
        drop(pending);
        assert!(ledger.snapshot().live_claimed_bytes > 0);
        assert_eq!(allocation.probe.retirements(), 0);
        drop(provider.physical.lock().unwrap().take());
        assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
        assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
        assert_eq!(provider.ordinary.load(Ordering::SeqCst), 0);
        drop(input);
        allocation.handle.release().unwrap();
    }
}
