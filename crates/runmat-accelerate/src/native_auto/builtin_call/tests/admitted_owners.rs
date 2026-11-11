//! Ownership conformance uses separate genuine provider allocation owners.
use super::*;
use runmat_accelerate_api::{
    allocation::WeakGpuTensorHandle, GpuTensorHandle, GpuTensorStorage, HostNumericTensorView,
    NumericDownloadStageLease, NumericElementType, PreparedNumericDownloadStage,
    ThreadProviderGuard,
};
use runmat_runtime::context::preparation::{
    AdmittedBuiltinPreparation, BuiltinPreparationAdmission,
};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Mutex,
};

struct ManagedCastOwner(runmat_runtime::context::host_effect::ManagedHostEffectLedger);

struct FixtureNumericStage;

impl NumericDownloadStageLease for FixtureNumericStage {
    fn clone_for_callback(&self) -> Box<dyn NumericDownloadStageLease> {
        Box::new(Self)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

async fn fixture_numeric_stage_future(
    owner: &Owner,
    source: &GpuTensorHandle,
    stage: Option<PreparedNumericDownloadStage>,
) -> anyhow::Result<runmat_accelerate_api::HostNumericTensorOwned> {
    let native = owner.inner.download_numeric(source).await;
    if let Ok(host) = &native {
        let payload = match &host.data {
            runmat_accelerate_api::HostNumericDataOwned::F32(values) => {
                Some((values.as_ptr() as usize, values.capacity()))
            }
            runmat_accelerate_api::HostNumericDataOwned::F64(values) => {
                Some((values.as_ptr() as usize, values.capacity()))
            }
            _ => None,
        };
        if let Some((payload_pointer, payload_capacity)) = payload {
            *owner.last_staged_native.lock().unwrap() = Some((
                payload_pointer,
                payload_capacity,
                host.shape.as_ptr() as usize,
                host.shape.capacity(),
            ));
        }
    }
    drop(stage);
    native
}

impl runmat_runtime::context::RuntimeParallelService for ManagedCastOwner {
    fn supports(&self, _: runmat_runtime::context::ParallelCapability) -> bool {
        false
    }

    fn placement_resources(&self) -> runmat_runtime::context::RuntimeParallelResources {
        let snapshot = self.0.snapshot();
        runmat_runtime::context::RuntimeParallelResources {
            cpu_millicores_available: 1_000,
            memory_available_bytes: Some(snapshot.free_bytes()),
            epoch: snapshot.epoch,
        }
    }

    fn managed_host_effect_ledger(
        &self,
    ) -> Option<runmat_runtime::context::host_effect::ManagedHostEffectLedger> {
        Some(self.0.clone())
    }
}

pub(super) struct Owner {
    inner: crate::simple_provider::InProcessProvider,
    uploads: AtomicUsize,
    downloads: AtomicUsize,
    staged_downloads: AtomicUsize,
    last_staged_native: Mutex<Option<(usize, usize, usize, usize)>>,
    allocations: Mutex<Vec<WeakGpuTensorHandle>>,
    reject_upload: bool,
    execution_ns: u64,
    unsupported_trig: bool,
    reject_cast_preparation: bool,
    cancel_download: Mutex<Option<runmat_execution::cancellation::CancellationSource>>,
}
impl Owner {
    fn publish(&self, handle: GpuTensorHandle) -> GpuTensorHandle {
        assert_eq!(handle.device_id(), self.inner.device_id());
        self.allocations.lock().unwrap().push(handle.downgrade());
        handle
    }
    fn live_count(&self) -> usize {
        self.allocations
            .lock()
            .unwrap()
            .iter()
            .filter(|weak| weak.acquire().is_ok())
            .count()
    }

    fn input(&self, data: &[f64]) -> GpuTensorHandle {
        let handle = self
            .inner
            .upload(&HostTensorView {
                data,
                shape: &[1, data.len()],
            })
            .unwrap();
        self.publish(handle)
    }
}
impl AccelProvider for Owner {
    fn device_id(&self) -> u32 {
        self.inner.device_id()
    }
    fn device_info(&self) -> String {
        "admitted-owner-fixture".into()
    }
    fn precision(&self) -> ProviderPrecision {
        ProviderPrecision::F64
    }
    fn capability_snapshot(&self) -> runmat_accelerate_api::ProviderCapabilitySnapshot {
        self.inner.capability_snapshot()
    }
    fn capability_snapshot_ref(
        &self,
    ) -> Option<&runmat_accelerate_api::ProviderCapabilitySnapshot> {
        self.inner.capability_snapshot_ref()
    }
    fn placement_resources(&self) -> runmat_execution::ProviderResourceSnapshot {
        self.inner.placement_resources()
    }
    fn upload(&self, host: &HostTensorView<'_>) -> Result<GpuTensorHandle> {
        self.uploads.fetch_add(1, Ordering::Relaxed);
        if self.reject_upload {
            return Err(anyhow!("test upload failure"));
        }
        Ok(self.publish(self.inner.upload(host)?))
    }
    fn upload_numeric_raw(
        &self,
        host: &HostNumericTensorView<'_>,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        self.uploads.fetch_add(1, Ordering::Relaxed);
        if self.reject_upload {
            return Err(runmat_accelerate_api::ProviderExecutionError::from(
                anyhow!("test upload failure"),
            ));
        }
        Ok(self.publish(self.inner.upload_numeric_raw(host)?))
    }
    fn upload_logical_raw(
        &self,
        host: &runmat_accelerate_api::HostLogicalTensorView<'_>,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        self.uploads.fetch_add(1, Ordering::Relaxed);
        Ok(self.publish(self.inner.upload_logical_raw(host)?))
    }
    fn download<'a>(
        &'a self,
        handle: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::AccelDownloadFuture<'a> {
        self.downloads.fetch_add(1, Ordering::Relaxed);
        self.inner.download(handle)
    }
    fn query_feasibility(
        &self,
        query: &runmat_accelerate_api::ProviderFeasibilityQuery,
    ) -> runmat_accelerate_api::ProviderFeasibility {
        if self.reject_cast_preparation
            && matches!(
                query.workload.preparation_route,
                Some(runmat_accelerate_api::ProviderPreparationRoute::FloatingCast { .. })
            )
        {
            return runmat_accelerate_api::ProviderFeasibility::rejected(
                runmat_accelerate_api::ProviderRejectionCode::UnsupportedOperation,
                "fixture cast preparation unavailable",
            );
        }
        // This fixture tests ownership before planning, not hook availability.
        known_fixture_resources(query)
    }
    fn estimate_cost(
        &self,
        _: &runmat_accelerate_api::ProviderCostQuery,
    ) -> Option<runmat_accelerate_api::ProviderCostEstimate> {
        Some(runmat_accelerate_api::ProviderCostEstimate::static_prior(
            ExecutionCostComponents {
                execution_ns: self.execution_ns,
                ..Default::default()
            },
            0,
        ))
    }
    fn download_numeric<'a>(
        &'a self,
        handle: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::AccelNumericDownloadFuture<'a> {
        self.downloads.fetch_add(1, Ordering::Relaxed);
        Box::pin(async move {
            let result = self.inner.download_numeric(handle).await;
            let cancellation = self.cancel_download.lock().unwrap().take();
            if let Some(cancellation) = cancellation {
                cancellation.cancel(runmat_execution::CancellationReason::User);
            }
            result
        })
    }
    fn download_numeric_completion_host_bound(
        &self,
        handle: &GpuTensorHandle,
    ) -> runmat_execution::ResourceByteBound {
        let native = self.inner.download_numeric(handle);
        let native_bytes = std::mem::size_of_val(native.as_ref().get_ref());
        let stage = fixture_numeric_stage_future(self, handle, None);
        native_bytes
            .checked_add(std::mem::size_of_val(&stage))
            .and_then(|bytes| u64::try_from(bytes).ok())
            .map(runmat_execution::ResourceByteBound::Known)
            .unwrap_or(runmat_execution::ResourceByteBound::Unavailable(
                runmat_execution::ResourceBoundUnavailable::NoScratchBound,
            ))
    }
    fn prepare_numeric_download_stage(
        &self,
        source: &GpuTensorHandle,
        query: &runmat_accelerate_api::ProviderFeasibilityQuery,
        claim: std::sync::Arc<dyn runmat_execution::resource::RetainedHostAllocationClaim>,
    ) -> anyhow::Result<PreparedNumericDownloadStage> {
        anyhow::ensure!(query.matches_numeric_download(source));
        anyhow::ensure!(source.allocation_lease().validate_live().is_ok());
        Ok(PreparedNumericDownloadStage::issued(
            Box::new(FixtureNumericStage),
            claim,
            source,
            0,
            0,
        ))
    }
    fn download_numeric_admitted_stage<'a>(
        &'a self,
        source: &'a GpuTensorHandle,
        stage: PreparedNumericDownloadStage,
    ) -> runmat_accelerate_api::AccelNumericDownloadFuture<'a> {
        self.staged_downloads.fetch_add(1, Ordering::Relaxed);
        Box::pin(fixture_numeric_stage_future(self, source, Some(stage)))
    }
    fn query_numeric_download_feasibility(
        &self,
        handle: &GpuTensorHandle,
        query: &runmat_accelerate_api::ProviderFeasibilityQuery,
    ) -> runmat_accelerate_api::ProviderFeasibility {
        self.inner.query_numeric_download_feasibility(handle, query)
    }
    fn download_numeric_host_workspace_bound(
        &self,
        handle: &GpuTensorHandle,
    ) -> runmat_execution::ResourceByteBound {
        self.inner.download_numeric_host_workspace_bound(handle)
    }
    fn download_numeric_provider_scratch_bound(
        &self,
        handle: &GpuTensorHandle,
    ) -> runmat_execution::ResourceByteBound {
        self.inner.download_numeric_provider_scratch_bound(handle)
    }
    fn download_numeric_queue_slots(&self, handle: &GpuTensorHandle) -> u32 {
        self.inner.download_numeric_queue_slots(handle)
    }
    fn cast_to_integer<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
        target: runmat_accelerate_api::IntegerElementType,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move { Ok(self.publish(self.inner.cast_to_integer(input, target).await?)) })
    }
    fn unary_sin<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::AccelProviderFuture<'a, GpuTensorHandle> {
        if self.unsupported_trig {
            return Box::pin(async {
                Err(runmat_accelerate_api::unsupported_provider_operation(
                    "fixture sine refusal",
                ))
            });
        }
        Box::pin(async move { Ok(self.publish(self.inner.unary_sin(input).await?)) })
    }
    fn unary_cos<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::AccelProviderFuture<'a, GpuTensorHandle> {
        if self.unsupported_trig {
            return Box::pin(async {
                Err(runmat_accelerate_api::unsupported_provider_operation(
                    "fixture cosine refusal",
                ))
            });
        }
        Box::pin(async move { Ok(self.publish(self.inner.unary_cos(input).await?)) })
    }
    fn elem_mul<'a>(
        &'a self,
        lhs: &'a GpuTensorHandle,
        rhs: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move { Ok(self.publish(self.inner.elem_mul(lhs, rhs).await?)) })
    }
}
pub(super) fn owner() -> &'static Owner {
    owner_with_upload_failure(false)
}
pub(super) fn owner_with_upload_failure(reject_upload: bool) -> &'static Owner {
    owner_with_quote_and_failure(1, reject_upload, false)
}
fn owner_with_quote_and_failure(
    execution_ns: u64,
    reject_upload: bool,
    unsupported_trig: bool,
) -> &'static Owner {
    owner_with_cast_preparation(execution_ns, reject_upload, unsupported_trig, false)
}

fn owner_with_cast_preparation(
    execution_ns: u64,
    reject_upload: bool,
    unsupported_trig: bool,
    reject_cast_preparation: bool,
) -> &'static Owner {
    let provider = Box::leak(Box::new(Owner {
        inner: crate::simple_provider::InProcessProvider::new(),
        uploads: AtomicUsize::new(0),
        downloads: AtomicUsize::new(0),
        staged_downloads: AtomicUsize::new(0),
        last_staged_native: Mutex::new(None),
        allocations: Mutex::new(Vec::new()),
        reject_upload,
        execution_ns,
        unsupported_trig,
        reject_cast_preparation,
        cancel_download: Mutex::new(None),
    }));
    // SAFETY: leaked owners have process lifetime and unique monotonically allocated IDs.
    unsafe {
        runmat_accelerate_api::register_device_provider(provider);
    }
    provider
}
fn request(values: Vec<Value>) -> AdmittedBuiltinPreparation {
    AdmittedBuiltinPreparation::new(
        BuiltinPreparationAdmission {
            binding: runmat_builtins::BuiltinBindingIdentity {
                builtin: runmat_builtins::POL2CART_CATALOG_ENTRY.identity,
                variant: "default",
            },
            form: runmat_builtins::BuiltinPreparationFormId(1),
            signature: 1,
            input_count: 3,
            requested_outputs: Some(3),
        },
        values,
    )
    .unwrap()
}

#[test]
fn automatic_resident_calls_keep_private_native_mirrors_and_publish_real_host_full_calls() {
    use runmat_accelerate_api::{GpuHandleProvenance, HostNumericDataView};
    for single in [false, true] {
        for operation in ["sin", "exp", "expm1", "times"] {
            // Deterministic fixture estimate selects the actual host route;
            // it is not measured profitability or production calibration.
            let provider = owner_with_quote_and_failure(1_000_000_000, false, false);
            let data = [0.0, 0.5, -0.5, 1.25];
            let native = data.map(|x| x as f32);
            let source = provider
                .upload_numeric_raw(&HostNumericTensorView {
                    data: if single {
                        HostNumericDataView::F32(&native)
                    } else {
                        HostNumericDataView::F64(&data)
                    },
                    shape: &[1, 4],
                    storage: GpuTensorStorage::Real,
                })
                .unwrap()
                .with_provenance(GpuHandleProvenance::Automatic)
                .unwrap();
            let session = Rc::new(crate::placement::PlacementSession::default());
            let context = RuntimeContext::new(Rc::new(
                runmat_runtime::execution::RuntimeExecutionService::new(),
            ))
            .with_service_ports(
                RuntimeServicePorts::default()
                    .with_placement(session.clone())
                    .with_acceleration(Rc::new(Adapter(provider, 0))),
            );
            let args = if operation == "times" {
                vec![
                    Value::GpuTensor(source.clone()),
                    Value::GpuTensor(source.clone()),
                ]
            } else {
                vec![Value::GpuTensor(source.clone())]
            };
            let pending = runmat_runtime::dispatcher::call_builtin_async_in_context(
                &context, operation, &args,
            );
            assert!(session.ordinary_feedback_for_test().is_empty());
            let output = futures::executor::block_on(context.scope(pending)).unwrap();
            let Value::Tensor(result) = output.value() else {
                panic!("private host numeric output");
            };
            assert_eq!(
                result.numeric_dtype(),
                if single {
                    runmat_value::NumericDType::F32
                } else {
                    runmat_value::NumericDType::F64
                }
            );
            for (actual, input) in result.materialize_f64().into_iter().zip(data) {
                let expected = if single {
                    let input = input as f32;
                    (match operation {
                        "sin" => input.sin(),
                        "exp" => input.exp(),
                        "expm1" => input.exp_m1(),
                        _ => input * input,
                    }) as f64
                } else {
                    match operation {
                        "sin" => input.sin(),
                        "exp" => input.exp(),
                        "expm1" => input.exp_m1(),
                        _ => input * input,
                    }
                };
                assert_eq!(actual.to_bits(), expected.to_bits());
            }
            assert_eq!(
                provider.downloads.load(Ordering::Relaxed),
                1,
                "one invocation-private mirror, including repeated data inputs"
            );
            assert_eq!(
                provider.uploads.load(Ordering::Relaxed),
                1,
                "no re-promotion of the selected host operand"
            );
            assert!(source.allocation_lease().validate_live().is_ok());
            assert!(args.iter().all(|value| matches!(value, Value::GpuTensor(handle) if handle.shares_allocation_with(&source))));
            let profile = session.ordinary_feedback_for_test();
            assert_eq!(profile.len(), 1);
            assert!(session.profile_snapshot().feedback.is_empty());
            assert_eq!(
                profile[0].kind,
                runmat_execution::ExecutionCandidateKind::SharedRuntime
            );
            assert_eq!(profile[0].samples, 1);
            assert_eq!(profile[0].failures, 0);
            assert!(matches!(
                profile[0].signature.revision.authority,
                runmat_execution::PlacementExecutionAuthorityRevision::RuntimeBindingSession(_)
            ));
            drop((args, source, output));
            assert_eq!(provider.live_count(), 0);
        }
    }
}

#[test]
fn promoting_resident_calls_preserve_native_domains_errors_and_full_call_boundary() {
    use runmat_accelerate_api::{GpuHandleProvenance, HostNumericDataView};
    for single in [false, true] {
        // One case per canonical boundary/engine; shared leaf adapters are not
        // separate planners. Both sides of each actual domain are exercised.
        for (operation, inside, outside) in [
            ("log", 0.5, -0.5),
            ("log1p", -0.5, -1.5),
            ("sqrt", 0.5, -0.5),
            ("realsqrt", 0.5, -0.5),
            ("asin", 0.5, 1.5),
            ("acosh", 1.5, 0.5),
            ("atanh", 0.5, 1.5),
        ] {
            for value in [inside, outside] {
                let provider = owner_with_quote_and_failure(1_000_000_000, false, false);
                let data = [value, value, f64::NAN, value];
                let native = data.map(|value| value as f32);
                let input = if single {
                    Value::Tensor(
                        runmat_value::Tensor::from_f32(native.to_vec(), vec![1, 4]).unwrap(),
                    )
                } else {
                    Value::Tensor(runmat_value::Tensor::new(data.to_vec(), vec![1, 4]).unwrap())
                };
                let source = provider
                    .upload_numeric_raw(&HostNumericTensorView {
                        data: if single {
                            HostNumericDataView::F32(&native)
                        } else {
                            HostNumericDataView::F64(&data)
                        },
                        shape: &[1, 4],
                        storage: GpuTensorStorage::Real,
                    })
                    .unwrap()
                    .with_provenance(GpuHandleProvenance::Automatic)
                    .unwrap();
                let session = Rc::new(crate::placement::PlacementSession::default());
                let ordinary = RuntimeContext::new(Rc::new(
                    runmat_runtime::execution::RuntimeExecutionService::new(),
                ));
                let context = ordinary.clone().with_service_ports(
                    RuntimeServicePorts::default()
                        .with_placement(session.clone())
                        .with_acceleration(Rc::new(Adapter(provider, 0))),
                );
                let expected = futures::executor::block_on(ordinary.scope(
                    runmat_runtime::dispatcher::call_builtin_async_in_context(
                        &ordinary,
                        operation,
                        &[input],
                    ),
                ));
                let args = [Value::GpuTensor(source.clone())];
                let actual = futures::executor::block_on(context.scope(
                    runmat_runtime::dispatcher::call_builtin_async_in_context(
                        &context, operation, &args,
                    ),
                ));
                match (&actual, &expected) {
                    (Ok(actual), Ok(expected)) => assert_native_result(actual.value(), expected.value()),
                    (Err(actual), Err(expected)) => {
                        assert_eq!(actual.identifier(), expected.identifier());
                        assert_eq!(actual.gpu_gather_retry, runmat_runtime::GpuGatherRetry::Never);
                    }
                    _ => panic!("{operation} native/private domain result disagrees: {actual:?} vs {expected:?}"),
                }
                assert_eq!(provider.downloads.load(Ordering::Relaxed), 1, "{operation}");
                assert_eq!(
                    provider.uploads.load(Ordering::Relaxed),
                    1,
                    "no restore/re-promotion"
                );
                assert!(source.allocation_lease().validate_live().is_ok());
                let profile = session.ordinary_feedback_for_test();
                assert_eq!(profile.len(), 1, "{operation}");
                assert!(session.profile_snapshot().feedback.is_empty());
                assert_eq!(
                    profile[0].kind,
                    runmat_execution::ExecutionCandidateKind::SharedRuntime
                );
                assert_eq!(profile[0].failures, u64::from(actual.is_err()));
                drop((actual, expected, args, source));
                assert_eq!(provider.live_count(), 0);
            }
        }
    }
}

fn assert_native_result(actual: &Value, expected: &Value) {
    match (actual, expected) {
        (Value::Tensor(actual), Value::Tensor(expected)) => {
            assert_eq!(actual.numeric_dtype(), expected.numeric_dtype());
            assert_eq!(actual.shape(), expected.shape());
            assert_eq!(
                actual
                    .materialize_f64()
                    .into_iter()
                    .map(f64::to_bits)
                    .collect::<Vec<_>>(),
                expected
                    .materialize_f64()
                    .into_iter()
                    .map(f64::to_bits)
                    .collect::<Vec<_>>()
            );
        }
        (Value::ComplexTensor(actual), Value::ComplexTensor(expected)) => {
            assert_eq!(actual.numeric_dtype(), expected.numeric_dtype());
            assert_eq!(actual.shape(), expected.shape());
            let bits = |values: Vec<(f64, f64)>| {
                values
                    .into_iter()
                    .map(|(real, imag)| (real.to_bits(), imag.to_bits()))
                    .collect::<Vec<_>>()
            };
            assert_eq!(
                bits(actual.materialize_f64()),
                bits(expected.materialize_f64())
            );
        }
        _ => panic!("native carrier/class changed: {actual:?} vs {expected:?}"),
    }
}

#[test]
fn mixed_power_private_host_route_keeps_double_arithmetic_and_single_publication() {
    use runmat_accelerate_api::{GpuHandleProvenance, HostNumericDataView};
    for base in [4.0_f32, -4.0] {
        let provider = owner_with_quote_and_failure(1_000_000_000, false, false);
        let data = [base, base];
        let source = provider
            .upload_numeric_raw(&HostNumericTensorView {
                data: HostNumericDataView::F32(&data),
                shape: &[1, 2],
                storage: GpuTensorStorage::Real,
            })
            .unwrap()
            .with_provenance(GpuHandleProvenance::Automatic)
            .unwrap();
        let exponent =
            Value::Tensor(runmat_value::Tensor::new(vec![0.5, 2.0], vec![1, 2]).unwrap());
        let ordinary = RuntimeContext::new(Rc::new(
            runmat_runtime::execution::RuntimeExecutionService::new(),
        ));
        let session = Rc::new(crate::placement::PlacementSession::default());
        let context = ordinary.clone().with_service_ports(
            RuntimeServicePorts::default()
                .with_placement(session.clone())
                .with_acceleration(Rc::new(Adapter(provider, 0))),
        );
        let expected = futures::executor::block_on(ordinary.scope(
            runmat_runtime::dispatcher::call_builtin_async_in_context(
                &ordinary,
                "power",
                &[
                    Value::Tensor(
                        runmat_value::Tensor::from_f32(data.to_vec(), vec![1, 2]).unwrap(),
                    ),
                    exponent.clone(),
                ],
            ),
        ))
        .unwrap();
        let args = [Value::GpuTensor(source.clone()), exponent];
        let actual = futures::executor::block_on(context.scope(
            runmat_runtime::dispatcher::call_builtin_async_in_context(&context, "power", &args),
        ))
        .unwrap();
        assert_native_result(actual.value(), expected.value());
        assert_eq!(provider.downloads.load(Ordering::Relaxed), 1);
        assert_eq!(provider.uploads.load(Ordering::Relaxed), 1);
        let profile = session.ordinary_feedback_for_test();
        assert_eq!(profile.len(), 1);
        assert!(session.profile_snapshot().feedback.is_empty());
        assert_eq!(
            profile[0].kind,
            runmat_execution::ExecutionCandidateKind::SharedRuntime
        );
        assert_eq!(profile[0].failures, 0);
        drop((actual, expected, args, source));
        assert_eq!(provider.live_count(), 0);
    }
}

#[test]
fn cancellation_during_private_call_readback_never_runs_or_publishes_the_body() {
    let provider = owner_with_quote_and_failure(1_000_000_000, false, false);
    let source = provider
        .input(&[0.25, 0.5])
        .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Automatic)
        .unwrap();
    let cancellation = runmat_execution::cancellation::CancellationSource::new();
    *provider.cancel_download.lock().unwrap() = Some(cancellation.clone());
    let session = Rc::new(crate::placement::PlacementSession::default());
    let context = RuntimeContext::new(Rc::new(
        runmat_runtime::execution::RuntimeExecutionService::new(),
    ))
    .with_cancellation_token(cancellation.token())
    .with_service_ports(
        RuntimeServicePorts::default()
            .with_placement(session.clone())
            .with_acceleration(Rc::new(Adapter(provider, 0))),
    );
    let args = [Value::GpuTensor(source.clone())];
    let error = futures::executor::block_on(context.scope(
        runmat_runtime::dispatcher::call_builtin_async_in_context(&context, "sin", &args),
    ))
    .unwrap_err();
    assert_eq!(
        error.gpu_gather_retry,
        runmat_runtime::GpuGatherRetry::Never
    );
    assert_eq!(provider.downloads.load(Ordering::Relaxed), 1);
    assert!(session.ordinary_feedback_for_test().is_empty());
    assert!(source.allocation_lease().validate_live().is_ok());
    drop((args, source));
    assert_eq!(provider.live_count(), 0);
}

#[test]
fn explicit_resident_call_keeps_provider_route_despite_a_cheaper_host_estimate() {
    let provider = owner_with_quote_and_failure(1_000_000_000, false, false);
    let source = provider
        .input(&[0.25, 0.5])
        .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
        .unwrap();
    let session = Rc::new(crate::placement::PlacementSession::default());
    let context = RuntimeContext::new(Rc::new(
        runmat_runtime::execution::RuntimeExecutionService::new(),
    ))
    .with_service_ports(
        RuntimeServicePorts::default()
            .with_placement(session.clone())
            .with_acceleration(Rc::new(Adapter(provider, 0))),
    );
    let args = [Value::GpuTensor(source.clone())];
    let output = futures::executor::block_on(context.scope(
        runmat_runtime::dispatcher::call_builtin_async_in_context(&context, "sin", &args),
    ))
    .unwrap();
    let Value::GpuTensor(result) = output.value() else {
        panic!("explicit GPU intent must retain resident output");
    };
    assert_eq!(provider.downloads.load(Ordering::Relaxed), 0);
    assert_eq!(provider.uploads.load(Ordering::Relaxed), 0);
    assert!(session
        .ordinary_feedback_for_test()
        .iter()
        .all(|sample| sample.kind.is_provider()));
    let actual = futures::executor::block_on(provider.inner.download_numeric(result)).unwrap();
    assert_eq!(
        actual.data,
        runmat_accelerate_api::HostNumericDataOwned::F64(vec![0.25f64.sin(), 0.5f64.sin()])
    );
    assert!(source.allocation_lease().validate_live().is_ok());
    drop((output, args, source));
    assert_eq!(provider.live_count(), 0);
}

#[test]
fn explicit_plain_cast_feasibility_refusal_retains_the_domain_fallback() {
    let provider = owner_with_cast_preparation(1, false, false, true);
    let source = provider
        .input(&[0.25, 0.5])
        .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
        .unwrap();
    let session = Rc::new(crate::placement::PlacementSession::default());
    let context = RuntimeContext::new(Rc::new(
        runmat_runtime::execution::RuntimeExecutionService::new(),
    ))
    .with_service_ports(
        RuntimeServicePorts::default()
            .with_placement(session.clone())
            .with_acceleration(Rc::new(Adapter(provider, 1))),
    );
    let request = AdmittedBuiltinPreparation::new(
        BuiltinPreparationAdmission {
            binding: runmat_builtins::BuiltinBindingIdentity {
                builtin: runmat_builtins::SINGLE_CATALOG_ENTRY.identity,
                variant: "default",
            },
            form: runmat_builtins::BuiltinPreparationFormId(1),
            signature: 0,
            input_count: 1,
            requested_outputs: None,
        },
        vec![Value::GpuTensor(source.clone())],
    )
    .unwrap();
    let auto = NativeAutoOffload::new(provider, ThresholdConfig::default());
    let prepared = futures::executor::block_on(super::super::admitted::prepare_async(
        &auto, &context, request,
    ))
    .expect("typed feasibility refusal leaves the original cast body available");
    assert!(prepared
        .observation
        .as_ref()
        .is_some_and(BuiltinCallObservation::shared_fallback_selected));
    assert!(
        matches!(prepared.args(), [Value::GpuTensor(handle)] if handle.view_identity() == source.view_identity())
    );
    assert!(session.ordinary_feedback_for_test().is_empty());
    drop(prepared);

    let args = [Value::GpuTensor(source.clone())];
    let output = futures::executor::block_on(context.scope(
        runmat_runtime::dispatcher::call_builtin_async_in_context(&context, "single", &args),
    ))
    .expect("single owns the unsupported provider cast fallback");
    let Value::GpuTensor(result) = output.value() else {
        panic!("explicit source must retain resident conversion output")
    };
    assert!(runmat_accelerate_api::handle_is_explicit(result));
    assert_eq!(result.shape(), source.shape());
    assert_eq!(provider.downloads.load(Ordering::Relaxed), 1);
    assert_eq!(provider.uploads.load(Ordering::Relaxed), 1);
    assert!(session.ordinary_feedback_for_test().is_empty());
}

#[test]
fn managed_plain_cast_retains_exact_output_and_refuses_before_underfunded_readback() {
    use runmat_runtime::context::host_effect::ManagedHostEffectLedger;

    for (name, input, expected_dtype, expected_shape, expected_values) in [
        (
            "single",
            Value::Tensor(runmat_value::Tensor::new(vec![0.25, 0.5], vec![1, 2]).unwrap()),
            Some(runmat_value::NumericDType::F32),
            vec![1, 2],
            vec![0.25, 0.5],
        ),
        (
            "double",
            Value::Tensor(runmat_value::Tensor::from_f32(vec![0.25, 0.5], vec![1, 2]).unwrap()),
            Some(runmat_value::NumericDType::F64),
            vec![1, 2],
            vec![0.25, 0.5],
        ),
        (
            "single",
            Value::Num(0.25),
            Some(runmat_value::NumericDType::F32),
            vec![1, 1],
            vec![0.25],
        ),
        ("double", Value::Num(0.25), None, vec![1, 1], vec![0.25]),
    ] {
        // Make the provider cast authentically unavailable. A high cost alone
        // does not select host execution: the legal resident implementation
        // may still win and correctly return a GpuTensor, which does not test
        // this witness's exact host-output owner.
        let provider = owner_with_cast_preparation(1_000_000_000, false, false, true);
        let ledger = ManagedHostEffectLedger::new_with_epoch(1 << 20, 31);
        let context = RuntimeContext::new(Rc::new(
            runmat_runtime::execution::RuntimeExecutionService::new(),
        ))
        .with_service_ports(
            RuntimeServicePorts::default()
                .with_parallel(Rc::new(ManagedCastOwner(ledger.clone())))
                .with_placement(Rc::new(crate::placement::PlacementSession::default()))
                .with_acceleration(Rc::new(Adapter(provider, 1))),
        );
        let output = futures::executor::block_on(context.scope(
            runmat_runtime::dispatcher::call_builtin_async_in_context(&context, name, &[input]),
        ))
        .unwrap_or_else(|error| {
            panic!(
                "managed {name} host cast expected {expected_dtype:?} {expected_shape:?}: {error:?}"
            )
        });
        if let Some(expected_dtype) = expected_dtype {
            let Value::Tensor(tensor) = output.value() else {
                panic!(
                    "managed {name} cast expected {expected_dtype:?} Tensor {expected_shape:?}, got {:?}; live claims={}, staged downloads={}",
                    output.value(),
                    ledger.snapshot().live_claimed_bytes,
                    provider.staged_downloads.load(Ordering::Relaxed),
                )
            };
            assert_eq!(tensor.numeric_dtype(), expected_dtype);
            assert_eq!(tensor.shape(), expected_shape);
            assert_eq!(tensor.materialize_f64(), expected_values);
        } else {
            assert_eq!(expected_shape, [1, 1]);
            assert_eq!(expected_values, [0.25]);
            assert!(
                matches!(output.value(), Value::Num(value) if value.to_bits() == 0.25_f64.to_bits()),
                "managed {name} scalar identity returned {:?}",
                output.value(),
            );
        }
        let retained = ledger.snapshot().live_claimed_bytes;
        if expected_dtype.is_some() {
            assert!(retained > 0, "published output must retain physical claims");
        } else {
            assert_eq!(retained, 0, "scalar identity retains no output allocation");
        }
        drop(output);
        assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
    }

    let provider = owner_with_cast_preparation(1_000_000_000, false, false, true);
    let source = provider
        .input(&[0.25, 0.5])
        .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Automatic)
        .unwrap();
    let ledger = ManagedHostEffectLedger::new_with_epoch(1 << 20, 32);
    let context = RuntimeContext::new(Rc::new(
        runmat_runtime::execution::RuntimeExecutionService::new(),
    ))
    .with_service_ports(
        RuntimeServicePorts::default()
            .with_parallel(Rc::new(ManagedCastOwner(ledger.clone())))
            .with_placement(Rc::new(crate::placement::PlacementSession::default()))
            .with_acceleration(Rc::new(Adapter(provider, 1))),
    );
    let output = futures::executor::block_on(context.scope(
        runmat_runtime::dispatcher::call_builtin_async_in_context(
            &context,
            "single",
            &[Value::GpuTensor(source)],
        ),
    ))
    .expect("managed automatic resident cast uses admitted staged readback");
    let Value::Tensor(tensor) = output.value() else {
        panic!("managed automatic resident cast publishes a host tensor")
    };
    assert_eq!(tensor.numeric_dtype(), runmat_value::NumericDType::F32);
    assert_eq!(tensor.shape(), [1, 2]);
    assert_eq!(tensor.materialize_f64(), [0.25, 0.5]);
    assert_eq!(provider.staged_downloads.load(Ordering::Relaxed), 1);
    assert_eq!(provider.downloads.load(Ordering::Relaxed), 0);
    assert!(ledger.snapshot().live_claimed_bytes > 0);
    drop(output);
    assert_eq!(ledger.snapshot().live_claimed_bytes, 0);

    let provider = owner_with_cast_preparation(1_000_000_000, false, false, true);
    let source = provider
        .input(&[0.25, 0.5])
        .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Automatic)
        .unwrap();
    let ledger = ManagedHostEffectLedger::new_with_epoch(0, 32);
    let context = RuntimeContext::new(Rc::new(
        runmat_runtime::execution::RuntimeExecutionService::new(),
    ))
    .with_service_ports(
        RuntimeServicePorts::default()
            .with_parallel(Rc::new(ManagedCastOwner(ledger)))
            .with_placement(Rc::new(crate::placement::PlacementSession::default()))
            .with_acceleration(Rc::new(Adapter(provider, 1))),
    );
    let result = futures::executor::block_on(context.scope(
        runmat_runtime::dispatcher::call_builtin_async_in_context(
            &context,
            "single",
            &[Value::GpuTensor(source)],
        ),
    ));
    assert!(
        result.is_err(),
        "unfunded cast cannot reach provider readback"
    );
    assert_eq!(provider.downloads.load(Ordering::Relaxed), 0);
    assert_eq!(provider.staged_downloads.load(Ordering::Relaxed), 0);
}

#[test]
fn managed_selected_readback_moves_native_vecs_into_retained_alias() {
    use crate::fusion_exec::host_readback::{CpuReadbackPlan, Source};
    use runmat_runtime::context::host_effect::ManagedHostEffectLedger;
    use std::borrow::Cow;

    let provider = owner_with_quote_and_failure(1_000_000_000, false, false);
    let source = provider
        .input(&[0.25, 0.5])
        .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Automatic)
        .unwrap();
    let ledger = ManagedHostEffectLedger::new_with_epoch(1 << 20, 35);
    let context = RuntimeContext::new(Rc::new(
        runmat_runtime::execution::RuntimeExecutionService::new(),
    ))
    .with_service_ports(
        RuntimeServicePorts::default().with_parallel(Rc::new(ManagedCastOwner(ledger.clone()))),
    );
    let stack = [Value::GpuTensor(source.clone())];
    let host_workspace = provider.download_numeric_host_workspace_bound(&source);
    let runmat_execution::ResourceByteBound::Known(host_workspace) = host_workspace else {
        panic!("fixture native download needs a finite workspace bound")
    };
    let plan = CpuReadbackPlan {
        sources: vec![Source {
            handle: source.clone(),
            owner: provider,
            offsets: vec![0],
            loads: Vec::new(),
            capabilities: Cow::Borrowed(provider.capability_snapshot_ref().unwrap()),
            payload_bytes: runmat_accelerate_api::checked_numeric_download_host_payload_bytes(
                &source,
            )
            .unwrap(),
            host_workspace,
            provider_scratch: provider.download_numeric_provider_scratch_bound(&source),
            queue_slots: provider.download_numeric_queue_slots(&source),
        }],
        stack_len: stack.len(),
        resident_offsets: vec![0],
    };
    let prepared =
        futures::executor::block_on(context.scope(plan.prepare_inputs_managed(&stack, &context)))
            .expect("selected native readback");
    let alias = prepared.entry[0].1.clone();
    let (payload_pointer, payload_capacity, shape_pointer, shape_capacity) = provider
        .last_staged_native
        .lock()
        .unwrap()
        .expect("fixture recorded native Vec ownership");
    let Value::Tensor(tensor) = &alias else {
        panic!("native real readback retains a Tensor")
    };
    assert_eq!(
        tensor.as_f64_slice().unwrap().as_ptr() as usize,
        payload_pointer
    );
    assert_eq!(tensor.shape().as_ptr() as usize, shape_pointer);
    assert_eq!(tensor.as_f64_slice().unwrap().len(), payload_capacity);
    assert_eq!(tensor.shape().len(), shape_capacity);
    prepared.cleanup().expect("release selected source");
    assert!(ledger.snapshot().live_claimed_bytes > 0);
    drop(alias);
    assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
    drop((stack, source));
    assert_eq!(provider.live_count(), 0);
}

#[test]
fn managed_selected_readback_adopts_logical_with_retained_owner() {
    use crate::fusion_exec::host_readback::{CpuReadbackPlan, Source};
    use runmat_runtime::context::host_effect::ManagedHostEffectLedger;

    let provider = owner_with_quote_and_failure(1_000_000_000, false, false);
    let source = provider
        .upload_logical_raw(&runmat_accelerate_api::HostLogicalTensorView {
            data: &[0, 1, 1],
            shape: &[1, 3],
        })
        .expect("fixture publishes a certified logical source")
        .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Automatic)
        .expect("fixture marks the owned source automatic");
    let ledger = ManagedHostEffectLedger::new_with_epoch(1 << 20, 36);
    let context = RuntimeContext::new(Rc::new(
        runmat_runtime::execution::RuntimeExecutionService::new(),
    ))
    .with_service_ports(
        RuntimeServicePorts::default().with_parallel(Rc::new(ManagedCastOwner(ledger.clone()))),
    );
    let stack = [Value::GpuTensor(source.clone())];
    let plan = CpuReadbackPlan {
        sources: vec![Source::admit(&source, provider, vec![0], Vec::new())
            .expect("validated automatic logical source has a finite readback quote")],
        stack_len: stack.len(),
        resident_offsets: vec![0],
    };
    let prepared =
        futures::executor::block_on(context.scope(plan.prepare_inputs_managed(&stack, &context)))
            .expect("managed logical readback completes");
    let alias = prepared.entry[0].1.clone();
    let Value::LogicalArray(logical) = &alias else {
        panic!("logical readback retains a logical host allocation")
    };
    assert_eq!(logical.shape, [1, 3]);
    assert_eq!(logical.data.as_slice(), [0, 1, 1]);
    assert_eq!(provider.staged_downloads.load(Ordering::Relaxed), 1);
    prepared.cleanup().expect("release selected source");
    assert!(ledger.snapshot().live_claimed_bytes > 0);
    drop(alias);
    assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
}

#[test]
fn managed_named_numerical_hosts_retain_outputs_and_stage_each_native_source() {
    use runmat_accelerate_api::{GpuHandleProvenance, HostNumericDataView};
    use runmat_runtime::context::host_effect::ManagedHostEffectLedger;

    // One source, a promoting complex result, a genuinely two-source
    // broadcast, and mixed-class power exercise the common selected owner.
    // The expected values are produced by the ordinary domain owners.
    for (name, operands, complex) in [
        ("sin", vec![(vec![0.0, 0.5], vec![1, 2], true)], false),
        ("log", vec![(vec![-0.5, 0.5], vec![1, 2], false)], true),
        (
            "plus",
            vec![
                (vec![1.0, 2.0], vec![1, 2], true),
                (vec![3.5], vec![1, 1], true),
            ],
            false,
        ),
        (
            "power",
            vec![
                (vec![-4.0, 9.0], vec![1, 2], true),
                (vec![0.5], vec![1, 1], false),
            ],
            true,
        ),
    ] {
        let provider = owner_with_quote_and_failure(1_000_000_000, false, false);
        let ledger = ManagedHostEffectLedger::new_with_epoch(1 << 22, 33);
        let session = Rc::new(crate::placement::PlacementSession::default());
        let ordinary = RuntimeContext::new(Rc::new(
            runmat_runtime::execution::RuntimeExecutionService::new(),
        ));
        let context = ordinary.clone().with_service_ports(
            RuntimeServicePorts::default()
                .with_parallel(Rc::new(ManagedCastOwner(ledger.clone())))
                .with_placement(session.clone())
                .with_acceleration(Rc::new(Adapter(provider, 0))),
        );
        let mut host_args = Vec::new();
        let mut resident_args = Vec::new();
        for (data, shape, single) in operands {
            let native = data.iter().map(|&value| value as f32).collect::<Vec<_>>();
            let host = if single {
                runmat_value::Tensor::from_f32(native.clone(), shape.clone()).unwrap()
            } else {
                runmat_value::Tensor::new(data.clone(), shape.clone()).unwrap()
            };
            host_args.push(Value::Tensor(host));
            let source = provider
                .upload_numeric_raw(&HostNumericTensorView {
                    data: if single {
                        HostNumericDataView::F32(&native)
                    } else {
                        HostNumericDataView::F64(&data)
                    },
                    shape: &shape,
                    storage: GpuTensorStorage::Real,
                })
                .unwrap()
                .with_provenance(GpuHandleProvenance::Automatic)
                .unwrap();
            resident_args.push(Value::GpuTensor(source));
        }
        let expected = futures::executor::block_on(ordinary.scope(
            runmat_runtime::dispatcher::call_builtin_async_in_context(&ordinary, name, &host_args),
        ))
        .unwrap();
        let actual = futures::executor::block_on(context.scope(
            runmat_runtime::dispatcher::call_builtin_async_in_context(
                &context,
                name,
                &resident_args,
            ),
        ))
        .expect("selected managed numerical host body");
        assert_native_result(actual.value(), expected.value());
        assert_eq!(matches!(actual.value(), Value::ComplexTensor(_)), complex);
        assert_eq!(provider.downloads.load(Ordering::Relaxed), 0, "{name}");
        assert_eq!(
            provider.staged_downloads.load(Ordering::Relaxed),
            resident_args.len(),
            "{name} stages every distinct source without ordinary readback"
        );
        assert_eq!(session.ordinary_feedback_for_test().len(), 1, "{name}");
        assert!(ledger.snapshot().live_claimed_bytes > 0, "{name}");
        drop(actual);
        assert_eq!(ledger.snapshot().live_claimed_bytes, 0, "{name}");
        drop((expected, host_args, resident_args));
        assert_eq!(provider.live_count(), 0, "{name}");
    }

    let provider = owner_with_quote_and_failure(1_000_000_000, false, false);
    let left = provider
        .input(&[1.0, 2.0])
        .with_provenance(GpuHandleProvenance::Automatic)
        .unwrap();
    let right = provider
        .input(&[3.5])
        .with_provenance(GpuHandleProvenance::Automatic)
        .unwrap();
    let ledger = ManagedHostEffectLedger::new_with_epoch(0, 34);
    let context = RuntimeContext::new(Rc::new(
        runmat_runtime::execution::RuntimeExecutionService::new(),
    ))
    .with_service_ports(
        RuntimeServicePorts::default()
            .with_parallel(Rc::new(ManagedCastOwner(ledger.clone())))
            .with_placement(Rc::new(crate::placement::PlacementSession::default()))
            .with_acceleration(Rc::new(Adapter(provider, 0))),
    );
    let args = [Value::GpuTensor(left), Value::GpuTensor(right)];
    assert!(
        futures::executor::block_on(context.scope(
            runmat_runtime::dispatcher::call_builtin_async_in_context(&context, "plus", &args),
        ))
        .is_err(),
        "two-source managed host route must refuse an unfunded output/readback"
    );
    assert_eq!(provider.staged_downloads.load(Ordering::Relaxed), 0);
    assert_eq!(provider.downloads.load(Ordering::Relaxed), 0);
    assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
    drop(args);
    assert_eq!(provider.live_count(), 0);
}

#[test]
fn admitted_mixed_or_other_owner_bypasses_selected_provider_and_preserves_runtime_route() {
    for mixed in [false, true] {
        let selected = owner();
        // Explicit pre-effect refusal preserves this case's direct fallback
        // purpose now that the shared owner also supports numeric fixtures.
        let other = owner_with_quote_and_failure(1, false, true);
        let theta_owner = if mixed { selected } else { other };
        let theta = theta_owner.input(&[0.0, 0.0]);
        let rho = other.input(&[2.0, 3.0]);
        let z = other.input(&[7.0]);
        let args = vec![
            Value::GpuTensor(theta.clone()),
            Value::GpuTensor(rho.clone()),
            Value::GpuTensor(z.clone()),
        ];
        let original = args.clone();
        let _ambient = ThreadProviderGuard::set(Some(selected));
        let probe = Rc::new(Probe::default());
        probe.provider.set(true);
        let context = runtime(&probe);
        let auto = NativeAutoOffload::new(selected, ThresholdConfig::default());
        let prepared =
            super::super::admitted::prepare_with(&auto, &context, request(args)).unwrap();
        assert_eq!(prepared.args(), &original);
        assert_eq!(selected.uploads.load(Ordering::Relaxed), 0);
        assert!(probe.requests.borrow().is_empty());
        if mixed {
            use runmat_runtime::context::{RuntimeAccelerationPolicy, RuntimeProviderPreference};
            let strict = context
                .clone()
                .with_acceleration_policy(RuntimeAccelerationPolicy {
                    enabled: true,
                    provider: RuntimeProviderPreference::Auto,
                    automatic_offload: true,
                    allow_host_fallback: false,
                    calibrate: false,
                });
            let error = futures::executor::block_on(strict.scope(
                runmat_runtime::dispatcher::call_builtin_async_with_outputs(
                    "pol2cart", &original, 3,
                ),
            ))
            .unwrap_err();
            assert_eq!(
                error.identifier(),
                Some("RunMat:Acceleration:FallbackDisabled")
            );
            assert_eq!(
                error.gpu_gather_retry,
                runmat_runtime::GpuGatherRetry::Never
            );
            assert_eq!(selected.downloads.load(Ordering::Relaxed), 0);
            assert_eq!(other.downloads.load(Ordering::Relaxed), 0);
            assert!(
                futures::executor::block_on(strict.scope(
                    runmat_runtime::dispatcher::call_builtin_async_with_outputs(
                        "pol2cart",
                        &[Value::Num(0.0), Value::Num(2.0)],
                        2,
                    ),
                ))
                .is_ok(),
                "ordinary host coordinates are not a provider fallback"
            );
        }
        let output =
            futures::executor::block_on(context.scope(prepared.execute(|args| async move {
                runmat_runtime::dispatcher::call_builtin_async_with_outputs("pol2cart", &args, 3)
                    .await
            })))
            .unwrap();
        let Value::OutputList(outputs) = output.value() else {
            panic!("coordinate outputs")
        };
        let expected = [vec![2.0, 3.0], vec![0.0, 0.0], vec![7.0, 7.0]];
        assert_eq!(outputs.len(), expected.len());
        for (actual, expected) in outputs.iter().zip(expected) {
            let Value::Tensor(tensor) = actual else {
                panic!("direct host fallback")
            };
            assert_eq!(tensor.materialize_f64(), expected);
        }
        assert_eq!(selected.uploads.load(Ordering::Relaxed), 0);
        assert!(probe.feedback.borrow().is_empty());
        drop(original);
        drop(theta);
        drop(rho);
        drop(z);
        assert_eq!(selected.live_count(), 0);
        assert_eq!(other.live_count(), 0);
    }
}

#[test]
fn admitted_other_owner_refusal_after_staging_is_terminal_and_retires_upload() {
    let selected = owner();
    let other = owner_with_quote_and_failure(1, false, true);
    let theta = other.input(&[0.0, 0.0]);
    let rho = other.input(&[2.0, 3.0]);
    let args = vec![
        Value::GpuTensor(theta.clone()),
        Value::GpuTensor(rho.clone()),
        Value::Tensor(Tensor::new(vec![7.0], vec![1, 1]).unwrap()),
    ];
    let _ambient = ThreadProviderGuard::set(Some(selected));
    let probe = Rc::new(Probe::default());
    probe.provider.set(true);
    let context = runtime(&probe);
    let auto = NativeAutoOffload::new(selected, ThresholdConfig::default());
    let prepared = super::super::admitted::prepare_with(&auto, &context, request(args)).unwrap();
    assert_eq!(selected.uploads.load(Ordering::Relaxed), 0);
    assert!(probe.requests.borrow().is_empty());
    let error = futures::executor::block_on(context.scope(prepared.execute(|args| async move {
        runmat_runtime::dispatcher::call_builtin_async_with_outputs("pol2cart", &args, 3).await
    })))
    .unwrap_err();
    let source = error
        .source
        .as_ref()
        .and_then(|source| source.downcast_ref::<runmat_accelerate_api::ProviderExecutionError>())
        .expect("provider failure retains its typed effect boundary");
    assert!(source.is_after_effect());
    assert!(
        !source.is_unsupported(),
        "post-staging failure cannot authorize replay"
    );
    assert_eq!(selected.uploads.load(Ordering::Relaxed), 0);
    assert_eq!(other.uploads.load(Ordering::Relaxed), 1);
    assert_eq!(other.downloads.load(Ordering::Relaxed), 0);
    assert!(probe.feedback.borrow().is_empty());
    assert_eq!(
        other.live_count(),
        2,
        "staging retired; both borrowed inputs survive"
    );
    assert_eq!(selected.live_count(), 0);
    assert_eq!(
        futures::executor::block_on(other.download(&theta))
            .unwrap()
            .data,
        vec![0.0, 0.0]
    );
    assert_eq!(
        futures::executor::block_on(other.download(&rho))
            .unwrap()
            .data,
        vec![2.0, 3.0]
    );
    drop(theta);
    drop(rho);
    assert_eq!(other.live_count(), 0);
}

#[test]
fn admitted_matching_owner_uses_retained_provider_not_ambient_selection() {
    let selected = owner();
    let ambient = owner();
    let borrowed = selected.input(&[0.0, 0.0]);
    let _ambient = ThreadProviderGuard::set(Some(ambient));
    let probe = Rc::new(Probe::default());
    probe.provider.set(true);
    let context = runtime(&probe);
    let auto = NativeAutoOffload::new(selected, ThresholdConfig::default());
    let args = vec![
        Value::GpuTensor(borrowed.clone()),
        Value::Tensor(Tensor::new(vec![2.0; 2], vec![1, 2]).unwrap()),
        Value::Tensor(Tensor::new(vec![7.0; 2], vec![1, 2]).unwrap()),
    ];
    let prepared = super::super::admitted::prepare_with(&auto, &context, request(args)).unwrap();
    assert_eq!(selected.uploads.load(Ordering::Relaxed), 2);
    assert_eq!(ambient.uploads.load(Ordering::Relaxed), 0);
    assert_eq!(probe.requests.borrow().len(), 1);
    assert!(prepared.args().iter().all(
        |value| matches!(value, Value::GpuTensor(handle) if handle.device_id() == selected.device_id())
    ));
    drop(prepared);
    assert_eq!(selected.live_count(), 1, "only the borrowed input survives");
    assert!(probe
        .feedback
        .borrow()
        .iter()
        .all(|feedback| !feedback.succeeded));
    assert!(futures::executor::block_on(selected.download(&borrowed)).is_ok());
    drop(borrowed);
    assert_eq!(selected.live_count(), 0);
}

#[test]
fn admitted_owner_proof_traverses_nested_values_and_rejects_unresolved_handles() {
    let selected = owner();
    let other = owner();
    let borrowed = other.input(&[2.0]);
    assert!(futures::executor::block_on(selected.download(&borrowed)).is_err());
    assert!(futures::executor::block_on(other.download(&borrowed)).is_ok());
    let nested = Value::output_list(vec![Value::output_list(vec![Value::GpuTensor(
        borrowed.clone(),
    )])]);
    assert!(!super::super::owner_admission::matches_selected_owner(selected, &[nested]).unwrap());
    let unregistered_owner = runmat_accelerate_api::test_support::TestAllocationOwner::new();
    let unowned = unregistered_owner
        .allocate_zeroed(vec![1, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    assert!(!super::super::owner_admission::matches_selected_owner(
        selected,
        &[Value::GpuTensor(unowned.handle)]
    )
    .unwrap());
    assert_eq!(unowned.probe.retirements(), 1);
    drop(borrowed);
    assert_eq!(other.live_count(), 0);
}
