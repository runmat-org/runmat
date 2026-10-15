//! Ownership conformance uses separate genuine provider allocation owners.
use super::*;
use runmat_accelerate_api::{
    allocation::WeakGpuTensorHandle, GpuTensorHandle, GpuTensorStorage, HostNumericTensorView,
    NumericElementType, ThreadProviderGuard,
};
use runmat_runtime::context::preparation::{
    AdmittedBuiltinPreparation, BuiltinPreparationAdmission,
};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Mutex,
};

pub(super) struct Owner {
    inner: crate::simple_provider::InProcessProvider,
    uploads: AtomicUsize,
    downloads: AtomicUsize,
    allocations: Mutex<Vec<WeakGpuTensorHandle>>,
    reject_upload: bool,
    execution_ns: u64,
    unsupported_trig: bool,
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
    let provider = Box::leak(Box::new(Owner {
        inner: crate::simple_provider::InProcessProvider::new(),
        uploads: AtomicUsize::new(0),
        downloads: AtomicUsize::new(0),
        allocations: Mutex::new(Vec::new()),
        reject_upload,
        execution_ns,
        unsupported_trig,
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
    let nested = Value::OutputList(vec![Value::OutputList(vec![Value::GpuTensor(
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
