use super::super::preparation::ArgumentOperation;
use super::*;
use runmat_accelerate_api::{
    allocation::WeakGpuTensorHandle,
    test_support::{RetirementProbe, TestAllocationOwner},
    GpuTensorHandle, GpuTensorStorage, HostNumericDataOwned, HostNumericDataView,
    HostNumericTensorOwned,
};
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Mutex,
};

struct CounterProvider {
    next: AtomicU64,
    fail_on: Option<u64>,
    allocations: Mutex<Vec<(u64, RetirementProbe, WeakGpuTensorHandle)>>,
    owner: TestAllocationOwner,
    alias_on: Mutex<Option<(u64, u64)>>,
}
impl CounterProvider {
    fn allocate(&self, id: u64, data: Vec<f64>, shape: Vec<usize>) -> Result<GpuTensorHandle> {
        let allocation = self.owner.allocate_numeric(HostNumericTensorOwned {
            data: HostNumericDataOwned::F64(data),
            shape,
            storage: GpuTensorStorage::Real,
        })?;
        self.allocations.lock().unwrap().push((
            id,
            allocation.probe,
            allocation.handle.downgrade(),
        ));
        Ok(allocation.handle)
    }
    fn retired(&self) -> Vec<u64> {
        self.allocations
            .lock()
            .unwrap()
            .iter()
            .filter_map(|(id, probe, _)| {
                let count = probe.retirements();
                assert!(count <= 1, "allocation retired more than once");
                (count == 1).then_some(*id)
            })
            .collect()
    }
}
impl AccelProvider for CounterProvider {
    fn upload_numeric_raw(
        &self,
        view: &runmat_accelerate_api::HostNumericTensorView<'_>,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        let id = self.next.fetch_add(1, Ordering::Relaxed);
        if self.fail_on == Some(id) {
            return Err(anyhow!("selected upload failure").into());
        }
        if let Some((at, alias)) = *self.alias_on.lock().unwrap() {
            if at == id {
                return self
                    .allocations
                    .lock()
                    .unwrap()
                    .iter()
                    .find(|(id, _, _)| *id == alias)
                    .expect("live alias target")
                    .2
                    .acquire()
                    .map_err(Into::into);
            }
        }
        let HostNumericDataView::F64(data) = view.data else {
            panic!("F64 fixture input")
        };
        Ok(self.allocate(id, data.to_vec(), view.shape.to_vec())?)
    }
    fn upload(&self, view: &HostTensorView<'_>) -> Result<GpuTensorHandle> {
        let id = self.next.fetch_add(1, Ordering::Relaxed);
        if self.fail_on == Some(id) {
            return Err(anyhow!("selected upload failure"));
        }
        self.allocate(id, view.data.to_vec(), view.shape.to_vec())
    }
    fn download<'a>(
        &'a self,
        _: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::AccelDownloadFuture<'a> {
        Box::pin(async { Err(anyhow!("unused")) })
    }
    fn device_info(&self) -> String {
        "owned-upload-test".into()
    }
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }
    fn precision(&self) -> ProviderPrecision {
        ProviderPrecision::F64
    }
    fn capability_snapshot(&self) -> runmat_accelerate_api::ProviderCapabilitySnapshot {
        let mut snapshot = runmat_accelerate_api::ProviderCapabilitySnapshot::conservative(self);
        // Uploads create private owned leases. Preparation publishes them only
        // after validation; failure or abandonment drops the unpublished leases.
        // The tests below exercise that transaction, including malformed aliases.
        snapshot.concurrency.transactional_results = true;
        snapshot
    }
    fn query_feasibility(
        &self,
        query: &runmat_accelerate_api::ProviderFeasibilityQuery,
    ) -> runmat_accelerate_api::ProviderFeasibility {
        known_fixture_resources(query)
    }
    fn estimate_cost(
        &self,
        _: &runmat_accelerate_api::ProviderCostQuery,
    ) -> Option<runmat_accelerate_api::ProviderCostEstimate> {
        // These fixtures force a staging route to inspect allocation lifetime;
        // the quote is not measured provider performance.
        Some(fixture_quote())
    }
}
fn provider(fail_on: Option<u64>) -> &'static CounterProvider {
    let provider = Box::leak(Box::new(CounterProvider {
        next: AtomicU64::new(1),
        fail_on,
        allocations: Mutex::new(vec![]),
        owner: TestAllocationOwner::new(),
        alias_on: Mutex::new(None),
    }));
    // SAFETY: this leaked facade owns a fresh test allocation namespace and
    // remains addressable for every retained borrowed handle in the fixture.
    unsafe { runmat_accelerate_api::register_device_provider(provider) };
    provider
}

#[test]
fn malformed_upload_alias_cannot_free_nested_borrowed_input() {
    let provider = provider(None);
    *provider.alias_on.lock().unwrap() = Some((1, 900));
    let borrowed = Value::GpuTensor(provider.allocate(900, vec![0.; 2], vec![1, 2]).unwrap());
    let args = [host(), host(), Value::output_list(vec![borrowed])];
    assert!(prepare(provider, &args).is_err());
    assert!(provider.retired().is_empty());
}

#[test]
fn second_upload_alias_releases_first_allocation_exactly_once() {
    let provider = provider(None);
    *provider.alias_on.lock().unwrap() = Some((2, 1));
    assert!(prepare(provider, &[host(), host()]).is_err());
    assert_eq!(provider.retired(), vec![1]);
}

#[test]
fn inaccessible_borrowed_graph_fails_before_any_upload() {
    runmat_gc::gc_test_context(|| {
        let provider = provider(None);
        let target = runmat_gc::gc_allocate(Value::Num(1.)).unwrap();
        let value = Value::HandleObject(runmat_value::HandleRef {
            class_name: "OwnershipFixture".into(),
            target,
            valid: true,
        });
        runmat_gc::gc_with_value_mut(&target, |_| {
            assert!(
                runmat_runtime::dispatcher::try_visit_gpu_handles(&value, &mut |_| {}).is_err()
            );
            assert!(prepare(provider, &[host(), host(), value]).is_err());
        })
        .unwrap();
        assert_eq!(provider.next.load(Ordering::Relaxed), 1);
        assert!(provider.retired().is_empty());
    });
}

#[test]
fn cpu_selection_preserves_inaccessible_control_and_result() {
    runmat_gc::gc_test_context(|| {
        let provider = provider(None);
        let probe = Rc::new(Probe::default());
        let target = runmat_gc::gc_allocate(Value::Num(1.)).unwrap();
        let value = Value::HandleObject(runmat_value::HandleRef {
            class_name: "OwnershipFixture".into(),
            target,
            valid: true,
        });
        let output = rooted(value.clone());
        runmat_gc::gc_with_value_mut(&target, |_| {
            let prepared = NativeAutoOffload::new(provider, ThresholdConfig::default())
                .prepare_operation(
                    "owned.binary.test",
                    &[host(), host(), value.clone()],
                    Some(&runtime(&probe)),
                    ArgumentOperation::Binary(BinaryOp::Elementwise),
                )
                .unwrap();
            let result = futures::executor::block_on(prepared.execute(|_| async { Ok(output) }));
            assert!(result.is_ok());
        })
        .unwrap();
        assert_eq!(provider.next.load(Ordering::Relaxed), 1);
        assert!(provider.retired().is_empty());
        assert!(probe
            .feedback
            .borrow()
            .iter()
            .all(|sample| !sample.succeeded));
    });
}

#[test]
fn inaccessible_result_retains_owned_payload_without_publication_scan() {
    runmat_gc::gc_test_context(|| {
        let provider = provider(None);
        let prepared = prepare(provider, &[host(), host()]).unwrap();
        let target = runmat_gc::gc_allocate(prepared.args()[0].clone()).unwrap();
        let value = Value::HandleObject(runmat_value::HandleRef {
            class_name: "OwnershipFixture".into(),
            target,
            valid: true,
        });
        let output = rooted(value.clone());
        let result = runmat_gc::gc_with_value_mut(&target, |_| {
            assert!(
                runmat_runtime::dispatcher::try_visit_gpu_handles(&value, &mut |_| {}).is_err()
            );
            assert!(runmat_runtime::dispatcher::value_contains_gpu(&value));
            let result = futures::executor::block_on(prepared.execute(|_| async { Ok(output) }))
                .expect("returned managed owner needs no publication scan");
            let Value::HandleObject(returned) = result.value() else {
                panic!("original managed owner");
            };
            assert_eq!(returned.target, target);
            result
        })
        .unwrap();
        // Retain the returned value while its managed payload owns the first upload.
        assert!(runmat_runtime::dispatcher::try_visit_gpu_handles(&result, &mut |_| {}).is_ok());
        assert_eq!(provider.retired(), vec![2]);
        runmat_gc::gc_with_value_mut(&target, |value| *value = Value::Num(0.)).unwrap();
        assert_eq!(provider.retired(), vec![1, 2]);
        assert!(!runmat_runtime::dispatcher::value_contains_gpu(&result));
        drop(result);
        assert_eq!(provider.retired(), vec![1, 2]);
    });
}
fn host() -> Value {
    Value::Tensor(Tensor::new(vec![1.; 2], vec![1, 2]).unwrap())
}
fn prepare(
    provider: &'static CounterProvider,
    args: &[Value],
) -> runmat_accelerate_api::ProviderExecutionResult<PreparedBuiltinCall> {
    let probe = Rc::new(Probe::default());
    probe.provider.set(true);
    NativeAutoOffload::new(provider, ThresholdConfig::default()).prepare_operation(
        "owned.binary.test",
        args,
        Some(&runtime(&probe)),
        ArgumentOperation::Binary(BinaryOp::Elementwise),
    )
}

#[test]
fn second_upload_failure_releases_first_success() {
    let provider = provider(Some(2));
    assert!(prepare(provider, &[host(), host()]).is_err());
    assert_eq!(provider.retired(), vec![1]);
}

#[test]
fn nary_preparation_releases_each_partial_upload_and_abandoned_execution() {
    use std::{
        future::Future,
        task::{Context, Poll},
    };
    let prepare_nary = |provider: &'static CounterProvider| {
        let probe = Rc::new(Probe::default());
        probe.provider.set(true);
        NativeAutoOffload::new(provider, ThresholdConfig::default()).prepare_operation(
            "owned.nary.test",
            &[host(), host(), host()],
            Some(&runtime(&probe)),
            ArgumentOperation::ElementwiseNary {
                data_arity: 3,
                requested_outputs: Some(3),
            },
        )
    };
    for failure in 1..=3 {
        let provider = provider(Some(failure));
        assert!(prepare_nary(provider).is_err());
        assert_eq!(provider.retired(), (1..failure).collect::<Vec<_>>());
    }
    for poll_once in [false, true] {
        let provider = provider(None);
        let prepared = prepare_nary(provider).unwrap();
        let mut future =
            Box::pin(
                prepared.execute(|_| {
                    std::future::pending::<
                        Result<runmat_runtime::execution::RootedValue, RuntimeError>,
                    >()
                }),
            );
        if poll_once {
            assert!(matches!(
                future
                    .as_mut()
                    .poll(&mut Context::from_waker(futures::task::noop_waker_ref())),
                Poll::Pending
            ));
        }
        drop(future);
        assert_eq!(provider.retired(), vec![1, 2, 3]);
    }
}

#[test]
fn abandoned_preparation_and_failed_call_release_uploads() {
    for fail_call in [false, true] {
        let provider = provider(None);
        let prepared = prepare(provider, &[host(), host()]).unwrap();
        if fail_call {
            prepared
                .into_execution()
                .1
                .finish(Err(&RuntimeError::new("call failure")))
                .unwrap();
        } else {
            drop(prepared);
        }
        assert_eq!(provider.retired(), vec![1, 2]);
    }
}

#[test]
fn nested_return_alias_adopts_only_its_own_upload() {
    let provider = provider(None);
    let prepared = prepare(provider, &[host(), host()]).unwrap();
    let output = Value::output_list(vec![Value::output_list(vec![prepared.args()[0].clone()])]);
    let output = Ok(output);
    prepared.into_execution().1.finish(output.as_ref()).unwrap();
    assert_eq!(provider.retired(), vec![2]);
    drop(output);
    assert_eq!(provider.retired(), vec![1, 2]);
}

#[test]
fn borrowed_input_is_never_released_on_call_failure() {
    let provider = provider(None);
    let borrowed = Value::GpuTensor(provider.allocate(900, vec![0.; 2], vec![1, 2]).unwrap());
    let prepared = prepare(provider, &[borrowed.clone(), host()]).unwrap();
    prepared
        .into_execution()
        .1
        .finish(Err(&RuntimeError::new("call failure")))
        .unwrap();
    assert_eq!(provider.retired(), vec![1]);
}

#[test]
fn public_execution_future_releases_uploads_when_dropped_before_or_after_poll() {
    use std::{
        future::Future,
        task::{Context, Poll},
    };
    for poll_once in [false, true] {
        let provider = provider(None);
        let prepared = prepare(provider, &[host(), host()]).unwrap();
        let mut future =
            Box::pin(
                prepared.execute(|_args| {
                    std::future::pending::<
                        Result<runmat_runtime::execution::RootedValue, RuntimeError>,
                    >()
                }),
            );
        if poll_once {
            let mut context = Context::from_waker(futures::task::noop_waker_ref());
            assert!(matches!(future.as_mut().poll(&mut context), Poll::Pending));
        }
        assert!(provider.retired().is_empty());
        drop(future);
        assert_eq!(provider.retired(), vec![1, 2]);
    }
}

#[test]
fn public_execution_adopts_nested_result_alias_and_releases_other_arguments() {
    let provider = provider(None);
    let prepared = prepare(provider, &[host(), host()]).unwrap();
    let result = futures::executor::block_on(
        prepared
            .execute(|args| async move { Ok(rooted(Value::output_list(vec![args[0].clone()]))) }),
    )
    .unwrap();
    assert_eq!(provider.retired(), vec![2]);
    assert!(runmat_runtime::dispatcher::value_contains_gpu(&result));
}

#[test]
fn failed_primary_leaves_original_arguments_valid_for_independent_preparation() {
    let provider = provider(None);
    let original = [host(), host()];
    let primary = prepare(provider, &original).unwrap();
    let result = futures::executor::block_on(
        primary.execute(|_| async { Err(RuntimeError::new("primary rejection")) }),
    );
    assert!(result.is_err());
    assert_eq!(provider.retired(), vec![1, 2]);
    assert!(original
        .iter()
        .all(|value| matches!(value, Value::Tensor(_))));
    let imported = prepare(provider, &original).unwrap();
    let result =
        futures::executor::block_on(imported.execute(|_| async { Ok(rooted(Value::Num(1.))) }));
    assert_eq!(result.unwrap().value(), &Value::Num(1.));
    assert_eq!(provider.retired(), vec![1, 2, 3, 4]);
}
