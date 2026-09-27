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

struct Owner {
    inner: crate::simple_provider::InProcessProvider,
    uploads: AtomicUsize,
    allocations: Mutex<Vec<WeakGpuTensorHandle>>,
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
    fn upload(&self, host: &HostTensorView<'_>) -> Result<GpuTensorHandle> {
        self.uploads.fetch_add(1, Ordering::Relaxed);
        Ok(self.publish(self.inner.upload(host)?))
    }
    fn upload_numeric_raw(&self, host: &HostNumericTensorView<'_>) -> Result<GpuTensorHandle> {
        self.uploads.fetch_add(1, Ordering::Relaxed);
        Ok(self.publish(self.inner.upload_numeric_raw(host)?))
    }
    fn download<'a>(
        &'a self,
        handle: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::AccelDownloadFuture<'a> {
        self.inner.download(handle)
    }
    fn query_feasibility(
        &self,
        query: &runmat_accelerate_api::ProviderFeasibilityQuery,
    ) -> runmat_accelerate_api::ProviderFeasibility {
        // This fixture tests ownership before planning, not hook availability.
        known_fixture_resources(query)
    }
}
fn owner() -> &'static Owner {
    let provider = Box::leak(Box::new(Owner {
        inner: crate::simple_provider::InProcessProvider::new(),
        uploads: AtomicUsize::new(0),
        allocations: Mutex::new(Vec::new()),
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
fn admitted_mixed_or_other_owner_bypasses_selected_provider_and_preserves_runtime_route() {
    for mixed in [false, true] {
        let selected = owner();
        let other = owner();
        let theta_owner = if mixed { selected } else { other };
        let theta = theta_owner.input(&[0.0, 0.0]);
        let rho = other.input(&[2.0, 3.0]);
        let args = vec![
            Value::GpuTensor(theta.clone()),
            Value::GpuTensor(rho.clone()),
            Value::Tensor(Tensor::new(vec![7.0], vec![1, 1]).unwrap()),
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
        let Value::OutputList(outputs) = output else {
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
        assert_eq!(selected.live_count(), 0);
        assert_eq!(other.live_count(), 0);
    }
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
