use futures::FutureExt;
use runmat_accelerate_api::test_support::{RetirementProbe, TestAllocationOwner};
use runmat_accelerate_api::{
    AccelDownloadFuture, AccelProvider, AccelProviderFuture, GpuHandleProvenance, GpuTensorHandle,
    GpuTensorStorage, HostNumericDataOwned, HostNumericDataView, HostNumericTensorOwned,
    HostTensorView, NumericElementType, ProviderPrecision, ThreadProviderGuard,
};
use runmat_value::Value;
use std::sync::Mutex;

#[path = "support/retained_provider.rs"]
mod retained_provider;

struct PendingProvider {
    owner: TestAllocationOwner,
    pending_allocation: Mutex<Option<RetirementProbe>>,
}

impl PendingProvider {
    fn pending(
        &self,
        handles: &[&GpuTensorHandle],
    ) -> AccelProviderFuture<'static, GpuTensorHandle> {
        for handle in handles {
            assert_eq!(handle.device_id(), self.owner.device_id());
            handle.allocation_lease().validate_live().unwrap();
        }
        // A real unpublished provider temporary is retained inside the suspended
        // future. Immutable view descriptors cannot be poisoned/restored anymore.
        let temporary = self
            .owner
            .allocate_zeroed(
                vec![1, 2],
                NumericElementType::F64,
                GpuTensorStorage::ComplexInterleaved,
            )
            .unwrap();
        *self.pending_allocation.lock().unwrap() = Some(temporary.probe);
        let borrowed_leases: Vec<_> = handles.iter().map(|handle| (*handle).clone()).collect();
        Box::pin(async move {
            let _owned = (temporary.handle, borrowed_leases);
            futures::future::pending().await
        })
    }
}

impl AccelProvider for PendingProvider {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }
    fn device_info(&self) -> String {
        "pending owned complex fixture".into()
    }
    fn precision(&self) -> ProviderPrecision {
        ProviderPrecision::F64
    }
    fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        anyhow::bail!("fixture inputs are admitted directly by their real owner")
    }
    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        Box::pin(async { anyhow::bail!("pending provider route must not gather") })
    }
    fn unary_conj<'a>(
        &'a self,
        handle: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, Option<GpuTensorHandle>> {
        let pending = self.pending(&[handle]);
        Box::pin(async move {
            pending.await.map(Some).map_err(|error| {
                runmat_accelerate_api::ProviderExecutionError::from(error).with_unproven_effects()
            })
        })
    }
    fn unary_real<'a>(
        &'a self,
        handle: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        self.pending(&[handle])
    }
    fn unary_imag<'a>(
        &'a self,
        handle: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        self.pending(&[handle])
    }
    fn complex_from_real<'a>(
        &'a self,
        handle: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        self.pending(&[handle])
    }
    fn complex_from_real_imag<'a>(
        &'a self,
        real: &'a GpuTensorHandle,
        imaginary: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        self.pending(&[real, imaginary])
    }
}

#[test]
fn abandoned_component_and_construction_calls_preserve_borrows_and_retire_pending_temporaries() {
    let provider = retained_provider::retained_test_provider!(PROVIDER: PendingProvider = PendingProvider {
        owner: TestAllocationOwner::new(),
        pending_allocation: Mutex::new(None),
    });
    let _current = ThreadProviderGuard::set(Some(provider));
    for (name, count) in [
        ("conj", 1),
        ("real", 1),
        ("imag", 1),
        ("complex", 1),
        ("complex", 2),
    ] {
        *provider.pending_allocation.lock().unwrap() = None;
        let allocations: Vec<_> = (0..count)
            .map(|_| {
                provider
                    .owner
                    .allocate_numeric(HostNumericTensorOwned {
                        data: HostNumericDataOwned::F64(vec![1., 2.]),
                        shape: vec![1, 2],
                        storage: GpuTensorStorage::Real,
                    })
                    .unwrap()
            })
            .collect();
        let probes: Vec<_> = allocations
            .iter()
            .map(|allocation| allocation.probe.clone())
            .collect();
        let handles: Vec<_> = allocations
            .into_iter()
            .map(|allocation| {
                allocation
                    .handle
                    .with_provenance(GpuHandleProvenance::Explicit)
                    .unwrap()
            })
            .collect();
        let descriptors: Vec<_> = handles.iter().map(|handle| handle.descriptor()).collect();
        let args: Vec<_> = handles.iter().cloned().map(Value::GpuTensor).collect();
        let mut call = Box::pin(runmat_runtime::call_builtin_async(name, &args));
        assert!(
            call.as_mut().now_or_never().is_none(),
            "{name} must reach pending provider"
        );
        let pending = provider
            .pending_allocation
            .lock()
            .unwrap()
            .clone()
            .expect("the actual provider hook must have allocated its pending temporary");
        assert_eq!(
            pending.retirements(),
            0,
            "{name}: temporary is live while pending"
        );
        assert!(probes.iter().all(|probe| probe.retirements() == 0));
        drop(call);
        assert_eq!(
            pending.retirements(),
            1,
            "{name}: cancellation retires unpublished temporary exactly once"
        );
        drop(args);
        for ((handle, descriptor), probe) in handles.iter().zip(&descriptors).zip(&probes) {
            assert_eq!(
                &handle.descriptor(),
                descriptor,
                "{name}: borrowed view is unchanged"
            );
            assert!(!runmat_accelerate_api::handle_is_logical(handle));
            assert_eq!(handle.shape(), &[1, 2]);
            assert!(runmat_accelerate_api::handle_transpose_info(handle).is_none());
            assert_eq!(
                probe.retirements(),
                0,
                "{name}: borrowed allocation remains live"
            );
            probe
                .with_data(handle, |data| {
                    let HostNumericDataView::F64(data) = data else {
                        panic!("real double input");
                    };
                    assert_eq!(data, &[1., 2.], "{name}: original owner payload is intact");
                })
                .unwrap();
        }
        drop(handles);
        assert!(
            probes.iter().all(|probe| probe.retirements() == 1),
            "{name}: last caller lease retires each input once"
        );
        assert_eq!(pending.retirements(), 1);
    }
}
