use super::*;
use futures::executor::block_on;
use runmat_accelerate_api::test_support::{RetirementProbe, TestAllocationOwner};
use runmat_accelerate_api::{
    AccelDownloadFuture, AccelNumericDownloadFuture, HostNumericDataOwned, HostNumericDataView,
    HostNumericTensorOwned, HostTensorView, ProviderPrecision,
};
use std::future::Future;
use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    Mutex,
};
use std::task::{Context, Poll};

struct DownloadOwner {
    allocations: TestAllocationOwner,
    probes: Mutex<Vec<RetirementProbe>>,
    downloads: AtomicUsize,
    workspace: AtomicUsize,
    pending_second: AtomicBool,
    release_second: AtomicBool,
}

impl DownloadOwner {
    fn registered() -> &'static Self {
        let owner = Box::leak(Box::new(Self {
            allocations: TestAllocationOwner::new(),
            probes: Mutex::new(Vec::new()),
            downloads: AtomicUsize::new(0),
            workspace: AtomicUsize::new(4),
            pending_second: AtomicBool::new(false),
            release_second: AtomicBool::new(false),
        }));
        // SAFETY: the leaked owner and unique device identity live for the
        // process/worker lifetime; no unrelated provider is replaced.
        unsafe { runmat_accelerate_api::register_device_provider(owner) };
        owner
    }

    fn resident(&self) -> GpuTensorHandle {
        let allocation = self
            .allocations
            .allocate_zeroed(vec![1, 1], NumericElementType::F32, GpuTensorStorage::Real)
            .unwrap();
        self.probes.lock().unwrap().push(allocation.probe);
        allocation
            .handle
            .with_provenance(GpuHandleProvenance::Automatic)
            .unwrap()
    }
}

impl AccelProvider for DownloadOwner {
    fn device_id(&self) -> u32 {
        self.allocations.device_id()
    }
    fn precision(&self) -> ProviderPrecision {
        ProviderPrecision::F32
    }
    fn device_info(&self) -> String {
        "checked-host-readback-fixture".into()
    }
    fn upload(&self, _: &HostTensorView<'_>) -> anyhow::Result<GpuTensorHandle> {
        panic!("selected host readback never uploads")
    }
    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        panic!("selected host readback must preserve native numeric class")
    }
    fn download_numeric<'a>(
        &'a self,
        handle: &'a GpuTensorHandle,
    ) -> AccelNumericDownloadFuture<'a> {
        Box::pin(async move {
            handle.allocation_lease().validate_live()?;
            let ordinal = self.downloads.fetch_add(1, Ordering::SeqCst);
            if ordinal == 1 && self.pending_second.load(Ordering::SeqCst) {
                std::future::poll_fn(|context| {
                    if self.release_second.load(Ordering::SeqCst) {
                        Poll::Ready(())
                    } else {
                        context.waker().wake_by_ref();
                        Poll::Pending
                    }
                })
                .await;
            }
            let words =
                self.probes.lock().unwrap()[ordinal].with_data(handle, |data| match data {
                    HostNumericDataView::F32(words) => words.to_vec(),
                    _ => panic!("fixture allocation must retain its F32 native words"),
                })?;
            Ok(HostNumericTensorOwned {
                data: HostNumericDataOwned::F32(words),
                shape: handle.shape().to_vec(),
                storage: GpuTensorStorage::Real,
            })
        })
    }
    fn download_numeric_host_workspace_bound(&self, _: &GpuTensorHandle) -> ResourceByteBound {
        ResourceByteBound::Known(self.workspace.load(Ordering::SeqCst) as u64)
    }
    fn download_numeric_provider_scratch_bound(&self, _: &GpuTensorHandle) -> ResourceByteBound {
        ResourceByteBound::Known(0)
    }
    fn download_numeric_queue_slots(&self, _: &GpuTensorHandle) -> u32 {
        0
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn changed_evidence_and_late_source_retirement_never_publish_private_downloads() {
    runmat_gc::gc_test_context(|| {
        let owner = DownloadOwner::registered();
        let first = owner.resident();
        let second = owner.resident();
        let source = |handle: GpuTensorHandle, offset| Source {
            handle,
            owner,
            offsets: vec![offset],
            payload_bytes: 4,
            host_workspace: 4,
            provider_scratch: ResourceByteBound::Known(0),
            queue_slots: 0,
        };
        let plan = CpuReadbackPlan {
            sources: vec![source(first.clone(), 0), source(second.clone(), 1)],
            stack_len: 2,
            resident_offsets: HashSet::from([0, 1]),
        };
        let stack = [
            Value::GpuTensor(first.clone()),
            Value::GpuTensor(second.clone()),
        ];

        let published =
            block_on(plan.prepare(&stack)).unwrap_or_else(|_| panic!("owned native F32 readbacks"));
        assert_eq!(published.len(), 2);
        for (offset, value) in &published {
            assert!(matches!(value, Value::Tensor(tensor)
                if tensor.numeric_dtype() == runmat_value::NumericDType::F32
                    && tensor.shape() == [1, 1]
                    && tensor.as_f32_slice() == Some(&[0.0][..])));
            assert!(*offset < 2);
        }
        assert_eq!(owner.downloads.load(Ordering::SeqCst), 2);
        owner.downloads.store(0, Ordering::SeqCst);
        let uncovered = [
            Value::GpuTensor(first.clone()),
            Value::GpuTensor(second.clone()),
            Value::GpuTensor(first.clone()),
        ];
        let incomplete = CpuReadbackPlan {
            sources: vec![source(first.clone(), 0), source(second.clone(), 1)],
            stack_len: 3,
            resident_offsets: HashSet::from([0, 1]),
        };
        assert!(!incomplete.covers_resident_slice(&uncovered));
        assert!(matches!(
            block_on(incomplete.prepare(&uncovered)),
            Err(CpuReadbackFailure::BeforeEffect(_))
        ));
        assert_eq!(owner.downloads.load(Ordering::SeqCst), 0);
        owner.workspace.store(8, Ordering::SeqCst);
        assert!(matches!(
            block_on(plan.prepare(&stack)),
            Err(CpuReadbackFailure::BeforeEffect(_))
        ));
        assert_eq!(owner.downloads.load(Ordering::SeqCst), 0);
        owner.workspace.store(4, Ordering::SeqCst);
        owner.pending_second.store(true, Ordering::SeqCst);
        let mut pending = Box::pin(plan.prepare(&stack));
        let waker = futures::task::noop_waker();
        let mut context = Context::from_waker(&waker);
        assert!(matches!(pending.as_mut().poll(&mut context), Poll::Pending));
        drop(pending);
        assert_eq!(owner.downloads.load(Ordering::SeqCst), 2);
        assert!(matches!(&stack[0], Value::GpuTensor(_)));
        assert!(matches!(&stack[1], Value::GpuTensor(_)));
        owner.downloads.store(0, Ordering::SeqCst);
        let mut pending = Box::pin(plan.prepare(&stack));
        assert!(matches!(pending.as_mut().poll(&mut context), Poll::Pending));
        owner.probes.lock().unwrap()[0].mark_unavailable();
        owner.release_second.store(true, Ordering::SeqCst);
        assert!(matches!(
            block_on(pending),
            Err(CpuReadbackFailure::AfterEffect(_))
        ));
        assert_eq!(owner.downloads.load(Ordering::SeqCst), 2);
        assert!(first.allocation_lease().validate_live().is_err());
        assert!(second.allocation_lease().validate_live().is_ok());
        assert!(
            matches!(&stack[0], Value::GpuTensor(handle) if handle.view_identity() == first.view_identity())
        );
        assert!(
            matches!(&stack[1], Value::GpuTensor(handle) if handle.view_identity() == second.view_identity())
        );
    });
}
