use super::*;
use futures::executor::block_on;
use runmat_accelerate_api::test_support::{RetirementProbe, TestAllocationOwner};
use runmat_accelerate_api::{
    AccelDownloadFuture, AccelNumericDownloadFuture, GpuTensorHandle, HostNumericDataOwned,
    HostNumericDataView, HostNumericTensorOwned, HostTensorView, ProviderPrecision,
};
use std::future::Future;
use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    Mutex,
};
use std::task::{Context, Poll};

struct DownloadOwner {
    allocations: TestAllocationOwner,
    probes: Mutex<
        Vec<(
            runmat_accelerate_api::allocation::AllocationIdentity,
            RetirementProbe,
        )>,
    >,
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
        self.probes
            .lock()
            .unwrap()
            .push((allocation.handle.allocation_identity(), allocation.probe));
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
            let probes = self.probes.lock().unwrap();
            let (_, probe) = probes
                .iter()
                .find(|(identity, _)| *identity == handle.allocation_identity())
                .ok_or_else(|| anyhow::anyhow!("fixture download has no owning allocation"))?;
            let words = probe.with_data(handle, |data| match data {
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
            loads: Vec::new(),
            capabilities: std::borrow::Cow::Owned(owner.capability_snapshot()),
            payload_bytes: 4,
            host_workspace: 4,
            provider_scratch: ResourceByteBound::Known(0),
            queue_slots: 0,
        };
        let plan = || CpuReadbackPlan {
            sources: vec![source(first.clone(), 0), source(second.clone(), 1)],
            stack_len: 2,
            resident_offsets: vec![0, 1],
        };
        let stack = [
            Value::GpuTensor(first.clone()),
            Value::GpuTensor(second.clone()),
        ];

        let published = block_on(plan().prepare_inputs(&stack))
            .unwrap_or_else(|_| panic!("owned native F32 readbacks"));
        assert_eq!(published.entry.len(), 2);
        for (offset, value) in &published.entry {
            assert!(matches!(value, Value::Tensor(tensor)
                if tensor.numeric_dtype() == runmat_value::NumericDType::F32
                    && tensor.shape() == [1, 1]
                    && tensor.as_f32_slice() == Some(&[0.0][..])));
            assert!(*offset < 2);
        }
        assert_eq!(owner.downloads.load(Ordering::SeqCst), 2);
        published
            .cleanup()
            .expect("retire published source references");
        // A missing comparative price must not revoke an independently
        // authenticated VM route or turn its retained readback into feedback.
        for operations in [false, true] {
            use runmat_runtime::context::{
                RuntimeContext, RuntimeServicePorts, VmFusionHostOperation,
                VmFusionInstalledRegion, VmFusionSemanticIdentity,
            };
            use std::rc::Rc;
            let session = Rc::new(crate::placement::PlacementSession::default());
            let runtime = RuntimeContext::new(Rc::new(
                runmat_runtime::execution::RuntimeExecutionService::new(),
            ))
            .with_service_ports(RuntimeServicePorts::default().with_placement(session.clone()));
            let installed = VmFusionInstalledRegion::install(
                &runtime,
                VmFusionSemanticIdentity::from_validated_digest(runmat_execution::Digest::sha256(
                    b"unpriced-authenticated-host-readback",
                )),
                0,
                0,
                0,
            )
            .unwrap();
            let handle = owner.resident();
            let stack = [Value::GpuTensor(handle.clone())];
            let readback = CpuReadbackPlan {
                sources: vec![Source::admit(&handle, owner, vec![0], Vec::new()).unwrap()],
                stack_len: 1,
                resident_offsets: vec![0],
            };
            let entry = &runmat_builtins::PLUS_CATALOG_ENTRY;
            let binding = entry.binding_identity(&entry.bindings[0]);
            let right = installed.right(&runtime).unwrap();
            let observation = crate::fusion_exec::FusionCpuObservation::new(None);
            let mut observation = if operations {
                observation.with_vm_host_operations_readback(
                    right,
                    Some(&runtime),
                    Some(vec![VmFusionHostOperation::Builtin(binding)]),
                    Some(readback),
                    None,
                )
            } else {
                observation.with_vm_host_route_readback(
                    right,
                    Some(&runtime),
                    Some(&[binding]),
                    Some(readback),
                )
            };
            assert!(observation.vm_host_route().is_some());
            let before = owner.downloads.load(Ordering::SeqCst);
            let prepared = block_on(observation.prepare_vm_host_inputs(&stack))
                .unwrap_or_else(|_| panic!("unpriced route retains its authenticated readback"))
                .unwrap();
            assert_eq!(owner.downloads.load(Ordering::SeqCst), before + 1);
            assert!(matches!(&prepared[..], [(0, Value::Tensor(tensor))]
                if tensor.as_f32_slice() == Some(&[0.0][..])));
            assert!(matches!(&stack[0], Value::GpuTensor(original)
                if original.view_identity() == handle.view_identity()));
            observation.complete_vm(runmat_time::Instant::now());
            assert!(session.profile_snapshot().feedback.is_empty());
        }
        owner.downloads.store(0, Ordering::SeqCst);
        let binding = crate::graph::VarBinding {
            kind: crate::graph::VarKind::Global,
            index: 0,
        };
        let mut repeated = source(first.clone(), 0);
        repeated.loads = [17, 23]
            .into_iter()
            .map(|pc| crate::fusion_exec::VmHostLoadUse {
                pc,
                binding: binding.clone(),
                value_id: 0,
            })
            .collect();
        let prepared = block_on(
            CpuReadbackPlan {
                sources: vec![repeated],
                stack_len: 1,
                resident_offsets: vec![0],
            }
            .prepare_inputs(&stack[..1]),
        )
        .unwrap_or_else(|_| panic!("one mirror for all exact load occurrences"));
        assert_eq!(owner.downloads.load(Ordering::SeqCst), 1);
        assert_eq!(prepared.entry.len(), 1);
        for pc in [17, 23] {
            let value = prepared
                .substitute_load(pc, &binding, &stack[0])
                .unwrap()
                .unwrap();
            assert!(
                matches!(value, Value::Tensor(tensor) if tensor.numeric_dtype() == runmat_value::NumericDType::F32)
            );
        }
        assert!(prepared
            .substitute_load(19, &binding, &stack[0])
            .unwrap()
            .is_none());
        assert!(
            prepared.substitute_load(23, &binding, &stack[1]).is_err(),
            "normal refresh to another allocation cannot reuse a stale mirror"
        );
        let explicit = first
            .clone()
            .with_provenance(GpuHandleProvenance::Explicit)
            .unwrap();
        assert!(prepared
            .substitute_load(23, &binding, &Value::GpuTensor(explicit.clone()))
            .is_err());
        assert!(
            Source::admit(&explicit, owner, vec![0], Vec::new()).is_none(),
            "explicit gpuArray intent cannot silently gather"
        );
        owner.workspace.store(8, Ordering::SeqCst);
        assert!(prepared.substitute_load(23, &binding, &stack[0]).is_err());
        owner.workspace.store(4, Ordering::SeqCst);
        assert!(
            matches!(&stack[0], Value::GpuTensor(handle) if handle.view_identity() == first.view_identity()),
            "host mirrors do not replace the workspace or original operand source"
        );
        drop(prepared);
        owner.downloads.store(0, Ordering::SeqCst);
        let retiring = owner.resident();
        let retirement = owner.probes.lock().unwrap().last().unwrap().1.clone();
        let retiring_stack = [Value::GpuTensor(retiring.clone())];
        let prepared = block_on(
            CpuReadbackPlan {
                sources: vec![source(retiring.clone(), 0)],
                stack_len: 1,
                resident_offsets: vec![0],
            }
            .prepare_inputs(&retiring_stack),
        )
        .unwrap_or_else(|_| panic!("entry-only source remains leased through publication"));
        drop(retiring_stack);
        drop(retiring);
        retirement.fail_retirement("checked selected CPU source cleanup");
        assert!(
            prepared.cleanup().is_err(),
            "last-source cleanup failure cannot certify successful completion"
        );
        assert_eq!(
            retirement.retirements(),
            1,
            "failed retirement is never replayed"
        );
        owner.downloads.store(0, Ordering::SeqCst);
        let uncovered = [
            Value::GpuTensor(first.clone()),
            Value::GpuTensor(second.clone()),
            Value::GpuTensor(first.clone()),
        ];
        let incomplete = CpuReadbackPlan {
            sources: vec![source(first.clone(), 0), source(second.clone(), 1)],
            stack_len: 3,
            resident_offsets: vec![0, 1],
        };
        assert!(!incomplete.covers_resident_slice(&uncovered));
        assert!(matches!(
            block_on(incomplete.prepare_inputs(&uncovered)),
            Err(CpuReadbackFailure::BeforeEffect(_))
        ));
        assert_eq!(owner.downloads.load(Ordering::SeqCst), 0);
        owner.workspace.store(8, Ordering::SeqCst);
        assert!(matches!(
            block_on(plan().prepare_inputs(&stack)),
            Err(CpuReadbackFailure::BeforeEffect(_))
        ));
        assert_eq!(owner.downloads.load(Ordering::SeqCst), 0);
        owner.workspace.store(4, Ordering::SeqCst);
        owner.pending_second.store(true, Ordering::SeqCst);
        let cancel_first = owner.resident();
        let cancel_second = owner.resident();
        let cancel_probes = {
            let probes = owner.probes.lock().unwrap();
            [
                probes[probes.len() - 2].1.clone(),
                probes[probes.len() - 1].1.clone(),
            ]
        };
        let cancel_stack = [
            Value::GpuTensor(cancel_first.clone()),
            Value::GpuTensor(cancel_second.clone()),
        ];
        let cancel_plan = CpuReadbackPlan {
            sources: vec![
                source(cancel_first.clone(), 0),
                source(cancel_second.clone(), 1),
            ],
            stack_len: 2,
            resident_offsets: vec![0, 1],
        };
        drop((cancel_first, cancel_second));
        let mut pending = Box::pin(cancel_plan.prepare_inputs(&cancel_stack));
        let waker = futures::task::noop_waker();
        let mut context = Context::from_waker(&waker);
        assert!(matches!(pending.as_mut().poll(&mut context), Poll::Pending));
        drop(pending);
        assert_eq!(owner.downloads.load(Ordering::SeqCst), 2);
        drop(cancel_stack);
        assert_eq!(cancel_probes[0].retirements(), 1);
        assert_eq!(cancel_probes[1].retirements(), 1);
        assert!(matches!(&stack[0], Value::GpuTensor(_)));
        assert!(matches!(&stack[1], Value::GpuTensor(_)));
        owner.downloads.store(0, Ordering::SeqCst);
        let mut pending = Box::pin(plan().prepare_inputs(&stack));
        assert!(matches!(pending.as_mut().poll(&mut context), Poll::Pending));
        owner.probes.lock().unwrap()[0].1.mark_unavailable();
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
