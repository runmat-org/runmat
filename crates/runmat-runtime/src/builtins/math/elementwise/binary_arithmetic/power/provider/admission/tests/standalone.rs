use super::super::*;
use runmat_accelerate_api::test_support::TestAllocationOwner;
use std::sync::atomic::{AtomicUsize, Ordering};

struct Recorder {
    owner: TestAllocationOwner,
    mode: AtomicUsize,
    queries: AtomicUsize,
    probes: AtomicUsize,
}

impl AccelProvider for Recorder {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }
    fn precision(&self) -> ProviderPrecision {
        ProviderPrecision::F64
    }
    fn device_info(&self) -> String {
        "power-admission-recorder".into()
    }
    fn upload(&self, _: &HostTensorView<'_>) -> anyhow::Result<GpuTensorHandle> {
        unreachable!()
    }
    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        unreachable!()
    }
    fn capability_snapshot(&self) -> ProviderCapabilitySnapshot {
        let mut snapshot = ProviderCapabilitySnapshot::conservative(self);
        if self.mode.load(Ordering::SeqCst) != 3 {
            snapshot.operations.push(ProviderCapabilityOperation {
                identity: ProviderOperationIdentity::new(
                    runmat_builtins::POWER_CATALOG_ENTRY.identity.name,
                ),
                family: ProviderOperationFamily::Elementwise,
            });
        }
        snapshot
    }
    fn query_feasibility(&self, query: &ProviderFeasibilityQuery) -> ProviderFeasibility {
        self.queries.fetch_add(1, Ordering::SeqCst);
        assert_eq!(query.inputs[0].element_type, ProviderElementType::F32);
        assert_eq!(query.inputs[0].layout, ProviderLayout::Strided);
        assert_eq!(query.inputs[0].shape, vec![3, 2]);
        let supports = match self.mode.load(Ordering::SeqCst) {
            1 => {
                query.workload.numerical_domain
                    == NumericalDomainEvidence::RealPower(PowerResultDomain::Real)
            }
            2 => {
                query.workload.numerical_domain
                    == NumericalDomainEvidence::RealPower(PowerResultDomain::Complex)
            }
            _ => false,
        };
        if supports {
            ProviderFeasibility::supported(ProviderResourceEstimate::default())
        } else {
            ProviderFeasibility::rejected(
                ProviderRejectionCode::UnsupportedElementType,
                "test.execution.unsupported",
            )
        }
    }
    fn classify_real_power<'a>(
        &'a self,
        _: &'a GpuTensorHandle,
        _: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, PowerResultDomain> {
        self.probes.fetch_add(1, Ordering::SeqCst);
        Box::pin(async { anyhow::bail!("probe was reached") })
    }
}

#[test]
fn standalone_admission_skips_only_explicitly_rejected_outcomes() {
    let _serial = crate::builtins::common::test_support::accel_test_lock();
    let owner = Box::leak(Box::new(Recorder {
        owner: TestAllocationOwner::new(),
        mode: AtomicUsize::new(0),
        queries: AtomicUsize::new(0),
        probes: AtomicUsize::new(0),
    }));
    let _provider = ThreadProviderGuard::set(Some(owner));
    for (mode, queries, probes) in [(0, 2, 0), (1, 1, 1), (2, 2, 1), (3, 0, 1)] {
        owner.mode.store(mode, Ordering::SeqCst);
        owner.queries.store(0, Ordering::SeqCst);
        owner.probes.store(0, Ordering::SeqCst);
        let left_allocation = owner
            .owner
            .allocate_zeroed(vec![2, 3], NumericElementType::F32, GpuTensorStorage::Real)
            .unwrap();
        let right_allocation = owner
            .owner
            .allocate_zeroed(vec![1, 2], NumericElementType::F32, GpuTensorStorage::Real)
            .unwrap();
        let left = left_allocation
            .handle
            .with_view(GpuTensorView::contiguous(vec![3, 2]).with_layout(
                GpuTensorLayout::Transpose(TransposeInfo {
                    base_rows: 2,
                    base_cols: 3,
                }),
            ))
            .expect("admitted transpose view");
        let right = right_allocation.handle;
        let before = left.wire_descriptor();
        let result = futures::executor::block_on(super::super::super::pair::try_real_pair(
            owner, &left, &right,
        ));
        if mode == 0 {
            assert!(result.unwrap().is_none());
        } else {
            assert!(result.unwrap_err().message().contains("probe was reached"));
        }
        assert_eq!(owner.queries.load(Ordering::SeqCst), queries);
        assert_eq!(owner.probes.load(Ordering::SeqCst), probes);
        assert_eq!(left.wire_descriptor(), before);
        assert_eq!(left_allocation.probe.retirements(), 0);
        assert_eq!(right_allocation.probe.retirements(), 0);
        drop((left, left_allocation.handle, right));
        assert_eq!(left_allocation.probe.retirements(), 1);
        assert_eq!(right_allocation.probe.retirements(), 1);
    }
}
