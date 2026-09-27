use super::*;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Mutex,
};

struct Recorder {
    mode: AtomicUsize,
    queries: AtomicUsize,
    probes: AtomicUsize,
}
static OWNER: Recorder = Recorder {
    mode: AtomicUsize::new(0),
    queries: AtomicUsize::new(0),
    probes: AtomicUsize::new(0),
};
static SERIAL: Mutex<()> = Mutex::new(());

impl AccelProvider for Recorder {
    fn device_id(&self) -> u32 {
        u32::MAX - 327
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
    fn free(&self, _: &GpuTensorHandle) -> anyhow::Result<()> {
        panic!("borrowed inputs must not be freed")
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
    let _serial = SERIAL.lock().unwrap();
    let _provider = ThreadProviderGuard::set(Some(&OWNER));
    for (mode, queries, probes) in [(0, 2, 0), (1, 1, 1), (2, 2, 1), (3, 0, 1)] {
        OWNER.mode.store(mode, Ordering::SeqCst);
        OWNER.queries.store(0, Ordering::SeqCst);
        OWNER.probes.store(0, Ordering::SeqCst);
        let left = GpuTensorHandle::new(vec![3, 2], OWNER.device_id(), 1)
            .with_numeric_descriptor(NumericElementType::F32, GpuTensorStorage::Real);
        let right = GpuTensorHandle::new(vec![1, 2], OWNER.device_id(), 2)
            .with_numeric_descriptor(NumericElementType::F32, GpuTensorStorage::Real);
        record_handle_transpose(&left, 2, 3);
        let before = snapshot_handle_metadata(&left);
        let result =
            futures::executor::block_on(super::super::pair::try_real_pair(&OWNER, &left, &right));
        if mode == 0 {
            assert!(result.unwrap().is_none());
        } else {
            assert!(result.unwrap_err().message().contains("probe was reached"));
        }
        assert_eq!(OWNER.queries.load(Ordering::SeqCst), queries);
        assert_eq!(OWNER.probes.load(Ordering::SeqCst), probes);
        assert_eq!(snapshot_handle_metadata(&left), before);
        clear_handle_metadata(&left);
        clear_handle_metadata(&right);
    }
}
