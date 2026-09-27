//! Shared test-only payload ownership. Probes observe the same allocation that
//! supplies downloads; they do not retain a tensor lease or duplicate payloads.
use runmat_accelerate_api::{
    test_support::{RetirementProbe, TestAllocationOwner},
    *,
};
use std::sync::Mutex;

pub(super) struct Owner {
    owner: TestAllocationOwner,
    probes: Mutex<Vec<RetirementProbe>>,
}
impl Owner {
    pub(super) fn new() -> Self {
        Self {
            owner: TestAllocationOwner::new(),
            probes: Mutex::new(Vec::new()),
        }
    }
    pub(super) fn device_id(&self) -> u32 {
        self.owner.device_id()
    }
    pub(super) fn upload(&self, host: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        self.upload_numeric(&HostNumericTensorView {
            data: HostNumericDataView::F64(host.data),
            shape: host.shape,
            storage: GpuTensorStorage::Real,
        })
    }
    pub(super) fn upload_numeric(
        &self,
        host: &HostNumericTensorView,
    ) -> anyhow::Result<GpuTensorHandle> {
        host.validate()?;
        macro_rules! owned { ($($variant:ident),*) => {
            match host.data { $(HostNumericDataView::$variant(values) => HostNumericDataOwned::$variant(values.to_vec()),)* }
        }; }
        let allocation = self.owner.allocate_numeric(HostNumericTensorOwned {
            data: owned!(F32, F64, I8, I16, I32, I64, U8, U16, U32, U64),
            shape: host.shape.to_vec(),
            storage: host.storage,
        })?;
        self.probes.lock().unwrap().push(allocation.probe);
        Ok(allocation.handle)
    }
    pub(super) fn probe(&self, handle: &GpuTensorHandle) -> RetirementProbe {
        self.probes
            .lock()
            .unwrap()
            .iter()
            .find(|probe| probe.with_data(handle, |_| ()).is_ok())
            .expect("live fixture owner")
            .clone()
    }
    pub(super) fn download(&self, handle: &GpuTensorHandle) -> anyhow::Result<HostTensorOwned> {
        self.probe(handle).with_data(handle, |data| {
            let data = match data {
                HostNumericDataView::F64(values) => values.to_vec(),
                HostNumericDataView::F32(values) => values.iter().map(|&v| f64::from(v)).collect(),
                _ => panic!("floating numerical fixture only"),
            };
            HostTensorOwned {
                data,
                shape: handle.shape().to_vec(),
                storage: handle.physical().storage(),
            }
        })
    }
}
