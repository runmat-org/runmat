//! A genuine Single/U8 fixture. Preferred floating precision does not imply
//! logical encoding: InProcessProvider deliberately uses F64 logical storage.
use runmat_accelerate_api::allocation::AllocationIdentity;
use runmat_accelerate_api::test_support::{RetirementProbe, TestAllocation, TestAllocationOwner};
use runmat_accelerate_api::*;
use std::sync::Mutex;

#[derive(Default)]
pub struct NativeLogicalProvider {
    owner: TestAllocationOwner,
    records: Mutex<Vec<(AllocationIdentity, RetirementProbe)>>,
}

impl NativeLogicalProvider {
    fn publish(&self, allocation: TestAllocation) -> GpuTensorHandle {
        self.records
            .lock()
            .unwrap()
            .push((allocation.handle.allocation_identity(), allocation.probe));
        allocation.handle
    }

    fn snapshot(&self, handle: &GpuTensorHandle) -> anyhow::Result<HostNumericTensorOwned> {
        let probe = self
            .records
            .lock()
            .unwrap()
            .iter()
            .find(|(identity, _)| *identity == handle.allocation_identity())
            .map(|(_, probe)| probe.clone())
            .ok_or_else(|| anyhow::anyhow!("fixture does not own allocation"))?;
        let data = probe.with_data(handle, |data| match data {
            HostNumericDataView::F32(data) => HostNumericDataOwned::F32(data.to_vec()),
            HostNumericDataView::U8(data) => HostNumericDataOwned::U8(data.to_vec()),
            _ => unreachable!("only native Single and logical U8 are published"),
        })?;
        Ok(HostNumericTensorOwned {
            data,
            shape: handle.shape().to_vec(),
            storage: handle.physical().storage(),
        })
    }
}

impl AccelProvider for NativeLogicalProvider {
    fn upload(&self, _: &HostTensorView<'_>) -> anyhow::Result<GpuTensorHandle> {
        Err(unsupported_provider_operation(
            "fixture native double upload",
        ))
    }
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }
    fn device_info(&self) -> String {
        "native Single/logical U8 fixture".into()
    }
    fn precision(&self) -> ProviderPrecision {
        ProviderPrecision::F32
    }
    fn upload_numeric_raw(
        &self,
        host: &HostNumericTensorView<'_>,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        host.validate()?;
        let HostNumericDataView::F32(data) = host.data else {
            return Err(unsupported_provider_operation("fixture native numeric class").into());
        };
        Ok(
            self.publish(self.owner.allocate_numeric(HostNumericTensorOwned {
                data: HostNumericDataOwned::F32(data.to_vec()),
                shape: host.shape.to_vec(),
                storage: host.storage,
            })?),
        )
    }
    fn upload_logical_raw(
        &self,
        host: &HostLogicalTensorView<'_>,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        Ok(self.publish(
            self.owner
                .allocate_logical(host.data.to_vec(), host.shape.to_vec())
                .map_err(|error| {
                    runmat_accelerate_api::ProviderExecutionError::from(error)
                        .with_unproven_effects()
                })?,
        ))
    }
    fn download_numeric<'a>(
        &'a self,
        handle: &'a GpuTensorHandle,
    ) -> AccelNumericDownloadFuture<'a> {
        Box::pin(async move { self.snapshot(handle) })
    }
    fn download<'a>(&'a self, handle: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        Box::pin(async move {
            let host = self.snapshot(handle)?;
            let data = match host.data {
                HostNumericDataOwned::F32(data) => data.into_iter().map(f64::from).collect(),
                HostNumericDataOwned::U8(data) => data.into_iter().map(f64::from).collect(),
                _ => unreachable!("snapshot validates storage"),
            };
            Ok(HostTensorOwned {
                data,
                shape: host.shape,
                storage: host.storage,
            })
        })
    }
}
