use std::collections::BTreeSet;

use anyhow::{Context, Result};
use runmat_execution::resource::{
    AcceleratorAllocationDomainId, AcceleratorClass, AcceleratorDevice, AcceleratorDeviceId,
    AcceleratorFeature, AcceleratorProvider, AcceleratorProviderId, AcceleratorProviderVersion,
};
use runmat_execution::Digest;

use crate::AccelProvider;

pub const EXECUTION_PROVIDER_ABI_SCHEMA_VERSION: u16 = 1;
pub const RUNMAT_CUDA_PROVIDER_ID: &str = "runmat.cuda";
pub const RUNMAT_WGPU_PROVIDER_ID: &str = "runmat.wgpu";
pub const RUNMAT_BUILTIN_PROVIDER_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Stable software contract exposed by a schedulable provider. Device
/// capacity and supported operation inventory remain device facts; they do
/// not change the ABI identity of compiled code or native extensions.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExecutionProviderContract {
    GenericComputeV1,
    CudaNativeV1,
}

impl ExecutionProviderContract {
    const fn tag(self) -> &'static [u8] {
        match self {
            Self::GenericComputeV1 => b"generic-compute-v1",
            Self::CudaNativeV1 => b"cuda-native-v1",
        }
    }
}

pub fn execution_provider_contract(
    provider_id: &str,
    provider_version: &str,
    contract: ExecutionProviderContract,
) -> Result<AcceleratorProvider> {
    let provider_id = AcceleratorProviderId::new(provider_id)?;
    let provider_version = AcceleratorProviderVersion::new(provider_version)?;
    let mut abi = b"runmat-execution-provider-abi-v1\0".to_vec();
    abi.extend_from_slice(&EXECUTION_PROVIDER_ABI_SCHEMA_VERSION.to_be_bytes());
    append_bytes(&mut abi, provider_id.as_str().as_bytes());
    append_bytes(&mut abi, provider_version.as_str().as_bytes());
    append_bytes(&mut abi, contract.tag());
    Ok(AcceleratorProvider {
        id: provider_id,
        version: provider_version,
        abi_fingerprint: Digest::sha256(abi),
    })
}

/// Builds a cluster-facing identity for a provider that explicitly opts in to
/// remote execution inventory. Process-local provider and buffer identifiers
/// never cross this boundary.
pub fn execution_accelerator_device(
    provider: &(impl AccelProvider + ?Sized),
    provider_id: &str,
    provider_version: &str,
    contract: ExecutionProviderContract,
    allocation_domain_selector: &str,
    device_selector: &str,
    inventory_epoch: u64,
) -> Result<AcceleratorDevice> {
    let info = provider.device_info_struct();
    let max_allocation_bytes = provider
        .placement_resources()
        .capacity_bytes
        .or(info.memory_bytes)
        .filter(|memory| *memory > 0)
        .context("provider does not declare schedulable accelerator memory")?;

    let execution_provider = execution_provider_contract(provider_id, provider_version, contract)?;
    let allocation_domain_selector = AcceleratorAllocationDomainId::new(format!(
        "allocation:{}",
        Digest::sha256(allocation_domain_selector.as_bytes())
    ))?;
    let device_selector = AcceleratorDeviceId::new(device_selector)?;

    let device_bytes = serde_json::to_vec(&(
        execution_provider.id.as_str(),
        device_selector.as_str(),
        &info.vendor,
        &info.name,
        info.backend.as_deref(),
    ))?;
    let device = AcceleratorDevice {
        id: AcceleratorDeviceId::new(format!("device:{}", Digest::sha256(device_bytes)))?,
        allocation_domain: allocation_domain_selector,
        class: AcceleratorClass::new("gpu")?,
        provider: execution_provider,
        max_allocation_bytes,
        features: BTreeSet::from([AcceleratorFeature::Compute]),
        inventory_epoch,
    };
    device.validate()?;
    Ok(device)
}

fn append_bytes(bytes: &mut Vec<u8>, value: &[u8]) {
    bytes.extend_from_slice(&(value.len() as u64).to_be_bytes());
    bytes.extend_from_slice(value);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        AccelDownloadFuture, ApiDeviceInfo, GpuTensorHandle, HostTensorView,
        ProviderCapabilitySnapshot,
    };
    use runmat_execution::ProviderResourceSnapshot;

    struct InventoryProvider {
        revision: u64,
    }

    impl AccelProvider for InventoryProvider {
        fn upload(&self, _host: &HostTensorView) -> Result<GpuTensorHandle> {
            anyhow::bail!("not used by inventory tests")
        }

        fn download<'a>(&'a self, _handle: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
            Box::pin(async { anyhow::bail!("not used by inventory tests") })
        }

        fn free(&self, _handle: &GpuTensorHandle) -> Result<()> {
            Ok(())
        }

        fn device_info(&self) -> String {
            "test device".into()
        }

        fn device_info_struct(&self) -> ApiDeviceInfo {
            ApiDeviceInfo {
                device_id: 72,
                name: "Test Device".into(),
                vendor: "RunMat".into(),
                memory_bytes: Some(16_384),
                backend: Some("test".into()),
            }
        }

        fn capability_snapshot(&self) -> ProviderCapabilitySnapshot {
            let mut snapshot = ProviderCapabilitySnapshot::conservative(self);
            snapshot.revision = self.revision;
            snapshot
        }

        fn placement_resources(&self) -> ProviderResourceSnapshot {
            ProviderResourceSnapshot {
                device_id: 72,
                capacity_bytes: Some(16_384),
                live_bytes: 0,
                reclaimable_bytes: 0,
                scratch_available_bytes: Some(16_384),
                queue_depth: None,
                queue_limit: None,
                lost: false,
                epoch: 4,
            }
        }
    }

    #[test]
    fn provider_abi_is_stable_across_inventory_epochs_and_capability_revisions() {
        let first = execution_accelerator_device(
            &InventoryProvider { revision: 1 },
            "runmat.test",
            "1.0.0",
            ExecutionProviderContract::GenericComputeV1,
            "test-device-0",
            "test-device-0",
            10,
        )
        .unwrap();
        let next_epoch = execution_accelerator_device(
            &InventoryProvider { revision: 1 },
            "runmat.test",
            "1.0.0",
            ExecutionProviderContract::GenericComputeV1,
            "test-device-0",
            "test-device-0",
            11,
        )
        .unwrap();
        let next_contract = execution_accelerator_device(
            &InventoryProvider { revision: 2 },
            "runmat.test",
            "1.0.0",
            ExecutionProviderContract::GenericComputeV1,
            "test-device-0",
            "test-device-0",
            11,
        )
        .unwrap();

        assert_eq!(first.id, next_epoch.id);
        assert_eq!(first.provider, next_epoch.provider);
        assert_ne!(first.inventory_epoch, next_epoch.inventory_epoch);
        assert_eq!(next_epoch.provider, next_contract.provider);
        assert_eq!(first.max_allocation_bytes, 16_384);

        let other_device = execution_accelerator_device(
            &InventoryProvider { revision: 1 },
            "runmat.test",
            "1.0.0",
            ExecutionProviderContract::GenericComputeV1,
            "test-device-1",
            "test-device-1",
            10,
        )
        .unwrap();
        assert_ne!(first.id, other_device.id);
        assert_ne!(first.allocation_domain, other_device.allocation_domain);
        assert_eq!(first.provider, other_device.provider);
    }

    #[test]
    fn native_provider_contracts_have_distinct_abi_identities() {
        let generic = execution_provider_contract(
            "runmat.test",
            "1.0.0",
            ExecutionProviderContract::GenericComputeV1,
        )
        .unwrap();
        let cuda = execution_provider_contract(
            "runmat.test",
            "1.0.0",
            ExecutionProviderContract::CudaNativeV1,
        )
        .unwrap();
        assert_ne!(generic.abi_fingerprint, cuda.abi_fingerprint);
    }

    #[test]
    fn provider_views_share_only_the_explicit_physical_allocation_domain() {
        let generic = execution_accelerator_device(
            &InventoryProvider { revision: 1 },
            "runmat.wgpu",
            "1.0.0",
            ExecutionProviderContract::GenericComputeV1,
            "physical-gpu-0",
            "wgpu-view-0",
            10,
        )
        .unwrap();
        let cuda = execution_accelerator_device(
            &InventoryProvider { revision: 1 },
            "runmat.cuda",
            "1.0.0",
            ExecutionProviderContract::CudaNativeV1,
            "physical-gpu-0",
            "cuda-view-0",
            10,
        )
        .unwrap();

        assert_eq!(generic.allocation_domain, cuda.allocation_domain);
        assert_ne!(generic.id, cuda.id);
        assert_ne!(generic.provider, cuda.provider);
    }
}
