use serde::{Deserialize, Serialize};

use crate::{AccelProvider, ApiDeviceInfo, SpawnHandleConcurrency};

use super::{ProviderElementType, ProviderOperationIdentity};

pub const PROVIDER_CAPABILITY_SCHEMA_VERSION: u16 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderOperationFamily {
    Upload,
    Download,
    Elementwise,
    Reduction,
    MatrixMultiply,
    Library,
    Fusion,
    Graph,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderCapabilityOperation {
    pub identity: ProviderOperationIdentity,
    pub family: ProviderOperationFamily,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderConcurrencyCapabilities {
    pub spawn_handles: SpawnHandleConcurrency,
    pub concurrent_dispatch: bool,
    pub cancellation: bool,
    pub transactional_results: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderCapabilitySnapshot {
    pub schema_version: u16,
    /// Provider-owned revision. A changed capability contract must change this
    /// value so cached placement decisions cannot silently survive it.
    pub revision: u64,
    pub device: ApiDeviceInfo,
    pub operations: Vec<ProviderCapabilityOperation>,
    pub element_types: Vec<ProviderElementType>,
    pub max_rank: Option<u32>,
    pub max_allocation_bytes: Option<u64>,
    pub concurrency: ProviderConcurrencyCapabilities,
}

impl ProviderCapabilitySnapshot {
    pub fn conservative(provider: &(impl AccelProvider + ?Sized)) -> Self {
        let element_type = ProviderElementType::from(provider.precision());
        let device = provider.device_info_struct();
        Self {
            schema_version: PROVIDER_CAPABILITY_SCHEMA_VERSION,
            revision: 0,
            device: device.clone(),
            operations: vec![
                ProviderCapabilityOperation {
                    identity: ProviderOperationIdentity::new("transfer.upload"),
                    family: ProviderOperationFamily::Upload,
                },
                ProviderCapabilityOperation {
                    identity: ProviderOperationIdentity::new("transfer.download"),
                    family: ProviderOperationFamily::Download,
                },
            ],
            element_types: vec![element_type],
            max_rank: None,
            max_allocation_bytes: device.memory_bytes,
            concurrency: ProviderConcurrencyCapabilities {
                spawn_handles: provider.spawn_handle_concurrency(),
                concurrent_dispatch: false,
                cancellation: false,
                transactional_results: false,
            },
        }
    }

    pub fn supports(&self, family: ProviderOperationFamily) -> bool {
        self.operations
            .iter()
            .any(|operation| operation.family == family)
    }

    pub fn supports_operation(
        &self,
        identity: &ProviderOperationIdentity,
        family: ProviderOperationFamily,
    ) -> bool {
        self.operations
            .iter()
            .any(|operation| operation.family == family && operation.identity == *identity)
    }
}
