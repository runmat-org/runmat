//! One qualified worker factory's semantic host and physical device inventory.

use crate::{ArtifactError, ArtifactResult};
use runmat_execution::host::ExecutionHostInventory;
use runmat_execution::resource::AcceleratorDevice;
use runmat_types::CapabilityRequirement;
use serde::{Deserialize, Serialize};

pub const PROGRAM_WORKER_INVENTORY_SCHEMA_VERSION: u16 = 1;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProgramWorkerInventory {
    pub schema_version: u16,
    pub host: ExecutionHostInventory,
    pub accelerators: Vec<AcceleratorDevice>,
}

impl ProgramWorkerInventory {
    pub fn validate(&self) -> ArtifactResult<()> {
        if self.schema_version != PROGRAM_WORKER_INVENTORY_SCHEMA_VERSION {
            return Err(ArtifactError::Invalid(
                "unsupported program worker inventory schema".into(),
            ));
        }
        self.host.validate().map_err(crate::error::contract_error)?;
        for device in &self.accelerators {
            device.validate().map_err(crate::error::contract_error)?;
        }
        if self
            .accelerators
            .windows(2)
            .any(|pair| pair[0].id >= pair[1].id)
        {
            return Err(ArtifactError::Invalid(
                "worker accelerator identities must be canonical".into(),
            ));
        }
        let advertised = self
            .host
            .capabilities
            .0
            .contains(&CapabilityRequirement::Accelerator);
        let has_devices = !self.accelerators.is_empty();
        if advertised != has_devices {
            return Err(ArtifactError::Invalid(
                "worker semantic accelerator capability contradicts its device inventory".into(),
            ));
        }
        Ok(())
    }
}
