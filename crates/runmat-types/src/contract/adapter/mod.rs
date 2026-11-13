mod dotnet;
mod model;
#[cfg(test)]
mod tests;
mod windows_com;

pub use dotnet::dotnet_contract;
pub use model::*;
pub use windows_com::windows_com_contract;

use super::SchemaValidationError;
use serde::{Deserialize, Serialize};

pub const FOREIGN_ADAPTER_CONTRACT_SCHEMA_VERSION: u16 = 1;
pub const DOTNET_ADAPTER_CONTRACT_VERSION: u32 = 1;
pub const WINDOWS_COM_ADAPTER_CONTRACT_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlannedForeignAdapter {
    DotNet,
    WindowsCom,
}

impl PlannedForeignAdapter {
    pub const fn adapter_id(self) -> &'static str {
        match self {
            Self::DotNet => "runmat-dotnet",
            Self::WindowsCom => "runmat-windows-com",
        }
    }

    pub const fn current_contract_version(self) -> u32 {
        match self {
            Self::DotNet => DOTNET_ADAPTER_CONTRACT_VERSION,
            Self::WindowsCom => WINDOWS_COM_ADAPTER_CONTRACT_VERSION,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ForeignAdapterDelivery {
    Planned,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ForeignAdapterContractReference {
    pub adapter: PlannedForeignAdapter,
    pub contract_version: u32,
    pub delivery: ForeignAdapterDelivery,
}

impl ForeignAdapterContractReference {
    pub const fn planned(adapter: PlannedForeignAdapter) -> Self {
        Self {
            adapter,
            contract_version: adapter.current_contract_version(),
            delivery: ForeignAdapterDelivery::Planned,
        }
    }

    pub fn validate(&self) -> Result<(), SchemaValidationError> {
        if self.contract_version != self.adapter.current_contract_version() {
            return Err(SchemaValidationError::new(
                "interop.adapter_contracts.contract_version",
                format!(
                    "unsupported {} contract version {}; expected {}",
                    self.adapter.adapter_id(),
                    self.contract_version,
                    self.adapter.current_contract_version()
                ),
            ));
        }
        Ok(())
    }

    pub fn definition(&self) -> ForeignAdapterContract {
        match self.adapter {
            PlannedForeignAdapter::DotNet => dotnet_contract(),
            PlannedForeignAdapter::WindowsCom => windows_com_contract(),
        }
    }
}
