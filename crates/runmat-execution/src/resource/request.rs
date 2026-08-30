use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use super::{
    matching::accelerator_leases_exactly_satisfy, AcceleratorDevice, AcceleratorDeviceLease,
    AcceleratorRequest, Capability,
};
use crate::ContractError;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResourceRequest {
    pub cpu_millicores: u32,
    pub memory_bytes: u64,
    pub scratch_bytes: u64,
    pub max_wall_millis: u64,
    pub max_artifact_bytes: u64,
    pub max_egress_bytes: u64,
    pub max_relay_bytes: u64,
    pub accelerators: Vec<AcceleratorRequest>,
    pub required_capabilities: BTreeSet<Capability>,
}

impl ResourceRequest {
    pub fn validate(&self) -> Result<(), ContractError> {
        if self.cpu_millicores == 0 {
            return Err(ContractError::invalid(
                "cpu_millicores",
                "must be greater than zero",
            ));
        }
        if self.memory_bytes == 0 || self.max_wall_millis == 0 {
            return Err(ContractError::invalid(
                "resource request",
                "memory and maximum duration must be greater than zero",
            ));
        }
        let accelerator_count = self.accelerators.iter().try_fold(0usize, |total, request| {
            total.checked_add(usize::from(request.count))
        });
        if self.accelerators.len() > 16 || accelerator_count.is_none_or(|count| count > 16) {
            return Err(ContractError::Limit {
                field: "accelerators",
                limit: 16,
            });
        }
        for accelerator in &self.accelerators {
            accelerator.validate()?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResourceInventory {
    pub cpu_millicores: u32,
    pub memory_bytes: u64,
    pub scratch_bytes: u64,
    pub accelerators: Vec<AcceleratorDevice>,
    pub capabilities: BTreeSet<Capability>,
}

impl ResourceInventory {
    pub fn validate(&self) -> Result<(), ContractError> {
        if self.cpu_millicores == 0 || self.memory_bytes == 0 {
            return Err(ContractError::invalid(
                "resource inventory",
                "CPU and memory must be non-zero",
            ));
        }
        if self
            .accelerators
            .windows(2)
            .any(|pair| pair[0].id >= pair[1].id)
        {
            return Err(ContractError::invalid(
                "resource inventory accelerators",
                "devices must be sorted and unique by identity",
            ));
        }
        for accelerator in &self.accelerators {
            accelerator.validate()?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResourceAssignment {
    pub accelerator_leases: Vec<AcceleratorDeviceLease>,
}

impl ResourceAssignment {
    pub fn validate(&self) -> Result<(), ContractError> {
        if self
            .accelerator_leases
            .windows(2)
            .any(|pair| pair[0].device_id >= pair[1].device_id)
        {
            return Err(ContractError::invalid(
                "resource assignment accelerator leases",
                "leases must be sorted and unique by device identity",
            ));
        }
        if self
            .accelerator_leases
            .iter()
            .map(|lease| &lease.allocation_domain)
            .collect::<BTreeSet<_>>()
            .len()
            != self.accelerator_leases.len()
        {
            return Err(ContractError::invalid(
                "resource assignment accelerator leases",
                "leases must be unique by physical allocation domain",
            ));
        }
        for lease in &self.accelerator_leases {
            lease.validate()?;
        }
        Ok(())
    }

    pub fn validate_for_request(&self, request: &ResourceRequest) -> Result<(), ContractError> {
        self.validate()?;
        request.validate()?;
        if !accelerator_leases_exactly_satisfy(&self.accelerator_leases, &request.accelerators) {
            return Err(ContractError::invalid(
                "resource assignment",
                "accelerator leases do not exactly satisfy the request",
            ));
        }
        Ok(())
    }
}
