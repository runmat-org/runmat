//! Compare a governed requirement with the reservation carried by its execution owner.
use super::ResourceRequest;
use crate::ContractError;

impl ResourceRequest {
    /// A reservation must cover every requested dimension; unused capacity is not authority
    /// for a different accelerator class or capability. This compares reservations, not RSS.
    pub fn require_within(&self, reservation: &Self) -> Result<(), ContractError> {
        self.validate()?;
        reservation.validate()?;
        for (name, required, available) in [
            (
                "CPU reservation",
                u64::from(self.cpu_millicores),
                u64::from(reservation.cpu_millicores),
            ),
            (
                "memory reservation",
                self.memory_bytes,
                reservation.memory_bytes,
            ),
            (
                "scratch reservation",
                self.scratch_bytes,
                reservation.scratch_bytes,
            ),
            (
                "duration reservation",
                self.max_wall_millis,
                reservation.max_wall_millis,
            ),
            (
                "artifact reservation",
                self.max_artifact_bytes,
                reservation.max_artifact_bytes,
            ),
            (
                "egress reservation",
                self.max_egress_bytes,
                reservation.max_egress_bytes,
            ),
            (
                "relay reservation",
                self.max_relay_bytes,
                reservation.max_relay_bytes,
            ),
        ] {
            if required > available {
                return Err(ContractError::invalid(
                    name,
                    "governed requirement exceeds reserved capacity",
                ));
            }
        }
        if !self
            .required_capabilities
            .is_subset(&reservation.required_capabilities)
        {
            return Err(ContractError::invalid(
                "capability reservation",
                "governed capability is absent from the reservation",
            ));
        }
        // Each reserved slot may cover only one required slot, including
        // overlapping class/provider/feature requirements. A class-name lookup
        // could spend the same capacity repeatedly and discard constraints.
        if !super::accelerator_requests_satisfy_requirements(
            &reservation.accelerators,
            &self.accelerators,
        ) {
            return Err(ContractError::invalid(
                "accelerator reservation",
                "governed accelerator requirements exceed the reservation",
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
