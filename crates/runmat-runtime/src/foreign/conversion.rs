use runmat_types::ForeignCapability;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ForeignTransferMode {
    Copy,
    SharedMemorySnapshot,
    Borrow,
    Adopt,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ForeignConversionConstraints {
    pub compatible_host_layout: bool,
    pub uniquely_owned: bool,
    pub crosses_process_boundary: bool,
    pub provider_resident: bool,
    pub output_can_be_adopted: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ForeignConversionPlan {
    pub transfer: ForeignTransferMode,
    pub materialize_to_host: bool,
    pub required_capabilities: Vec<ForeignCapability>,
}

pub fn plan_foreign_input(constraints: ForeignConversionConstraints) -> ForeignConversionPlan {
    if constraints.crosses_process_boundary {
        return ForeignConversionPlan {
            transfer: ForeignTransferMode::SharedMemorySnapshot,
            materialize_to_host: constraints.provider_resident,
            required_capabilities: vec![ForeignCapability::Transfer],
        };
    }
    if constraints.compatible_host_layout && constraints.uniquely_owned {
        return ForeignConversionPlan {
            transfer: ForeignTransferMode::Borrow,
            materialize_to_host: constraints.provider_resident,
            required_capabilities: vec![ForeignCapability::Read, ForeignCapability::ZeroCopy],
        };
    }
    ForeignConversionPlan {
        transfer: ForeignTransferMode::Copy,
        materialize_to_host: constraints.provider_resident,
        required_capabilities: vec![ForeignCapability::Read],
    }
}

pub fn plan_foreign_output(constraints: ForeignConversionConstraints) -> ForeignConversionPlan {
    if !constraints.crosses_process_boundary
        && constraints.compatible_host_layout
        && constraints.output_can_be_adopted
    {
        return ForeignConversionPlan {
            transfer: ForeignTransferMode::Adopt,
            materialize_to_host: false,
            required_capabilities: vec![ForeignCapability::Write, ForeignCapability::ZeroCopy],
        };
    }
    ForeignConversionPlan {
        transfer: if constraints.crosses_process_boundary {
            ForeignTransferMode::SharedMemorySnapshot
        } else {
            ForeignTransferMode::Copy
        },
        materialize_to_host: false,
        required_capabilities: vec![ForeignCapability::Write],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn copies_are_the_default_and_zero_copy_requires_complete_proof() {
        let ordinary = plan_foreign_input(ForeignConversionConstraints {
            compatible_host_layout: true,
            uniquely_owned: false,
            crosses_process_boundary: false,
            provider_resident: false,
            output_can_be_adopted: false,
        });
        assert_eq!(ordinary.transfer, ForeignTransferMode::Copy);

        let unique = plan_foreign_input(ForeignConversionConstraints {
            compatible_host_layout: true,
            uniquely_owned: true,
            crosses_process_boundary: false,
            provider_resident: false,
            output_can_be_adopted: false,
        });
        assert_eq!(unique.transfer, ForeignTransferMode::Borrow);
        assert!(unique
            .required_capabilities
            .contains(&ForeignCapability::ZeroCopy));
    }

    #[test]
    fn isolation_uses_a_snapshot_even_for_compatible_unique_storage() {
        let plan = plan_foreign_input(ForeignConversionConstraints {
            compatible_host_layout: true,
            uniquely_owned: true,
            crosses_process_boundary: true,
            provider_resident: true,
            output_can_be_adopted: false,
        });
        assert_eq!(plan.transfer, ForeignTransferMode::SharedMemorySnapshot);
        assert!(plan.materialize_to_host);
    }
}
