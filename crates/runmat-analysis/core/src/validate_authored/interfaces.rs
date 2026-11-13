use std::collections::HashSet;

use crate::{AnalysisInterface, AnalysisInterfaceKind, AnalysisValidationError};

use super::{validate_id, validate_target};

pub(super) fn validate_interfaces(
    values: &[AnalysisInterface],
) -> Result<(), AnalysisValidationError> {
    let mut ids = HashSet::new();
    for interface in values {
        validate_id(&interface.interface_id, "interface", &mut ids)?;
        validate_target(&interface.primary).map_err(|detail| {
            AnalysisValidationError::InvalidTarget {
                owner_id: interface.interface_id.clone(),
                detail,
            }
        })?;
        validate_target(&interface.secondary).map_err(|detail| {
            AnalysisValidationError::InvalidTarget {
                owner_id: interface.interface_id.clone(),
                detail,
            }
        })?;
        let valid = interface.primary.region_id != interface.secondary.region_id
            && match &interface.kind {
                AnalysisInterfaceKind::Contact(value) => match value.friction {
                    crate::ContactFriction::Frictionless => true,
                    crate::ContactFriction::Coulomb { coefficient } => {
                        coefficient.is_finite() && coefficient.canonical_value() > 0.0
                    }
                },
                AnalysisInterfaceKind::FluidStructure(_) => true,
                AnalysisInterfaceKind::ConjugateHeatTransfer(value) => match value.transfer {
                    crate::ThermalInterfaceTransfer::Perfect => true,
                    crate::ThermalInterfaceTransfer::Conductance(value) => {
                        value.is_finite() && value.canonical_value() > 0.0
                    }
                    crate::ThermalInterfaceTransfer::Resistance(value) => {
                        value.is_finite() && value.canonical_value() > 0.0
                    }
                },
            };
        if !valid {
            return Err(AnalysisValidationError::InvalidInterface {
                interface_id: interface.interface_id.clone(),
                detail: "targets or interface parameters are outside their admissible range"
                    .to_string(),
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ContactFriction, ContactInterfaceModel, FrictionCoefficient, RegionTarget};

    fn contact(id: &str, primary: &str, secondary: &str) -> AnalysisInterface {
        AnalysisInterface {
            interface_id: id.into(),
            primary: RegionTarget::boundary(primary),
            secondary: RegionTarget::boundary(secondary),
            kind: AnalysisInterfaceKind::Contact(ContactInterfaceModel {
                friction: ContactFriction::Frictionless,
            }),
        }
    }

    #[test]
    fn interfaces_require_unique_ids_distinct_targets_and_valid_parameters() {
        validate_interfaces(&[contact("contact", "left", "right")]).unwrap();

        assert!(matches!(
            validate_interfaces(&[
                contact("contact", "left", "right"),
                contact("contact", "top", "bottom"),
            ]),
            Err(AnalysisValidationError::DuplicateId {
                kind: "interface",
                ..
            })
        ));
        assert!(matches!(
            validate_interfaces(&[contact("contact", "left", "left")]),
            Err(AnalysisValidationError::InvalidInterface { .. })
        ));
        let mut invalid = contact("contact", "left", "right");
        let AnalysisInterfaceKind::Contact(parameters) = &mut invalid.kind else {
            unreachable!()
        };
        parameters.friction = ContactFriction::Coulomb {
            coefficient: FrictionCoefficient::new(0.0),
        };
        assert!(matches!(
            validate_interfaces(&[invalid]),
            Err(AnalysisValidationError::InvalidInterface { .. })
        ));
    }
}
