use std::collections::HashSet;

use crate::{
    AnalysisConnector, AnalysisValidationError, ConnectorKind, EntityDimension, RegionTarget,
};

use super::{validate_id, validate_target};

pub(super) fn validate_connectors(
    values: &[AnalysisConnector],
) -> Result<(), AnalysisValidationError> {
    let mut ids = HashSet::new();
    for connector in values {
        validate_id(&connector.connector_id, "connector", &mut ids)?;
        match &connector.kind {
            ConnectorKind::AxialSpring {
                primary,
                secondary,
                stiffness,
            } => {
                validate_connector_target(connector, primary, EntityDimension::Point)?;
                validate_connector_target(connector, secondary, EntityDimension::Point)?;
                if primary == secondary {
                    return Err(AnalysisValidationError::InvalidConnector {
                        connector_id: connector.connector_id.clone(),
                        detail:
                            "an axial connector requires distinct primary and secondary targets"
                                .to_string(),
                    });
                }
                validate_connector_coefficient(connector, stiffness.canonical_value())?;
            }
            ConnectorKind::AxialDashpot {
                primary,
                secondary,
                damping,
            } => {
                validate_connector_target(connector, primary, EntityDimension::Point)?;
                validate_connector_target(connector, secondary, EntityDimension::Point)?;
                if primary == secondary {
                    return Err(AnalysisValidationError::InvalidConnector {
                        connector_id: connector.connector_id.clone(),
                        detail:
                            "an axial connector requires distinct primary and secondary targets"
                                .to_string(),
                    });
                }
                validate_connector_coefficient(connector, damping.canonical_value())?;
            }
            ConnectorKind::IsotropicGroundSpring { target, stiffness } => {
                validate_connector_target(connector, target, EntityDimension::Point)?;
                validate_connector_coefficient(connector, stiffness.canonical_value())?;
            }
            ConnectorKind::SurfaceNormalFoundation { target, modulus } => {
                validate_connector_target(connector, target, EntityDimension::Face)?;
                validate_connector_coefficient(connector, modulus.canonical_value())?;
            }
        }
    }
    Ok(())
}

fn validate_connector_target(
    connector: &AnalysisConnector,
    target: &RegionTarget,
    expected_dimension: EntityDimension,
) -> Result<(), AnalysisValidationError> {
    validate_target(target).map_err(|detail| AnalysisValidationError::InvalidTarget {
        owner_id: connector.connector_id.clone(),
        detail,
    })?;
    if target.entity_dimension != expected_dimension {
        return Err(AnalysisValidationError::InvalidConnector {
            connector_id: connector.connector_id.clone(),
            detail: format!(
                "connector target must have {expected_dimension:?} dimension, found {:?}",
                target.entity_dimension
            ),
        });
    }
    Ok(())
}

fn validate_connector_coefficient(
    connector: &AnalysisConnector,
    coefficient: f64,
) -> Result<(), AnalysisValidationError> {
    if coefficient.is_finite() && coefficient >= 0.0 {
        Ok(())
    } else {
        Err(AnalysisValidationError::InvalidConnector {
            connector_id: connector.connector_id.clone(),
            detail: "coefficient must be finite and non-negative".to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Damping, FoundationModulus, Stiffness};

    fn connector(kind: ConnectorKind) -> AnalysisConnector {
        AnalysisConnector {
            connector_id: "connector".into(),
            kind,
        }
    }

    #[test]
    fn connector_attachment_semantics_are_unambiguous() {
        validate_connectors(&[connector(ConnectorKind::AxialSpring {
            primary: RegionTarget::point("node:1"),
            secondary: RegionTarget::point("node:2"),
            stiffness: Stiffness::new(1.0),
        })])
        .expect("distinct axial spring endpoints");
        validate_connectors(&[connector(ConnectorKind::AxialDashpot {
            primary: RegionTarget::point("node:1"),
            secondary: RegionTarget::point("node:2"),
            damping: Damping::new(1.0),
        })])
        .expect("distinct axial dashpot endpoints");
        validate_connectors(&[connector(ConnectorKind::IsotropicGroundSpring {
            target: RegionTarget::point("node:1"),
            stiffness: Stiffness::new(1.0),
        })])
        .expect("point ground spring");
        validate_connectors(&[connector(ConnectorKind::SurfaceNormalFoundation {
            target: RegionTarget::boundary("support"),
            modulus: FoundationModulus::new(1.0),
        })])
        .expect("face foundation");

        assert!(matches!(
            validate_connectors(&[connector(ConnectorKind::AxialSpring {
                primary: RegionTarget::point("node:1"),
                secondary: RegionTarget::point("node:1"),
                stiffness: Stiffness::new(1.0),
            })]),
            Err(AnalysisValidationError::InvalidConnector { .. })
        ));
        assert!(matches!(
            validate_connectors(&[connector(ConnectorKind::IsotropicGroundSpring {
                target: RegionTarget::boundary("face"),
                stiffness: Stiffness::new(1.0),
            })]),
            Err(AnalysisValidationError::InvalidConnector { .. })
        ));
        assert!(matches!(
            validate_connectors(&[connector(ConnectorKind::SurfaceNormalFoundation {
                target: RegionTarget::point("node:1"),
                modulus: FoundationModulus::new(1.0),
            })]),
            Err(AnalysisValidationError::InvalidConnector { .. })
        ));
        assert!(matches!(
            validate_connectors(&[connector(ConnectorKind::AxialDashpot {
                primary: RegionTarget::point("node:1"),
                secondary: RegionTarget::point("node:2"),
                damping: Damping::new(f64::NAN),
            })]),
            Err(AnalysisValidationError::InvalidConnector { .. })
        ));
    }
}
