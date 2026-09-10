use std::collections::HashSet;

use crate::{
    AnalysisBoundary, AnalysisConnector, AnalysisConstraint, AnalysisInterface,
    AnalysisInterfaceKind, AnalysisModel, AnalysisValidationError, BoundaryOperator, ConnectorKind,
    CoordinateFrame, EntityDimension, ExpectedTopology, FluidConstraintKind,
    MagneticConstraintKind, PairingPolicy, PeriodicConstraint, PeriodicTransform, PhysicalSignal,
    StructuralConstraintKind, ThermalConstraintKind, ValidatedAnalysisSequence,
};

pub(crate) fn validate_authored_model(
    model: &AnalysisModel,
) -> Result<ValidatedAnalysisSequence, AnalysisValidationError> {
    validate_coordinate_frame(&model.frame)?;
    crate::validate_materials::validate_materials(&model.materials, &model.material_assignments)?;
    crate::validate_signals::validate_signals(&model.signals)?;
    crate::validate_domains::validate_domains(model)?;
    validate_constraints(&model.constraints, &model.signals)?;
    validate_boundaries(&model.boundaries, &model.signals)?;
    validate_connectors(&model.connectors)?;
    validate_interfaces(&model.interfaces)?;
    crate::validate_excitations::validate_model_excitations(model)?;
    crate::validate_sequence::validate_sequence(model)
}

fn validate_interfaces(values: &[AnalysisInterface]) -> Result<(), AnalysisValidationError> {
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
                        coefficient.is_finite() && coefficient.ratio > 0.0
                    }
                },
                AnalysisInterfaceKind::FluidStructure(_) => true,
                AnalysisInterfaceKind::ConjugateHeatTransfer(value) => match value.transfer {
                    crate::ThermalInterfaceTransfer::Perfect => true,
                    crate::ThermalInterfaceTransfer::Conductance(value) => {
                        value.is_finite() && value.watts_per_square_meter_kelvin > 0.0
                    }
                    crate::ThermalInterfaceTransfer::Resistance(value) => {
                        value.is_finite() && value.square_meter_kelvin_per_watt > 0.0
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

pub fn validate_coordinate_frame(frame: &CoordinateFrame) -> Result<(), AnalysisValidationError> {
    let valid = match frame {
        CoordinateFrame::Global
        | CoordinateFrame::Material
        | CoordinateFrame::Element
        | CoordinateFrame::BoundaryNormalTangent => true,
        CoordinateFrame::Cartesian(frame) => {
            !frame.frame_id.trim().is_empty()
                && frame.origin.is_finite()
                && basis_pair_is_valid(frame.x_axis, frame.y_axis)
        }
        CoordinateFrame::Cylindrical(frame) => {
            !frame.frame_id.trim().is_empty()
                && frame.origin.is_finite()
                && basis_pair_is_valid(frame.axis, frame.radial_reference)
        }
        CoordinateFrame::Spherical(frame) => {
            !frame.frame_id.trim().is_empty()
                && frame.origin.is_finite()
                && basis_pair_is_valid(frame.polar_axis, frame.azimuth_reference)
        }
    };
    if valid {
        Ok(())
    } else {
        Err(AnalysisValidationError::InvalidFrame)
    }
}

fn basis_pair_is_valid(a: [f64; 3], b: [f64; 3]) -> bool {
    if !a.into_iter().chain(b).all(f64::is_finite) {
        return false;
    }
    let norm_a = dot(a, a).sqrt();
    let norm_b = dot(b, b).sqrt();
    if norm_a <= f64::EPSILON || norm_b <= f64::EPSILON {
        return false;
    }
    let cosine = (dot(a, b) / (norm_a * norm_b)).abs();
    cosine < 1.0 - 1.0e-10
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a.into_iter().zip(b).map(|(lhs, rhs)| lhs * rhs).sum()
}

fn validate_constraints(
    values: &[AnalysisConstraint],
    signals: &[PhysicalSignal],
) -> Result<(), AnalysisValidationError> {
    let mut ids = HashSet::new();
    for constraint in values {
        let id = constraint.constraint_id();
        validate_id(id, "constraint", &mut ids)?;
        let mut target_error = None;
        constraint.visit_targets(|target| {
            if target_error.is_none() {
                target_error = validate_target(target).err();
            }
        });
        if let Some(detail) = target_error {
            return Err(AnalysisValidationError::InvalidTarget {
                owner_id: id.to_string(),
                detail,
            });
        }
        validate_constraint_values(constraint, signals)?;
    }
    Ok(())
}

fn validate_constraint_values(
    value: &AnalysisConstraint,
    signals: &[PhysicalSignal],
) -> Result<(), AnalysisValidationError> {
    let invalid = match value {
        AnalysisConstraint::Structural(value) => match &value.kind {
            StructuralConstraintKind::Prescribed(values) => {
                (values.translations.is_empty() && values.rotations.is_empty())
                    || has_duplicates(values.translations.iter().map(|entry| entry.dof))
                    || has_duplicates(values.rotations.iter().map(|entry| entry.dof))
                    || values
                        .translations
                        .iter()
                        .any(|entry| !entry.value.is_well_formed(signals))
                    || values
                        .rotations
                        .iter()
                        .any(|entry| !entry.value.is_well_formed(signals))
            }
            StructuralConstraintKind::NormalDisplacement { value, .. } => {
                !value.is_well_formed(signals)
            }
            StructuralConstraintKind::RigidCoupling(value)
            | StructuralConstraintKind::DistributingCoupling(value) => {
                value.reference_point.reference_point_id.trim().is_empty()
                    || !value.reference_point.position.is_finite()
            }
            StructuralConstraintKind::Tie(pair) => !pairing_is_valid(&pair.pairing),
            StructuralConstraintKind::Periodic(periodic) => !periodic_is_valid(periodic),
            StructuralConstraintKind::Equation(equation) => {
                equation.terms.is_empty()
                    || !equation.right_hand_side.is_finite()
                    || equation
                        .terms
                        .iter()
                        .any(|term| !term.coefficient.is_finite())
            }
            _ => false,
        },
        AnalysisConstraint::Thermal(value) => match &value.kind {
            ThermalConstraintKind::Prescribed { value, .. } => !value.is_well_formed(signals),
            ThermalConstraintKind::Periodic(periodic) => !periodic_is_valid(periodic),
            ThermalConstraintKind::Tie(pair) => !pairing_is_valid(&pair.pairing),
            ThermalConstraintKind::AffineJump {
                periodic,
                temperature_jump,
            } => !periodic_is_valid(periodic) || !temperature_jump.is_well_formed(signals),
        },
        AnalysisConstraint::Electrical(value) => match &value.kind {
            crate::ElectricalConstraintKind::PrescribedPotential { value, .. } => {
                !value.is_well_formed(signals)
            }
            crate::ElectricalConstraintKind::IntegralCurrent { value, .. } => {
                !value.is_well_formed(signals)
            }
            crate::ElectricalConstraintKind::Periodic(periodic) => !periodic_is_valid(periodic),
            _ => false,
        },
        AnalysisConstraint::Magnetic(value) => match &value.kind {
            MagneticConstraintKind::PrescribedVectorPotential { value, .. } => {
                !value.is_well_formed(signals)
            }
            MagneticConstraintKind::Periodic(periodic) => !periodic_is_valid(periodic),
            MagneticConstraintKind::PhaseShiftedPeriodic(periodic) => {
                !periodic_is_valid(&periodic.periodic) || !periodic.phase.is_finite()
            }
            _ => false,
        },
        AnalysisConstraint::Acoustic(value) => match &value.kind {
            crate::AcousticConstraintKind::PrescribedPressure { value, .. } => {
                !value.is_well_formed(signals)
            }
            crate::AcousticConstraintKind::Periodic(periodic) => !periodic_is_valid(periodic),
            crate::AcousticConstraintKind::PhaseShiftedPeriodic(periodic) => {
                !periodic_is_valid(&periodic.periodic) || !periodic.phase.is_finite()
            }
        },
        AnalysisConstraint::Fluid(value) => match &value.kind {
            FluidConstraintKind::PrescribedVelocity { value, .. }
            | FluidConstraintKind::MovingWall {
                velocity: value, ..
            } => !value.is_well_formed(signals),
            FluidConstraintKind::PressureReference { value, .. } => !value.is_well_formed(signals),
            FluidConstraintKind::Periodic(periodic) => !periodic_is_valid(periodic),
            _ => false,
        },
    };
    if invalid {
        Err(AnalysisValidationError::InvalidConstraint {
            constraint_id: value.constraint_id().to_string(),
            detail: "prescribed values, pairing, or degrees of freedom are invalid".to_string(),
        })
    } else {
        Ok(())
    }
}

fn validate_boundaries(
    values: &[AnalysisBoundary],
    signals: &[PhysicalSignal],
) -> Result<(), AnalysisValidationError> {
    let mut ids = HashSet::new();
    for boundary in values {
        validate_id(&boundary.boundary_id, "boundary", &mut ids)?;
        validate_target(&boundary.target).map_err(|detail| {
            AnalysisValidationError::InvalidTarget {
                owner_id: boundary.boundary_id.clone(),
                detail,
            }
        })?;
        let valid = match &boundary.operator {
            BoundaryOperator::StructuralPressure { value }
            | BoundaryOperator::FluidOutletPressure { value } => value.is_well_formed(signals),
            BoundaryOperator::ThermalHeatFlux { value } => value.is_well_formed(signals),
            BoundaryOperator::ThermalConvection {
                ambient_temperature,
                coefficient,
            } => {
                ambient_temperature.is_well_formed(signals)
                    && coefficient.is_finite()
                    && coefficient.watts_per_square_meter_kelvin >= 0.0
            }
            BoundaryOperator::AcousticImpedance { impedance } => {
                impedance.is_finite() && impedance.pascal_seconds_per_meter > 0.0
            }
            _ => true,
        };
        if !valid {
            return Err(AnalysisValidationError::InvalidBoundary {
                boundary_id: boundary.boundary_id.clone(),
                detail: "boundary operator or prescribed value is invalid".to_string(),
            });
        }
    }
    Ok(())
}

fn validate_connectors(values: &[AnalysisConnector]) -> Result<(), AnalysisValidationError> {
    let mut ids = HashSet::new();
    for connector in values {
        validate_id(&connector.connector_id, "connector", &mut ids)?;
        validate_target(&connector.primary).map_err(|detail| {
            AnalysisValidationError::InvalidTarget {
                owner_id: connector.connector_id.clone(),
                detail,
            }
        })?;
        if let Some(secondary) = &connector.secondary {
            validate_target(secondary).map_err(|detail| {
                AnalysisValidationError::InvalidTarget {
                    owner_id: connector.connector_id.clone(),
                    detail,
                }
            })?;
        }
        match (&connector.kind, &connector.secondary) {
            (ConnectorKind::ElasticFoundation { .. }, Some(_)) => {
                return Err(AnalysisValidationError::InvalidConnector {
                    connector_id: connector.connector_id.clone(),
                    detail: "an elastic foundation is attached to ground and cannot have a secondary target"
                        .to_string(),
                });
            }
            (
                ConnectorKind::TranslationalSpring { .. }
                | ConnectorKind::TranslationalDashpot { .. },
                Some(secondary),
            ) if secondary == &connector.primary => {
                return Err(AnalysisValidationError::InvalidConnector {
                    connector_id: connector.connector_id.clone(),
                    detail: "a two-ended connector requires distinct primary and secondary targets"
                        .to_string(),
                });
            }
            _ => {}
        }
        let valid = match connector.kind {
            ConnectorKind::TranslationalSpring { stiffness }
            | ConnectorKind::ElasticFoundation { stiffness } => {
                stiffness.is_finite() && stiffness.newtons_per_meter >= 0.0
            }
            ConnectorKind::TranslationalDashpot { damping } => {
                damping.is_finite() && damping.newton_seconds_per_meter >= 0.0
            }
        };
        if !valid {
            return Err(AnalysisValidationError::InvalidConnector {
                connector_id: connector.connector_id.clone(),
                detail: "coefficient must be finite and non-negative".to_string(),
            });
        }
    }
    Ok(())
}

fn validate_id(
    id: &str,
    kind: &'static str,
    seen: &mut HashSet<String>,
) -> Result<(), AnalysisValidationError> {
    if id.trim().is_empty() {
        return Err(AnalysisValidationError::InvalidId { kind });
    }
    if !seen.insert(id.to_string()) {
        return Err(AnalysisValidationError::DuplicateId {
            kind,
            id: id.to_string(),
        });
    }
    Ok(())
}

pub(crate) fn validate_target(target: &crate::RegionTarget) -> Result<(), String> {
    if target.region_id.trim().is_empty() {
        return Err("region id is empty".to_string());
    }
    let topology_matches_dimension = matches!(
        (target.entity_dimension, target.expected_topology),
        (
            EntityDimension::Point,
            ExpectedTopology::GeometryEntity | ExpectedTopology::MeshNode
        ) | (EntityDimension::Edge, ExpectedTopology::GeometryEntity)
            | (
                EntityDimension::Face,
                ExpectedTopology::GeometryEntity | ExpectedTopology::BoundaryFace
            )
            | (
                EntityDimension::Volume,
                ExpectedTopology::GeometryEntity
                    | ExpectedTopology::MeshElement
                    | ExpectedTopology::Cell
            )
    );
    if topology_matches_dimension {
        Ok(())
    } else {
        Err(format!(
            "entity dimension {:?} is incompatible with expected topology {:?}",
            target.entity_dimension, target.expected_topology
        ))
    }
}

fn pairing_is_valid(pairing: &PairingPolicy) -> bool {
    match pairing {
        PairingPolicy::Matching => true,
        PairingPolicy::GeometricTolerance(value) => value.is_finite() && value.meters >= 0.0,
    }
}

fn periodic_is_valid(periodic: &PeriodicConstraint) -> bool {
    pairing_is_valid(&periodic.pair.pairing)
        && match periodic.transform {
            PeriodicTransform::Identity => true,
            PeriodicTransform::Translation(value) => value.is_finite(),
            PeriodicTransform::Rotation {
                origin,
                axis,
                angle,
            } => {
                origin.is_finite()
                    && angle.is_finite()
                    && axis.into_iter().all(f64::is_finite)
                    && axis.into_iter().any(|value| value != 0.0)
            }
        }
}

fn has_duplicates<T: Eq + std::hash::Hash>(values: impl IntoIterator<Item = T>) -> bool {
    let mut seen = HashSet::new();
    values.into_iter().any(|value| !seen.insert(value))
}

#[cfg(test)]
mod interface_tests {
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

#[cfg(test)]
mod connector_tests {
    use super::*;
    use crate::{Damping, RegionTarget, Stiffness};

    fn connector(kind: ConnectorKind, secondary: Option<&str>) -> AnalysisConnector {
        AnalysisConnector {
            connector_id: "connector".into(),
            primary: RegionTarget::point("node:1"),
            secondary: secondary.map(RegionTarget::point),
            kind,
        }
    }

    #[test]
    fn connector_attachment_semantics_are_unambiguous() {
        validate_connectors(&[connector(
            ConnectorKind::TranslationalSpring {
                stiffness: Stiffness::new(1.0),
            },
            Some("node:2"),
        )])
        .expect("distinct two-ended spring");
        validate_connectors(&[connector(
            ConnectorKind::TranslationalDashpot {
                damping: Damping::new(1.0),
            },
            None,
        )])
        .expect("ground dashpot");
        validate_connectors(&[connector(
            ConnectorKind::ElasticFoundation {
                stiffness: Stiffness::new(1.0),
            },
            None,
        )])
        .expect("ground foundation");

        assert!(matches!(
            validate_connectors(&[connector(
                ConnectorKind::TranslationalSpring {
                    stiffness: Stiffness::new(1.0),
                },
                Some("node:1"),
            )]),
            Err(AnalysisValidationError::InvalidConnector { .. })
        ));
        assert!(matches!(
            validate_connectors(&[connector(
                ConnectorKind::ElasticFoundation {
                    stiffness: Stiffness::new(1.0),
                },
                Some("node:2"),
            )]),
            Err(AnalysisValidationError::InvalidConnector { .. })
        ));
    }
}
