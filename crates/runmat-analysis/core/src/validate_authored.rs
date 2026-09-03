use std::collections::HashSet;

use crate::{
    AnalysisBoundary, AnalysisConnector, AnalysisConstraint, AnalysisModel,
    AnalysisValidationError, BoundaryOperator, ConnectorKind, CoordinateFrame, EntityDimension,
    ExpectedTopology, FluidConstraintKind, MagneticConstraintKind, PairingPolicy,
    StructuralConstraintKind, ThermalConstraintKind,
};

pub(crate) fn validate_authored_model(
    model: &AnalysisModel,
) -> Result<(), AnalysisValidationError> {
    validate_frame(&model.frame)?;
    validate_constraints(&model.constraints)?;
    validate_boundaries(&model.boundaries)?;
    validate_connectors(&model.connectors)?;
    Ok(())
}

fn validate_frame(frame: &CoordinateFrame) -> Result<(), AnalysisValidationError> {
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

fn validate_constraints(values: &[AnalysisConstraint]) -> Result<(), AnalysisValidationError> {
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
        validate_constraint_values(constraint)?;
    }
    Ok(())
}

fn validate_constraint_values(value: &AnalysisConstraint) -> Result<(), AnalysisValidationError> {
    let invalid = match value {
        AnalysisConstraint::Structural(value) => match &value.kind {
            StructuralConstraintKind::PrescribedTranslations(values) => {
                values.values.is_empty()
                    || has_duplicates(values.values.iter().map(|entry| entry.dof))
                    || values
                        .values
                        .iter()
                        .any(|entry| !entry.value.is_well_formed())
            }
            StructuralConstraintKind::PrescribedRotations(values) => {
                values.values.is_empty()
                    || has_duplicates(values.values.iter().map(|entry| entry.dof))
                    || values
                        .values
                        .iter()
                        .any(|entry| !entry.value.is_well_formed())
            }
            StructuralConstraintKind::NormalDisplacement { value, .. } => !value.is_well_formed(),
            StructuralConstraintKind::RigidCoupling(value)
            | StructuralConstraintKind::DistributingCoupling(value) => {
                value.reference_point.reference_point_id.trim().is_empty()
                    || !value.reference_point.position.is_finite()
            }
            StructuralConstraintKind::Tie(pair) | StructuralConstraintKind::Periodic(pair) => {
                !pairing_is_valid(&pair.pairing)
            }
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
            ThermalConstraintKind::Prescribed { value, .. } => !value.is_well_formed(),
            ThermalConstraintKind::Periodic(pair) | ThermalConstraintKind::Tie(pair) => {
                !pairing_is_valid(&pair.pairing)
            }
            ThermalConstraintKind::AffineJump {
                pair,
                temperature_jump,
            } => !pairing_is_valid(&pair.pairing) || !temperature_jump.is_well_formed(),
        },
        AnalysisConstraint::Electrical(value) => match &value.kind {
            crate::ElectricalConstraintKind::PrescribedPotential { value, .. } => {
                !value.is_well_formed()
            }
            crate::ElectricalConstraintKind::IntegralCurrent { value, .. } => {
                !value.is_well_formed()
            }
            crate::ElectricalConstraintKind::Periodic(pair) => !pairing_is_valid(&pair.pairing),
            _ => false,
        },
        AnalysisConstraint::Magnetic(value) => match &value.kind {
            MagneticConstraintKind::PrescribedVectorPotential { value, .. } => {
                !value.is_well_formed()
            }
            MagneticConstraintKind::Periodic(pair) => !pairing_is_valid(&pair.pairing),
            _ => false,
        },
        AnalysisConstraint::Acoustic(value) => match &value.kind {
            crate::AcousticConstraintKind::PrescribedPressure { value, .. } => {
                !value.is_well_formed()
            }
            crate::AcousticConstraintKind::Periodic(pair) => !pairing_is_valid(&pair.pairing),
        },
        AnalysisConstraint::Fluid(value) => match &value.kind {
            FluidConstraintKind::PrescribedVelocity { value, .. }
            | FluidConstraintKind::MovingWall {
                velocity: value, ..
            } => !value.is_well_formed(),
            FluidConstraintKind::PressureReference { value, .. } => !value.is_well_formed(),
            FluidConstraintKind::Periodic(pair) => !pairing_is_valid(&pair.pairing),
            _ => false,
        },
    };
    if invalid {
        Err(AnalysisValidationError::InvalidConstraint {
            constraint_id: value.constraint_id().to_string(),
            detail: "values, schedules, pairing, or degrees of freedom are invalid".to_string(),
        })
    } else {
        Ok(())
    }
}

fn validate_boundaries(values: &[AnalysisBoundary]) -> Result<(), AnalysisValidationError> {
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
            | BoundaryOperator::FluidOutletPressure { value } => value.is_well_formed(),
            BoundaryOperator::ThermalHeatFlux { value } => value.is_well_formed(),
            BoundaryOperator::ThermalConvection {
                ambient_temperature,
                coefficient,
            } => {
                ambient_temperature.is_well_formed()
                    && coefficient.is_finite()
                    && coefficient.watts_per_square_meter_kelvin >= 0.0
            }
            BoundaryOperator::AcousticImpedance { impedance } => {
                impedance.is_finite() && impedance.pascal_seconds_per_meter > 0.0
            }
            BoundaryOperator::FluidInletVelocity { value } => value.is_well_formed(),
            _ => true,
        };
        if !valid {
            return Err(AnalysisValidationError::InvalidBoundary {
                boundary_id: boundary.boundary_id.clone(),
                detail: "operator values or schedule are invalid".to_string(),
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

fn validate_target(target: &crate::RegionTarget) -> Result<(), String> {
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

fn has_duplicates<T: Eq + std::hash::Hash>(values: impl IntoIterator<Item = T>) -> bool {
    let mut seen = HashSet::new();
    values.into_iter().any(|value| !seen.insert(value))
}
