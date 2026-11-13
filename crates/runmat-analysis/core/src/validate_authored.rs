use std::collections::HashSet;

mod connectors;
mod interfaces;

use crate::{
    materialized_motion_acceleration_is_compatible, AnalysisBoundary, AnalysisConstraint,
    AnalysisModel, AnalysisValidationError, BoundaryOperator, CoordinateFrame, EntityDimension,
    ExpectedTopology, FluidConstraintKind, MagneticConstraintKind, PairingPolicy,
    PeriodicConstraint, PeriodicTransform, PhysicalSignal, StructuralConstraintKind,
    StructuralKinematicHistory, StructuralMotionReconstruction, ThermalConstraintKind,
};

pub(crate) fn validate_authored_model(
    model: &AnalysisModel,
) -> Result<(), AnalysisValidationError> {
    validate_coordinate_frame(&model.frame)?;
    crate::validate_materials::validate_materials(&model.materials, &model.material_assignments)?;
    crate::validate_signals::validate_signals(&model.signals)?;
    crate::validate_domains::validate_domains(model)?;
    validate_constraints(&model.constraints, &model.signals)?;
    validate_boundaries(&model.boundaries, &model.signals)?;
    connectors::validate_connectors(&model.connectors)?;
    interfaces::validate_interfaces(&model.interfaces)?;
    crate::validate_excitations::validate_model_excitations(model)?;
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
            StructuralConstraintKind::PrescribedMotion(values) => {
                values.translations.is_empty()
                    || value.frame != CoordinateFrame::Global
                    || has_duplicates(values.translations.iter().map(|entry| entry.dof))
                    || values.translations.iter().any(|entry| {
                        let StructuralKinematicHistory::AccelerationDriven {
                            acceleration,
                            initial_displacement,
                            initial_velocity,
                            reconstruction:
                                StructuralMotionReconstruction::PiecewiseLinearAccelerationExact,
                        } = &entry.history;
                        !initial_displacement.is_finite()
                            || !initial_velocity.is_finite()
                            || !signals.iter().any(|signal| {
                                signal.signal_id() == &acceleration.0
                                    && materialized_motion_acceleration_is_compatible(signal)
                            })
                    })
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
                    && coefficient.canonical_value() >= 0.0
            }
            BoundaryOperator::ThermalRadiation {
                ambient_temperature,
                emissivity,
            } => {
                ambient_temperature.is_well_formed(signals)
                    && match ambient_temperature {
                        crate::PrescribedValue::Constant(value) => value.canonical_value() >= 0.0,
                        crate::PrescribedValue::Signal(_) => true,
                    }
                    && emissivity.is_finite()
                    && (0.0..=1.0).contains(&emissivity.canonical_value())
            }
            BoundaryOperator::AcousticImpedance { impedance } => {
                impedance.is_finite() && impedance.canonical_value() > 0.0
            }
            BoundaryOperator::AcousticImpedanceLaw { law } => law.is_valid(),
            BoundaryOperator::AcousticSphericalRadiation { center_m } => {
                center_m.iter().all(|value| value.is_finite())
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

pub(super) fn validate_id(
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
    if !topology_matches_dimension {
        return Err(format!(
            "entity dimension {:?} is incompatible with expected topology {:?}",
            target.entity_dimension, target.expected_topology
        ));
    }
    target
        .validate_resolution()
        .map_err(|error| error.to_string())
}

fn pairing_is_valid(pairing: &PairingPolicy) -> bool {
    match pairing {
        PairingPolicy::Matching => true,
        PairingPolicy::GeometricTolerance(value) => {
            value.is_finite() && value.canonical_value() >= 0.0
        }
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
mod prescribed_motion_tests {
    use super::*;
    use crate::{
        Acceleration, AccelerationSignalId, Displacement, PrescribedStructuralMotion, RegionTarget,
        Signal, SignalApproximationEvidence, SignalExtrapolation, SignalId, SignalInterpolation,
        SignalKnot, SignalOrdinate, SignalPrecision, SignalProvenance, SignalRepresentation,
        SignalShape, SignalValue, StructuralConstraint, StructuralKinematicHistory,
        StructuralMotionReconstruction, StructuralTranslationDof, TabulatedSignal,
        TabulatedSignalDomain, TranslationDofMotion, Velocity,
    };

    fn acceleration(shape: SignalShape) -> PhysicalSignal {
        PhysicalSignal::Acceleration(Signal {
            signal_id: SignalId("motion".into()),
            shape,
            frame: Some(CoordinateFrame::Global),
            representation: SignalRepresentation::Tabulated(TabulatedSignal {
                domain: TabulatedSignalDomain::TimeSeconds,
                knots: [0.0, 1.0]
                    .into_iter()
                    .map(|time| {
                        SignalKnot::continuous(
                            time,
                            SignalOrdinate::Real(SignalValue::Vector3([
                                Acceleration::new(1.0),
                                Acceleration::new(0.0),
                                Acceleration::new(0.0),
                            ])),
                        )
                    })
                    .collect(),
                interpolation: SignalInterpolation::Linear,
                extrapolation: SignalExtrapolation::Reject,
            }),
            precision: SignalPrecision::Real64,
            evidence: SignalApproximationEvidence {
                max_absolute_error: Some(0.0),
                max_relative_error: Some(0.0),
                derivative_max_absolute_error: None,
                sample_count: 2,
            },
            provenance: SignalProvenance::Authored,
        })
    }

    fn constraint() -> AnalysisConstraint {
        AnalysisConstraint::Structural(StructuralConstraint {
            constraint_id: "motion".into(),
            frame: CoordinateFrame::Global,
            kind: StructuralConstraintKind::PrescribedMotion(PrescribedStructuralMotion {
                target: RegionTarget::boundary("support"),
                translations: vec![TranslationDofMotion {
                    dof: StructuralTranslationDof::Ux,
                    history: StructuralKinematicHistory::AccelerationDriven {
                        acceleration: AccelerationSignalId(SignalId("motion".into())),
                        initial_displacement: Displacement::new(0.0),
                        initial_velocity: Velocity::new(0.0),
                        reconstruction:
                            StructuralMotionReconstruction::PiecewiseLinearAccelerationExact,
                    },
                }],
            }),
        })
    }

    #[test]
    fn prescribed_motion_requires_a_typed_global_vector_acceleration_history() {
        validate_constraint_values(&constraint(), &[acceleration(SignalShape::Vector3)])
            .expect("vector acceleration is the public support-motion contract");
        assert!(matches!(
            validate_constraint_values(&constraint(), &[acceleration(SignalShape::Scalar)]),
            Err(AnalysisValidationError::InvalidConstraint { .. })
        ));
        let mut local_signal = acceleration(SignalShape::Vector3);
        let PhysicalSignal::Acceleration(signal) = &mut local_signal else {
            unreachable!()
        };
        signal.frame = Some(CoordinateFrame::Material);
        assert!(matches!(
            validate_constraint_values(&constraint(), &[local_signal]),
            Err(AnalysisValidationError::InvalidConstraint { .. })
        ));
        assert!(matches!(
            validate_constraint_values(&constraint(), &[]),
            Err(AnalysisValidationError::InvalidConstraint { .. })
        ));
        let mut local = constraint();
        let AnalysisConstraint::Structural(structural) = &mut local else {
            unreachable!()
        };
        structural.frame = CoordinateFrame::Material;
        assert!(matches!(
            validate_constraint_values(&local, &[acceleration(SignalShape::Vector3)]),
            Err(AnalysisValidationError::InvalidConstraint { .. })
        ));
    }
}
