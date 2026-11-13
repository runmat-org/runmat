use serde::{Deserialize, Serialize};

use super::{
    authored_inputs::AuthoredInputType,
    model_authoring::AuthoredRegionSource,
    prescribed_value::PrescribedValue,
    quantity::{
        ElectricCurrent, ElectricPotential, MagneticVectorPotential, Pressure, Temperature,
        Velocity,
    },
    signal::PhysicalSignal,
};
use runmat_geometry_core::GeometryRegionRole;

mod motion;
mod relational;

pub use motion::{
    authored_motion_acceleration_is_compatible, materialized_motion_acceleration_is_compatible,
    AuthoredPrescribedStructuralMotion,
};
pub use relational::{
    AuthoredPairedConstraint, AuthoredPairingPolicy, AuthoredPeriodicConstraint,
    AuthoredPeriodicTransform, AuthoredPhaseShiftedPeriodicConstraint,
    AuthoredReferencePointCoupling, AuthoredRotationDofValue, AuthoredStructuralConstraintEquation,
    AuthoredStructuralConstraintEquationTerm, AuthoredStructuralDofs, AuthoredThermalJump,
    AuthoredTranslationDofValue,
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AuthoredConstraint {
    pub constraint_id: String,
    pub kind: AuthoredConstraintKind,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum AuthoredConstraintKind {
    StructuralFixed {
        target: AuthoredConstraintTarget,
    },
    StructuralSymmetry {
        target: AuthoredConstraintTarget,
    },
    StructuralAntisymmetry {
        target: AuthoredConstraintTarget,
    },
    StructuralPrescribed(AuthoredStructuralDofs),
    StructuralPrescribedMotion(AuthoredPrescribedStructuralMotion),
    StructuralNormalDisplacement {
        target: AuthoredConstraintTarget,
        value: PrescribedValue<super::quantity::Displacement>,
    },
    StructuralRigidCoupling(AuthoredReferencePointCoupling),
    StructuralDistributingCoupling(AuthoredReferencePointCoupling),
    StructuralTie(AuthoredPairedConstraint),
    StructuralPeriodic(AuthoredPeriodicConstraint),
    StructuralEquation(AuthoredStructuralConstraintEquation),
    ThermalTemperature {
        target: AuthoredConstraintTarget,
        value: PrescribedValue<Temperature>,
    },
    ThermalPeriodic(AuthoredPeriodicConstraint),
    ThermalTie(AuthoredPairedConstraint),
    ThermalAffineJump(AuthoredThermalJump),
    ElectricalPotential {
        target: AuthoredConstraintTarget,
        value: PrescribedValue<ElectricPotential>,
    },
    ElectricalEquipotential {
        target: AuthoredConstraintTarget,
    },
    ElectricalFloatingConductor {
        target: AuthoredConstraintTarget,
    },
    ElectricalIntegralCurrent {
        target: AuthoredConstraintTarget,
        value: PrescribedValue<ElectricCurrent>,
    },
    ElectricalPotentialReference {
        target: AuthoredConstraintTarget,
    },
    ElectricalPeriodic(AuthoredPeriodicConstraint),
    ElectricPerfectConductor {
        target: AuthoredConstraintTarget,
    },
    MagneticVectorPotential {
        target: AuthoredConstraintTarget,
        value: PrescribedValue<MagneticVectorPotential>,
    },
    MagneticVectorPotentialGround {
        target: AuthoredConstraintTarget,
    },
    MagneticCoulombGauge {
        target: AuthoredConstraintTarget,
    },
    MagneticPeriodic(AuthoredPeriodicConstraint),
    MagneticPhaseShiftedPeriodic(AuthoredPhaseShiftedPeriodicConstraint),
    AcousticPressure {
        target: AuthoredConstraintTarget,
        value: PrescribedValue<Pressure>,
    },
    AcousticPeriodic(AuthoredPeriodicConstraint),
    AcousticPhaseShiftedPeriodic(AuthoredPhaseShiftedPeriodicConstraint),
    FluidVelocity {
        target: AuthoredConstraintTarget,
        value: PrescribedValue<[Velocity; 3]>,
    },
    FluidMovingWall {
        target: AuthoredConstraintTarget,
        velocity: PrescribedValue<[Velocity; 3]>,
    },
    FluidNoSlipWall {
        target: AuthoredConstraintTarget,
    },
    FluidSlipWall {
        target: AuthoredConstraintTarget,
    },
    FluidSymmetry {
        target: AuthoredConstraintTarget,
    },
    FluidPressureReference {
        target: AuthoredConstraintTarget,
        value: PrescribedValue<Pressure>,
    },
    FluidPeriodic(AuthoredPeriodicConstraint),
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AuthoredConstraintTarget {
    pub region: AuthoredRegionSource,
    pub cardinality: AuthoredTargetCardinality,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuthoredTargetCardinality {
    One,
    OneOrMore,
}

impl AuthoredConstraint {
    pub fn validate(&self) -> Result<(), AuthoredConstraintError> {
        if self.constraint_id.trim().is_empty() {
            return Err(AuthoredConstraintError::EmptyId);
        }
        let mut target_error = None;
        self.kind.visit_targets(|target| {
            if target_error.is_some() {
                return;
            }
            target_error = target.validate().err().or_else(|| {
                let boundary_required = !matches!(
                    self.kind,
                    AuthoredConstraintKind::MagneticCoulombGauge { .. }
                );
                if boundary_required && target.role() == GeometryRegionRole::Solid {
                    Some(AuthoredConstraintError::BoundaryRequired)
                } else if target.role() == GeometryRegionRole::Cell {
                    Some(AuthoredConstraintError::UnsupportedCellTarget)
                } else {
                    None
                }
            });
        });
        if let Some(error) = target_error {
            return Err(error);
        }
        self.kind.validate_value()?;
        relational::validate(self)
    }

    pub fn validate_signal_references(
        &self,
        signals: &[PhysicalSignal],
    ) -> Result<(), AuthoredConstraintError> {
        let valid = match &self.kind {
            AuthoredConstraintKind::StructuralPrescribed(values) => {
                values
                    .translations
                    .iter()
                    .all(|entry| entry.value.is_well_formed(signals))
                    && values
                        .rotations
                        .iter()
                        .all(|entry| entry.value.is_well_formed(signals))
            }
            AuthoredConstraintKind::StructuralPrescribedMotion(motion) => {
                motion.translations.iter().all(|translation| {
                    let super::constraint::StructuralKinematicHistory::AccelerationDriven {
                        acceleration,
                        ..
                    } = &translation.history;
                    signals.iter().any(|signal| {
                        signal.signal_id() == &acceleration.0
                            && authored_motion_acceleration_is_compatible(signal)
                    })
                })
            }
            AuthoredConstraintKind::StructuralNormalDisplacement { value, .. } => {
                value.is_well_formed(signals)
            }
            AuthoredConstraintKind::ThermalTemperature { value, .. } => {
                value.is_well_formed(signals)
            }
            AuthoredConstraintKind::ElectricalPotential { value, .. } => {
                value.is_well_formed(signals)
            }
            AuthoredConstraintKind::ElectricalIntegralCurrent { value, .. } => {
                value.is_well_formed(signals)
            }
            AuthoredConstraintKind::MagneticVectorPotential { value, .. } => {
                value.is_well_formed(signals)
            }
            AuthoredConstraintKind::AcousticPressure { value, .. }
            | AuthoredConstraintKind::FluidPressureReference { value, .. } => {
                value.is_well_formed(signals)
            }
            AuthoredConstraintKind::FluidVelocity { value, .. } => value.is_well_formed(signals),
            AuthoredConstraintKind::FluidMovingWall { velocity, .. } => {
                velocity.is_well_formed(signals)
            }
            AuthoredConstraintKind::ThermalAffineJump(value) => {
                value.temperature_jump.is_well_formed(signals)
            }
            _ => true,
        };
        valid
            .then_some(())
            .ok_or(AuthoredConstraintError::InvalidSignalReference)
    }
}

impl AuthoredConstraintKind {
    pub fn visit_targets(&self, mut visit: impl FnMut(&AuthoredConstraintTarget)) {
        match self {
            Self::StructuralFixed { target }
            | Self::StructuralSymmetry { target }
            | Self::StructuralAntisymmetry { target }
            | Self::ThermalTemperature { target, .. }
            | Self::ElectricalPotential { target, .. }
            | Self::ElectricalEquipotential { target }
            | Self::ElectricalFloatingConductor { target }
            | Self::ElectricalIntegralCurrent { target, .. }
            | Self::ElectricalPotentialReference { target }
            | Self::ElectricPerfectConductor { target }
            | Self::MagneticVectorPotential { target, .. }
            | Self::MagneticVectorPotentialGround { target }
            | Self::MagneticCoulombGauge { target }
            | Self::AcousticPressure { target, .. }
            | Self::FluidVelocity { target, .. }
            | Self::FluidMovingWall { target, .. }
            | Self::FluidNoSlipWall { target }
            | Self::FluidSlipWall { target }
            | Self::FluidSymmetry { target }
            | Self::FluidPressureReference { target, .. } => visit(target),
            Self::StructuralPrescribed(value) => visit(&value.target),
            Self::StructuralPrescribedMotion(value) => visit(&value.target),
            Self::StructuralNormalDisplacement { target, .. } => visit(target),
            Self::StructuralRigidCoupling(value) | Self::StructuralDistributingCoupling(value) => {
                visit(&value.target)
            }
            Self::StructuralTie(pair) | Self::ThermalTie(pair) => {
                visit(&pair.primary);
                visit(&pair.secondary);
            }
            Self::StructuralPeriodic(periodic)
            | Self::ThermalPeriodic(periodic)
            | Self::ElectricalPeriodic(periodic)
            | Self::MagneticPeriodic(periodic)
            | Self::AcousticPeriodic(periodic)
            | Self::FluidPeriodic(periodic) => {
                visit(&periodic.pair.primary);
                visit(&periodic.pair.secondary);
            }
            Self::ThermalAffineJump(value) => {
                visit(&value.periodic.pair.primary);
                visit(&value.periodic.pair.secondary);
            }
            Self::MagneticPhaseShiftedPeriodic(value)
            | Self::AcousticPhaseShiftedPeriodic(value) => {
                visit(&value.periodic.pair.primary);
                visit(&value.periodic.pair.secondary);
            }
            Self::StructuralEquation(equation) => {
                for term in &equation.terms {
                    visit(&term.target);
                }
            }
        }
    }

    pub fn visit_targets_mut(&mut self, mut visit: impl FnMut(&mut AuthoredConstraintTarget)) {
        match self {
            Self::StructuralFixed { target }
            | Self::StructuralSymmetry { target }
            | Self::StructuralAntisymmetry { target }
            | Self::ThermalTemperature { target, .. }
            | Self::ElectricalPotential { target, .. }
            | Self::ElectricalEquipotential { target }
            | Self::ElectricalFloatingConductor { target }
            | Self::ElectricalIntegralCurrent { target, .. }
            | Self::ElectricalPotentialReference { target }
            | Self::ElectricPerfectConductor { target }
            | Self::MagneticVectorPotential { target, .. }
            | Self::MagneticVectorPotentialGround { target }
            | Self::MagneticCoulombGauge { target }
            | Self::AcousticPressure { target, .. }
            | Self::FluidVelocity { target, .. }
            | Self::FluidMovingWall { target, .. }
            | Self::FluidNoSlipWall { target }
            | Self::FluidSlipWall { target }
            | Self::FluidSymmetry { target }
            | Self::FluidPressureReference { target, .. } => visit(target),
            Self::StructuralPrescribed(value) => visit(&mut value.target),
            Self::StructuralPrescribedMotion(value) => visit(&mut value.target),
            Self::StructuralNormalDisplacement { target, .. } => visit(target),
            Self::StructuralRigidCoupling(value) | Self::StructuralDistributingCoupling(value) => {
                visit(&mut value.target)
            }
            Self::StructuralTie(pair) | Self::ThermalTie(pair) => {
                visit(&mut pair.primary);
                visit(&mut pair.secondary);
            }
            Self::StructuralPeriodic(periodic)
            | Self::ThermalPeriodic(periodic)
            | Self::ElectricalPeriodic(periodic)
            | Self::MagneticPeriodic(periodic)
            | Self::AcousticPeriodic(periodic)
            | Self::FluidPeriodic(periodic) => {
                visit(&mut periodic.pair.primary);
                visit(&mut periodic.pair.secondary);
            }
            Self::ThermalAffineJump(value) => {
                visit(&mut value.periodic.pair.primary);
                visit(&mut value.periodic.pair.secondary);
            }
            Self::MagneticPhaseShiftedPeriodic(value)
            | Self::AcousticPhaseShiftedPeriodic(value) => {
                visit(&mut value.periodic.pair.primary);
                visit(&mut value.periodic.pair.secondary);
            }
            Self::StructuralEquation(equation) => {
                for term in &mut equation.terms {
                    visit(&mut term.target);
                }
            }
        }
    }

    fn validate_value(&self) -> Result<(), AuthoredConstraintError> {
        let finite = match self {
            Self::ThermalTemperature {
                value: PrescribedValue::Constant(value),
                ..
            } => value.canonical_value().is_finite() && value.canonical_value() >= 0.0,
            Self::ElectricalPotential {
                value: PrescribedValue::Constant(value),
                ..
            } => value.canonical_value().is_finite(),
            Self::ElectricalIntegralCurrent {
                value: PrescribedValue::Constant(value),
                ..
            } => value.canonical_value().is_finite(),
            Self::MagneticVectorPotential {
                value: PrescribedValue::Constant(value),
                ..
            } => value.canonical_value().is_finite(),
            Self::AcousticPressure {
                value: PrescribedValue::Constant(value),
                ..
            }
            | Self::FluidPressureReference {
                value: PrescribedValue::Constant(value),
                ..
            } => value.canonical_value().is_finite(),
            Self::FluidVelocity {
                value: PrescribedValue::Constant(value),
                ..
            }
            | Self::FluidMovingWall {
                velocity: PrescribedValue::Constant(value),
                ..
            } => value.iter().all(|component| component.is_finite()),
            Self::StructuralNormalDisplacement {
                value: PrescribedValue::Constant(value),
                ..
            } => value.canonical_value().is_finite(),
            Self::ThermalAffineJump(AuthoredThermalJump {
                temperature_jump: PrescribedValue::Constant(value),
                ..
            }) => value.canonical_value().is_finite(),
            _ => true,
        };
        if finite {
            Ok(())
        } else {
            Err(AuthoredConstraintError::InvalidValue)
        }
    }
}

impl AuthoredConstraintTarget {
    pub fn role(&self) -> GeometryRegionRole {
        match &self.region {
            AuthoredRegionSource::Geometry(region) => region.role,
            AuthoredRegionSource::ModelInput(input) => match input.value_type {
                AuthoredInputType::Region(role) => role,
                _ => GeometryRegionRole::Vertex,
            },
        }
    }

    fn validate(&self) -> Result<(), AuthoredConstraintError> {
        match &self.region {
            AuthoredRegionSource::Geometry(region) => {
                if region.module_path.trim().is_empty()
                    || region.geometry_id.trim().is_empty()
                    || region.region_id.trim().is_empty()
                {
                    return Err(AuthoredConstraintError::InvalidTarget);
                }
            }
            AuthoredRegionSource::ModelInput(input) => {
                if input.input_id.trim().is_empty()
                    || !matches!(input.value_type, AuthoredInputType::Region(_))
                {
                    return Err(AuthoredConstraintError::InvalidTarget);
                }
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum AuthoredConstraintError {
    #[error("constraint id must not be empty")]
    EmptyId,
    #[error("constraint target is invalid")]
    InvalidTarget,
    #[error("constraint kind requires boundary topology rather than a solid domain")]
    BoundaryRequired,
    #[error("cell targets are not supported by this essential-constraint surface")]
    UnsupportedCellTarget,
    #[error("constraint value is non-finite or outside its physical domain")]
    InvalidValue,
    #[error("constraint references an incompatible or unknown physical signal")]
    InvalidSignalReference,
    #[error("constraint frame is invalid")]
    InvalidFrame,
    #[error("prescribed constraint requires at least one degree of freedom")]
    MissingDofs,
    #[error("prescribed constraint repeats a degree of freedom")]
    DuplicateDof,
    #[error("paired constraint targets must be distinct")]
    IdenticalTargets,
    #[error("periodic transform is invalid")]
    InvalidTransform,
    #[error("constraint equation requires at least one term")]
    MissingTerms,
    #[error("constraint equation coefficients must not be zero")]
    ZeroCoefficient,
}
