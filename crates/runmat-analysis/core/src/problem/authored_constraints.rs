use serde::{Deserialize, Serialize};

use super::{
    model_authoring::{AuthoredModelInputKind, AuthoredRegionSource},
    prescribed_value::PrescribedValue,
    quantity::{
        ElectricCurrent, ElectricPotential, MagneticVectorPotential, Pressure, Temperature,
        Velocity,
    },
};
use runmat_geometry_core::GeometryRegionRole;

mod relational;

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
        value: PrescribedValue<Velocity>,
    },
    FluidMovingWall {
        target: AuthoredConstraintTarget,
        velocity: PrescribedValue<Velocity>,
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

    fn validate_value(&self) -> Result<(), AuthoredConstraintError> {
        let finite = match self {
            Self::ThermalTemperature {
                value: PrescribedValue::Constant(value),
                ..
            } => value.kelvin.is_finite() && value.kelvin >= 0.0,
            Self::ElectricalPotential {
                value: PrescribedValue::Constant(value),
                ..
            } => value.volts.is_finite(),
            Self::ElectricalIntegralCurrent {
                value: PrescribedValue::Constant(value),
                ..
            } => value.amperes.is_finite(),
            Self::MagneticVectorPotential {
                value: PrescribedValue::Constant(value),
                ..
            } => value.webers_per_meter.is_finite(),
            Self::AcousticPressure {
                value: PrescribedValue::Constant(value),
                ..
            }
            | Self::FluidPressureReference {
                value: PrescribedValue::Constant(value),
                ..
            } => value.pascals.is_finite(),
            Self::FluidVelocity {
                value: PrescribedValue::Constant(value),
                ..
            }
            | Self::FluidMovingWall {
                velocity: PrescribedValue::Constant(value),
                ..
            } => value.meters_per_second.is_finite(),
            Self::StructuralNormalDisplacement {
                value: PrescribedValue::Constant(value),
                ..
            } => value.meters.is_finite(),
            Self::ThermalAffineJump(AuthoredThermalJump {
                temperature_jump: PrescribedValue::Constant(value),
                ..
            }) => value.kelvin.is_finite(),
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
            AuthoredRegionSource::ModelInput(input) => match input.kind {
                AuthoredModelInputKind::Region(role) => role,
                AuthoredModelInputKind::Material => GeometryRegionRole::Vertex,
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
                    || !matches!(input.kind, AuthoredModelInputKind::Region(_))
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
