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
    ThermalTemperature {
        target: AuthoredConstraintTarget,
        value: PrescribedValue<Temperature>,
    },
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
    AcousticPressure {
        target: AuthoredConstraintTarget,
        value: PrescribedValue<Pressure>,
    },
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
        let target = self.kind.target();
        target.validate()?;
        let boundary_required = !matches!(
            self.kind,
            AuthoredConstraintKind::MagneticCoulombGauge { .. }
        );
        if boundary_required && target.role() == GeometryRegionRole::Solid {
            return Err(AuthoredConstraintError::BoundaryRequired);
        }
        if matches!(target.role(), GeometryRegionRole::Cell) {
            return Err(AuthoredConstraintError::UnsupportedCellTarget);
        }
        self.kind.validate_value()
    }
}

impl AuthoredConstraintKind {
    pub fn target(&self) -> &AuthoredConstraintTarget {
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
            | Self::FluidPressureReference { target, .. } => target,
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
}
