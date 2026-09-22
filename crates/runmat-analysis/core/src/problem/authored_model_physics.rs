use serde::{Deserialize, Serialize};

use super::{
    model_authoring::{AuthoredModelError, AuthoredRegionSource},
    prescribed_value::PrescribedValue,
    quantity::{
        Damping, FoundationModulus, FrictionCoefficient, HeatFlux, HeatTransferCoefficient,
        Pressure, SpecificAcousticImpedance, Stiffness, Temperature, ThermalContactResistance,
    },
};
use runmat_geometry_core::GeometryRegionRole;
use std::collections::BTreeSet;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AuthoredBoundary {
    pub boundary_id: String,
    pub target: AuthoredRegionSource,
    pub kind: AuthoredBoundaryKind,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuthoredBoundaryKind {
    StructuralPressure {
        value: PrescribedValue<Pressure>,
    },
    ThermalHeatFlux {
        value: PrescribedValue<HeatFlux>,
    },
    ThermalConvection {
        ambient_temperature: PrescribedValue<Temperature>,
        coefficient: HeatTransferCoefficient,
    },
    MagneticInsulation,
    AcousticRigidWall,
    AcousticRadiation,
    AcousticImpedance {
        impedance: SpecificAcousticImpedance,
    },
    FluidOutletPressure {
        value: PrescribedValue<Pressure>,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AuthoredConnector {
    pub connector_id: String,
    pub kind: AuthoredConnectorKind,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuthoredConnectorKind {
    AxialSpring {
        primary: AuthoredRegionSource,
        secondary: AuthoredRegionSource,
        stiffness: Stiffness,
    },
    AxialDashpot {
        primary: AuthoredRegionSource,
        secondary: AuthoredRegionSource,
        damping: Damping,
    },
    IsotropicGroundSpring {
        target: AuthoredRegionSource,
        stiffness: Stiffness,
    },
    SurfaceNormalFoundation {
        target: AuthoredRegionSource,
        modulus: FoundationModulus,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AuthoredInterface {
    pub interface_id: String,
    pub primary: AuthoredRegionSource,
    pub secondary: AuthoredRegionSource,
    pub kind: AuthoredInterfaceKind,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuthoredInterfaceKind {
    ContactFrictionless,
    ContactCoulomb {
        coefficient: FrictionCoefficient,
    },
    FluidStructureNoSlip,
    ConjugateHeatTransferPerfect,
    ConjugateHeatTransferConductance {
        coefficient: HeatTransferCoefficient,
    },
    ConjugateHeatTransferResistance {
        resistance: ThermalContactResistance,
    },
}

impl AuthoredBoundary {
    pub fn validate(&self) -> Result<(), AuthoredModelPhysicsError> {
        id(&self.boundary_id)?;
        require_role(&self.target, &[GeometryRegionRole::Face])?;
        let valid = match &self.kind {
            AuthoredBoundaryKind::StructuralPressure {
                value: PrescribedValue::Constant(value),
            }
            | AuthoredBoundaryKind::FluidOutletPressure {
                value: PrescribedValue::Constant(value),
            } => value.is_finite(),
            AuthoredBoundaryKind::ThermalHeatFlux {
                value: PrescribedValue::Constant(value),
            } => value.is_finite(),
            AuthoredBoundaryKind::ThermalConvection {
                ambient_temperature: PrescribedValue::Constant(temperature),
                coefficient,
            } => {
                temperature.is_finite()
                    && temperature.kelvin >= 0.0
                    && coefficient.is_finite()
                    && coefficient.watts_per_square_meter_kelvin >= 0.0
            }
            AuthoredBoundaryKind::AcousticImpedance { impedance } => {
                impedance.is_finite() && impedance.pascal_seconds_per_meter > 0.0
            }
            _ => true,
        };
        valid
            .then_some(())
            .ok_or(AuthoredModelPhysicsError::InvalidValue)
    }
}

impl AuthoredConnector {
    pub fn validate(&self) -> Result<(), AuthoredModelPhysicsError> {
        id(&self.connector_id)?;
        match &self.kind {
            AuthoredConnectorKind::AxialSpring {
                primary,
                secondary,
                stiffness,
            } => {
                require_distinct_vertices(primary, secondary)?;
                finite_nonnegative(stiffness.newtons_per_meter)?;
            }
            AuthoredConnectorKind::AxialDashpot {
                primary,
                secondary,
                damping,
            } => {
                require_distinct_vertices(primary, secondary)?;
                finite_nonnegative(damping.newton_seconds_per_meter)?;
            }
            AuthoredConnectorKind::IsotropicGroundSpring { target, stiffness } => {
                require_role(target, &[GeometryRegionRole::Vertex])?;
                finite_nonnegative(stiffness.newtons_per_meter)?;
            }
            AuthoredConnectorKind::SurfaceNormalFoundation { target, modulus } => {
                require_role(target, &[GeometryRegionRole::Face])?;
                finite_nonnegative(modulus.newtons_per_cubic_meter)?;
            }
        }
        Ok(())
    }
}

impl AuthoredConnectorKind {
    pub fn visit_targets(&self, mut visit: impl FnMut(&AuthoredRegionSource)) {
        match self {
            Self::AxialSpring {
                primary, secondary, ..
            }
            | Self::AxialDashpot {
                primary, secondary, ..
            } => {
                visit(primary);
                visit(secondary);
            }
            Self::IsotropicGroundSpring { target, .. }
            | Self::SurfaceNormalFoundation { target, .. } => visit(target),
        }
    }

    pub fn visit_targets_mut(&mut self, mut visit: impl FnMut(&mut AuthoredRegionSource)) {
        match self {
            Self::AxialSpring {
                primary, secondary, ..
            }
            | Self::AxialDashpot {
                primary, secondary, ..
            } => {
                visit(primary);
                visit(secondary);
            }
            Self::IsotropicGroundSpring { target, .. }
            | Self::SurfaceNormalFoundation { target, .. } => visit(target),
        }
    }
}

fn require_distinct_vertices(
    primary: &AuthoredRegionSource,
    secondary: &AuthoredRegionSource,
) -> Result<(), AuthoredModelPhysicsError> {
    require_role(primary, &[GeometryRegionRole::Vertex])?;
    require_role(secondary, &[GeometryRegionRole::Vertex])?;
    if primary == secondary {
        return Err(AuthoredModelPhysicsError::IdenticalTargets);
    }
    Ok(())
}

impl AuthoredInterface {
    pub fn validate(&self) -> Result<(), AuthoredModelPhysicsError> {
        id(&self.interface_id)?;
        require_role(&self.primary, &[GeometryRegionRole::Face])?;
        require_role(&self.secondary, &[GeometryRegionRole::Face])?;
        if self.primary == self.secondary {
            return Err(AuthoredModelPhysicsError::IdenticalTargets);
        }
        match self.kind {
            AuthoredInterfaceKind::ContactCoulomb { coefficient } => {
                finite_positive(coefficient.ratio)?
            }
            AuthoredInterfaceKind::ConjugateHeatTransferConductance { coefficient } => {
                finite_positive(coefficient.watts_per_square_meter_kelvin)?
            }
            AuthoredInterfaceKind::ConjugateHeatTransferResistance { resistance } => {
                finite_positive(resistance.square_meter_kelvin_per_watt)?
            }
            _ => {}
        }
        Ok(())
    }
}

fn id(value: &str) -> Result<(), AuthoredModelPhysicsError> {
    if value.trim().is_empty() {
        Err(AuthoredModelPhysicsError::EmptyId)
    } else {
        Ok(())
    }
}

fn require_role(
    target: &AuthoredRegionSource,
    accepted: &[GeometryRegionRole],
) -> Result<(), AuthoredModelPhysicsError> {
    target
        .validate()
        .map_err(|_| AuthoredModelPhysicsError::InvalidTarget)?;
    if accepted.contains(&target.role()) {
        Ok(())
    } else {
        Err(AuthoredModelPhysicsError::InvalidTopology)
    }
}

fn finite_nonnegative(value: f64) -> Result<(), AuthoredModelPhysicsError> {
    if value.is_finite() && value >= 0.0 {
        Ok(())
    } else {
        Err(AuthoredModelPhysicsError::InvalidValue)
    }
}

fn finite_positive(value: f64) -> Result<(), AuthoredModelPhysicsError> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(AuthoredModelPhysicsError::InvalidValue)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum AuthoredModelPhysicsError {
    #[error("definition id must not be empty")]
    EmptyId,
    #[error("region target is invalid")]
    InvalidTarget,
    #[error("region target has incompatible topology")]
    InvalidTopology,
    #[error("primary and secondary targets must be distinct")]
    IdenticalTargets,
    #[error("physical value is non-finite or outside its admissible range")]
    InvalidValue,
}

pub(super) fn validate_unique<T, E>(
    values: &[T],
    identity: impl Fn(&T) -> &str,
    validate: impl Fn(&T) -> Result<(), E>,
    kind: &'static str,
) -> Result<(), AuthoredModelError>
where
    E: std::fmt::Display,
{
    let mut ids = BTreeSet::new();
    for value in values {
        validate(value).map_err(|error| AuthoredModelError::InvalidPhysics {
            kind,
            detail: error.to_string(),
        })?;
        let id = identity(value);
        if !ids.insert(id) {
            return Err(AuthoredModelError::DuplicatePhysics {
                kind,
                id: id.to_string(),
            });
        }
    }
    Ok(())
}
