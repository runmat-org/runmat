use runmat_analysis_core::{
    CoordinateFrame, ExcitationCaseId, ExcitationDistribution, ExcitationId, ExcitationKinematics,
    PhysicalQuantityKind, Position, ProjectionRequirement, RegionTarget, SignalId,
};
use serde::{Deserialize, Serialize};

use super::ProjectionSample;

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub(crate) struct ComplexScalar {
    pub real: f64,
    pub imaginary: f64,
}

impl ComplexScalar {
    pub(crate) const fn real(value: f64) -> Self {
        Self {
            real: value,
            imaginary: 0.0,
        }
    }

    pub(super) fn component(self, component: ComplexComponent) -> Self {
        Self::real(match component {
            ComplexComponent::Real => self.real,
            ComplexComponent::Imaginary => self.imaginary,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub(crate) struct ComplexVector3 {
    pub real: [f64; 3],
    pub imaginary: [f64; 3],
}

impl ComplexVector3 {
    pub(crate) const fn real(value: [f64; 3]) -> Self {
        Self {
            real: value,
            imaginary: [0.0; 3],
        }
    }

    pub(super) fn component(self, component: ComplexComponent) -> Self {
        Self::real(match component {
            ComplexComponent::Real => self.real,
            ComplexComponent::Imaginary => self.imaginary,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ComplexComponent {
    Real,
    Imaginary,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "value")]
pub(crate) enum RealizedExcitationTarget {
    Region(RegionTarget),
    ReferencePoint {
        reference_point_id: String,
        position: Position,
    },
    Port(String),
    Interface(String),
    WholeModel,
}

impl RealizedExcitationTarget {
    pub(crate) fn solver_region_id(&self) -> Option<&str> {
        match self {
            Self::Region(target) => Some(&target.region_id),
            Self::ReferencePoint {
                reference_point_id, ..
            } => Some(reference_point_id),
            Self::Port(_) | Self::Interface(_) | Self::WholeModel => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "value")]
pub(crate) enum RealizedExcitationKind {
    Force(ComplexVector3),
    Moment(ComplexVector3),
    Traction(ComplexVector3),
    Pressure(ComplexScalar),
    BodyForce(ComplexVector3),
    BodyForceField(super::field::NodalVectorField),
    HeatGeneration(ComplexScalar),
    HeatGenerationField(super::field::ElementScalarField),
    HeatFlux(ComplexScalar),
    TemperatureField(ComplexScalar),
    AmbientTemperature(ComplexScalar),
    TerminalVoltage(ComplexScalar),
    ElectricCurrent(ComplexScalar),
    ElectricCharge(ComplexScalar),
    ElectricChargeDensity(ComplexScalar),
    CurrentDensity(ComplexVector3),
    MagneticVolumeCurrentDensity(ComplexVector3),
    MagneticVectorPotential(ComplexVector3),
    AcousticPressure(ComplexScalar),
    AcousticIncidentPressure(ComplexScalar),
    NormalVelocity(ComplexScalar),
    VolumeVelocity(ComplexScalar),
    InletVelocity(ComplexVector3),
    MassFlow(ComplexScalar),
    WallMotion(ComplexVector3),
}

impl RealizedExcitationKind {
    pub(crate) fn is_structural_load(&self) -> bool {
        matches!(
            self,
            Self::Force(_)
                | Self::Moment(_)
                | Self::Traction(_)
                | Self::Pressure(_)
                | Self::BodyForce(_)
                | Self::BodyForceField(_)
        )
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct RealizedExcitation {
    pub excitation_id: ExcitationId,
    pub case_ids: Vec<ExcitationCaseId>,
    pub signal_id: SignalId,
    pub quantity: PhysicalQuantityKind,
    pub sample: ProjectionSample,
    pub target: RealizedExcitationTarget,
    pub kind: RealizedExcitationKind,
    pub policy: RealizedExcitationPolicy,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct RealizedExcitationPolicy {
    pub frame: CoordinateFrame,
    pub distribution: ExcitationDistribution,
    pub kinematics: ExcitationKinematics,
    pub projection: ProjectionRequirement,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct RealizedExcitations {
    entries: Vec<RealizedExcitation>,
}

impl RealizedExcitations {
    pub(crate) fn new(entries: Vec<RealizedExcitation>) -> Self {
        Self { entries }
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = &RealizedExcitation> {
        self.entries.iter()
    }

    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }

    pub(crate) fn filtered(
        &self,
        retain: impl Fn(&RealizedExcitation) -> bool,
    ) -> RealizedExcitations {
        RealizedExcitations::new(
            self.entries
                .iter()
                .filter(|entry| retain(entry))
                .cloned()
                .collect(),
        )
    }

    pub(crate) fn component(&self, component: ComplexComponent) -> Self {
        Self::new(
            self.entries
                .iter()
                .cloned()
                .map(|mut entry| {
                    entry.kind = entry.kind.component(component);
                    entry
                })
                .collect(),
        )
    }
}

impl RealizedExcitationKind {
    fn component(self, component: ComplexComponent) -> Self {
        match self {
            Self::Force(value) => Self::Force(value.component(component)),
            Self::Moment(value) => Self::Moment(value.component(component)),
            Self::Traction(value) => Self::Traction(value.component(component)),
            Self::Pressure(value) => Self::Pressure(value.component(component)),
            Self::BodyForce(value) => Self::BodyForce(value.component(component)),
            Self::BodyForceField(value) => Self::BodyForceField(value.component(component)),
            Self::HeatGeneration(value) => Self::HeatGeneration(value.component(component)),
            Self::HeatGenerationField(value) => {
                Self::HeatGenerationField(value.component(component))
            }
            Self::HeatFlux(value) => Self::HeatFlux(value.component(component)),
            Self::TemperatureField(value) => Self::TemperatureField(value.component(component)),
            Self::AmbientTemperature(value) => Self::AmbientTemperature(value.component(component)),
            Self::TerminalVoltage(value) => Self::TerminalVoltage(value.component(component)),
            Self::ElectricCurrent(value) => Self::ElectricCurrent(value.component(component)),
            Self::ElectricCharge(value) => Self::ElectricCharge(value.component(component)),
            Self::ElectricChargeDensity(value) => {
                Self::ElectricChargeDensity(value.component(component))
            }
            Self::CurrentDensity(value) => Self::CurrentDensity(value.component(component)),
            Self::MagneticVolumeCurrentDensity(value) => {
                Self::MagneticVolumeCurrentDensity(value.component(component))
            }
            Self::MagneticVectorPotential(value) => {
                Self::MagneticVectorPotential(value.component(component))
            }
            Self::AcousticPressure(value) => Self::AcousticPressure(value.component(component)),
            Self::AcousticIncidentPressure(value) => {
                Self::AcousticIncidentPressure(value.component(component))
            }
            Self::NormalVelocity(value) => Self::NormalVelocity(value.component(component)),
            Self::VolumeVelocity(value) => Self::VolumeVelocity(value.component(component)),
            Self::InletVelocity(value) => Self::InletVelocity(value.component(component)),
            Self::MassFlow(value) => Self::MassFlow(value.component(component)),
            Self::WallMotion(value) => Self::WallMotion(value.component(component)),
        }
    }
}
