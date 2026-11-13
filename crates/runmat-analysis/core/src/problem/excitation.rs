use serde::{Deserialize, Serialize};

use super::{
    coordinate_frame::CoordinateFrame,
    reference_point::ReferencePoint,
    signal::{PhysicalQuantityKind, SignalId},
    target::RegionTarget,
};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ExcitationId(pub String);

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ExcitationCaseId(pub String);

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnalysisExcitation {
    pub excitation_id: ExcitationId,
    pub target: ExcitationTarget,
    pub frame: CoordinateFrame,
    pub direction: Option<Direction>,
    pub distribution: ExcitationDistribution,
    pub kinematics: ExcitationKinematics,
    pub signal_id: SignalId,
    pub active_step_ids: Vec<String>,
    pub combination_group: Option<String>,
    pub projection: ProjectionRequirement,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExcitationTarget {
    Region(RegionTarget),
    ReferencePoint(ReferencePoint),
    Port { port_id: String },
    Interface { interface_id: String },
    WholeModel,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Direction {
    pub vector: [f64; 3],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExcitationDistribution {
    Concentrated,
    PreserveResultant,
    IntensityPerArea,
    IntensityPerVolume,
    UniformKinematic,
    TopologyField,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExcitationKinematics {
    FixedFrame,
    Follower,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectionRequirement {
    pub preserve_resultant: bool,
    pub preserve_moment: bool,
    pub relative_tolerance: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExcitationAction {
    Force(AnalysisExcitation),
    Moment(AnalysisExcitation),
    Traction(AnalysisExcitation),
    Pressure(AnalysisExcitation),
    BodyForce(AnalysisExcitation),
    HeatGeneration(AnalysisExcitation),
    HeatFlux(AnalysisExcitation),
    TemperatureField(AnalysisExcitation),
    AmbientTemperature(AnalysisExcitation),
    TerminalVoltage(AnalysisExcitation),
    ElectricCurrent(AnalysisExcitation),
    ElectricCharge(AnalysisExcitation),
    ElectricChargeDensity(AnalysisExcitation),
    CurrentDensity(AnalysisExcitation),
    /// Volumetric magnetic source J in A/m², integrated over volume against
    /// the vector-potential test basis. This is not surface current intensity.
    MagneticVolumeCurrentDensity(AnalysisExcitation),
    MagneticVectorPotential(AnalysisExcitation),
    AcousticPressure(AnalysisExcitation),
    /// Uniform incoming pressure; propagation is opposite the exterior outward normal.
    AcousticIncidentPressure(AnalysisExcitation),
    NormalVelocity(AnalysisExcitation),
    /// Total outward volume flow through one finite acoustic boundary patch.
    VolumeVelocity(AnalysisExcitation),
    InletVelocity(AnalysisExcitation),
    /// Signed total mass input into the selected fluid boundary, in kg/s.
    /// Canonical realization uses inward normals; outward balance reporting has the opposite sign.
    MassFlow(AnalysisExcitation),
    WallMotion(AnalysisExcitation),
}

impl ExcitationAction {
    pub fn excitation(&self) -> &AnalysisExcitation {
        match self {
            Self::Force(value)
            | Self::Moment(value)
            | Self::Traction(value)
            | Self::Pressure(value)
            | Self::BodyForce(value)
            | Self::HeatGeneration(value)
            | Self::HeatFlux(value)
            | Self::TemperatureField(value)
            | Self::AmbientTemperature(value)
            | Self::TerminalVoltage(value)
            | Self::ElectricCurrent(value)
            | Self::ElectricCharge(value)
            | Self::ElectricChargeDensity(value)
            | Self::CurrentDensity(value)
            | Self::MagneticVolumeCurrentDensity(value)
            | Self::MagneticVectorPotential(value)
            | Self::AcousticPressure(value)
            | Self::AcousticIncidentPressure(value)
            | Self::NormalVelocity(value)
            | Self::VolumeVelocity(value)
            | Self::InletVelocity(value)
            | Self::MassFlow(value)
            | Self::WallMotion(value) => value,
        }
    }

    pub fn expected_quantity(&self) -> PhysicalQuantityKind {
        match self {
            Self::Force(_) => PhysicalQuantityKind::Force,
            Self::Moment(_) => PhysicalQuantityKind::Moment,
            Self::Traction(_) => PhysicalQuantityKind::Traction,
            Self::Pressure(_) | Self::AcousticPressure(_) | Self::AcousticIncidentPressure(_) => {
                PhysicalQuantityKind::Pressure
            }
            Self::BodyForce(_) => PhysicalQuantityKind::BodyForceDensity,
            Self::NormalVelocity(_) | Self::InletVelocity(_) | Self::WallMotion(_) => {
                PhysicalQuantityKind::Velocity
            }
            Self::HeatGeneration(action)
                if action.distribution == ExcitationDistribution::PreserveResultant =>
            {
                PhysicalQuantityKind::Power
            }
            Self::HeatGeneration(_) => PhysicalQuantityKind::HeatGenerationDensity,
            Self::HeatFlux(_) => PhysicalQuantityKind::HeatFlux,
            Self::TemperatureField(_) | Self::AmbientTemperature(_) => {
                PhysicalQuantityKind::Temperature
            }
            Self::TerminalVoltage(_) => PhysicalQuantityKind::ElectricPotential,
            Self::ElectricCurrent(_) => PhysicalQuantityKind::ElectricCurrent,
            Self::ElectricCharge(_) => PhysicalQuantityKind::ElectricCharge,
            Self::ElectricChargeDensity(_) => PhysicalQuantityKind::ElectricChargeDensity,
            Self::CurrentDensity(_) => PhysicalQuantityKind::ElectricCurrentDensity,
            Self::MagneticVolumeCurrentDensity(_) => PhysicalQuantityKind::ElectricCurrentDensity,
            Self::MagneticVectorPotential(_) => PhysicalQuantityKind::MagneticVectorPotential,
            Self::MassFlow(_) => PhysicalQuantityKind::MassFlowRate,
            Self::VolumeVelocity(_) => PhysicalQuantityKind::VolumeFlowRate,
        }
    }

    pub fn excitation_mut(&mut self) -> &mut AnalysisExcitation {
        match self {
            Self::Force(value)
            | Self::Moment(value)
            | Self::Traction(value)
            | Self::Pressure(value)
            | Self::BodyForce(value)
            | Self::HeatGeneration(value)
            | Self::HeatFlux(value)
            | Self::TemperatureField(value)
            | Self::AmbientTemperature(value)
            | Self::TerminalVoltage(value)
            | Self::ElectricCurrent(value)
            | Self::ElectricCharge(value)
            | Self::ElectricChargeDensity(value)
            | Self::CurrentDensity(value)
            | Self::MagneticVolumeCurrentDensity(value)
            | Self::MagneticVectorPotential(value)
            | Self::AcousticPressure(value)
            | Self::AcousticIncidentPressure(value)
            | Self::NormalVelocity(value)
            | Self::VolumeVelocity(value)
            | Self::InletVelocity(value)
            | Self::MassFlow(value)
            | Self::WallMotion(value) => value,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExcitationCase {
    pub case_id: ExcitationCaseId,
    pub action_ids: Vec<ExcitationId>,
    pub activation: ExcitationCaseActivation,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExcitationCaseActivation {
    Always,
    Steps(Vec<String>),
    MutuallyExclusiveGroup { group_id: String },
}
