//! Shared source provenance; retained physics inputs require authenticated source closure.
use super::PhysicalSignal;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SignalProvenance {
    Authored,
    Imported {
        source_digest: String,
    },
    /// Original retained physical field values, not an arbitrary imported external signal.
    /// Consumers must authenticate the corresponding immutable source binding before assembly.
    Retained {
        source_digest: String,
    },
    Generated(GeneratorProvenance),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeneratorProvenance {
    pub project_revision: String,
    pub callable_id: String,
    pub frozen_input_digest: String,
}

impl PhysicalSignal {
    pub fn provenance(&self) -> &SignalProvenance {
        match self {
            Self::Displacement(signal) => &signal.provenance,
            Self::Rotation(signal) => &signal.provenance,
            Self::Velocity(signal) => &signal.provenance,
            Self::Acceleration(signal) => &signal.provenance,
            Self::AngularVelocity(signal) => &signal.provenance,
            Self::AngularAcceleration(signal) => &signal.provenance,
            Self::Force(signal) => &signal.provenance,
            Self::Moment(signal) => &signal.provenance,
            Self::Traction(signal) => &signal.provenance,
            Self::Pressure(signal) => &signal.provenance,
            Self::BodyForceDensity(signal) => &signal.provenance,
            Self::Temperature(signal) => &signal.provenance,
            Self::TemperatureDifference(signal) => &signal.provenance,
            Self::HeatFlux(signal) => &signal.provenance,
            Self::HeatGenerationDensity(signal) => &signal.provenance,
            Self::HeatRate(signal) => &signal.provenance,
            Self::ElectricPotential(signal) => &signal.provenance,
            Self::ElectricCurrent(signal) => &signal.provenance,
            Self::ElectricCharge(signal) => &signal.provenance,
            Self::ElectricCurrentDensity(signal) => &signal.provenance,
            Self::ElectricChargeDensity(signal) => &signal.provenance,
            Self::MagneticVectorPotential(signal) => &signal.provenance,
            Self::MassFlowRate(signal) => &signal.provenance,
            Self::VolumeFlowRate(signal) => &signal.provenance,
        }
    }
}
