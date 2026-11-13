use serde::{Deserialize, Serialize};

use super::{
    quantity::{FrictionCoefficient, HeatTransferCoefficient, ThermalContactResistance},
    target::RegionTarget,
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContactInterfaceModel {
    pub friction: ContactFriction,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContactFriction {
    Frictionless,
    Coulomb { coefficient: FrictionCoefficient },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FluidStructureInterfaceModel {
    pub kinematics: FluidStructureKinematics,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FluidStructureKinematics {
    NoSlip,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConjugateHeatTransferInterfaceModel {
    pub transfer: ThermalInterfaceTransfer,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ThermalInterfaceTransfer {
    Perfect,
    Conductance(HeatTransferCoefficient),
    Resistance(ThermalContactResistance),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnalysisInterfaceKind {
    Contact(ContactInterfaceModel),
    FluidStructure(FluidStructureInterfaceModel),
    ConjugateHeatTransfer(ConjugateHeatTransferInterfaceModel),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnalysisInterface {
    pub interface_id: String,
    pub primary: RegionTarget,
    pub secondary: RegionTarget,
    pub kind: AnalysisInterfaceKind,
}
