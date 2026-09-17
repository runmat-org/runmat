use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum QuantityDimension {
    Dimensionless,
    Length,
    Area,
    Volume,
    Time,
    Frequency,
    Mass,
    Density,
    Temperature,
    Velocity,
    Acceleration,
    Force,
    Moment,
    Pressure,
    Energy,
    Power,
    HeatFlux,
    ThermalConductivity,
    ElectricPotential,
    ElectricCurrent,
    MagneticFluxDensity,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum OperationQuantityRule {
    NotApplicable,
    PreserveInput {
        input: usize,
    },
    RequiredInput {
        input: usize,
        dimension: QuantityDimension,
    },
    FixedOutput {
        output: usize,
        dimension: QuantityDimension,
    },
    CheckedByLowering,
}
