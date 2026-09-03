use serde::{Deserialize, Serialize};

use super::{
    quantity::{
        HeatFlux, HeatTransferCoefficient, Pressure, SpecificAcousticImpedance, Temperature,
        Velocity,
    },
    schedule::ValueSchedule,
    target::RegionTarget,
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnalysisBoundary {
    pub boundary_id: String,
    pub target: RegionTarget,
    pub operator: BoundaryOperator,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BoundaryOperator {
    StructuralPressure {
        value: ValueSchedule<Pressure>,
    },
    ThermalHeatFlux {
        value: ValueSchedule<HeatFlux>,
    },
    ThermalConvection {
        ambient_temperature: ValueSchedule<Temperature>,
        coefficient: HeatTransferCoefficient,
    },
    MagneticInsulation,
    AcousticRigidWall,
    AcousticRadiation,
    AcousticImpedance {
        impedance: SpecificAcousticImpedance,
    },
    FluidInletVelocity {
        value: ValueSchedule<Velocity>,
    },
    FluidOutletPressure {
        value: ValueSchedule<Pressure>,
    },
    FluidNoSlipWall,
    FluidSlipWall,
    FluidSymmetry,
}
