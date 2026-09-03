use serde::{Deserialize, Serialize};

use super::{
    prescribed_value::PrescribedValue,
    quantity::{
        HeatFlux, HeatTransferCoefficient, Pressure, SpecificAcousticImpedance, Temperature,
        Velocity,
    },
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
    FluidInletVelocity {
        value: PrescribedValue<Velocity>,
    },
    FluidOutletPressure {
        value: PrescribedValue<Pressure>,
    },
    FluidNoSlipWall,
    FluidSlipWall,
    FluidSymmetry,
}
