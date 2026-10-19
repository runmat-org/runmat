use serde::{Deserialize, Serialize};

use super::{
    prescribed_value::PrescribedValue,
    quantity::{
        HeatFlux, HeatTransferCoefficient, Pressure, SpecificAcousticImpedance, SurfaceEmissivity,
        Temperature,
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
    /// Diffuse gray surface exchanging with a prescribed, nonparticipating ambient.
    /// This is not enclosure/view-factor or participating-medium radiation.
    ThermalRadiation {
        ambient_temperature: PrescribedValue<Temperature>,
        emissivity: SurfaceEmissivity,
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
