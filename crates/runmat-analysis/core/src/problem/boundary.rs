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
    /// Outgoing spherical-wave radiation centered at `center_m`: the first-order
    /// Bayliss-Turkel condition dp/dn = -(ik + 1/r) p, exact for a monopole field.
    AcousticSphericalRadiation {
        center_m: [f64; 3],
    },
    AcousticImpedance {
        impedance: SpecificAcousticImpedance,
    },
    /// Complex, frequency-dependent specific impedance (a table or a porous-layer model).
    AcousticImpedanceLaw {
        law: super::acoustic_impedance::AcousticImpedanceLaw,
    },
    FluidOutletPressure {
        value: PrescribedValue<Pressure>,
    },
}
