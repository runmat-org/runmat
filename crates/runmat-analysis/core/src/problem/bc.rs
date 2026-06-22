use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BoundaryConditionKind {
    Fixed,
    PrescribedDisplacement,
    PrescribedRotation {
        rx: f64,
        ry: f64,
        rz: f64,
    },
    MagneticInsulation,
    VectorPotentialGround,
    AcousticRigidWall,
    AcousticRadiation,
    AcousticImpedance {
        specific_impedance_pa_s_per_m: f64,
    },
    ThermalPrescribedTemperature {
        temperature_k: f64,
    },
    ThermalHeatFlux {
        heat_flux_w_per_m2: f64,
    },
    ThermalConvection {
        ambient_temperature_k: f64,
        coefficient_w_per_m2k: f64,
    },
    CfdInletVelocity {
        velocity_m_per_s: f64,
    },
    CfdOutletPressure {
        pressure_pa: f64,
    },
    CfdNoSlipWall,
    CfdSlipWall,
    CfdSymmetry,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BoundaryCondition {
    pub bc_id: String,
    pub region_id: String,
    pub kind: BoundaryConditionKind,
}
