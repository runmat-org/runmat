use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BoundaryConditionKind {
    Fixed,
    PrescribedDisplacement,
    MagneticInsulation,
    VectorPotentialGround,
    AcousticRigidWall,
    AcousticRadiation,
    AcousticImpedance { specific_impedance_pa_s_per_m: f64 },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BoundaryCondition {
    pub bc_id: String,
    pub region_id: String,
    pub kind: BoundaryConditionKind,
}
