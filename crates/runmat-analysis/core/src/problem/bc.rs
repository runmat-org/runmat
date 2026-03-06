use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BoundaryConditionKind {
    Fixed,
    PrescribedDisplacement,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BoundaryCondition {
    pub bc_id: String,
    pub region_id: String,
    pub kind: BoundaryConditionKind,
}
