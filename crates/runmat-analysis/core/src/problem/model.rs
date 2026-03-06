use runmat_geometry_core::UnitSystem;
use serde::{Deserialize, Serialize};

use super::{
    bc::BoundaryCondition, loads::LoadCase, material_assignment::MaterialAssignment,
    materials::MaterialModel, steps::AnalysisStep,
};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnalysisModelId(pub String);

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReferenceFrame {
    Global,
    Local(String),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnalysisModel {
    pub model_id: AnalysisModelId,
    pub geometry_id: String,
    pub geometry_revision: u32,
    pub units: UnitSystem,
    pub frame: ReferenceFrame,
    pub materials: Vec<MaterialModel>,
    #[serde(default)]
    pub material_assignments: Vec<MaterialAssignment>,
    pub boundary_conditions: Vec<BoundaryCondition>,
    pub loads: Vec<LoadCase>,
    pub steps: Vec<AnalysisStep>,
}
