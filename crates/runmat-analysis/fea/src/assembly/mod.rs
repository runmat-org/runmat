use runmat_analysis_core::AnalysisModel;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AssemblySummary {
    pub dof_count: usize,
    pub constrained_dof_count: usize,
    pub load_count: usize,
}

pub fn assemble_linear_system(model: &AnalysisModel) -> AssemblySummary {
    AssemblySummary {
        dof_count: model.loads.len() * 3,
        constrained_dof_count: model.boundary_conditions.len() * 3,
        load_count: model.loads.len(),
    }
}
