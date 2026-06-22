pub mod electro_thermal;
pub mod electromagnetic;
pub mod linear_static;
pub mod modal;
pub mod nonlinear;
pub mod thermal;
pub mod thermo_mechanical;
pub mod transient;

use runmat_analysis_core::AnalysisModel;

use crate::{assembly::AssemblySummary, contracts::FeaRunError};

pub(crate) fn reject_moment_loads_without_rotational_dofs(
    model: &AnalysisModel,
    summary: &AssemblySummary,
) -> Result<(), FeaRunError> {
    crate::assembly::dofs::validate_moment_loads_against_layout(
        model,
        &summary.structural_dof_layout,
    )
}
