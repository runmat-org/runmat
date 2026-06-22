pub mod electro_thermal;
pub mod electromagnetic;
pub mod linear_static;
pub mod modal;
pub mod nonlinear;
pub mod thermal;
pub mod thermo_mechanical;
pub mod transient;

use runmat_analysis_core::{AnalysisModel, LoadKind};

use crate::contracts::FeaRunError;

const MOMENT_REQUIRES_ROTATIONAL_DOF_MESSAGE: &str =
    "moment loads require rotational-DOF structural elements";

pub(crate) fn reject_moment_loads_without_rotational_dofs(
    model: &AnalysisModel,
) -> Result<(), FeaRunError> {
    if let Some(load) = model
        .loads
        .iter()
        .find(|load| matches!(load.kind, LoadKind::Moment { .. }))
    {
        return Err(FeaRunError::InvalidModel(format!(
            "{}; load_id={} region_id={}",
            MOMENT_REQUIRES_ROTATIONAL_DOF_MESSAGE, load.load_id, load.region_id
        )));
    }
    Ok(())
}
