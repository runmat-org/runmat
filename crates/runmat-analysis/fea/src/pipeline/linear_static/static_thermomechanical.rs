//! Physical solid and reduced static thermomechanical result dispatch.

use runmat_analysis_core::AnalysisField;

use crate::{
    assembly::AssemblySummary,
    contracts::{FeaRunError, FEA_FIELD_STRUCTURAL_DISPLACEMENT, FEA_FIELD_STRUCTURAL_VON_MISES},
    pipeline::thermo_mechanical::{
        recover_physical_static_thermo_mechanical_fields, recover_thermo_mechanical_snapshots,
        ThermoMechanicalSnapshotFields,
    },
};

pub(super) fn recover(
    summary: &AssemblySummary,
    fields: &[AnalysisField],
    residual_norm: f64,
) -> Result<ThermoMechanicalSnapshotFields, FeaRunError> {
    let displacement = super::field_snapshot(fields, FEA_FIELD_STRUCTURAL_DISPLACEMENT);
    let von_mises = super::field_snapshot(fields, FEA_FIELD_STRUCTURAL_VON_MISES);
    if summary
        .structural_solid_recovery
        .iter()
        .any(|element| element.solved_thermal_quadrature.is_some())
    {
        return crate::pipeline::thermo_mechanical::recover_solved_static_fields(
            summary,
            &displacement,
            &von_mises,
            residual_norm,
        )
        .map_err(FeaRunError::Assembly);
    }
    if summary
        .structural_solid_recovery
        .iter()
        .any(|element| element.thermal_eigenstrain.is_some())
    {
        recover_physical_static_thermo_mechanical_fields(
            summary,
            &displacement,
            &von_mises,
            residual_norm,
        )
        .map_err(FeaRunError::Assembly)
    } else {
        Ok(recover_thermo_mechanical_snapshots(
            summary.thermo_mechanical.as_ref(),
            &[1.0],
            &displacement,
            &von_mises,
            &[residual_norm],
            super::element_count_for_dofs(summary.dof_count),
        ))
    }
}
