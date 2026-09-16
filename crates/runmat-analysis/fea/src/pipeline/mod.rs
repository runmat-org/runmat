pub mod electro_thermal;
pub mod electromagnetic;
pub mod electromagnetic_frequency;
mod electromagnetic_frequency_history;
mod electromagnetic_frequency_result;
pub mod linear_static;
pub mod modal;
pub mod modal_basis;
pub mod modal_governance;
pub mod modal_residual;
pub mod nonlinear;
pub(crate) mod prestress;
pub mod structural_buckling;
pub mod structural_frequency;
pub mod structural_linearization;
pub mod structural_random_response;
pub mod structural_response_spectrum;
pub mod thermal;
pub mod thermo_mechanical;
pub mod transient;

mod activation;
mod admission;
mod observation_history;
pub(crate) mod structural_nodal;

use runmat_analysis_core::{
    AnalysisModel, ExcitationAction, ExcitationTarget, StructuralConstraint,
    StructuralConstraintKind,
};

use crate::{
    assembly::{dofs::StructuralDofKind, AssemblySummary},
    contracts::FeaRunError,
    physics::structural::topology::explicit_target_node_indices,
};

const ROTATION_REQUIRES_ROTATIONAL_DOF_MESSAGE: &str =
    "prescribed rotations require rotational-DOF structural elements";

pub(crate) fn validate_rotational_dof_targets(
    model: &AnalysisModel,
    summary: &AssemblySummary,
) -> Result<(), FeaRunError> {
    for constraint in &model.constraints {
        let Some(constraint) = constraint.structural() else {
            continue;
        };
        if !matches!(
            &constraint.kind,
            StructuralConstraintKind::Prescribed(values) if !values.rotations.is_empty()
        ) {
            continue;
        }
        if !summary.structural_dof_layout.has_rotational_dofs() {
            return Err(rotation_constraint_error(constraint));
        }
        let Some(structural) = model.structural.as_ref() else {
            continue;
        };
        let target = constraint
            .kind
            .target()
            .expect("rotation constraint has a target");
        let target_nodes = explicit_target_node_indices(structural, &target.region_id);
        if target_nodes.is_empty()
            || target_nodes
                .iter()
                .any(|node_index| !node_has_rotational_dofs(summary, *node_index))
        {
            return Err(rotation_constraint_error(constraint));
        }
    }
    Ok(())
}

pub(crate) fn reject_moment_loads_for_nonstructural_pipeline(
    model: &AnalysisModel,
    family: &str,
) -> Result<(), FeaRunError> {
    let Some(action) = model
        .excitations
        .iter()
        .find(|action| matches!(action, ExcitationAction::Moment(_)))
    else {
        return Ok(());
    };
    let excitation = action.excitation();
    let region_id = match &excitation.target {
        ExcitationTarget::Region(target) => target.region_id.as_str(),
        _ => "<non-region target>",
    };

    Err(FeaRunError::InvalidModel(format!(
        "moment loads are structural loads and cannot be used in {family} solves; load_id={} region_id={}",
        excitation.excitation_id.0,
        region_id,
    )))
}

fn rotation_constraint_error(constraint: &StructuralConstraint) -> FeaRunError {
    let region_id = constraint
        .kind
        .target()
        .map(|target| target.region_id.as_str())
        .unwrap_or("<equation>");
    FeaRunError::InvalidModel(format!(
        "{}; constraint_id={} region_id={}",
        ROTATION_REQUIRES_ROTATIONAL_DOF_MESSAGE, constraint.constraint_id, region_id
    ))
}

fn node_has_rotational_dofs(summary: &AssemblySummary, node_index: usize) -> bool {
    [
        StructuralDofKind::Rx,
        StructuralDofKind::Ry,
        StructuralDofKind::Rz,
    ]
    .iter()
    .any(|kind| {
        summary
            .structural_dof_layout
            .index(node_index, *kind)
            .is_some()
    })
}
