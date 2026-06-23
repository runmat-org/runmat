pub mod electro_thermal;
pub mod electromagnetic;
pub mod linear_static;
pub mod modal;
pub mod nonlinear;
pub mod thermal;
pub mod thermo_mechanical;
pub mod transient;

use runmat_analysis_core::{
    AnalysisModel, BoundaryConditionKind, LoadKind, StructuralElementKind, StructuralModel,
};

use crate::{
    assembly::{dofs::StructuralDofKind, AssemblySummary},
    contracts::FeaRunError,
};

const MOMENT_REQUIRES_ROTATIONAL_DOF_MESSAGE: &str =
    "moment loads require rotational-DOF structural elements";
const ROTATION_REQUIRES_ROTATIONAL_DOF_MESSAGE: &str =
    "prescribed rotations require rotational-DOF structural elements";

pub(crate) fn validate_rotational_dof_targets(
    model: &AnalysisModel,
    summary: &AssemblySummary,
) -> Result<(), FeaRunError> {
    for load in &model.loads {
        if !matches!(load.kind, LoadKind::Moment { .. }) {
            continue;
        }
        if !summary.structural_dof_layout.has_rotational_dofs() {
            return Err(moment_load_error(load));
        }
        let Some(structural) = model.structural.as_ref() else {
            continue;
        };
        let target_nodes = structural_target_nodes(structural, &load.region_id);
        if target_nodes.is_empty()
            || target_nodes
                .iter()
                .any(|node_index| !node_has_rotational_dofs(summary, *node_index))
        {
            return Err(moment_load_error(load));
        }
    }
    for bc in &model.boundary_conditions {
        if !matches!(bc.kind, BoundaryConditionKind::PrescribedRotation { .. }) {
            continue;
        }
        if !summary.structural_dof_layout.has_rotational_dofs() {
            return Err(rotation_boundary_condition_error(bc));
        }
        let Some(structural) = model.structural.as_ref() else {
            continue;
        };
        let target_nodes = structural_target_nodes(structural, &bc.region_id);
        if target_nodes.is_empty()
            || target_nodes
                .iter()
                .any(|node_index| !node_has_rotational_dofs(summary, *node_index))
        {
            return Err(rotation_boundary_condition_error(bc));
        }
    }
    Ok(())
}

fn moment_load_error(load: &runmat_analysis_core::LoadCase) -> FeaRunError {
    FeaRunError::InvalidModel(format!(
        "{}; load_id={} region_id={}",
        MOMENT_REQUIRES_ROTATIONAL_DOF_MESSAGE, load.load_id, load.region_id
    ))
}

fn rotation_boundary_condition_error(bc: &runmat_analysis_core::BoundaryCondition) -> FeaRunError {
    FeaRunError::InvalidModel(format!(
        "{}; bc_id={} region_id={}",
        ROTATION_REQUIRES_ROTATIONAL_DOF_MESSAGE, bc.bc_id, bc.region_id
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

fn structural_target_nodes(structural: &StructuralModel, region_id: &str) -> Vec<usize> {
    if let Some(node_id) = structural_node_selector(region_id) {
        return structural_node_index(structural, node_id)
            .into_iter()
            .collect();
    }
    let mut nodes = Vec::new();
    for element in &structural.elements {
        if element.region_id != region_id {
            continue;
        }
        match &element.kind {
            StructuralElementKind::Beam(beam) => {
                for node_id in beam.node_ids {
                    if let Some(index) = structural_node_index(structural, node_id) {
                        if !nodes.contains(&index) {
                            nodes.push(index);
                        }
                    }
                }
            }
            StructuralElementKind::Shell(shell) => {
                for node_id in shell.node_ids {
                    if let Some(index) = structural_node_index(structural, node_id) {
                        if !nodes.contains(&index) {
                            nodes.push(index);
                        }
                    }
                }
            }
        }
    }
    nodes
}

fn structural_node_index(structural: &StructuralModel, node_id: u32) -> Option<usize> {
    structural
        .nodes
        .iter()
        .position(|node| node.node_id == node_id)
}

fn structural_node_selector(region_id: &str) -> Option<u32> {
    region_id
        .strip_prefix("node:")
        .unwrap_or(region_id)
        .parse::<u32>()
        .ok()
}
