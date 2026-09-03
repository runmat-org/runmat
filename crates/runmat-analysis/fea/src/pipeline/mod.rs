pub mod electro_thermal;
pub mod electromagnetic;
pub mod linear_static;
pub mod modal;
pub mod nonlinear;
pub mod thermal;
pub mod thermo_mechanical;
pub mod transient;

use runmat_analysis_core::{
    AnalysisModel, StructuralConstraint, StructuralConstraintKind, StructuralElementKind,
    StructuralModel,
};

use crate::{
    assembly::{dofs::StructuralDofKind, AssemblySummary},
    contracts::FeaRunError,
    excitation::{realize_model_excitations, RealizedExcitation, RealizedExcitationKind},
};

const MOMENT_REQUIRES_ROTATIONAL_DOF_MESSAGE: &str =
    "moment loads require rotational-DOF structural elements; use a wrench load for equivalent force-couple moments on solid-only face regions";
const ROTATION_REQUIRES_ROTATIONAL_DOF_MESSAGE: &str =
    "prescribed rotations require rotational-DOF structural elements";

pub(crate) fn validate_rotational_dof_targets(
    model: &AnalysisModel,
    summary: &AssemblySummary,
) -> Result<(), FeaRunError> {
    let excitations = realize_model_excitations(model)
        .map_err(|error| FeaRunError::InvalidModel(error.to_string()))?;
    for load in excitations.iter() {
        if !matches!(load.kind, RealizedExcitationKind::Moment(_)) {
            continue;
        }
        if !summary.structural_dof_layout.has_rotational_dofs() {
            return Err(moment_load_error(load));
        }
        let Some(structural) = model.structural.as_ref() else {
            continue;
        };
        let Some(region_id) = load.target.solver_region_id() else {
            return Err(moment_load_error(load));
        };
        let target_nodes = structural_target_nodes(structural, region_id);
        if target_nodes.is_empty()
            || target_nodes
                .iter()
                .any(|node_index| !node_has_rotational_dofs(summary, *node_index))
        {
            return Err(moment_load_error(load));
        }
    }
    for constraint in &model.constraints {
        let Some(constraint) = constraint.structural() else {
            continue;
        };
        if !matches!(
            constraint.kind,
            StructuralConstraintKind::PrescribedRotations(_)
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
        let target_nodes = structural_target_nodes(structural, &target.region_id);
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
    let excitations = realize_model_excitations(model)
        .map_err(|error| FeaRunError::InvalidModel(error.to_string()))?;
    let Some(load) = excitations.iter().find(|load| {
        matches!(
            load.kind,
            RealizedExcitationKind::Moment(_) | RealizedExcitationKind::Wrench { .. }
        )
    }) else {
        return Ok(());
    };
    let load_kind = match load.kind {
        RealizedExcitationKind::Moment(_) => "moment",
        RealizedExcitationKind::Wrench { .. } => "wrench",
        _ => "structural",
    };

    Err(FeaRunError::InvalidModel(format!(
        "{load_kind} loads are structural loads and cannot be used in {family} solves; load_id={} region_id={}",
        load.excitation_id.0,
        load.target.solver_region_id().unwrap_or("<non-region target>")
    )))
}

fn moment_load_error(load: &RealizedExcitation) -> FeaRunError {
    FeaRunError::InvalidModel(format!(
        "{}; load_id={} region_id={}",
        MOMENT_REQUIRES_ROTATIONAL_DOF_MESSAGE,
        load.excitation_id.0,
        load.target
            .solver_region_id()
            .unwrap_or("<non-region target>")
    ))
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
