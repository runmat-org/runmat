//! Additional lowering inventory from borrowed, already resolved target occurrences.
//! Prepared selection/topology retention, group accumulation and final rank are separate.
use crate::constraints::{
    coupling,
    lowering::{constraint_row_construction_profile, ConstraintRowConstructionProfile},
    normal, pairing,
    prepared::PreparedTargets,
    workspace::{add, check_cancelled, count_add, multiply, slots},
    ConstraintCouplingKind, ConstraintCouplingWeight, ConstraintPlanError,
    ResolvedConstraintTarget,
};
use crate::physics::structural::topology::{
    explicit_surface::shell_surface_workspace_bytes, node_space::CanonicalStructuralNodeSpace,
};
use runmat_analysis_core::{
    CoordinateFrame, StructuralConstraint, StructuralConstraintKind, StructuralModel,
};
use runmat_meshing_core::SolverMeshArtifact;

#[derive(Debug)]
pub(super) struct LoweringWorkspaceProfile {
    pub rows: ConstraintRowConstructionProfile,
    pub geometry_bytes: u64,
    pub target_record_bytes: u64,
    pub prescribed_value_bytes: u64,
}

pub(super) fn derive(
    constraint: &StructuralConstraint,
    mesh: Option<&SolverMeshArtifact>,
    mixed: Option<(&StructuralModel, &CanonicalStructuralNodeSpace)>,
    prepared: &PreparedTargets<'_>,
) -> Result<LoweringWorkspaceProfile, ConstraintPlanError> {
    check_cancelled()?;
    let mut selections = prepared.selections();
    let first = selections.next();
    let nodes = first.map_or(0, |(target, nodes)| {
        if prepared.requires_mixed() {
            nodes.len()
        } else {
            target.matched_node_identities.len()
        }
    });
    let mut geometry_bytes = 0;
    let mut prescribed_value_bytes = 0;
    let (rows, terms, maximum) = match &constraint.kind {
        StructuralConstraintKind::Fixed { .. } => {
            let rows = if prepared.requires_mixed() {
                let topology = mixed.ok_or_else(crate::constraints::workspace::overflow)?.1;
                first
                    .ok_or_else(crate::constraints::workspace::overflow)?
                    .1
                    .iter()
                    .try_fold(0, |sum, node| {
                        let node = topology
                            .nodes
                            .get(*node)
                            .ok_or_else(crate::constraints::workspace::overflow)?;
                        count_add(sum, if node.owns_explicit_rotation { 6 } else { 3 })
                    })?
            } else {
                multiply(nodes, 3)?
            };
            (rows, rows, usize::from(rows != 0))
        }
        StructuralConstraintKind::Prescribed(values) => {
            let count = if prepared.requires_mixed() {
                count_add(values.translations.len(), values.rotations.len())?
            } else {
                values.translations.len()
            };
            if !prepared.requires_mixed() {
                prescribed_value_bytes = slots::<(crate::constraints::CanonicalStructuralDof, f64)>(
                    values.translations.len(),
                )?;
            }
            if constraint.frame == CoordinateFrame::BoundaryNormalTangent {
                geometry_bytes = surface_bytes(mesh, mixed, prepared, first.map(|v| v.0))?;
            }
            let rows = multiply(nodes, count)?;
            (rows, multiply(rows, 3)?, if rows == 0 { 0 } else { 3 })
        }
        StructuralConstraintKind::PrescribedMotion(values) => {
            let rows = multiply(nodes, values.translations.len())?;
            (rows, rows, usize::from(rows != 0))
        }
        StructuralConstraintKind::Equation(_) => {
            let terms = prepared.remaining_node_count()?;
            (1, terms, terms)
        }
        StructuralConstraintKind::NormalDisplacement { .. }
        | StructuralConstraintKind::Symmetry { .. }
        | StructuralConstraintKind::Antisymmetry { .. } => {
            geometry_bytes = surface_bytes(mesh, mixed, prepared, first.map(|v| v.0))?;
            let rows = multiply(
                nodes,
                if matches!(
                    constraint.kind,
                    StructuralConstraintKind::Antisymmetry { .. }
                ) {
                    2
                } else {
                    1
                },
            )?;
            (rows, multiply(rows, 3)?, if rows == 0 { 0 } else { 3 })
        }
        StructuralConstraintKind::RigidCoupling(_)
        | StructuralConstraintKind::DistributingCoupling(_) => {
            let kind = if matches!(constraint.kind, StructuralConstraintKind::RigidCoupling(_)) {
                ConstraintCouplingKind::Rigid
            } else {
                ConstraintCouplingKind::Distributing
            };
            let profile =
                coupling::profile::rows_profile(nodes, kind, constraint.constraint_id.len())?;
            geometry_bytes = if prepared.requires_mixed() {
                add(
                    surface_bytes(mesh, mixed, prepared, first.map(|v| v.0))?,
                    profile.geometry_workspace_bytes()?,
                )?
            } else {
                coupling::constraint_coupling_workspace_profile(
                    mesh.ok_or_else(crate::constraints::workspace::overflow)?,
                    kind,
                    &constraint.constraint_id,
                )?
                .geometry_workspace_bytes()?
            };
            (
                profile.row_slots,
                profile.retained_term_slots,
                profile.compaction_term_slots,
            )
        }
        StructuralConstraintKind::Tie(pair)
        | StructuralConstraintKind::Periodic(runmat_analysis_core::PeriodicConstraint {
            pair,
            ..
        }) => {
            let mesh = mesh.ok_or_else(|| ConstraintPlanError::UnsupportedTarget { constraint_id: constraint.constraint_id.clone(), reason: "explicit-only tie and periodic pairing requires a qualified geometric topology" })?;
            let primary = first.ok_or_else(crate::constraints::workspace::overflow)?.0;
            let secondary = selections
                .next()
                .ok_or_else(crate::constraints::workspace::overflow)?
                .0;
            let count = if prepared.requires_mixed() {
                mixed
                    .ok_or_else(crate::constraints::workspace::overflow)?
                    .1
                    .nodes
                    .len()
            } else {
                mesh.topology.nodes.len()
            };
            let profile = pairing::profile::derive(mesh, count, primary, secondary, &pair.pairing)?;
            geometry_bytes = profile.total_requested_bytes;
            let rows = multiply(profile.pairing_slots, 3)?;
            let basis = match &constraint.kind {
                StructuralConstraintKind::Periodic(periodic) => {
                    pairing::displacement_transform(&periodic.transform)
                }
                _ => [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            };
            let mut components = 0;
            let mut maximum_components = 0;
            for direction in basis {
                let count = direction.iter().filter(|value| **value != 0.0).count();
                components = count_add(components, count)?;
                maximum_components = maximum_components.max(count);
            }
            // Use the row owner's original transform, including its exact zero decisions.
            // Pairing weights remain bounded by the selected original facets.
            let maximum = count_add(
                1,
                multiply(profile.maximum_weights_per_pairing, maximum_components)?,
            )?;
            let terms = count_add(rows, multiply(profile.retained_weight_slots, components)?)?;
            (rows, terms, if rows == 0 { 0 } else { maximum })
        }
    };
    let rows =
        constraint_row_construction_profile(rows, terms, maximum, &constraint.constraint_id)?;
    let target_record_bytes = slots::<ResolvedConstraintTarget>(prepared.selections().len())?;
    let profile = LoweringWorkspaceProfile {
        rows,
        geometry_bytes,
        target_record_bytes,
        prescribed_value_bytes,
    };
    profile.total_requested_bytes()?;
    Ok(profile)
}

fn surface_bytes(
    mesh: Option<&SolverMeshArtifact>,
    mixed: Option<(&StructuralModel, &CanonicalStructuralNodeSpace)>,
    prepared: &PreparedTargets<'_>,
    target: Option<&ResolvedConstraintTarget>,
) -> Result<u64, ConstraintPlanError> {
    let target = target.ok_or_else(crate::constraints::workspace::overflow)?;
    if prepared.requires_mixed() {
        let structural = mixed.ok_or_else(crate::constraints::workspace::overflow)?.0;
        add(
            shell_surface_workspace_bytes(structural, &target.authored.region_id)?,
            slots::<ConstraintCouplingWeight>(target.matched_node_identities.len())?,
        )
    } else {
        Ok(normal::constraint_normal_workspace_profile(
            mesh.ok_or_else(crate::constraints::workspace::overflow)?,
            target,
        )?
        .total_requested_bytes)
    }
}

#[cfg(test)]
mod tests;

impl LoweringWorkspaceProfile {
    pub(super) fn total_requested_bytes(&self) -> Result<u64, ConstraintPlanError> {
        let total = [
            self.rows.total_requested_bytes,
            self.geometry_bytes,
            self.target_record_bytes,
            self.prescribed_value_bytes,
        ]
        .into_iter()
        .try_fold(0, add)?;
        usize::try_from(total).map_err(|_| crate::constraints::workspace::overflow())?;
        Ok(total)
    }
}
