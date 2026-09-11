use runmat_analysis_core::{
    AnalysisModel, StructuralConstraintKind, StructuralModel, StructuralRotationDof,
    StructuralTranslationDof,
};

use super::{
    direct_constraints::DirectConstraintAssignments, dofs::StructuralDofKind,
    structural_constraint_kind_name, structural_target_nodes, StructuralAssemblyError,
};

pub(super) fn apply_explicit_constraints(
    model: &AnalysisModel,
    structural: &StructuralModel,
    canonical_node_indices: &[usize],
    assignments: &mut DirectConstraintAssignments<'_>,
    strict_mapping: bool,
) -> Result<(), StructuralAssemblyError> {
    if canonical_node_indices.len() != structural.nodes.len() {
        return Err(StructuralAssemblyError::InvalidCanonicalNodeMap {
            expected: structural.nodes.len(),
            actual: canonical_node_indices.len(),
        });
    }
    for constraint in model
        .constraints
        .iter()
        .filter_map(|value| value.structural())
    {
        let target = constraint.kind.target().ok_or_else(|| {
            StructuralAssemblyError::UnsupportedStructuralConstraint {
                constraint_id: constraint.constraint_id.clone(),
                constraint_kind: structural_constraint_kind_name(&constraint.kind),
            }
        })?;
        let explicit_nodes = structural_target_nodes(structural, &target.region_id);
        if explicit_nodes.is_empty() {
            if strict_mapping {
                return Err(StructuralAssemblyError::UnmappedExplicitConstraint {
                    constraint_id: constraint.constraint_id.clone(),
                    region_id: target.region_id.clone(),
                });
            }
            continue;
        }
        if !matches!(
            constraint.kind,
            StructuralConstraintKind::Fixed { .. } | StructuralConstraintKind::Prescribed(_)
        ) {
            return Err(StructuralAssemblyError::UnsupportedStructuralConstraint {
                constraint_id: constraint.constraint_id.clone(),
                constraint_kind: structural_constraint_kind_name(&constraint.kind),
            });
        }
        for explicit_node in explicit_nodes {
            let canonical_node = canonical_node_indices[explicit_node];
            match &constraint.kind {
                StructuralConstraintKind::Fixed { .. } => {
                    for kind in StructuralDofKind::ORDER {
                        assignments.assign(&constraint.constraint_id, canonical_node, kind, 0.0)?;
                    }
                }
                StructuralConstraintKind::Prescribed(values) => {
                    for prescribed in &values.translations {
                        let value =
                            prescribed
                                .value
                                .study_constant(&model.signals)
                                .ok_or_else(|| {
                                    StructuralAssemblyError::UnsupportedScheduledConstraint {
                                        constraint_id: constraint.constraint_id.clone(),
                                    }
                                })?;
                        let kind = match prescribed.dof {
                            StructuralTranslationDof::Ux => StructuralDofKind::Ux,
                            StructuralTranslationDof::Uy => StructuralDofKind::Uy,
                            StructuralTranslationDof::Uz => StructuralDofKind::Uz,
                        };
                        assignments.assign(
                            &constraint.constraint_id,
                            canonical_node,
                            kind,
                            value.meters,
                        )?;
                    }
                    for prescribed in &values.rotations {
                        let value =
                            prescribed
                                .value
                                .study_constant(&model.signals)
                                .ok_or_else(|| {
                                    StructuralAssemblyError::UnsupportedScheduledConstraint {
                                        constraint_id: constraint.constraint_id.clone(),
                                    }
                                })?;
                        let kind = match prescribed.dof {
                            StructuralRotationDof::Rx => StructuralDofKind::Rx,
                            StructuralRotationDof::Ry => StructuralDofKind::Ry,
                            StructuralRotationDof::Rz => StructuralDofKind::Rz,
                        };
                        assignments.assign(
                            &constraint.constraint_id,
                            canonical_node,
                            kind,
                            value.radians,
                        )?;
                    }
                }
                _ => unreachable!("constraint kind was checked above"),
            }
        }
    }
    Ok(())
}
