//! One-way lowering from authored structural constraints to solver-topology rows.

use std::collections::BTreeSet;

use runmat_analysis_core::{
    AnalysisModel, CoordinateFrame, PrescribedStructuralDofs, PrescribedValue,
    StructuralConstraint, StructuralConstraintKind,
};
use runmat_meshing_core::{SolverMeshArtifact, StableDigest};

use super::contract::{
    CanonicalStructuralConstraintPlan, CanonicalStructuralConstraintRow,
    CanonicalStructuralConstraintTerm, CanonicalStructuralDof, ConstraintEnforcement,
    ConstraintGroupEvidence, ConstraintPlanError, ConstraintPlanEvidence, ResolvedConstraintTarget,
};
use super::frame::{boundary_tangent_basis, translation_basis};
use super::normal::boundary_node_normals;
use super::plan::{authored_constraints_digest, source_digest, structural_constraints};
use super::row;
use super::target::resolve_solid_target;

pub(super) fn compile(
    model: &AnalysisModel,
    mesh: &SolverMeshArtifact,
) -> Result<CanonicalStructuralConstraintPlan, ConstraintPlanError> {
    mesh.validate()
        .map_err(|error| ConstraintPlanError::Invalid(error.to_string()))?;
    let authored = structural_constraints(model);
    validate_unique_ids(&authored)?;
    let authored_constraints_digest = authored_constraints_digest(&authored)?;
    let mut groups = Vec::with_capacity(authored.len());
    let mut rows = Vec::new();

    for constraint in authored {
        runmat_analysis_core::validate_coordinate_frame(&constraint.frame).map_err(|_| {
            ConstraintPlanError::DegenerateFrame {
                constraint_id: constraint.constraint_id.clone(),
            }
        })?;
        let (mut targets, compiled_rows) = compile_constraint(model, mesh, constraint)?;
        targets.sort_by_key(|target| target.selection_digest);
        targets.dedup_by_key(|target| target.selection_digest);
        let enforcement = enforcement_for(&compiled_rows);
        let first_row = rows.len();
        rows.extend(compiled_rows);
        groups.push(ConstraintGroupEvidence {
            constraint_id: constraint.constraint_id.clone(),
            source_digest: source_digest(constraint)?,
            targets,
            enforcement,
            row_count: rows.len() - first_row,
        });
    }

    groups.sort_by(|left, right| left.constraint_id.cmp(&right.constraint_id));
    rows.sort_by(row::compare_rows);
    let (active_row_count, redundant_row_count) = row::classify(&mut rows)?;
    let plan = CanonicalStructuralConstraintPlan {
        schema_version: 1,
        topology_digest: mesh.canonical_digest,
        geometry_source_digest: mesh.geometry.source_digest,
        geometry_revision: mesh.geometry.geometry_revision,
        persistent_mapping_version: mesh.geometry.persistent_mapping_version,
        authored_constraints_digest,
        evidence: ConstraintPlanEvidence {
            authored_constraint_count: groups.len(),
            active_row_count,
            redundant_row_count,
            conflicting_row_count: 0,
            rank: active_row_count,
        },
        groups,
        rows,
    };
    plan.validate()?;
    Ok(plan)
}

fn validate_unique_ids(constraints: &[&StructuralConstraint]) -> Result<(), ConstraintPlanError> {
    let mut ids = BTreeSet::new();
    for constraint in constraints {
        if !ids.insert(constraint.constraint_id.as_str()) {
            return Err(ConstraintPlanError::DuplicateConstraintId {
                constraint_id: constraint.constraint_id.clone(),
            });
        }
    }
    Ok(())
}

fn compile_constraint(
    model: &AnalysisModel,
    mesh: &SolverMeshArtifact,
    constraint: &StructuralConstraint,
) -> Result<
    (
        Vec<ResolvedConstraintTarget>,
        Vec<CanonicalStructuralConstraintRow>,
    ),
    ConstraintPlanError,
> {
    match &constraint.kind {
        StructuralConstraintKind::Fixed { target } => compile_fixed(mesh, constraint, target),
        StructuralConstraintKind::Prescribed(values) => {
            compile_prescribed(model, mesh, constraint, values)
        }
        StructuralConstraintKind::Equation(equation) => {
            require_global_frame(constraint)?;
            if equation.terms.is_empty() {
                return Err(ConstraintPlanError::Invalid(format!(
                    "constraint `{}` has no equation terms",
                    constraint.constraint_id
                )));
            }
            let mut targets = Vec::with_capacity(equation.terms.len());
            let mut terms = Vec::new();
            for term in &equation.terms {
                if !term.coefficient.is_finite() || term.coefficient == 0.0 {
                    return Err(ConstraintPlanError::Invalid(format!(
                        "constraint `{}` contains a zero or non-finite equation coefficient",
                        constraint.constraint_id
                    )));
                }
                let resolved = resolve_solid_target(mesh, &term.target, &constraint.constraint_id)?;
                terms.extend(
                    resolved
                        .matched_node_identities
                        .iter()
                        .map(|node_identity| CanonicalStructuralConstraintTerm {
                            node_identity: *node_identity,
                            dof: term.dof.into(),
                            coefficient: term.coefficient,
                        }),
                );
                targets.push(resolved);
            }
            let right_hand_side = equation.right_hand_side.meters;
            if !right_hand_side.is_finite() {
                return Err(ConstraintPlanError::NonFiniteValue {
                    constraint_id: constraint.constraint_id.clone(),
                });
            }
            Ok((
                targets,
                vec![row::new_sparse(
                    &constraint.constraint_id,
                    terms,
                    right_hand_side,
                )?],
            ))
        }
        StructuralConstraintKind::NormalDisplacement { target, value } => {
            require_normal_frame(constraint)?;
            let resolved = resolve_solid_target(mesh, target, &constraint.constraint_id)?;
            let normals = boundary_node_normals(mesh, &resolved, &constraint.constraint_id)?;
            let right_hand_side =
                study_constant_displacement(value, model, &constraint.constraint_id)?;
            let rows = resolved
                .matched_node_identities
                .iter()
                .map(|node_identity| {
                    row::new_direction(
                        &constraint.constraint_id,
                        *node_identity,
                        normals[node_identity],
                        right_hand_side,
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok((vec![resolved], rows))
        }
        StructuralConstraintKind::Symmetry { target } => {
            compile_boundary_directions(mesh, constraint, target, false)
        }
        StructuralConstraintKind::Antisymmetry { target } => {
            compile_boundary_directions(mesh, constraint, target, true)
        }
        kind => Err(ConstraintPlanError::UnsupportedKind {
            constraint_id: constraint.constraint_id.clone(),
            kind: structural_kind_name(kind),
        }),
    }
}

fn compile_fixed(
    mesh: &SolverMeshArtifact,
    constraint: &StructuralConstraint,
    target: &runmat_analysis_core::RegionTarget,
) -> Result<
    (
        Vec<ResolvedConstraintTarget>,
        Vec<CanonicalStructuralConstraintRow>,
    ),
    ConstraintPlanError,
> {
    let resolved = resolve_solid_target(mesh, target, &constraint.constraint_id)?;
    let rows = resolved
        .matched_node_identities
        .iter()
        .flat_map(|node_identity| {
            CanonicalStructuralDof::TRANSLATIONS.map(|dof| {
                row::new_singleton(&constraint.constraint_id, *node_identity, dof, 1.0, 0.0)
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok((vec![resolved], rows))
}

fn compile_prescribed(
    model: &AnalysisModel,
    mesh: &SolverMeshArtifact,
    constraint: &StructuralConstraint,
    values: &PrescribedStructuralDofs,
) -> Result<
    (
        Vec<ResolvedConstraintTarget>,
        Vec<CanonicalStructuralConstraintRow>,
    ),
    ConstraintPlanError,
> {
    if !values.rotations.is_empty() {
        return Err(ConstraintPlanError::UnavailableRotationalDof {
            constraint_id: constraint.constraint_id.clone(),
        });
    }
    let prescribed = values
        .translations
        .iter()
        .map(|value| {
            Ok((
                CanonicalStructuralDof::from(value.dof),
                study_constant_displacement(&value.value, model, &constraint.constraint_id)?,
            ))
        })
        .collect::<Result<Vec<_>, ConstraintPlanError>>()?;
    let resolved = resolve_solid_target(mesh, &values.target, &constraint.constraint_id)?;
    let normals = if constraint.frame == CoordinateFrame::BoundaryNormalTangent {
        Some(boundary_node_normals(
            mesh,
            &resolved,
            &constraint.constraint_id,
        )?)
    } else {
        None
    };
    let mut rows = Vec::new();
    for node_identity in &resolved.matched_node_identities {
        let point = node_coordinates(mesh, *node_identity, &constraint.constraint_id)?;
        let normal = normals
            .as_ref()
            .and_then(|normals| normals.get(node_identity))
            .copied();
        let basis = translation_basis(&constraint.frame, point, normal, &constraint.constraint_id)?;
        for &(dof, value) in &prescribed {
            rows.push(row::new_direction(
                &constraint.constraint_id,
                *node_identity,
                translation_direction(dof, basis),
                value,
            )?);
        }
    }
    Ok((vec![resolved], rows))
}

fn compile_boundary_directions(
    mesh: &SolverMeshArtifact,
    constraint: &StructuralConstraint,
    target: &runmat_analysis_core::RegionTarget,
    tangential: bool,
) -> Result<
    (
        Vec<ResolvedConstraintTarget>,
        Vec<CanonicalStructuralConstraintRow>,
    ),
    ConstraintPlanError,
> {
    require_normal_frame(constraint)?;
    let resolved = resolve_solid_target(mesh, target, &constraint.constraint_id)?;
    let normals = boundary_node_normals(mesh, &resolved, &constraint.constraint_id)?;
    let mut rows = Vec::new();
    for node_identity in &resolved.matched_node_identities {
        let directions = if tangential {
            boundary_tangent_basis(normals[node_identity], &constraint.constraint_id)?.to_vec()
        } else {
            vec![normals[node_identity]]
        };
        for direction in directions {
            rows.push(row::new_direction(
                &constraint.constraint_id,
                *node_identity,
                direction,
                0.0,
            )?);
        }
    }
    Ok((vec![resolved], rows))
}

fn enforcement_for(rows: &[CanonicalStructuralConstraintRow]) -> ConstraintEnforcement {
    if rows.iter().all(|row| row.terms.len() == 1) {
        ConstraintEnforcement::DirectElimination
    } else {
        ConstraintEnforcement::AffineReduction
    }
}

fn require_global_frame(constraint: &StructuralConstraint) -> Result<(), ConstraintPlanError> {
    if constraint.frame != CoordinateFrame::Global {
        return Err(ConstraintPlanError::UnsupportedFrame {
            constraint_id: constraint.constraint_id.clone(),
            frame: constraint.frame.context_label(),
        });
    }
    Ok(())
}

fn require_normal_frame(constraint: &StructuralConstraint) -> Result<(), ConstraintPlanError> {
    if !matches!(
        constraint.frame,
        CoordinateFrame::Global | CoordinateFrame::BoundaryNormalTangent
    ) {
        return Err(ConstraintPlanError::UnsupportedFrame {
            constraint_id: constraint.constraint_id.clone(),
            frame: constraint.frame.context_label(),
        });
    }
    Ok(())
}

fn node_coordinates(
    mesh: &SolverMeshArtifact,
    identity: StableDigest,
    constraint_id: &str,
) -> Result<[f64; 3], ConstraintPlanError> {
    mesh.topology
        .nodes
        .iter()
        .find(|node| node.stable_identity == identity)
        .map(|node| node.coordinates_m)
        .ok_or_else(|| {
            ConstraintPlanError::Invalid(format!(
                "constraint `{constraint_id}` references a node absent from solver topology"
            ))
        })
}

fn translation_direction(dof: CanonicalStructuralDof, basis: [[f64; 3]; 3]) -> [f64; 3] {
    match dof {
        CanonicalStructuralDof::Translation(runmat_analysis_core::StructuralTranslationDof::Ux) => {
            basis[0]
        }
        CanonicalStructuralDof::Translation(runmat_analysis_core::StructuralTranslationDof::Uy) => {
            basis[1]
        }
        CanonicalStructuralDof::Translation(runmat_analysis_core::StructuralTranslationDof::Uz) => {
            basis[2]
        }
        CanonicalStructuralDof::Rotation(_) => {
            unreachable!("translation values contain no rotational DOF")
        }
    }
}

fn study_constant_displacement(
    value: &PrescribedValue<runmat_analysis_core::Displacement>,
    model: &AnalysisModel,
    constraint_id: &str,
) -> Result<f64, ConstraintPlanError> {
    let value = value
        .study_constant(&model.signals)
        .ok_or_else(|| ConstraintPlanError::ScheduledValue {
            constraint_id: constraint_id.into(),
        })?
        .meters;
    if !value.is_finite() {
        return Err(ConstraintPlanError::NonFiniteValue {
            constraint_id: constraint_id.into(),
        });
    }
    Ok(value)
}

fn structural_kind_name(kind: &StructuralConstraintKind) -> &'static str {
    match kind {
        StructuralConstraintKind::Fixed { .. } => "fixed",
        StructuralConstraintKind::Prescribed(_) => "prescribed",
        StructuralConstraintKind::NormalDisplacement { .. } => "normal_displacement",
        StructuralConstraintKind::Symmetry { .. } => "symmetry",
        StructuralConstraintKind::Antisymmetry { .. } => "antisymmetry",
        StructuralConstraintKind::RigidCoupling(_) => "rigid_coupling",
        StructuralConstraintKind::DistributingCoupling(_) => "distributing_coupling",
        StructuralConstraintKind::Tie(_) => "tie",
        StructuralConstraintKind::Periodic(_) => "periodic",
        StructuralConstraintKind::Equation(_) => "equation",
    }
}
