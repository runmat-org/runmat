//! One-way lowering from authored structural constraints to solver-topology rows.

use std::collections::{BTreeMap, BTreeSet};

use runmat_analysis_core::{
    AnalysisModel, CoordinateFrame, PairedConstraint, PeriodicConstraint, PeriodicTransform,
    PrescribedStructuralDofs, PrescribedStructuralMotion, PrescribedValue, StructuralConstraint,
    StructuralConstraintKind,
};
use runmat_meshing_core::SolverMeshArtifact;

use super::contract::{
    CanonicalStructuralConstraintPlan, CanonicalStructuralConstraintRow,
    CanonicalStructuralConstraintTerm, CanonicalStructuralDof, CanonicalStructuralDofOwner,
    CanonicalStructuralReferencePoint, ConstraintCouplingEvidence, ConstraintCouplingKind,
    ConstraintEnforcement, ConstraintGroupEvidence, ConstraintPairingEvidence, ConstraintPlanError,
    ConstraintPlanEvidence, ResolvedConstraintTarget,
};
use super::coupling::compile_reference_coupling;
use super::enforcement;
use super::frame::{
    boundary_tangent_basis, require_global, require_normal_capable, translation_basis,
    translation_direction,
};
use super::normal::boundary_node_normals;
use super::pairing::{pair_targets, pairing_rows};
use super::plan::{
    authored_constraints_digest, reference_point_identity, source_digest, structural_constraints,
};
use super::row;
use super::target::{node_coordinates, resolve_solid_target};
use crate::physics::structural::topology::CanonicalStructuralTopologyIdentity;

type CompiledConstraint = (
    Vec<ResolvedConstraintTarget>,
    Vec<CanonicalStructuralConstraintRow>,
    Vec<ConstraintPairingEvidence>,
    Option<ConstraintCouplingEvidence>,
);

pub(super) fn compile(
    model: &AnalysisModel,
    mesh: &SolverMeshArtifact,
) -> Result<CanonicalStructuralConstraintPlan, ConstraintPlanError> {
    mesh.validate()
        .map_err(|error| ConstraintPlanError::Invalid(error.to_string()))?;
    let authored = structural_constraints(model);
    validate_unique_ids(&authored)?;
    let reference_points = canonical_reference_points(&authored)?;
    let authored_constraints_digest = authored_constraints_digest(&authored)?;
    let mut groups = Vec::with_capacity(authored.len());
    let mut rows = Vec::new();
    let mut enforcement = ConstraintEnforcement::DirectElimination;

    for constraint in authored {
        runmat_analysis_core::validate_coordinate_frame(&constraint.frame).map_err(|_| {
            ConstraintPlanError::DegenerateFrame {
                constraint_id: constraint.constraint_id.clone(),
            }
        })?;
        let (mut targets, compiled_rows, pairings, coupling) =
            compile_constraint(model, mesh, constraint)?;
        targets.sort_by_key(|target| target.selection_digest);
        targets.dedup_by_key(|target| target.selection_digest);
        enforcement = enforcement::combine(
            enforcement,
            enforcement::select(&constraint.kind, &compiled_rows),
        );
        let first_row = rows.len();
        rows.extend(compiled_rows);
        groups.push(ConstraintGroupEvidence {
            constraint_id: constraint.constraint_id.clone(),
            source_digest: source_digest(constraint)?,
            implementation:
                runmat_analysis_core::BoundaryTransferImplementation::CanonicalBoundaryProjection,
            targets,
            reference_point_identities: coupling
                .iter()
                .map(|coupling| coupling.reference_point_identity)
                .collect(),
            coupling,
            pairings,
            row_count: rows.len() - first_row,
        });
    }

    groups.sort_by(|left, right| left.constraint_id.cmp(&right.constraint_id));
    rows.sort_by(row::compare_rows);
    let (active_row_count, redundant_row_count) = row::classify(&mut rows)?;
    let plan = CanonicalStructuralConstraintPlan {
        schema_version: 1,
        topology: CanonicalStructuralTopologyIdentity::Solid {
            mesh_digest: mesh.canonical_digest,
        },
        authored_constraints_digest,
        enforcement,
        reference_points,
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

pub(super) fn canonical_reference_points(
    constraints: &[&StructuralConstraint],
) -> Result<Vec<CanonicalStructuralReferencePoint>, ConstraintPlanError> {
    let mut by_id = BTreeMap::<String, [f64; 3]>::new();
    for constraint in constraints {
        let coupling = match &constraint.kind {
            StructuralConstraintKind::RigidCoupling(coupling)
            | StructuralConstraintKind::DistributingCoupling(coupling) => coupling,
            _ => continue,
        };
        let id = coupling.reference_point.reference_point_id.trim();
        let position = coupling.reference_point.position.meters;
        if id.is_empty() || position.iter().any(|value| !value.is_finite()) {
            return Err(ConstraintPlanError::Invalid(format!(
                "constraint `{}` has an invalid reference point",
                constraint.constraint_id
            )));
        }
        if by_id
            .insert(id.into(), position)
            .is_some_and(|existing| existing != position)
        {
            return Err(ConstraintPlanError::Invalid(format!(
                "reference point `{id}` has conflicting positions"
            )));
        }
    }
    let mut points = by_id
        .into_iter()
        .map(|(reference_point_id, position_meters)| {
            Ok(CanonicalStructuralReferencePoint {
                identity: reference_point_identity(&reference_point_id, position_meters)?,
                reference_point_id,
                position_meters,
            })
        })
        .collect::<Result<Vec<_>, ConstraintPlanError>>()?;
    points.sort_by_key(|point| point.identity);
    if points
        .windows(2)
        .any(|pair| pair[0].identity == pair[1].identity)
    {
        return Err(ConstraintPlanError::Invalid(
            "reference-point identities collide".into(),
        ));
    }
    Ok(points)
}

fn compile_constraint(
    model: &AnalysisModel,
    mesh: &SolverMeshArtifact,
    constraint: &StructuralConstraint,
) -> Result<CompiledConstraint, ConstraintPlanError> {
    match &constraint.kind {
        StructuralConstraintKind::Fixed { target } => compile_fixed(mesh, constraint, target),
        StructuralConstraintKind::Prescribed(values) => {
            compile_prescribed(model, mesh, constraint, values)
        }
        StructuralConstraintKind::PrescribedMotion(values) => {
            compile_prescribed_motion(mesh, constraint, values)
        }
        StructuralConstraintKind::Equation(equation) => {
            require_global(constraint)?;
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
                            owner: CanonicalStructuralDofOwner::StructuralNode(*node_identity),
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
                Vec::new(),
                None,
            ))
        }
        StructuralConstraintKind::NormalDisplacement { target, value } => {
            require_normal_capable(constraint)?;
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
            Ok((vec![resolved], rows, Vec::new(), None))
        }
        StructuralConstraintKind::Symmetry { target } => {
            compile_boundary_directions(mesh, constraint, target, false)
        }
        StructuralConstraintKind::Antisymmetry { target } => {
            compile_boundary_directions(mesh, constraint, target, true)
        }
        StructuralConstraintKind::RigidCoupling(coupling) => {
            compile_coupling(mesh, constraint, coupling, ConstraintCouplingKind::Rigid)
        }
        StructuralConstraintKind::DistributingCoupling(coupling) => compile_coupling(
            mesh,
            constraint,
            coupling,
            ConstraintCouplingKind::Distributing,
        ),
        StructuralConstraintKind::Tie(pair) => {
            compile_pair(mesh, constraint, pair, &PeriodicTransform::Identity, false)
        }
        StructuralConstraintKind::Periodic(periodic) => {
            compile_periodic(mesh, constraint, periodic)
        }
    }
}

fn compile_fixed(
    mesh: &SolverMeshArtifact,
    constraint: &StructuralConstraint,
    target: &runmat_analysis_core::RegionTarget,
) -> Result<CompiledConstraint, ConstraintPlanError> {
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
    Ok((vec![resolved], rows, Vec::new(), None))
}

fn compile_prescribed(
    model: &AnalysisModel,
    mesh: &SolverMeshArtifact,
    constraint: &StructuralConstraint,
    values: &PrescribedStructuralDofs,
) -> Result<CompiledConstraint, ConstraintPlanError> {
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
    Ok((vec![resolved], rows, Vec::new(), None))
}

fn compile_prescribed_motion(
    mesh: &SolverMeshArtifact,
    constraint: &StructuralConstraint,
    motion: &PrescribedStructuralMotion,
) -> Result<CompiledConstraint, ConstraintPlanError> {
    require_global(constraint)?;
    let resolved = resolve_solid_target(mesh, &motion.target, &constraint.constraint_id)?;
    let mut rows = Vec::with_capacity(
        resolved
            .matched_node_identities
            .len()
            .saturating_mul(motion.translations.len()),
    );
    for node_identity in &resolved.matched_node_identities {
        for translation in &motion.translations {
            rows.push(row::new_singleton(
                &constraint.constraint_id,
                *node_identity,
                translation.dof.into(),
                1.0,
                0.0,
            )?);
        }
    }
    Ok((vec![resolved], rows, Vec::new(), None))
}

fn compile_boundary_directions(
    mesh: &SolverMeshArtifact,
    constraint: &StructuralConstraint,
    target: &runmat_analysis_core::RegionTarget,
    tangential: bool,
) -> Result<CompiledConstraint, ConstraintPlanError> {
    require_normal_capable(constraint)?;
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
    Ok((vec![resolved], rows, Vec::new(), None))
}

fn compile_periodic(
    mesh: &SolverMeshArtifact,
    constraint: &StructuralConstraint,
    periodic: &PeriodicConstraint,
) -> Result<CompiledConstraint, ConstraintPlanError> {
    compile_pair(mesh, constraint, &periodic.pair, &periodic.transform, true)
}

fn compile_pair(
    mesh: &SolverMeshArtifact,
    constraint: &StructuralConstraint,
    pair: &PairedConstraint,
    transform: &PeriodicTransform,
    transform_displacement: bool,
) -> Result<CompiledConstraint, ConstraintPlanError> {
    require_global(constraint)?;
    let primary = resolve_solid_target(mesh, &pair.primary, &constraint.constraint_id)?;
    let secondary = resolve_solid_target(mesh, &pair.secondary, &constraint.constraint_id)?;
    let pairings = pair_targets(
        mesh,
        &primary,
        &secondary,
        &pair.pairing,
        transform,
        &constraint.constraint_id,
    )?;
    let rows = pairing_rows(
        &constraint.constraint_id,
        &pairings,
        transform,
        transform_displacement,
    )?;
    Ok((vec![primary, secondary], rows, pairings, None))
}

fn compile_coupling(
    mesh: &SolverMeshArtifact,
    constraint: &StructuralConstraint,
    coupling: &runmat_analysis_core::ReferencePointCoupling,
    kind: ConstraintCouplingKind,
) -> Result<CompiledConstraint, ConstraintPlanError> {
    require_global(constraint)?;
    let target = resolve_solid_target(mesh, &coupling.target, &constraint.constraint_id)?;
    let identity = reference_point_identity(
        &coupling.reference_point.reference_point_id,
        coupling.reference_point.position.meters,
    )?;
    let (rows, evidence) = compile_reference_coupling(
        mesh,
        &target,
        &coupling.reference_point,
        identity,
        kind,
        &constraint.constraint_id,
    )?;
    Ok((vec![target], rows, Vec::new(), Some(evidence)))
}

pub(super) fn study_constant_displacement(
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
