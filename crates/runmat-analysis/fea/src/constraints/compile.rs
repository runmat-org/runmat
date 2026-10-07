//! One-way lowering from authored structural constraints to solver-topology rows.

use runmat_analysis_core::{
    AnalysisModel, CoordinateFrame, PairedConstraint, PeriodicConstraint, PeriodicTransform,
    PrescribedStructuralDofs, PrescribedStructuralMotion, PrescribedValue, StructuralConstraint,
    StructuralConstraintKind,
};
use runmat_meshing_core::SolverMeshArtifact;

use super::contract::{
    CanonicalStructuralConstraintTerm, CanonicalStructuralDof, CanonicalStructuralDofOwner,
    CanonicalStructuralReferencePoint, ConstraintCouplingKind, ConstraintPlanError,
};
use super::coupling::compile_reference_coupling;
use super::frame::{
    boundary_tangent_basis, require_global, require_normal_capable, translation_basis,
    translation_direction,
};
use super::normal::boundary_node_normals;
use super::pairing::{pair_targets, pairing_rows};
use super::plan::reference_point_identity;
use super::row;
use super::target::{node_coordinates, resolve_nodal_target};
use super::workspace::{check_cancelled, reserve};

use super::compilation::CompiledConstraint;
use super::lowering;
use super::workspace::multiply;

mod reference_points;
pub(super) use reference_points::canonical_reference_points;
pub use reference_points::{
    constraint_reference_point_workspace_profile, ConstraintReferencePointWorkspaceProfile,
};

pub(super) fn compile_constraint(
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
            let mut targets = reserve(equation.terms.len())?;
            let mut terms = Vec::new();
            for term in &equation.terms {
                if !term.coefficient.is_finite() || term.coefficient == 0.0 {
                    return Err(ConstraintPlanError::Invalid(format!(
                        "constraint `{}` contains a zero or non-finite equation coefficient",
                        constraint.constraint_id
                    )));
                }
                let resolved = resolve_nodal_target(mesh, &term.target, &constraint.constraint_id)?;
                terms
                    .try_reserve_exact(resolved.matched_node_identities.len())
                    .map_err(|_| super::workspace::overflow())?;
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
            let mut rows = lowering::rows(1, terms.len(), &constraint.constraint_id)?;
            rows.push(row::new_sparse(
                &constraint.constraint_id,
                terms,
                right_hand_side,
            )?);
            Ok(CompiledConstraint::rows(targets, rows))
        }
        StructuralConstraintKind::NormalDisplacement { target, value } => {
            require_normal_capable(constraint)?;
            let resolved = resolve_nodal_target(mesh, target, &constraint.constraint_id)?;
            let normals = boundary_node_normals(mesh, &resolved, &constraint.constraint_id)?;
            let right_hand_side =
                study_constant_displacement(value, model, &constraint.constraint_id)?;
            let mut rows = lowering::rows(
                resolved.matched_node_identities.len(),
                3,
                &constraint.constraint_id,
            )?;
            for node_identity in &resolved.matched_node_identities {
                rows.push(row::new_direction(
                    &constraint.constraint_id,
                    *node_identity,
                    normals[node_identity],
                    right_hand_side,
                )?);
            }
            Ok(CompiledConstraint::rows(
                lowering::targets([resolved])?,
                rows,
            ))
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
    let resolved = resolve_nodal_target(mesh, target, &constraint.constraint_id)?;
    let mut rows = lowering::rows(
        multiply(resolved.matched_node_identities.len(), 3)?,
        1,
        &constraint.constraint_id,
    )?;
    for node_identity in &resolved.matched_node_identities {
        for dof in CanonicalStructuralDof::TRANSLATIONS {
            rows.push(row::new_singleton(
                &constraint.constraint_id,
                *node_identity,
                dof,
                1.0,
                0.0,
            )?);
        }
    }
    Ok(CompiledConstraint::rows(
        lowering::targets([resolved])?,
        rows,
    ))
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
    let mut prescribed = reserve(values.translations.len())?;
    for value in &values.translations {
        prescribed.push((
            CanonicalStructuralDof::from(value.dof),
            study_constant_displacement(&value.value, model, &constraint.constraint_id)?,
        ));
    }
    let resolved = resolve_nodal_target(mesh, &values.target, &constraint.constraint_id)?;
    let normals = if constraint.frame == CoordinateFrame::BoundaryNormalTangent {
        Some(boundary_node_normals(
            mesh,
            &resolved,
            &constraint.constraint_id,
        )?)
    } else {
        None
    };
    let mut rows = lowering::rows(
        multiply(resolved.matched_node_identities.len(), prescribed.len())?,
        3,
        &constraint.constraint_id,
    )?;
    for node_identity in &resolved.matched_node_identities {
        check_cancelled()?;
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
    Ok(CompiledConstraint::rows(
        lowering::targets([resolved])?,
        rows,
    ))
}

fn compile_prescribed_motion(
    mesh: &SolverMeshArtifact,
    constraint: &StructuralConstraint,
    motion: &PrescribedStructuralMotion,
) -> Result<CompiledConstraint, ConstraintPlanError> {
    require_global(constraint)?;
    let resolved = resolve_nodal_target(mesh, &motion.target, &constraint.constraint_id)?;
    let mut rows = lowering::rows(
        multiply(
            resolved.matched_node_identities.len(),
            motion.translations.len(),
        )?,
        1,
        &constraint.constraint_id,
    )?;
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
    Ok(CompiledConstraint::rows(
        lowering::targets([resolved])?,
        rows,
    ))
}

fn compile_boundary_directions(
    mesh: &SolverMeshArtifact,
    constraint: &StructuralConstraint,
    target: &runmat_analysis_core::RegionTarget,
    tangential: bool,
) -> Result<CompiledConstraint, ConstraintPlanError> {
    require_normal_capable(constraint)?;
    let resolved = resolve_nodal_target(mesh, target, &constraint.constraint_id)?;
    let normals = boundary_node_normals(mesh, &resolved, &constraint.constraint_id)?;
    let count = multiply(
        resolved.matched_node_identities.len(),
        if tangential { 2 } else { 1 },
    )?;
    let mut rows = lowering::rows(count, 3, &constraint.constraint_id)?;
    for node_identity in &resolved.matched_node_identities {
        check_cancelled()?;
        let directions = if tangential {
            boundary_tangent_basis(normals[node_identity], &constraint.constraint_id)?
        } else {
            [normals[node_identity]; 2]
        };
        for direction in directions.into_iter().take(if tangential { 2 } else { 1 }) {
            rows.push(row::new_direction(
                &constraint.constraint_id,
                *node_identity,
                direction,
                0.0,
            )?);
        }
    }
    Ok(CompiledConstraint::rows(
        lowering::targets([resolved])?,
        rows,
    ))
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
    let primary = resolve_nodal_target(mesh, &pair.primary, &constraint.constraint_id)?;
    let secondary = resolve_nodal_target(mesh, &pair.secondary, &constraint.constraint_id)?;
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
    Ok(CompiledConstraint {
        targets: lowering::targets([primary, secondary])?,
        rows,
        pairings,
        coupling: None,
    })
}

fn compile_coupling(
    mesh: &SolverMeshArtifact,
    constraint: &StructuralConstraint,
    coupling: &runmat_analysis_core::ReferencePointCoupling,
    kind: ConstraintCouplingKind,
) -> Result<CompiledConstraint, ConstraintPlanError> {
    require_global(constraint)?;
    let target = resolve_nodal_target(mesh, &coupling.target, &constraint.constraint_id)?;
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
    Ok(CompiledConstraint {
        targets: lowering::targets([target])?,
        rows,
        pairings: Vec::new(),
        coupling: Some(evidence),
    })
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
