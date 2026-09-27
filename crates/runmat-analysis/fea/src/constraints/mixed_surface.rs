//! Explicit-shell surface constraints in explicit and conformal-mixed node spaces.

use runmat_analysis_core::{
    AnalysisModel, CoordinateFrame, PrescribedStructuralDofs, StructuralConstraint,
};
use runmat_meshing_core::SolverMeshArtifact;

use crate::physics::structural::topology::node_space::CanonicalStructuralNodeSpace;

use super::{
    compilation::CompiledConstraint,
    coupling::compile_reference_coupling_for_nodes,
    explicit_surface::resolve_explicit_surface,
    frame::{
        boundary_tangent_basis, require_global, require_normal_capable, translation_basis,
        translation_direction,
    },
    mixed::resolve_target,
    plan::reference_point_identity,
    row, CanonicalStructuralDof, ConstraintCouplingKind, ConstraintPlanError,
};

pub(super) fn compile_prescribed(
    model: &AnalysisModel,
    mesh: Option<&SolverMeshArtifact>,
    structural: &runmat_analysis_core::StructuralModel,
    topology: &CanonicalStructuralNodeSpace,
    constraint: &StructuralConstraint,
    values: &PrescribedStructuralDofs,
) -> Result<CompiledConstraint, ConstraintPlanError> {
    let (resolved, nodes) = resolve_target(mesh, structural, topology, constraint, &values.target)?;
    let boundary_normals = if constraint.frame == CoordinateFrame::BoundaryNormalTangent {
        Some(
            resolve_explicit_surface(structural, topology, &resolved, &constraint.constraint_id)?
                .normals,
        )
    } else {
        None
    };
    let mut rows = Vec::new();
    for node in nodes {
        let canonical = &topology.nodes[node];
        let basis = translation_basis(
            &constraint.frame,
            canonical.coordinates_m,
            boundary_normals
                .as_ref()
                .and_then(|normals| normals.get(&canonical.identity))
                .copied(),
            &constraint.constraint_id,
        )?;
        for prescribed in &values.translations {
            let value = prescribed
                .value
                .study_constant(&model.signals)
                .ok_or_else(|| ConstraintPlanError::ScheduledValue {
                    constraint_id: constraint.constraint_id.clone(),
                })?
                .meters;
            rows.push(row::new_direction(
                &constraint.constraint_id,
                canonical.identity,
                translation_direction(prescribed.dof.into(), basis),
                finite_value(value, constraint)?,
            )?);
        }
        if !values.rotations.is_empty() && !canonical.owns_explicit_rotation {
            return Err(ConstraintPlanError::UnavailableRotationalDof {
                constraint_id: constraint.constraint_id.clone(),
            });
        }
        if !values.rotations.is_empty() && constraint.frame != CoordinateFrame::Global {
            return Err(ConstraintPlanError::UnsupportedFrame {
                constraint_id: constraint.constraint_id.clone(),
                frame: constraint.frame.context_label(),
            });
        }
        for prescribed in &values.rotations {
            let value = prescribed
                .value
                .study_constant(&model.signals)
                .ok_or_else(|| ConstraintPlanError::ScheduledValue {
                    constraint_id: constraint.constraint_id.clone(),
                })?
                .radians;
            rows.push(row::new_singleton(
                &constraint.constraint_id,
                canonical.identity,
                CanonicalStructuralDof::Rotation(prescribed.dof),
                1.0,
                finite_value(value, constraint)?,
            )?);
        }
    }
    Ok(CompiledConstraint::rows(vec![resolved], rows))
}

pub(super) fn compile_normal(
    model: &AnalysisModel,
    mesh: Option<&SolverMeshArtifact>,
    structural: &runmat_analysis_core::StructuralModel,
    topology: &CanonicalStructuralNodeSpace,
    constraint: &StructuralConstraint,
    target: &runmat_analysis_core::RegionTarget,
    value: &runmat_analysis_core::PrescribedValue<runmat_analysis_core::Displacement>,
) -> Result<CompiledConstraint, ConstraintPlanError> {
    require_normal_capable(constraint)?;
    let (resolved, _) = resolve_target(mesh, structural, topology, constraint, target)?;
    let surface =
        resolve_explicit_surface(structural, topology, &resolved, &constraint.constraint_id)?;
    let right_hand_side =
        super::compile::study_constant_displacement(value, model, &constraint.constraint_id)?;
    let rows = resolved
        .matched_node_identities
        .iter()
        .map(|identity| {
            row::new_direction(
                &constraint.constraint_id,
                *identity,
                surface.normals[identity],
                right_hand_side,
            )
        })
        .collect::<Result<_, _>>()?;
    Ok(CompiledConstraint::rows(vec![resolved], rows))
}

pub(super) fn compile_directions(
    mesh: Option<&SolverMeshArtifact>,
    structural: &runmat_analysis_core::StructuralModel,
    topology: &CanonicalStructuralNodeSpace,
    constraint: &StructuralConstraint,
    target: &runmat_analysis_core::RegionTarget,
    tangential: bool,
) -> Result<CompiledConstraint, ConstraintPlanError> {
    require_normal_capable(constraint)?;
    let (resolved, _) = resolve_target(mesh, structural, topology, constraint, target)?;
    let surface =
        resolve_explicit_surface(structural, topology, &resolved, &constraint.constraint_id)?;
    let mut rows = Vec::new();
    for identity in &resolved.matched_node_identities {
        let directions = if tangential {
            boundary_tangent_basis(surface.normals[identity], &constraint.constraint_id)?.to_vec()
        } else {
            vec![surface.normals[identity]]
        };
        for direction in directions {
            rows.push(row::new_direction(
                &constraint.constraint_id,
                *identity,
                direction,
                0.0,
            )?);
        }
    }
    Ok(CompiledConstraint::rows(vec![resolved], rows))
}

pub(super) fn compile_coupling(
    mesh: Option<&SolverMeshArtifact>,
    structural: &runmat_analysis_core::StructuralModel,
    topology: &CanonicalStructuralNodeSpace,
    constraint: &StructuralConstraint,
    coupling: &runmat_analysis_core::ReferencePointCoupling,
    kind: ConstraintCouplingKind,
) -> Result<CompiledConstraint, ConstraintPlanError> {
    require_global(constraint)?;
    let (target, _) = resolve_target(mesh, structural, topology, constraint, &coupling.target)?;
    let surface =
        resolve_explicit_surface(structural, topology, &target, &constraint.constraint_id)?;
    let identity = reference_point_identity(
        &coupling.reference_point.reference_point_id,
        coupling.reference_point.position.meters,
    )?;
    let (rows, evidence) = compile_reference_coupling_for_nodes(
        &surface.coordinates,
        surface.weights,
        &coupling.reference_point,
        identity,
        kind,
        &constraint.constraint_id,
    )?;
    Ok(CompiledConstraint::coupling(target, rows, evidence))
}

fn finite_value(value: f64, constraint: &StructuralConstraint) -> Result<f64, ConstraintPlanError> {
    value
        .is_finite()
        .then_some(value)
        .ok_or_else(|| ConstraintPlanError::NonFiniteValue {
            constraint_id: constraint.constraint_id.clone(),
        })
}
