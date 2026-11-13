//! Explicit-shell surface constraints in explicit and conformal-mixed node spaces.

use runmat_analysis_core::{
    AnalysisModel, CoordinateFrame, PrescribedStructuralDofs, StructuralConstraint,
};

use crate::physics::structural::topology::node_space::CanonicalStructuralNodeSpace;

use super::{
    compilation::CompiledConstraint,
    coupling::compile_reference_coupling_for_nodes,
    explicit_surface::resolve_explicit_surface,
    frame::{
        boundary_tangent_basis, require_global, require_normal_capable, translation_basis,
        translation_direction,
    },
    lowering,
    plan::reference_point_identity,
    prepared::PreparedTargets,
    row,
    workspace::{count_add, multiply},
    CanonicalStructuralDof, ConstraintCouplingKind, ConstraintPlanError,
};

pub(super) fn compile_prescribed(
    model: &AnalysisModel,
    structural: &runmat_analysis_core::StructuralModel,
    topology: &CanonicalStructuralNodeSpace,
    constraint: &StructuralConstraint,
    prepared: &mut PreparedTargets<'_>,
    values: &PrescribedStructuralDofs,
) -> Result<CompiledConstraint, ConstraintPlanError> {
    let (resolved, nodes) = prepared.take(&values.target)?;
    let boundary_normals = if constraint.frame == CoordinateFrame::BoundaryNormalTangent {
        Some(
            resolve_explicit_surface(structural, topology, &resolved, &constraint.constraint_id)?
                .nodes,
        )
    } else {
        None
    };
    let count = multiply(
        nodes.len(),
        count_add(values.translations.len(), values.rotations.len())?,
    )?;
    let mut rows = lowering::rows(count, 3, &constraint.constraint_id)?;
    for node in nodes {
        let canonical = &topology.nodes[node];
        let basis = translation_basis(
            &constraint.frame,
            canonical.coordinates_m.canonical_value(),
            boundary_normals
                .as_ref()
                .and_then(|normals| normals.get(&canonical.identity))
                .map(|node| node.oriented_normal),
            &constraint.constraint_id,
        )?;
        for prescribed in &values.translations {
            let value = prescribed
                .value
                .study_constant(&model.signals)
                .ok_or_else(|| ConstraintPlanError::ScheduledValue {
                    constraint_id: constraint.constraint_id.clone(),
                })?
                .canonical_value();
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
                .canonical_value();
            rows.push(row::new_singleton(
                &constraint.constraint_id,
                canonical.identity,
                CanonicalStructuralDof::Rotation(prescribed.dof),
                1.0,
                finite_value(value, constraint)?,
            )?);
        }
    }
    Ok(CompiledConstraint::rows(
        lowering::targets([resolved])?,
        rows,
    ))
}

pub(super) fn compile_normal(
    model: &AnalysisModel,
    structural: &runmat_analysis_core::StructuralModel,
    topology: &CanonicalStructuralNodeSpace,
    constraint: &StructuralConstraint,
    prepared: &mut PreparedTargets<'_>,
    target: &runmat_analysis_core::RegionTarget,
    value: &runmat_analysis_core::PrescribedValue<runmat_analysis_core::Displacement>,
) -> Result<CompiledConstraint, ConstraintPlanError> {
    require_normal_capable(constraint)?;
    let (resolved, _) = prepared.take(target)?;
    let surface =
        resolve_explicit_surface(structural, topology, &resolved, &constraint.constraint_id)?;
    let right_hand_side =
        super::compile::study_constant_displacement(value, model, &constraint.constraint_id)?;
    let mut rows = lowering::rows(
        resolved.matched_node_identities.len(),
        3,
        &constraint.constraint_id,
    )?;
    for identity in &resolved.matched_node_identities {
        rows.push(row::new_direction(
            &constraint.constraint_id,
            *identity,
            surface.nodes[identity].oriented_normal,
            right_hand_side,
        )?);
    }
    Ok(CompiledConstraint::rows(
        lowering::targets([resolved])?,
        rows,
    ))
}

pub(super) fn compile_directions(
    structural: &runmat_analysis_core::StructuralModel,
    topology: &CanonicalStructuralNodeSpace,
    constraint: &StructuralConstraint,
    prepared: &mut PreparedTargets<'_>,
    target: &runmat_analysis_core::RegionTarget,
    tangential: bool,
) -> Result<CompiledConstraint, ConstraintPlanError> {
    require_normal_capable(constraint)?;
    let (resolved, _) = prepared.take(target)?;
    let surface =
        resolve_explicit_surface(structural, topology, &resolved, &constraint.constraint_id)?;
    let count = multiply(
        resolved.matched_node_identities.len(),
        if tangential { 2 } else { 1 },
    )?;
    let mut rows = lowering::rows(count, 3, &constraint.constraint_id)?;
    for identity in &resolved.matched_node_identities {
        let directions = if tangential {
            boundary_tangent_basis(
                surface.nodes[identity].oriented_normal,
                &constraint.constraint_id,
            )?
        } else {
            [surface.nodes[identity].oriented_normal; 2]
        };
        for direction in directions.into_iter().take(if tangential { 2 } else { 1 }) {
            rows.push(row::new_direction(
                &constraint.constraint_id,
                *identity,
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

pub(super) fn compile_coupling(
    structural: &runmat_analysis_core::StructuralModel,
    topology: &CanonicalStructuralNodeSpace,
    constraint: &StructuralConstraint,
    prepared: &mut PreparedTargets<'_>,
    coupling: &runmat_analysis_core::ReferencePointCoupling,
    kind: ConstraintCouplingKind,
) -> Result<CompiledConstraint, ConstraintPlanError> {
    require_global(constraint)?;
    let (target, _) = prepared.take(&coupling.target)?;
    let surface =
        resolve_explicit_surface(structural, topology, &target, &constraint.constraint_id)?;
    let identity = reference_point_identity(
        &coupling.reference_point.reference_point_id,
        coupling.reference_point.position.canonical_value(),
    )?;
    let (rows, evidence) = compile_reference_coupling_for_nodes(
        &surface.nodes,
        surface.weights,
        &coupling.reference_point,
        identity,
        kind,
        &constraint.constraint_id,
    )?;
    CompiledConstraint::coupling(target, rows, evidence)
}

fn finite_value(value: f64, constraint: &StructuralConstraint) -> Result<f64, ConstraintPlanError> {
    value
        .is_finite()
        .then_some(value)
        .ok_or_else(|| ConstraintPlanError::NonFiniteValue {
            constraint_id: constraint.constraint_id.clone(),
        })
}
