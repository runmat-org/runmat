//! Mapping between canonical constraint owners and the generalized solver state.

use std::collections::BTreeMap;

use runmat_analysis_core::{
    AnalysisField, AnalysisFieldValues, StructuralRotationDof, StructuralTranslationDof,
};
use runmat_meshing_core::{SolverMeshArtifact, StableDigest};

use super::{
    CanonicalStructuralConstraintPlan, CanonicalStructuralDof, CanonicalStructuralDofOwner,
    ConstraintResultError,
};

#[derive(Clone, Copy, Debug)]
pub(super) struct ConstraintNodeState {
    pub coordinates_m: [f64; 3],
    pub dof_indices: [Option<usize>; 6],
}

pub(super) fn nodal_vector_field<'a>(
    fields: &'a [AnalysisField],
    field_id: &'static str,
    mesh: &SolverMeshArtifact,
) -> Result<&'a [f64], ConstraintResultError> {
    let field = fields
        .iter()
        .find(|field| field.field_id == field_id)
        .ok_or(ConstraintResultError::MissingField { field_id })?;
    let AnalysisFieldValues::HostF64(values) = &field.values else {
        return Err(ConstraintResultError::InvalidField { field_id });
    };
    if field.shape != [mesh.topology.nodes.len(), 3]
        || values.len() != mesh.topology.nodes.len() * 3
        || values.iter().any(|value| !value.is_finite())
    {
        return Err(ConstraintResultError::InvalidField { field_id });
    }
    Ok(values)
}

pub(super) fn nodal_vector_field_for_count<'a>(
    fields: &'a [AnalysisField],
    field_id: &'static str,
    node_count: usize,
) -> Result<&'a [f64], ConstraintResultError> {
    let field = fields
        .iter()
        .find(|field| field.field_id == field_id)
        .ok_or(ConstraintResultError::MissingField { field_id })?;
    let AnalysisFieldValues::HostF64(values) = &field.values else {
        return Err(ConstraintResultError::InvalidField { field_id });
    };
    if field.shape != [node_count, 3]
        || values.len() != node_count * 3
        || values.iter().any(|value| !value.is_finite())
    {
        return Err(ConstraintResultError::InvalidField { field_id });
    }
    Ok(values)
}

pub(super) fn owner_dof_index(
    plan: &CanonicalStructuralConstraintPlan,
    nodes: &BTreeMap<StableDigest, ConstraintNodeState>,
    reference_dof_offset: usize,
    owner: CanonicalStructuralDofOwner,
    dof: CanonicalStructuralDof,
) -> Result<usize, ConstraintResultError> {
    match owner {
        CanonicalStructuralDofOwner::StructuralNode(identity) => {
            let node = nodes
                .get(&identity)
                .copied()
                .ok_or(ConstraintResultError::NodeAbsent)?;
            node.dof_indices[dof_component(dof)].ok_or(ConstraintResultError::InvalidEvidence)
        }
        CanonicalStructuralDofOwner::ReferencePoint(identity) => {
            let point_index = plan
                .reference_points
                .iter()
                .position(|point| point.identity == identity)
                .ok_or(ConstraintResultError::InvalidEvidence)?;
            let component = match dof {
                CanonicalStructuralDof::Translation(StructuralTranslationDof::Ux) => 0,
                CanonicalStructuralDof::Translation(StructuralTranslationDof::Uy) => 1,
                CanonicalStructuralDof::Translation(StructuralTranslationDof::Uz) => 2,
                CanonicalStructuralDof::Rotation(StructuralRotationDof::Rx) => 3,
                CanonicalStructuralDof::Rotation(StructuralRotationDof::Ry) => 4,
                CanonicalStructuralDof::Rotation(StructuralRotationDof::Rz) => 5,
            };
            Ok(reference_dof_offset + point_index * 6 + component)
        }
    }
}

pub(super) const fn dof_component(dof: CanonicalStructuralDof) -> usize {
    match dof {
        CanonicalStructuralDof::Translation(StructuralTranslationDof::Ux) => 0,
        CanonicalStructuralDof::Translation(StructuralTranslationDof::Uy) => 1,
        CanonicalStructuralDof::Translation(StructuralTranslationDof::Uz) => 2,
        CanonicalStructuralDof::Rotation(StructuralRotationDof::Rx) => 3,
        CanonicalStructuralDof::Rotation(StructuralRotationDof::Ry) => 4,
        CanonicalStructuralDof::Rotation(StructuralRotationDof::Rz) => 5,
    }
}

pub(super) fn translation_component(
    dof: CanonicalStructuralDof,
) -> Result<usize, ConstraintResultError> {
    match dof {
        CanonicalStructuralDof::Translation(StructuralTranslationDof::Ux) => Ok(0),
        CanonicalStructuralDof::Translation(StructuralTranslationDof::Uy) => Ok(1),
        CanonicalStructuralDof::Translation(StructuralTranslationDof::Uz) => Ok(2),
        CanonicalStructuralDof::Rotation(_) => Err(ConstraintResultError::InvalidEvidence),
    }
}
