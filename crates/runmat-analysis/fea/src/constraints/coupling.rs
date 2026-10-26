//! Reference-point coupling equations and conservative target weights.

use runmat_analysis_core::{ReferencePoint, StructuralRotationDof};
use runmat_execution::Digest;
use runmat_meshing_core::{BoundaryTriangleOrder, SolverMeshArtifact, StableDigest};

use super::contract::{
    CanonicalStructuralConstraintRow, CanonicalStructuralConstraintTerm, CanonicalStructuralDof,
    CanonicalStructuralDofOwner, ConstraintCouplingEvidence, ConstraintCouplingKind,
    ConstraintCouplingWeight, ConstraintPlanError, ResolvedConstraintTarget,
};
use super::row;
use super::workspace::{check_cancelled, reserve};
use crate::topology::storage::CanonicalNodeValues as NodeValues;
mod coordinates;
use coordinates::CoordinateLookup;
mod geometry;
pub(in crate::constraints) mod profile;
mod rows;
mod weights;
use super::vector::{cross, magnitude, scale, subtract};
use geometry::validate_rotational_geometry;
pub use profile::{constraint_coupling_workspace_profile, ConstraintCouplingWorkspaceProfile};
use rows::{distributing_rows, rigid_rows};

pub(super) fn compile_reference_coupling(
    mesh: &SolverMeshArtifact,
    target: &ResolvedConstraintTarget,
    reference_point: &ReferencePoint,
    reference_point_identity: Digest,
    kind: ConstraintCouplingKind,
    constraint_id: &str,
) -> Result<
    (
        Vec<CanonicalStructuralConstraintRow>,
        ConstraintCouplingEvidence,
    ),
    ConstraintPlanError,
> {
    check_cancelled()?;
    let _profile = constraint_coupling_workspace_profile(mesh, kind, constraint_id)?;
    let coordinates = NodeValues::derive(
        mesh.topology
            .nodes
            .iter()
            .map(|n| (n.stable_identity, n.coordinates_m.canonical_value())),
        mesh.topology.nodes.len(),
    )?;
    let weights = weights::target_weights(mesh, target, &coordinates, constraint_id)?;
    compile_reference_coupling_for_nodes(
        &coordinates,
        weights,
        reference_point,
        reference_point_identity,
        kind,
        constraint_id,
    )
}

pub(super) fn compile_reference_coupling_for_nodes(
    coordinates: &impl CoordinateLookup,
    weights: Vec<ConstraintCouplingWeight>,
    reference_point: &ReferencePoint,
    reference_point_identity: Digest,
    kind: ConstraintCouplingKind,
    constraint_id: &str,
) -> Result<
    (
        Vec<CanonicalStructuralConstraintRow>,
        ConstraintCouplingEvidence,
    ),
    ConstraintPlanError,
> {
    check_cancelled()?;
    let _rows_profile = profile::rows_profile(weights.len(), kind, constraint_id.len())?;
    let mut radii = reserve(weights.len())?;
    for weight in &weights {
        check_cancelled()?;
        let position = coordinates.position(&weight.node_identity).ok_or_else(|| {
            ConstraintPlanError::Invalid(format!(
                "constraint `{constraint_id}` coupling node is absent from solver topology"
            ))
        })?;
        radii.push(
            weight.weight
                * magnitude(subtract(
                    position,
                    reference_point.position.canonical_value(),
                ))
                .powi(2),
        );
    }
    let characteristic_length_meters = radii.into_iter().sum::<f64>().sqrt();
    validate_rotational_geometry(
        &weights,
        coordinates,
        characteristic_length_meters,
        constraint_id,
    )?;
    let rows = match kind {
        ConstraintCouplingKind::Rigid => rigid_rows(
            coordinates,
            &weights,
            reference_point,
            reference_point_identity,
            constraint_id,
        )?,
        ConstraintCouplingKind::Distributing => distributing_rows(
            coordinates,
            &weights,
            reference_point,
            reference_point_identity,
            constraint_id,
        )?,
    };
    Ok((
        rows,
        ConstraintCouplingEvidence {
            kind,
            reference_point_identity,
            target_weights: weights,
            characteristic_length_meters,
        },
    ))
}
