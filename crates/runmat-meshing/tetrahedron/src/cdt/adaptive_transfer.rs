//! Meshing-owned projection from checked adaptive lineage to solver field-transfer maps.

use std::collections::{BTreeMap, BTreeSet};

use runmat_meshing_core::{
    solver_volume_element_identity, MeshingCancellationSignal, SolverEntityTransfer,
    SolverMeshArtifact, SolverMeshTransferMap, SolverTransferMethod, SolverTransferSource,
    StableDigest, SOLVER_MESH_TRANSFER_SCHEMA_VERSION,
};

mod point_location;
mod preserved_boundary;
#[cfg(test)]
mod tests;

use point_location::build_volume_element_transfers;
use preserved_boundary::require_unchanged_boundaries;

use super::{
    adaptive_refinement::decision_mark, validate_marked_delaunay_volume_coarsening,
    validate_marked_delaunay_volume_refinement, DelaunayAdaptiveCoarseningErrorKind,
    DelaunayAdaptiveCoarseningOptions, DelaunayAdaptiveCoarseningResult,
    DelaunayAdaptiveRefinementDecision, DelaunayAdaptiveRefinementErrorKind,
    DelaunayAdaptiveRefinementOptions, DelaunayAdaptiveRefinementResult,
    DelaunayVolumeRefinementInput, DelaunayVolumeTopology,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DelaunayAdaptiveTransferOptions {
    pub maximum_point_location_predicates: u64,
    pub cancellation_check_interval: u64,
}

impl Default for DelaunayAdaptiveTransferOptions {
    fn default() -> Self {
        Self {
            maximum_point_location_predicates: 100_000_000,
            cancellation_check_interval: 1_024,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DelaunayAdaptiveTransferErrorKind {
    InvalidOptions,
    InvalidLineage,
    InvalidArtifact,
    UnsupportedBoundaryChange,
    ProjectionFailure,
    ResourceLimit,
    Cancelled,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DelaunayAdaptiveTransferError {
    pub kind: DelaunayAdaptiveTransferErrorKind,
    pub reason: String,
}

impl std::fmt::Display for DelaunayAdaptiveTransferError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "adaptive solver transfer {:?}: {}",
            self.kind, self.reason
        )
    }
}

impl std::error::Error for DelaunayAdaptiveTransferError {}

#[derive(Clone, Copy)]
pub struct DelaunayAdaptiveCoarseningTransferInput<'a> {
    pub original: DelaunayVolumeRefinementInput<'a>,
    pub refinement: &'a DelaunayAdaptiveRefinementResult,
    pub removal_node_identities: &'a [StableDigest],
    pub coarsening: &'a DelaunayAdaptiveCoarseningResult,
    pub source_artifact: &'a SolverMeshArtifact,
    pub target_artifact: &'a SolverMeshArtifact,
}

pub fn build_refinement_solver_transfer_map(
    original: DelaunayVolumeRefinementInput<'_>,
    refinement: &DelaunayAdaptiveRefinementResult,
    source_artifact: &SolverMeshArtifact,
    target_artifact: &SolverMeshArtifact,
    refinement_options: DelaunayAdaptiveRefinementOptions,
    transfer_options: DelaunayAdaptiveTransferOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<SolverMeshTransferMap, DelaunayAdaptiveTransferError> {
    validate_options(transfer_options)?;
    let marks = refinement
        .decisions
        .iter()
        .map(decision_mark)
        .collect::<Vec<_>>();
    validate_marked_delaunay_volume_refinement(
        original,
        &marks,
        refinement,
        refinement_options,
        cancellation,
    )
    .map_err(|failure| {
        let kind = match failure.kind {
            DelaunayAdaptiveRefinementErrorKind::InvalidOptions => {
                DelaunayAdaptiveTransferErrorKind::InvalidOptions
            }
            DelaunayAdaptiveRefinementErrorKind::ResourceLimit => {
                DelaunayAdaptiveTransferErrorKind::ResourceLimit
            }
            DelaunayAdaptiveRefinementErrorKind::Cancelled => {
                DelaunayAdaptiveTransferErrorKind::Cancelled
            }
            _ => DelaunayAdaptiveTransferErrorKind::InvalidLineage,
        };
        error(kind, failure.to_string())
    })?;
    let node_transfers = refinement
        .decisions
        .iter()
        .filter_map(|decision| match decision {
            DelaunayAdaptiveRefinementDecision::Inserted { mark, lineage } => {
                let mut sources = mark.node_identities;
                sources.sort_unstable();
                Some(SolverEntityTransfer {
                    target_stable_identity: lineage.node.identity,
                    method: SolverTransferMethod::BarycentricInterpolation,
                    sources: sources
                        .map(|stable_identity| SolverTransferSource {
                            stable_identity,
                            weight: 0.25,
                        })
                        .into(),
                })
            }
            DelaunayAdaptiveRefinementDecision::CoveredByPriorInsertion { .. } => None,
        })
        .collect::<Vec<_>>();
    build_map(
        original.topology,
        &refinement.topology,
        source_artifact,
        target_artifact,
        node_transfers,
        transfer_options,
        cancellation,
    )
}

pub fn build_coarsening_solver_transfer_map(
    input: DelaunayAdaptiveCoarseningTransferInput<'_>,
    coarsening_options: DelaunayAdaptiveCoarseningOptions,
    transfer_options: DelaunayAdaptiveTransferOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<SolverMeshTransferMap, DelaunayAdaptiveTransferError> {
    validate_options(transfer_options)?;
    validate_marked_delaunay_volume_coarsening(
        input.original,
        input.refinement,
        input.removal_node_identities,
        input.coarsening,
        coarsening_options,
        cancellation,
    )
    .map_err(|failure| {
        let kind = match failure.kind {
            DelaunayAdaptiveCoarseningErrorKind::InvalidOptions => {
                DelaunayAdaptiveTransferErrorKind::InvalidOptions
            }
            DelaunayAdaptiveCoarseningErrorKind::ResourceLimit => {
                DelaunayAdaptiveTransferErrorKind::ResourceLimit
            }
            DelaunayAdaptiveCoarseningErrorKind::Cancelled => {
                DelaunayAdaptiveTransferErrorKind::Cancelled
            }
            _ => DelaunayAdaptiveTransferErrorKind::InvalidLineage,
        };
        error(kind, failure.to_string())
    })?;
    build_map(
        &input.refinement.topology,
        &input.coarsening.topology,
        input.source_artifact,
        input.target_artifact,
        Vec::new(),
        transfer_options,
        cancellation,
    )
}

fn build_map(
    source_topology: &DelaunayVolumeTopology,
    target_topology: &DelaunayVolumeTopology,
    source_artifact: &SolverMeshArtifact,
    target_artifact: &SolverMeshArtifact,
    mut node_transfers: Vec<SolverEntityTransfer>,
    options: DelaunayAdaptiveTransferOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<SolverMeshTransferMap, DelaunayAdaptiveTransferError> {
    validate_projection(source_topology, source_artifact)?;
    validate_projection(target_topology, target_artifact)?;
    if source_artifact.geometry != target_artifact.geometry {
        return Err(invalid_artifact(
            "adaptive solver artifacts use different exact geometry revisions",
        ));
    }
    require_unchanged_boundaries(source_artifact, target_artifact)?;
    node_transfers.sort_by_key(|transfer| transfer.target_stable_identity);
    let source_corners = corner_identities(source_artifact);
    let target_corners = corner_identities(target_artifact);
    let mapped_nodes = node_transfers
        .iter()
        .map(|transfer| transfer.target_stable_identity)
        .collect::<BTreeSet<_>>();
    if mapped_nodes
        != target_corners
            .difference(&source_corners)
            .copied()
            .collect()
    {
        return Err(error(
            DelaunayAdaptiveTransferErrorKind::InvalidLineage,
            "adaptive insertion lineage does not exactly cover new solver corner nodes",
        ));
    }

    let volume_element_transfers =
        build_volume_element_transfers(source_artifact, target_artifact, options, cancellation)?;
    let transfer = SolverMeshTransferMap {
        schema_version: SOLVER_MESH_TRANSFER_SCHEMA_VERSION,
        source_artifact_digest: source_artifact.canonical_digest,
        target_artifact_digest: target_artifact.canonical_digest,
        geometry: source_artifact.geometry.clone(),
        node_transfers,
        volume_element_transfers,
        boundary_face_transfers: Vec::new(),
        boundary_edge_transfers: Vec::new(),
    };
    transfer
        .validate_against(source_artifact, target_artifact)
        .map_err(|failure| invalid_artifact(failure.to_string()))?;
    Ok(transfer)
}

fn validate_projection(
    topology: &DelaunayVolumeTopology,
    artifact: &SolverMeshArtifact,
) -> Result<(), DelaunayAdaptiveTransferError> {
    artifact
        .validate()
        .map_err(|failure| invalid_artifact(failure.to_string()))?;
    let topology_nodes = topology
        .nodes
        .iter()
        .map(|node| node.identity)
        .collect::<BTreeSet<_>>();
    if corner_identities(artifact) != topology_nodes {
        return Err(invalid_artifact(
            "solver corner identities do not exactly project the adaptive CDT nodes",
        ));
    }
    let artifact_nodes = artifact
        .topology
        .nodes
        .iter()
        .map(|node| (node.stable_identity, node.coordinates_m))
        .collect::<BTreeMap<_, _>>();
    if topology
        .nodes
        .iter()
        .any(|node| artifact_nodes.get(&node.identity).copied() != Some(node.coordinates_m))
    {
        return Err(invalid_artifact(
            "solver corner coordinates do not exactly project the adaptive CDT nodes",
        ));
    }
    let topology_elements = topology
        .tetrahedra
        .iter()
        .map(|tetrahedron| {
            solver_volume_element_identity(
                tetrahedron
                    .vertex_indices
                    .map(|vertex| topology.nodes[vertex as usize].identity),
            )
        })
        .collect::<BTreeSet<_>>();
    let artifact_elements = artifact
        .topology
        .volume_elements
        .iter()
        .map(|element| element.stable_identity)
        .collect::<BTreeSet<_>>();
    if topology_elements != artifact_elements {
        return Err(invalid_artifact(
            "solver element identities do not exactly project the adaptive CDT cells",
        ));
    }
    Ok(())
}

fn corner_identities(artifact: &SolverMeshArtifact) -> BTreeSet<StableDigest> {
    let nodes = artifact
        .topology
        .nodes
        .iter()
        .map(|node| (node.node_id, node.stable_identity))
        .collect::<BTreeMap<_, _>>();
    artifact
        .topology
        .volume_elements
        .iter()
        .flat_map(|element| element.node_ids[..4].iter().map(|node| nodes[node]))
        .collect()
}

fn validate_options(
    options: DelaunayAdaptiveTransferOptions,
) -> Result<(), DelaunayAdaptiveTransferError> {
    if options.maximum_point_location_predicates == 0 || options.cancellation_check_interval == 0 {
        return Err(error(
            DelaunayAdaptiveTransferErrorKind::InvalidOptions,
            "point-location and cancellation limits must be nonzero",
        ));
    }
    Ok(())
}

fn invalid_artifact(reason: impl Into<String>) -> DelaunayAdaptiveTransferError {
    error(DelaunayAdaptiveTransferErrorKind::InvalidArtifact, reason)
}

fn error(
    kind: DelaunayAdaptiveTransferErrorKind,
    reason: impl Into<String>,
) -> DelaunayAdaptiveTransferError {
    DelaunayAdaptiveTransferError {
        kind,
        reason: reason.into(),
    }
}
