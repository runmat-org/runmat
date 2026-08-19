use runmat_meshing_core::{
    solver_volume_element_identity, CanonicalMeshingContract, SolverMeshAdaptationCell,
    SolverMeshAdaptationKind, SolverMeshAdaptationLineage, SolverMeshAdaptationMark,
    SolverMeshAdaptationMutation, SolverMeshArtifact, SolverMeshTransferMap,
    SOLVER_MESH_ADAPTATION_LINEAGE_SCHEMA_VERSION,
};

use super::{
    error, DelaunayAdaptiveCoarseningInput, DelaunayAdaptiveTransferError,
    DelaunayAdaptiveTransferErrorKind,
};
use crate::cdt::{
    DelaunayAdaptiveInsertionLineage, DelaunayAdaptiveRefinementDecision,
    DelaunayAdaptiveRefinementResult, DelaunayAdaptiveTetrahedronRecord,
};

pub(super) fn build_refinement_lineage(
    refinement: &DelaunayAdaptiveRefinementResult,
    source: &SolverMeshArtifact,
    target: &SolverMeshArtifact,
    transfer: &SolverMeshTransferMap,
) -> Result<SolverMeshAdaptationLineage, DelaunayAdaptiveTransferError> {
    let marks = refinement
        .decisions
        .iter()
        .map(|decision| {
            let mark = match decision {
                DelaunayAdaptiveRefinementDecision::Inserted { mark, .. }
                | DelaunayAdaptiveRefinementDecision::CoveredByPriorInsertion { mark } => mark,
            };
            SolverMeshAdaptationMark {
                element_stable_identity: solver_volume_element_identity(mark.node_identities),
                indicator_value: mark.indicator_value,
            }
        })
        .collect();
    let mutations = refinement
        .decisions
        .iter()
        .filter_map(|decision| match decision {
            DelaunayAdaptiveRefinementDecision::Inserted { mark, lineage } => Some((mark, lineage)),
            DelaunayAdaptiveRefinementDecision::CoveredByPriorInsertion { .. } => None,
        })
        .map(|(mark, lineage)| {
            mutation(
                Some(solver_volume_element_identity(mark.node_identities)),
                lineage,
                false,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    finish(
        SolverMeshAdaptationKind::HRefinement,
        source,
        target,
        transfer,
        marks,
        Vec::new(),
        mutations,
    )
}

pub(super) fn build_coarsening_lineage(
    input: DelaunayAdaptiveCoarseningInput<'_>,
    transfer: &SolverMeshTransferMap,
) -> Result<SolverMeshAdaptationLineage, DelaunayAdaptiveTransferError> {
    let inserted = input
        .refinement
        .decisions
        .iter()
        .filter_map(|decision| match decision {
            DelaunayAdaptiveRefinementDecision::Inserted { lineage, .. } => {
                Some((lineage.node.identity, lineage))
            }
            DelaunayAdaptiveRefinementDecision::CoveredByPriorInsertion { .. } => None,
        })
        .collect::<std::collections::BTreeMap<_, _>>();
    let mutations = input
        .coarsening
        .removed_node_identities
        .iter()
        .map(|identity| {
            inserted
                .get(identity)
                .ok_or_else(|| {
                    error(
                        DelaunayAdaptiveTransferErrorKind::InvalidLineage,
                        "coarsening references an absent insertion lineage",
                    )
                })
                .and_then(|lineage| mutation(None, lineage, true))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut removals = input.removal_node_identities.to_vec();
    removals.sort_unstable();
    finish(
        SolverMeshAdaptationKind::HCoarsening,
        input.source_artifact,
        input.target_artifact,
        transfer,
        Vec::new(),
        removals,
        mutations,
    )
}

fn mutation(
    source_mark_identity: Option<runmat_meshing_core::StableDigest>,
    lineage: &DelaunayAdaptiveInsertionLineage,
    inverse: bool,
) -> Result<SolverMeshAdaptationMutation, DelaunayAdaptiveTransferError> {
    let (removed, created) = if inverse {
        (&lineage.created_tetrahedra, &lineage.removed_tetrahedra)
    } else {
        (&lineage.removed_tetrahedra, &lineage.created_tetrahedra)
    };
    Ok(SolverMeshAdaptationMutation {
        source_mark_identity,
        node_identity: lineage.node.identity,
        node_coordinates_m: lineage.node.coordinates_m,
        removed_cells: cells(removed)?,
        created_cells: cells(created)?,
    })
}

fn cells(
    records: &[DelaunayAdaptiveTetrahedronRecord],
) -> Result<Vec<SolverMeshAdaptationCell>, DelaunayAdaptiveTransferError> {
    let mut cells = records
        .iter()
        .map(|record| {
            let region_id = record.region_id.clone().ok_or_else(|| {
                error(
                    DelaunayAdaptiveTransferErrorKind::InvalidLineage,
                    "adaptive cell lineage is missing its region",
                )
            })?;
            Ok(SolverMeshAdaptationCell {
                stable_identity: solver_volume_element_identity(record.node_identities),
                node_identities: record.node_identities,
                region_id,
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    cells.sort_by_key(|cell| cell.stable_identity);
    Ok(cells)
}

fn finish(
    kind: SolverMeshAdaptationKind,
    source: &SolverMeshArtifact,
    target: &SolverMeshArtifact,
    transfer: &SolverMeshTransferMap,
    marks: Vec<SolverMeshAdaptationMark>,
    requested_removal_node_identities: Vec<runmat_meshing_core::StableDigest>,
    mutations: Vec<SolverMeshAdaptationMutation>,
) -> Result<SolverMeshAdaptationLineage, DelaunayAdaptiveTransferError> {
    let lineage = SolverMeshAdaptationLineage {
        schema_version: SOLVER_MESH_ADAPTATION_LINEAGE_SCHEMA_VERSION,
        source_artifact_digest: source.canonical_digest,
        target_artifact_digest: target.canonical_digest,
        transfer_map_digest: transfer.canonical_digest().map_err(|failure| {
            error(
                DelaunayAdaptiveTransferErrorKind::InvalidArtifact,
                failure.to_string(),
            )
        })?,
        geometry: source.geometry.clone(),
        kind,
        marks,
        requested_removal_node_identities,
        mutations,
    };
    lineage
        .validate_against(source, target, transfer)
        .map_err(|failure| {
            error(
                DelaunayAdaptiveTransferErrorKind::InvalidLineage,
                failure.to_string(),
            )
        })?;
    Ok(lineage)
}
