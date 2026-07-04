use std::collections::BTreeSet;

use crate::contracts::{AnalysisMeshArtifact, SourceEntityKind};

use super::AnalysisMeshValidationError;

const SOLID_BACKEND: &str = "solid";

pub(super) fn validate_boundary_source_provenance(
    mesh: &AnalysisMeshArtifact,
    require_source_edge_provenance: bool,
) -> Result<(), AnalysisMeshValidationError> {
    if mesh.backend.backend != SOLID_BACKEND {
        return Ok(());
    }
    for face in &mesh.boundary_faces {
        if !face.provenance.iter().any(|provenance| {
            provenance.source_entity_kind == SourceEntityKind::Face
                && !provenance.source_entity_id.is_empty()
        }) {
            return Err(
                AnalysisMeshValidationError::MissingBoundarySourceFaceProvenance {
                    face_id: face.face_id.clone(),
                },
            );
        }
    }

    if !require_source_edge_provenance {
        return Ok(());
    }
    let required_edge_count = mesh.backend.plc_input_protected_edge_count;
    if required_edge_count == 0 {
        return Ok(());
    }
    let recovered_edge_ids = mesh
        .boundary_edges
        .iter()
        .flat_map(|edge| edge.provenance.iter())
        .filter(|provenance| {
            provenance.source_entity_kind == SourceEntityKind::Edge
                && !provenance.source_entity_id.is_empty()
        })
        .map(|provenance| provenance.source_entity_id.clone())
        .collect::<BTreeSet<_>>();

    if recovered_edge_ids.len() < required_edge_count {
        return Err(
            AnalysisMeshValidationError::MissingBoundarySourceEdgeProvenance {
                recovered_edge_count: recovered_edge_ids.len(),
                required_edge_count,
            },
        );
    }
    Ok(())
}
