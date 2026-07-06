use std::collections::{BTreeMap, BTreeSet};

use runmat_meshing_core::contracts::UNCLASSIFIED_MATERIAL_REGION_ID;
use runmat_meshing_tetrahedron::{
    generate::TetrahedronMesh,
    recover::{TetrahedronRecoveryKind, TetrahedronRecoveryQueue, TetrahedronRecoveryStatus},
};

const MAX_REPORTED_RECOVERY_IDS: usize = 64;

pub(super) fn tetrahedron_entity_count(tetrahedron_mesh: &TetrahedronMesh, key: &str) -> usize {
    tetrahedron_mesh
        .evidence
        .entity_counts
        .get(key)
        .copied()
        .unwrap_or_default()
}

pub(super) fn tetrahedron_rejection_counts_by_prefix(
    tetrahedron_mesh: &TetrahedronMesh,
    prefix: &str,
) -> BTreeMap<String, usize> {
    tetrahedron_mesh
        .evidence
        .rejection_counts
        .iter()
        .filter_map(|(reason, count)| {
            reason
                .strip_prefix(prefix)
                .map(|stripped| (stripped.to_string(), *count))
        })
        .collect()
}

pub(super) fn tetrahedron_material_region_count(tetrahedron_mesh: &TetrahedronMesh) -> usize {
    tetrahedron_mesh
        .elements
        .iter()
        .map(|element| element.material_region_id.as_str())
        .collect::<BTreeSet<_>>()
        .len()
}

pub(super) fn tetrahedron_unclassified_material_element_count(
    tetrahedron_mesh: &TetrahedronMesh,
) -> usize {
    tetrahedron_mesh
        .elements
        .iter()
        .filter(|element| element.material_region_id == UNCLASSIFIED_MATERIAL_REGION_ID)
        .count()
}

pub(super) fn recovery_entity_count(recovery_queue: &TetrahedronRecoveryQueue, key: &str) -> usize {
    recovery_queue
        .evidence
        .entity_counts
        .get(key)
        .copied()
        .unwrap_or_default()
}

pub(super) struct BoundedRecoveryIds {
    pub(super) ids: Vec<String>,
    pub(super) omitted_count: usize,
}

pub(super) fn bounded_missing_recovery_ids(
    recovery_queue: &TetrahedronRecoveryQueue,
    kind: TetrahedronRecoveryKind,
) -> BoundedRecoveryIds {
    let all_ids = recovery_queue
        .items
        .iter()
        .filter(|item| item.kind == kind && item.status == TetrahedronRecoveryStatus::Missing)
        .filter_map(|item| match kind {
            TetrahedronRecoveryKind::SourceFace | TetrahedronRecoveryKind::SourceEdge => item
                .source_entity_id
                .as_ref()
                .map(|source_id| source_id.id.clone()),
            TetrahedronRecoveryKind::MaterialInterface => item.material_interface_id.clone(),
        })
        .collect::<BTreeSet<_>>();
    let total_count = all_ids.len();
    let ids = all_ids
        .into_iter()
        .take(MAX_REPORTED_RECOVERY_IDS)
        .collect::<Vec<_>>();
    BoundedRecoveryIds {
        omitted_count: total_count.saturating_sub(ids.len()),
        ids,
    }
}
