use std::collections::BTreeMap;

use runmat_geometry_core::PersistentEntityId;
use runmat_meshing_core::StableDigest;

use super::{DelaunayVolumeNode, DelaunayVolumeTopology};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DelaunayAdaptiveTetrahedronRecord {
    pub node_identities: [StableDigest; 4],
    pub region_id: Option<PersistentEntityId>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DelaunayAdaptiveInsertionLineage {
    pub node: DelaunayVolumeNode,
    pub removed_tetrahedra: Vec<DelaunayAdaptiveTetrahedronRecord>,
    pub created_tetrahedra: Vec<DelaunayAdaptiveTetrahedronRecord>,
}

pub(super) fn tetrahedron_records(
    topology: &DelaunayVolumeTopology,
) -> BTreeMap<[StableDigest; 4], DelaunayAdaptiveTetrahedronRecord> {
    topology
        .tetrahedra
        .iter()
        .map(|tetrahedron| {
            let node_identities = tetrahedron
                .vertex_indices
                .map(|vertex| topology.nodes[vertex as usize].identity);
            (
                tetrahedron_record_key(node_identities),
                DelaunayAdaptiveTetrahedronRecord {
                    node_identities,
                    region_id: tetrahedron.region_id.clone(),
                },
            )
        })
        .collect()
}

pub(super) fn tetrahedron_record_key(mut identities: [StableDigest; 4]) -> [StableDigest; 4] {
    identities.sort_unstable();
    identities
}

pub(super) fn insertion_lineage(
    before: BTreeMap<[StableDigest; 4], DelaunayAdaptiveTetrahedronRecord>,
    after: &DelaunayVolumeTopology,
    node: DelaunayVolumeNode,
) -> DelaunayAdaptiveInsertionLineage {
    let after = tetrahedron_records(after);
    DelaunayAdaptiveInsertionLineage {
        node,
        removed_tetrahedra: before
            .iter()
            .filter(|(key, _)| !after.contains_key(*key))
            .map(|(_, record)| record.clone())
            .collect(),
        created_tetrahedra: after
            .iter()
            .filter(|(key, _)| !before.contains_key(*key))
            .map(|(_, record)| record.clone())
            .collect(),
    }
}
