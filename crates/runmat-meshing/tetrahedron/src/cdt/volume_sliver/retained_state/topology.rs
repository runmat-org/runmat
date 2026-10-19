//! Exact connectivity and inserted-node differences against an admitted baseline.
//! Full incidence is materialized only for the active retained search state.
use super::*;
use runmat_geometry_core::PersistentEntityId;

type CellKey = [StableDigest; 4];
type CreatedCell = (CellKey, [StableDigest; 4], Option<PersistentEntityId>);

pub(in crate::cdt::volume_sliver) struct TopologyPatch {
    inserted_nodes: Vec<DelaunayVolumeNode>,
    removed: Vec<CellKey>,
    created: Vec<CreatedCell>,
}

impl TopologyPatch {
    pub(in crate::cdt::volume_sliver) fn between(
        base: &DelaunayVolumeTopology,
        candidate: &DelaunayVolumeTopology,
    ) -> Result<Self, DelaunayVolumeSliverError> {
        if base.nodes.iter().any(|node| {
            candidate
                .nodes
                .binary_search_by_key(&node.identity, |other| other.identity)
                .map_or(true, |index| candidate.nodes[index] != *node)
        }) {
            return Err(error(
                DelaunayVolumeSliverErrorKind::InvalidTopology,
                "retained state removed or changed an admitted baseline node",
            ));
        }
        let inserted_nodes = candidate
            .nodes
            .iter()
            .filter(|node| {
                base.nodes
                    .binary_search_by_key(&node.identity, |other| other.identity)
                    .is_err()
            })
            .copied()
            .collect();
        let (removed, created_keys) = connectivity_delta(base, candidate);
        let mut baseline = base.tetrahedra.iter().peekable();
        for cell in &candidate.tetrahedra {
            let key = cell_key(candidate, cell.vertex_indices);
            while baseline
                .peek()
                .is_some_and(|old| cell_key(base, old.vertex_indices) < key)
            {
                baseline.next();
            }
            if let Some(old) = baseline
                .peek()
                .filter(|old| cell_key(base, old.vertex_indices) == key)
            {
                if old.region_id != cell.region_id {
                    return Err(error(
                        DelaunayVolumeSliverErrorKind::InvalidTopology,
                        "retained state reassigned an unchanged cell's region",
                    ));
                }
            }
        }
        let created = candidate
            .tetrahedra
            .iter()
            .filter_map(|cell| {
                let mut key = identities(candidate, cell.vertex_indices);
                key.sort_unstable();
                created_keys.binary_search(&key).is_ok().then(|| {
                    (
                        key,
                        identities(candidate, cell.vertex_indices),
                        cell.region_id.clone(),
                    )
                })
            })
            .collect();
        Ok(Self {
            inserted_nodes,
            removed,
            created,
        })
    }

    pub(in crate::cdt::volume_sliver) fn compare(
        &self,
        other: &Self,
        base: &DelaunayVolumeTopology,
    ) -> std::cmp::Ordering {
        self.keys(base).cmp(other.keys(base))
    }

    fn keys<'a>(&'a self, base: &'a DelaunayVolumeTopology) -> impl Iterator<Item = CellKey> + 'a {
        let mut retained = connectivity::keys(base)
            .filter(|key| self.removed.binary_search(key).is_err())
            .peekable();
        let mut created = self.created.iter().map(|cell| cell.0).peekable();
        std::iter::from_fn(move || match (retained.peek(), created.peek()) {
            (Some(left), Some(right)) if left <= right => retained.next(),
            (Some(_), Some(_)) => created.next(),
            (Some(_), None) => retained.next(),
            (None, Some(_)) => created.next(),
            (None, None) => None,
        })
    }

    pub(in crate::cdt::volume_sliver) fn materialize(
        &self,
        base: &DelaunayVolumeTopology,
        options: DelaunayVolumeSliverOptions,
        cancellation: &dyn MeshingCancellationSignal,
    ) -> Result<DelaunayVolumeTopology, DelaunayVolumeSliverError> {
        let mut nodes = base.nodes.clone();
        nodes.extend(self.inserted_nodes.iter().copied());
        nodes.sort_unstable_by_key(|node| node.identity);
        let remap = |corners: [StableDigest; 4]| {
            corners.map(|identity| {
                nodes
                    .binary_search_by_key(&identity, |node| node.identity)
                    .expect("retained cell corners belong to baseline or inserted nodes")
                    as u32
            })
        };
        let mut retained =
            Vec::with_capacity(base.tetrahedra.len() - self.removed.len() + self.created.len());
        for cell in &base.tetrahedra {
            let mut key = identities(base, cell.vertex_indices);
            key.sort_unstable();
            if self.removed.binary_search(&key).is_err() {
                let vertices = if self.inserted_nodes.is_empty() {
                    cell.vertex_indices
                } else {
                    remap(identities(base, cell.vertex_indices))
                };
                retained.push((vertices, cell.region_id.clone()));
            }
        }
        retained.extend(
            self.created
                .iter()
                .map(|cell| (remap(cell.1), cell.2.clone())),
        );
        build_delaunay_volume_topology_with_regions(
            nodes,
            retained,
            options.insertion.topology,
            cancellation,
        )
        .map_err(|failure| {
            error(
                match failure.kind {
                    DelaunayTopologyErrorKind::ResourceLimit => {
                        DelaunayVolumeSliverErrorKind::ResourceLimit
                    }
                    DelaunayTopologyErrorKind::Cancelled => {
                        DelaunayVolumeSliverErrorKind::Cancelled
                    }
                    _ => DelaunayVolumeSliverErrorKind::InvalidTopology,
                },
                failure.to_string(),
            )
        })
    }
}

fn cell_key(topology: &DelaunayVolumeTopology, vertices: [u32; 4]) -> CellKey {
    let mut key = identities(topology, vertices);
    key.sort_unstable();
    key
}

#[cfg(test)]
#[path = "topology/tests.rs"]
mod tests;

#[cfg(test)]
#[path = "topology/insertion_tests.rs"]
mod insertion_tests;
