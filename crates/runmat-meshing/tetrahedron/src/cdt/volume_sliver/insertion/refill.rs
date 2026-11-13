//! A local cone refill retains every enclosed point through the shared insertion
//! kernel. Only cavity data is materialized; the enclosing mesh stays borrowed.
use super::*;
use crate::cdt::{
    insertion::insert_conforming_volume_node_into_mutation_with_barriers,
    mutation::{TopologyMutation, TopologyMutationErrorKind},
    topology::build_canonical_volume_topology,
    DelaunayInsertionErrorKind, DelaunayTopologyErrorKind,
};

pub(in crate::cdt::volume_sliver) struct RefillPoint<'a> {
    pub(in crate::cdt::volume_sliver) node: DelaunayVolumeNode,
    pub(in crate::cdt::volume_sliver) replaces: Option<StableDigest>,
    pub(in crate::cdt::volume_sliver) relocations: &'a [(StableDigest, DelaunayVolumeNode)],
}
impl RefillPoint<'_> {
    pub(super) fn insert(node: DelaunayVolumeNode) -> Self {
        Self {
            node,
            replaces: None,
            relocations: &[],
        }
    }
}

pub(in crate::cdt::volume_sliver) fn preserve_vertices(
    topology: &(impl VolumeTopologyRead + ?Sized),
    removed: &BTreeSet<u32>,
    boundary: &[[u32; 3]],
    point: RefillPoint<'_>,
    options: DelaunayVolumeSliverOptions,
    work: &mut OptimizationWork<'_>,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<Vec<[StableDigest; 4]>, DelaunayVolumeSliverError> {
    let node = point.node;
    let boundary_vertices = boundary.iter().flatten().copied().collect::<BTreeSet<_>>();
    let enclosed = removed
        .iter()
        .flat_map(|cell| topology.cell(*cell).vertex_indices)
        .filter(|vertex| {
            !boundary_vertices.contains(vertex)
                && Some(topology.node(*vertex).identity) != point.replaces
        })
        .collect::<BTreeSet<_>>();
    let enclosed_identities = enclosed
        .iter()
        .map(|index| topology.node(*index).identity)
        .collect::<BTreeSet<_>>();
    let mut named = BTreeSet::new();
    if point
        .relocations
        .iter()
        .any(|(id, _)| !enclosed_identities.contains(id) || !named.insert(*id))
    {
        return Err(error(
            DelaunayVolumeSliverErrorKind::InvalidTopology,
            "cavity relocation must name distinct original enclosed points",
        ));
    }
    if enclosed.is_empty() {
        return Ok(boundary
            .iter()
            .map(|face| {
                [
                    topology.node(face[0]).identity,
                    topology.node(face[1]).identity,
                    topology.node(face[2]).identity,
                    node.identity,
                ]
            })
            .collect());
    }
    if boundary_vertices
        .len()
        .checked_add(enclosed.len())
        .and_then(|count| count.checked_add(1))
        .is_none_or(|count| count as u64 > options.insertion.topology.maximum_nodes)
    {
        return Err(error(
            DelaunayVolumeSliverErrorKind::ResourceLimit,
            "local cavity refill exceeds its declared point inventory",
        ));
    }
    let region = topology
        .cell(*removed.first().expect("nonempty checked cavity"))
        .region_id
        .clone();
    let mut nodes = boundary_vertices
        .iter()
        .map(|vertex| topology.node(*vertex))
        .collect::<Vec<_>>();
    nodes.push(node);
    nodes.sort_unstable_by_key(|node| node.identity);
    let index = |identity| {
        nodes
            .binary_search_by_key(&identity, |node| node.identity)
            .expect("cavity boundary identity") as u32
    };
    let inserted = index(node.identity);
    let cells = boundary
        .iter()
        .map(|face| {
            (
                [
                    index(topology.node(face[0]).identity),
                    index(topology.node(face[1]).identity),
                    index(topology.node(face[2]).identity),
                    inserted,
                ],
                region.clone(),
            )
        })
        .collect();
    let local =
        build_canonical_volume_topology(nodes, cells, options.insertion.topology, cancellation)
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
            })?
            .into_inner();
    let barriers = boundary
        .iter()
        .map(|face| {
            let mut identities = face.map(|vertex| topology.node(vertex).identity);
            identities.sort_unstable();
            identities
        })
        .collect();
    let mut local = TopologyMutation::from_canonical(local);
    let mut enclosed = enclosed
        .into_iter()
        .map(|vertex| {
            let original = topology.node(vertex);
            point
                .relocations
                .iter()
                .find(|(id, _)| *id == original.identity)
                .map_or(original, |(_, node)| *node)
        })
        .collect::<Vec<_>>();
    enclosed.sort_unstable_by_key(|node| node.identity);
    for existing in enclosed {
        work.evaluate()?;
        insert_conforming_volume_node_into_mutation_with_barriers(
            &mut local,
            existing,
            &barriers,
            options.insertion,
            cancellation,
        )
        .map_err(|failure| {
            error(
                match failure.kind {
                    DelaunayInsertionErrorKind::ResourceLimit => {
                        DelaunayVolumeSliverErrorKind::ResourceLimit
                    }
                    DelaunayInsertionErrorKind::Cancelled => {
                        DelaunayVolumeSliverErrorKind::Cancelled
                    }
                    _ => DelaunayVolumeSliverErrorKind::InvalidTopology,
                },
                failure.to_string(),
            )
        })?;
    }
    let local = local
        .freeze(options.insertion.topology, cancellation)
        .map_err(|failure| {
            error(
                match failure.kind {
                    TopologyMutationErrorKind::ResourceLimit => {
                        DelaunayVolumeSliverErrorKind::ResourceLimit
                    }
                    TopologyMutationErrorKind::Cancelled => {
                        DelaunayVolumeSliverErrorKind::Cancelled
                    }
                    TopologyMutationErrorKind::InvalidTopology => {
                        DelaunayVolumeSliverErrorKind::InvalidTopology
                    }
                },
                failure.to_string(),
            )
        })?;
    Ok(local
        .tetrahedra
        .iter()
        .map(|cell| {
            cell.vertex_indices
                .map(|vertex| local.nodes[vertex as usize].identity)
        })
        .collect())
}

#[cfg(test)]
mod tests;
