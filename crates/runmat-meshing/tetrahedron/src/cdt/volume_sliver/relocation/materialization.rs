//! Explicit complete fixed-star materialization for the standalone strategy.
use super::*;

pub(super) fn rebuild_with_relocated_node(
    topology: &DelaunayVolumeTopology,
    source_index: usize,
    replacement: DelaunayVolumeNode,
    options: DelaunayVolumeSliverOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<Option<DelaunayVolumeTopology>, DelaunayVolumeSliverError> {
    super::super::checkpoint(0, options, cancellation)?;
    if topology.nodes.len() as u64 > options.insertion.topology.maximum_nodes
        || topology.tetrahedra.len() as u64 > options.insertion.topology.maximum_tetrahedra
    {
        return Err(error(
            DelaunayVolumeSliverErrorKind::ResourceLimit,
            "relocation inventory exceeds original topology limits",
        ));
    }
    let mut ordered = topology
        .nodes
        .iter()
        .copied()
        .enumerate()
        .map(|(index, node)| {
            (
                index,
                if index == source_index {
                    replacement
                } else {
                    node
                },
            )
        })
        .collect::<Vec<_>>();
    ordered.sort_by_key(|(_, node)| node.identity);
    if ordered
        .windows(2)
        .any(|pair| pair[0].1.identity >= pair[1].1.identity)
    {
        return Ok(None);
    }
    let mut remap = vec![0_u32; ordered.len()];
    let nodes = ordered
        .into_iter()
        .enumerate()
        .map(|(canonical, (original, node))| {
            remap[original] = canonical as u32;
            node
        })
        .collect::<Vec<_>>();
    let tetrahedra = topology
        .tetrahedra
        .iter()
        .map(|tetrahedron| {
            (
                tetrahedron
                    .vertex_indices
                    .map(|vertex| remap[vertex as usize]),
                tetrahedron.region_id.clone(),
            )
        })
        .collect::<Vec<_>>();
    match build_delaunay_volume_topology_with_regions(
        nodes,
        tetrahedra,
        options.insertion.topology,
        cancellation,
    ) {
        Ok(topology) => Ok(Some(topology)),
        Err(failure) => match failure.kind {
            DelaunayTopologyErrorKind::ResourceLimit => Err(error(
                DelaunayVolumeSliverErrorKind::ResourceLimit,
                failure.to_string(),
            )),
            DelaunayTopologyErrorKind::Cancelled => Err(error(
                DelaunayVolumeSliverErrorKind::Cancelled,
                failure.to_string(),
            )),
            _ => Ok(None),
        },
    }
}
