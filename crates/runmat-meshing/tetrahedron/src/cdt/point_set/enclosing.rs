use std::collections::{BTreeMap, BTreeSet};

use runmat_meshing_core::{MeshingCancellationSignal, StableDigest};

use super::{
    build_delaunay_volume_topology, error, insertion_error, insertion_error_at_node, resource,
    topology_error, DelaunayPointSetError, DelaunayPointSetErrorKind, DelaunayPointSetOptions,
    DelaunayVolumeNode, DelaunayVolumeTopology,
};
use crate::cdt::insertion::{
    insert_delaunay_volume_node_into_mutation,
    legalize_delaunay_volume_topology_completely_with_barriers,
    validate_locally_delaunay_volume_topology,
};
use crate::cdt::mutation::TopologyMutation;

pub(super) fn enclosing_point_set(
    nodes: &[DelaunayVolumeNode],
    options: DelaunayPointSetOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<DelaunayVolumeTopology, DelaunayPointSetError> {
    let enclosing = enclosing_tetrahedron(nodes)?;
    let enclosing_identities = enclosing
        .iter()
        .map(|node| node.identity)
        .collect::<BTreeSet<_>>();
    let mut working_options = options.insertion;
    working_options.topology.maximum_nodes = working_options
        .topology
        .maximum_nodes
        .checked_add(4)
        .ok_or_else(|| resource("working node limit overflow"))?;
    let topology = build_delaunay_volume_topology(
        enclosing.to_vec(),
        vec![[0, 1, 2, 3]],
        working_options.topology,
        cancellation,
    )
    .map_err(topology_error)?;
    let mut topology = TopologyMutation::from_canonical(topology);
    let mut insertion_nodes = nodes.to_vec();
    insertion_nodes.sort_by(super::compare_geometric_nodes);
    for (index, node) in insertion_nodes.iter().copied().enumerate() {
        insert_delaunay_volume_node_into_mutation(
            &mut topology,
            node,
            working_options,
            cancellation,
        )
        .map_err(|failure| insertion_error_at_node(failure, index, node))?;
    }
    let topology = topology
        .freeze(working_options.topology, cancellation)
        .map_err(crate::cdt::insertion::mutation_topology_error)
        .map_err(insertion_error)?;
    let result_index = nodes
        .iter()
        .enumerate()
        .map(|(index, node)| (node.identity, index as u32))
        .collect::<BTreeMap<_, _>>();
    let tetrahedra = topology
        .tetrahedra
        .iter()
        .filter(|tetrahedron| {
            tetrahedron.vertex_indices.iter().all(|vertex| {
                !enclosing_identities.contains(&topology.nodes[*vertex as usize].identity)
            })
        })
        .map(|tetrahedron| {
            tetrahedron
                .vertex_indices
                .map(|vertex| result_index[&topology.nodes[vertex as usize].identity])
        })
        .collect::<Vec<_>>();
    if tetrahedra.is_empty() {
        return Err(error(
            DelaunayPointSetErrorKind::InsufficientDimension,
            "point set does not contain a noncoplanar tetrahedron",
        ));
    }
    let result = build_delaunay_volume_topology(
        nodes.to_vec(),
        tetrahedra,
        options.insertion.topology,
        cancellation,
    )
    .map_err(topology_error)?;
    let result = legalize_delaunay_volume_topology_completely_with_barriers(
        result,
        &[],
        options.insertion,
        cancellation,
    )
    .map_err(insertion_error)?;
    validate_locally_delaunay_volume_topology(&result, options.insertion, cancellation)
        .map_err(insertion_error)?;
    Ok(result)
}

fn enclosing_tetrahedron(
    nodes: &[DelaunayVolumeNode],
) -> Result<[DelaunayVolumeNode; 4], DelaunayPointSetError> {
    let mut minimum = [f64::INFINITY; 3];
    let mut maximum = [f64::NEG_INFINITY; 3];
    for node in nodes {
        for axis in 0..3 {
            minimum[axis] = minimum[axis].min(node.coordinates_m[axis]);
            maximum[axis] = maximum[axis].max(node.coordinates_m[axis]);
        }
    }
    let center: [f64; 3] = std::array::from_fn(|axis| minimum[axis] * 0.5 + maximum[axis] * 0.5);
    let extent = nodes
        .iter()
        .flat_map(|node| (0..3).map(|axis| (node.coordinates_m[axis] - center[axis]).abs()))
        .fold(0.0_f64, f64::max);
    if center.iter().any(|value| !value.is_finite()) || !extent.is_finite() || extent == 0.0 {
        return Err(error(
            DelaunayPointSetErrorKind::InsufficientDimension,
            "point extent cannot define a finite enclosing tetrahedron",
        ));
    }
    let radius = extent * 1_048_576.0;
    let sqrt_two = 2.0_f64.sqrt();
    let sqrt_six = 6.0_f64.sqrt();
    let offsets = [
        [0.0, 0.0, 3.0 * radius],
        [2.0 * sqrt_two * radius, 0.0, -radius],
        [-sqrt_two * radius, sqrt_six * radius, -radius],
        [-sqrt_two * radius, -sqrt_six * radius, -radius],
    ];
    let occupied = nodes
        .iter()
        .map(|node| node.identity)
        .collect::<BTreeSet<_>>();
    let identities = construction_identities(&occupied);
    let result = std::array::from_fn(|index| {
        DelaunayVolumeNode::explicit(
            identities[index],
            std::array::from_fn(|axis| center[axis] + offsets[index][axis]),
        )
    });
    if result
        .iter()
        .flat_map(|node| node.coordinates_m)
        .any(|value| !value.is_finite())
    {
        return Err(resource(
            "coordinate range cannot form a finite enclosing tetrahedron",
        ));
    }
    Ok(result)
}

fn construction_identities(occupied: &BTreeSet<StableDigest>) -> [StableDigest; 4] {
    let mut identities = [StableDigest::ZERO; 4];
    let mut admitted = 0;
    let mut counter = 1_u64;
    while admitted < identities.len() {
        let mut bytes = [0; 32];
        bytes[24..].copy_from_slice(&counter.to_be_bytes());
        let identity = StableDigest::from_bytes(bytes);
        if !occupied.contains(&identity) {
            identities[admitted] = identity;
            admitted += 1;
        }
        counter += 1;
    }
    identities
}
