use std::collections::BTreeSet;

use runmat_meshing_core::{
    quality::predicate::{orient3d, PredicateSign},
    MeshingCancellationSignal, StableDigest,
};

use super::insertion::{
    insert_delaunay_volume_node_into_mutation,
    legalize_delaunay_volume_topology_completely_with_barriers,
    validate_locally_delaunay_volume_topology,
};
use super::mutation::TopologyMutation;
use super::{
    build_delaunay_volume_topology, DelaunayInsertionError, DelaunayInsertionErrorKind,
    DelaunayInsertionOptions, DelaunayTopologyError, DelaunayTopologyErrorKind, DelaunayVolumeNode,
    DelaunayVolumeTopology,
};

mod enclosing;
mod hull;

use enclosing::enclosing_point_set;
use hull::convex_hull_seed;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DelaunayPointSetOptions {
    pub insertion: DelaunayInsertionOptions,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DelaunayPointSetErrorKind {
    InvalidOptions,
    InvalidNode,
    InsufficientDimension,
    ResourceLimit,
    Cancelled,
    InvalidTopology,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DelaunayPointSetError {
    pub kind: DelaunayPointSetErrorKind,
    pub reason: String,
}

impl std::fmt::Display for DelaunayPointSetError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "3D Delaunay point set {:?}: {}",
            self.kind, self.reason
        )
    }
}

impl std::error::Error for DelaunayPointSetError {}

/// Builds the canonical unconstrained Delaunay tetrahedralization of a point
/// set. Its exact convex hull is constructed first, then non-hull points are inserted internally.
pub fn build_delaunay_volume_point_set(
    mut nodes: Vec<DelaunayVolumeNode>,
    options: DelaunayPointSetOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<DelaunayVolumeTopology, DelaunayPointSetError> {
    validate_options(options)?;
    validate_and_order_nodes(&mut nodes, options, cancellation)?;
    validate_coordinate_range(&nodes)?;
    match enclosing_point_set(&nodes, options, cancellation) {
        Ok(topology) => match validate_convex_hull_closure(&topology, options, cancellation) {
            Ok(()) => return Ok(topology),
            Err(failure) if failure.kind == DelaunayPointSetErrorKind::InvalidTopology => {}
            Err(failure) => return Err(failure),
        },
        Err(failure) if failure.kind == DelaunayPointSetErrorKind::InvalidTopology => {}
        Err(failure) => return Err(failure),
    }

    let mut geometric_nodes = nodes.clone();
    geometric_nodes.sort_by(compare_geometric_nodes);
    let seed = convex_hull_seed(&geometric_nodes, options, cancellation)?;
    let working_options = options.insertion;
    let topology = build_delaunay_volume_topology(
        seed.nodes,
        seed.tetrahedra,
        working_options.topology,
        cancellation,
    )
    .map_err(topology_error)?;
    let topology = ensure_locally_delaunay(topology, working_options, cancellation)?;
    validate_convex_hull_closure(&topology, options, cancellation)?;

    let mut topology = TopologyMutation::from_canonical(topology);
    for (index, node) in geometric_nodes.iter().copied().enumerate() {
        if seed.identities.contains(&node.identity) {
            continue;
        }
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
        .map_err(super::insertion::mutation_topology_error)
        .map_err(insertion_error)?;
    let topology = ensure_locally_delaunay(topology, working_options, cancellation)?;

    if topology.nodes.len() != nodes.len() {
        return Err(error(
            DelaunayPointSetErrorKind::InvalidTopology,
            "convex-hull bootstrap did not retain every admitted point",
        ));
    }
    validate_locally_delaunay_volume_topology(&topology, options.insertion, cancellation)
        .map_err(insertion_error)?;
    validate_convex_hull_closure(&topology, options, cancellation)?;
    Ok(topology)
}

/// Bowyer-Watson insertion normally leaves the point set locally Delaunay. The
/// complete flip search is a deterministic repair path for degenerate cases,
/// not work every successful construction should pay unconditionally.
fn ensure_locally_delaunay(
    topology: DelaunayVolumeTopology,
    options: DelaunayInsertionOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<DelaunayVolumeTopology, DelaunayPointSetError> {
    match validate_locally_delaunay_volume_topology(&topology, options, cancellation) {
        Ok(()) => Ok(topology),
        Err(failure) if failure.kind == DelaunayInsertionErrorKind::InvalidTopology => {
            legalize_delaunay_volume_topology_completely_with_barriers(
                topology,
                &[],
                options,
                cancellation,
            )
            .map_err(insertion_error)
        }
        Err(failure) => Err(insertion_error(failure)),
    }
}

fn validate_coordinate_range(nodes: &[DelaunayVolumeNode]) -> Result<(), DelaunayPointSetError> {
    let mut minimum = [f64::INFINITY; 3];
    let mut maximum = [f64::NEG_INFINITY; 3];
    for node in nodes {
        for axis in 0..3 {
            minimum[axis] = minimum[axis].min(node.coordinates_m[axis]);
            maximum[axis] = maximum[axis].max(node.coordinates_m[axis]);
        }
    }
    if (0..3).any(|axis| !(maximum[axis] - minimum[axis]).is_finite()) {
        return Err(resource(
            "coordinate range cannot support finite spatial predicates",
        ));
    }
    Ok(())
}

fn validate_convex_hull_closure(
    topology: &DelaunayVolumeTopology,
    options: DelaunayPointSetOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<(), DelaunayPointSetError> {
    let mut predicates = 0_u64;
    for (tetrahedron_index, tetrahedron) in topology.tetrahedra.iter().enumerate() {
        for opposite in 0..4 {
            if tetrahedron.neighbors[opposite].is_some() {
                continue;
            }
            let mut face = [0_u32; 3];
            let mut cursor = 0;
            for (slot, vertex) in tetrahedron.vertex_indices.iter().enumerate() {
                if slot != opposite {
                    face[cursor] = *vertex;
                    cursor += 1;
                }
            }
            let face_points = face.map(|vertex| topology.nodes[vertex as usize].coordinates_m);
            let interior_index = tetrahedron.vertex_indices[opposite] as usize;
            let interior_sign = orient3d([
                face_points[0],
                face_points[1],
                face_points[2],
                topology.nodes[interior_index].coordinates_m,
            ])
            .map_err(|failure| {
                error(
                    DelaunayPointSetErrorKind::InvalidTopology,
                    format!("convex-hull predicate rejected an interior node: {failure:?}"),
                )
            })?;
            for (node_index, node) in topology.nodes.iter().enumerate() {
                predicates = predicates
                    .checked_add(1)
                    .ok_or_else(|| resource("convex-hull validation predicate count overflow"))?;
                if predicates > options.insertion.maximum_predicate_evaluations {
                    return Err(resource("convex-hull validation predicate limit exceeded"));
                }
                if predicates.is_multiple_of(options.insertion.topology.cancellation_check_interval)
                    && cancellation.is_cancelled()
                {
                    return Err(error(DelaunayPointSetErrorKind::Cancelled, "cancelled"));
                }
                let sign = orient3d([
                    face_points[0],
                    face_points[1],
                    face_points[2],
                    node.coordinates_m,
                ])
                .map_err(|failure| {
                    error(
                        DelaunayPointSetErrorKind::InvalidTopology,
                        format!("convex-hull predicate rejected an admitted node: {failure:?}"),
                    )
                })?;
                if sign != PredicateSign::Zero && sign != interior_sign {
                    return Err(error(
                        DelaunayPointSetErrorKind::InvalidTopology,
                        format!(
                            "boundary face {face:?} of tetrahedron {tetrahedron_index} does not support the admitted convex hull; node {node_index} lies on its exterior side"
                        ),
                    ));
                }
            }
        }
    }
    Ok(())
}

fn validate_and_order_nodes(
    nodes: &mut [DelaunayVolumeNode],
    options: DelaunayPointSetOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<(), DelaunayPointSetError> {
    if nodes.len() < 4 {
        return Err(error(
            DelaunayPointSetErrorKind::InsufficientDimension,
            "at least four nodes are required",
        ));
    }
    if nodes.len() as u64 > options.insertion.topology.maximum_nodes {
        return Err(resource("node limit exceeded"));
    }
    // Persistent identities deliberately bind nodes to one source revision. They
    // must not decide the geometry produced for an equivalent point cloud.
    // Canonical volume coordinates are exact dyadic values, so their total order
    // provides a stable insertion schedule across source units and rigid motion.
    nodes.sort_by_key(|node| node.identity);
    let mut positions = BTreeSet::new();
    for (index, node) in nodes.iter().enumerate() {
        if (index as u64).is_multiple_of(options.insertion.topology.cancellation_check_interval)
            && cancellation.is_cancelled()
        {
            return Err(error(DelaunayPointSetErrorKind::Cancelled, "cancelled"));
        }
        if node.identity == StableDigest::ZERO
            || node.coordinates_m.iter().any(|value| !value.is_finite())
            || index > 0 && nodes[index - 1].identity == node.identity
            || !positions.insert(node.coordinates_m.map(coordinate_bits))
        {
            return Err(error(
                DelaunayPointSetErrorKind::InvalidNode,
                "nodes require unique nonzero identities and unique finite positions",
            ));
        }
    }
    Ok(())
}

fn compare_geometric_nodes(
    left: &DelaunayVolumeNode,
    right: &DelaunayVolumeNode,
) -> std::cmp::Ordering {
    left.coordinates_m
        .iter()
        .zip(right.coordinates_m.iter())
        .find_map(|(left, right)| (left != right).then(|| left.total_cmp(right)))
        .unwrap_or_else(|| left.identity.cmp(&right.identity))
}

fn coordinate_bits(value: f64) -> u64 {
    if value == 0.0 {
        0
    } else {
        value.to_bits()
    }
}

fn validate_options(options: DelaunayPointSetOptions) -> Result<(), DelaunayPointSetError> {
    if options.insertion.topology.maximum_nodes < 4
        || options.insertion.topology.maximum_tetrahedra == 0
        || options.insertion.topology.cancellation_check_interval == 0
        || options.insertion.maximum_cavity_tetrahedra == 0
        || options.insertion.maximum_cavity_boundary_faces == 0
        || options.insertion.maximum_predicate_evaluations == 0
    {
        return Err(error(
            DelaunayPointSetErrorKind::InvalidOptions,
            "all point-set, topology, and insertion limits must be nonzero",
        ));
    }
    Ok(())
}

fn insertion_error(error_value: DelaunayInsertionError) -> DelaunayPointSetError {
    let kind = match error_value.kind {
        DelaunayInsertionErrorKind::InvalidOptions => DelaunayPointSetErrorKind::InvalidOptions,
        DelaunayInsertionErrorKind::InvalidNode => DelaunayPointSetErrorKind::InvalidNode,
        DelaunayInsertionErrorKind::ResourceLimit => DelaunayPointSetErrorKind::ResourceLimit,
        DelaunayInsertionErrorKind::Cancelled => DelaunayPointSetErrorKind::Cancelled,
        DelaunayInsertionErrorKind::PointOutsideTopology
        | DelaunayInsertionErrorKind::InvalidTopology => DelaunayPointSetErrorKind::InvalidTopology,
    };
    error(kind, error_value.to_string())
}

fn insertion_error_at_node(
    error_value: DelaunayInsertionError,
    index: usize,
    node: DelaunayVolumeNode,
) -> DelaunayPointSetError {
    let mut mapped = insertion_error(error_value);
    mapped.reason = format!(
        "{} while inserting canonical node {index} ({:?}, [{}, {}, {}])",
        mapped.reason,
        node.identity,
        node.coordinates_m[0],
        node.coordinates_m[1],
        node.coordinates_m[2]
    );
    mapped
}

fn topology_error(error_value: DelaunayTopologyError) -> DelaunayPointSetError {
    let kind = match error_value.kind {
        DelaunayTopologyErrorKind::InvalidOptions => DelaunayPointSetErrorKind::InvalidOptions,
        DelaunayTopologyErrorKind::InvalidNode => DelaunayPointSetErrorKind::InvalidNode,
        DelaunayTopologyErrorKind::ResourceLimit => DelaunayPointSetErrorKind::ResourceLimit,
        DelaunayTopologyErrorKind::Cancelled => DelaunayPointSetErrorKind::Cancelled,
        DelaunayTopologyErrorKind::InvalidTetrahedron
        | DelaunayTopologyErrorKind::InvalidRegion
        | DelaunayTopologyErrorKind::DegenerateTetrahedron
        | DelaunayTopologyErrorKind::NonManifoldFace => DelaunayPointSetErrorKind::InvalidTopology,
    };
    error(kind, error_value.to_string())
}

fn resource(reason: impl Into<String>) -> DelaunayPointSetError {
    error(DelaunayPointSetErrorKind::ResourceLimit, reason)
}

fn error(kind: DelaunayPointSetErrorKind, reason: impl Into<String>) -> DelaunayPointSetError {
    DelaunayPointSetError {
        kind,
        reason: reason.into(),
    }
}

#[cfg(test)]
#[path = "point_set/tests.rs"]
mod tests;
