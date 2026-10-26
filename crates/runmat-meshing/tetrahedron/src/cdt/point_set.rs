use std::collections::BTreeSet;

use runmat_meshing_core::{MeshingCancellationSignal, StableDigest};

use super::insertion::{validate_locally_delaunay_volume_topology, UnconstrainedHull};
use super::{
    build_delaunay_volume_topology, DelaunayInsertionError, DelaunayInsertionErrorKind,
    DelaunayInsertionOptions, DelaunayTopologyError, DelaunayTopologyErrorKind, DelaunayVolumeNode,
    DelaunayVolumeTopology,
};

mod hull_validation;
#[cfg(test)]
mod scaling;
mod seed;
mod work;
use hull_validation::validate_convex_hull_closure;

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
/// set by growing its real convex hull; no artificial enclosing vertices are used.
pub fn build_delaunay_volume_point_set(
    nodes: Vec<DelaunayVolumeNode>,
    options: DelaunayPointSetOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<DelaunayVolumeTopology, DelaunayPointSetError> {
    build_delaunay_volume_point_set_with_order(
        nodes,
        options,
        cancellation,
        compare_geometric_nodes,
    )
}

/// Insertion ordering is a scheduling hint only. The original point inventory
/// remains authoritative for every predicate, cavity and published coordinate.
pub(in crate::cdt) fn build_delaunay_volume_point_set_with_order(
    mut nodes: Vec<DelaunayVolumeNode>,
    options: DelaunayPointSetOptions,
    cancellation: &dyn MeshingCancellationSignal,
    compare: impl Fn(&DelaunayVolumeNode, &DelaunayVolumeNode) -> std::cmp::Ordering,
) -> Result<DelaunayVolumeTopology, DelaunayPointSetError> {
    validate_options(options)?;
    validate_and_order_nodes(&mut nodes, options, cancellation)?;
    validate_coordinate_range(&nodes)?;
    let mut geometric_nodes = nodes.clone();
    geometric_nodes.sort_by(compare);
    let initial = seed::initial_tetrahedron(
        &geometric_nodes,
        &mut work::HullWork::new(options, cancellation),
    )?;
    let mut seed_nodes = initial.map(|index| geometric_nodes[index]).to_vec();
    seed_nodes.sort_by_key(|node| node.identity);
    let topology = build_delaunay_volume_topology(
        seed_nodes,
        vec![[0, 1, 2, 3]],
        options.insertion.topology,
        cancellation,
    )
    .map_err(topology_error)?;
    let mut hull = UnconstrainedHull::new(topology).map_err(insertion_error)?;
    for (index, node) in geometric_nodes.iter().copied().enumerate() {
        if initial.contains(&index) {
            continue;
        }
        hull.insert(node, options.insertion, cancellation)
            .map_err(|failure| insertion_error_at_node(failure, index, node))?;
    }

    let topology = hull
        .freeze(options.insertion, cancellation)
        .map_err(insertion_error)?;
    if topology.nodes.len() != nodes.len()
        || topology
            .incidence
            .vertex_stars
            .iter()
            .any(|star| star.is_empty())
    {
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

fn validate_coordinate_range(nodes: &[DelaunayVolumeNode]) -> Result<(), DelaunayPointSetError> {
    let mut minimum = [f64::INFINITY; 3];
    let mut maximum = [f64::NEG_INFINITY; 3];
    for node in nodes {
        for axis in 0..3 {
            minimum[axis] = minimum[axis].min(node.coordinates_m.canonical_value()[axis]);
            maximum[axis] = maximum[axis].max(node.coordinates_m.canonical_value()[axis]);
        }
    }
    if (0..3).any(|axis| !(maximum[axis] - minimum[axis]).is_finite()) {
        return Err(resource(
            "coordinate range cannot support finite spatial predicates",
        ));
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
    // Geometric order gives equivalent inputs a deterministic insertion schedule.
    // Rigid transformations may reorder cospherical choices; independent physical
    // admission, rather than identical connectivity, governs those cases.
    nodes.sort_by_key(|node| node.identity);
    let mut positions = BTreeSet::new();
    for (index, node) in nodes.iter().enumerate() {
        if (index as u64).is_multiple_of(options.insertion.topology.cancellation_check_interval)
            && cancellation.is_cancelled()
        {
            return Err(error(DelaunayPointSetErrorKind::Cancelled, "cancelled"));
        }
        if node.identity == StableDigest::ZERO
            || node
                .coordinates_m
                .canonical_value()
                .iter()
                .any(|value| !value.is_finite())
            || index > 0 && nodes[index - 1].identity == node.identity
            || !positions.insert(node.coordinates_m.canonical_value().map(coordinate_bits))
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
        .canonical_value()
        .iter()
        .zip(right.coordinates_m.canonical_value().iter())
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
        node.coordinates_m.canonical_value()[0],
        node.coordinates_m.canonical_value()[1],
        node.coordinates_m.canonical_value()[2]
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
