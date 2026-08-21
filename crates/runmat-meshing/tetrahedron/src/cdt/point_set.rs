use std::collections::{BTreeMap, BTreeSet};

use runmat_meshing_core::{
    quality::predicate::{orient2d, orient3d, PredicateSign},
    MeshingCancellationSignal, StableDigest,
};

use super::insertion::insert_delaunay_volume_node_mutation;
use super::{
    build_delaunay_volume_topology, validate_delaunay_volume_topology, DelaunayInsertionError,
    DelaunayInsertionErrorKind, DelaunayInsertionOptions, DelaunayTopologyError,
    DelaunayTopologyErrorKind, DelaunayVolumeNode, DelaunayVolumeTopology,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DelaunayPointSetOptions {
    pub insertion: DelaunayInsertionOptions,
    /// Full local-Delaunay validation cadence in optimized builds. Debug and
    /// test builds validate after every insertion.
    pub validation_check_interval: u64,
}

impl Default for DelaunayPointSetOptions {
    fn default() -> Self {
        Self {
            insertion: DelaunayInsertionOptions::default(),
            validation_check_interval: 256,
        }
    }
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
/// set. Construction-only enclosing nodes are removed before return.
pub fn build_delaunay_volume_point_set(
    mut nodes: Vec<DelaunayVolumeNode>,
    options: DelaunayPointSetOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<DelaunayVolumeTopology, DelaunayPointSetError> {
    validate_options(options)?;
    validate_and_order_nodes(&mut nodes, options, cancellation)?;
    let enclosing = enclosing_tetrahedron(&nodes)?;
    validate_spatial_dimension(&nodes, options, cancellation)?;
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
    let mut topology = build_delaunay_volume_topology(
        enclosing.to_vec(),
        vec![[0, 1, 2, 3]],
        working_options.topology,
        cancellation,
    )
    .map_err(topology_error)?;

    for (index, node) in nodes.iter().copied().enumerate() {
        topology =
            insert_delaunay_volume_node_mutation(topology, node, working_options, cancellation)
                .map_err(insertion_error)?;
        if cfg!(debug_assertions)
            || (index as u64 + 1).is_multiple_of(options.validation_check_interval)
        {
            validate_delaunay_volume_topology(&topology, working_options, cancellation)
                .map_err(insertion_error)?;
        }
    }

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
    let result =
        build_delaunay_volume_topology(nodes, tetrahedra, options.insertion.topology, cancellation)
            .map_err(topology_error)?;
    validate_delaunay_volume_topology(&result, options.insertion, cancellation)
        .map_err(insertion_error)?;
    Ok(result)
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

fn coordinate_bits(value: f64) -> u64 {
    if value == 0.0 {
        0
    } else {
        value.to_bits()
    }
}

fn validate_spatial_dimension(
    nodes: &[DelaunayVolumeNode],
    options: DelaunayPointSetOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<(), DelaunayPointSetError> {
    let first = nodes[0].coordinates_m;
    let second = nodes[1].coordinates_m;
    let interval = options.insertion.topology.cancellation_check_interval;
    let mut third = None;
    for (index, node) in nodes.iter().enumerate().skip(2) {
        if (index as u64).is_multiple_of(interval) && cancellation.is_cancelled() {
            return Err(error(DelaunayPointSetErrorKind::Cancelled, "cancelled"));
        }
        let point = node.coordinates_m;
        let projections = [
            [
                [first[0], first[1]],
                [second[0], second[1]],
                [point[0], point[1]],
            ],
            [
                [first[1], first[2]],
                [second[1], second[2]],
                [point[1], point[2]],
            ],
            [
                [first[2], first[0]],
                [second[2], second[0]],
                [point[2], point[0]],
            ],
        ];
        if projections.into_iter().any(|projection| {
            matches!(
                orient2d(projection),
                Ok(PredicateSign::Positive | PredicateSign::Negative)
            )
        }) {
            third = Some(index);
            break;
        }
    }
    let third = third.ok_or_else(|| {
        error(
            DelaunayPointSetErrorKind::InsufficientDimension,
            "all point-set nodes are collinear",
        )
    })?;
    for (index, node) in nodes.iter().enumerate() {
        if (index as u64).is_multiple_of(interval) && cancellation.is_cancelled() {
            return Err(error(DelaunayPointSetErrorKind::Cancelled, "cancelled"));
        }
        let sign = orient3d([
            first,
            second,
            nodes[third].coordinates_m,
            node.coordinates_m,
        ]);
        if matches!(sign, Ok(PredicateSign::Positive | PredicateSign::Negative)) {
            return Ok(());
        }
    }
    Err(error(
        DelaunayPointSetErrorKind::InsufficientDimension,
        "all point-set nodes are coplanar",
    ))
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
    let radius = extent * 4.0;
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
    let result = std::array::from_fn(|index| DelaunayVolumeNode {
        identity: identities[index],
        coordinates_m: std::array::from_fn(|axis| center[axis] + offsets[index][axis]),
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

fn validate_options(options: DelaunayPointSetOptions) -> Result<(), DelaunayPointSetError> {
    if options.validation_check_interval == 0
        || options.insertion.topology.maximum_nodes < 4
        || options.insertion.topology.maximum_tetrahedra == 0
        || options.insertion.topology.cancellation_check_interval == 0
        || options.insertion.maximum_cavity_tetrahedra == 0
        || options.insertion.maximum_cavity_boundary_faces == 0
        || options.insertion.maximum_predicate_evaluations == 0
    {
        return Err(error(
            DelaunayPointSetErrorKind::InvalidOptions,
            "all point-set, topology, insertion, and validation limits must be nonzero",
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
