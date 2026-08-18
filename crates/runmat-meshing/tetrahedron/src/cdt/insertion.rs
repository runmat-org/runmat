use std::collections::{BTreeSet, VecDeque};

use runmat_meshing_core::{
    quality::predicate::{
        insphere3d_symbolic, orient3d, PredicateSign, SpatialPredicateError, SpatialPredicatePoint,
    },
    MeshingCancellationSignal,
};

use super::{
    topology::build_delaunay_volume_topology_with_regions, DelaunayTopologyError,
    DelaunayTopologyOptions, DelaunayVolumeNode, DelaunayVolumeTopology,
};

mod validation;
mod work;

pub use validation::validate_delaunay_volume_topology;
use work::InsertionWork;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DelaunayInsertionOptions {
    pub topology: DelaunayTopologyOptions,
    pub maximum_cavity_tetrahedra: u64,
    pub maximum_cavity_boundary_faces: u64,
    pub maximum_predicate_evaluations: u64,
}

impl Default for DelaunayInsertionOptions {
    fn default() -> Self {
        Self {
            topology: DelaunayTopologyOptions::default(),
            maximum_cavity_tetrahedra: 10_000_000,
            maximum_cavity_boundary_faces: 20_000_000,
            maximum_predicate_evaluations: 100_000_000,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DelaunayInsertionErrorKind {
    InvalidOptions,
    InvalidTopology,
    InvalidNode,
    PointOutsideTopology,
    ResourceLimit,
    Cancelled,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DelaunayInsertionError {
    pub kind: DelaunayInsertionErrorKind,
    pub reason: String,
}

impl std::fmt::Display for DelaunayInsertionError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "3D Delaunay insertion {:?}: {}",
            self.kind, self.reason
        )
    }
}

impl std::error::Error for DelaunayInsertionError {}

/// Inserts one node with a connected Bowyer-Watson cavity and returns a fully
/// rebuilt canonical topology. The input must already be locally Delaunay.
pub fn insert_delaunay_volume_node(
    topology: DelaunayVolumeTopology,
    node: DelaunayVolumeNode,
    options: DelaunayInsertionOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<DelaunayVolumeTopology, DelaunayInsertionError> {
    validate_options(options)?;
    validate_delaunay_volume_topology(&topology, options, cancellation)?;
    let rebuilt = insert_delaunay_volume_node_mutation(topology, node, options, cancellation)?;
    validate_delaunay_volume_topology(&rebuilt, options, cancellation)?;
    Ok(rebuilt)
}

pub(super) fn insert_delaunay_volume_node_mutation(
    topology: DelaunayVolumeTopology,
    node: DelaunayVolumeNode,
    options: DelaunayInsertionOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<DelaunayVolumeTopology, DelaunayInsertionError> {
    validate_options(options)?;
    validate_new_node(&topology, node)?;

    let mut work = InsertionWork::new(options, cancellation);
    let mut seed = None;
    for (index, tetrahedron) in topology.tetrahedra.iter().enumerate() {
        work.checkpoint()?;
        if point_in_tetrahedron(&topology, tetrahedron.vertex_indices, node, &mut work)? {
            seed = Some(index);
            break;
        }
    }
    let seed = seed.ok_or_else(|| {
        error(
            DelaunayInsertionErrorKind::PointOutsideTopology,
            "node is not contained by the admitted tetrahedron complex",
        )
    })?;

    let cavity = connected_cavity(&topology, node, seed, &mut work)?;
    let boundary = cavity_boundary(&topology, &cavity, options)?;
    let cavity_regions = cavity
        .iter()
        .map(|index| topology.tetrahedra[*index].region_id.clone())
        .collect::<BTreeSet<_>>();
    if cavity_regions.len() != 1 {
        return Err(error(
            DelaunayInsertionErrorKind::InvalidTopology,
            "insertion cavity crosses assigned region incidence",
        ));
    }
    let cavity_region = cavity_regions.into_iter().next().flatten();
    let (nodes, remap, inserted_index) = insert_node_canonically(&topology.nodes, node);
    let mut tetrahedra =
        Vec::with_capacity(topology.tetrahedra.len() - cavity.len() + boundary.len());
    for (index, tetrahedron) in topology.tetrahedra.iter().enumerate() {
        if !cavity.contains(&index) {
            tetrahedra.push((
                tetrahedron
                    .vertex_indices
                    .map(|vertex| remap[vertex as usize]),
                tetrahedron.region_id.clone(),
            ));
        }
    }
    for face in boundary {
        tetrahedra.push((
            [
                remap[face[0] as usize],
                remap[face[1] as usize],
                remap[face[2] as usize],
                inserted_index,
            ],
            cavity_region.clone(),
        ));
    }
    build_delaunay_volume_topology_with_regions(nodes, tetrahedra, options.topology, cancellation)
        .map_err(topology_error)
}

fn connected_cavity(
    topology: &DelaunayVolumeTopology,
    node: DelaunayVolumeNode,
    seed: usize,
    work: &mut InsertionWork<'_>,
) -> Result<BTreeSet<usize>, DelaunayInsertionError> {
    let mut cavity = BTreeSet::new();
    let mut examined = BTreeSet::new();
    let mut queue = VecDeque::from([(seed, false)]);
    while let Some((index, forced)) = queue.pop_front() {
        work.checkpoint()?;
        if cavity.contains(&index) || !forced && !examined.insert(index) {
            continue;
        }
        let tetrahedron = &topology.tetrahedra[index];
        if !forced && !in_circumsphere(topology, tetrahedron.vertex_indices, node, work)? {
            continue;
        }
        cavity.insert(index);
        if cavity.len() as u64 > work.options.maximum_cavity_tetrahedra {
            return Err(resource("cavity tetrahedron limit exceeded"));
        }
        for (opposite, neighbor) in tetrahedron.neighbors.iter().enumerate() {
            if let Some(neighbor) = neighbor {
                // A symbolic in-sphere tie must never leave a physical zero-volume
                // replacement across a face containing the inserted node.
                let coplanar = node_coplanar_with_face(
                    topology,
                    tetrahedron.vertex_indices,
                    opposite,
                    node,
                    work,
                )?;
                queue.push_back((*neighbor as usize, coplanar));
            }
        }
    }
    if !cavity.contains(&seed) {
        return Err(error(
            DelaunayInsertionErrorKind::InvalidTopology,
            "the containing tetrahedron does not contain the node in its circumsphere",
        ));
    }
    Ok(cavity)
}

fn node_coplanar_with_face(
    topology: &DelaunayVolumeTopology,
    vertices: [u32; 4],
    opposite: usize,
    node: DelaunayVolumeNode,
    work: &mut InsertionWork<'_>,
) -> Result<bool, DelaunayInsertionError> {
    work.predicate()?;
    let mut points = [[0.0; 3]; 4];
    let mut cursor = 0;
    for (vertex_index, vertex) in vertices.iter().enumerate() {
        if vertex_index != opposite {
            points[cursor] = topology.nodes[*vertex as usize].coordinates_m;
            cursor += 1;
        }
    }
    points[3] = node.coordinates_m;
    orient3d(points)
        .map(|sign| sign == PredicateSign::Zero)
        .map_err(predicate_error)
}

fn cavity_boundary(
    topology: &DelaunayVolumeTopology,
    cavity: &BTreeSet<usize>,
    options: DelaunayInsertionOptions,
) -> Result<Vec<[u32; 3]>, DelaunayInsertionError> {
    let mut boundary = BTreeSet::new();
    for index in cavity {
        let tetrahedron = &topology.tetrahedra[*index];
        for opposite in 0..4 {
            if tetrahedron.neighbors[opposite]
                .is_some_and(|neighbor| cavity.contains(&(neighbor as usize)))
            {
                continue;
            }
            let mut face = [0; 3];
            let mut cursor = 0;
            for (vertex_index, vertex) in tetrahedron.vertex_indices.iter().enumerate() {
                if vertex_index != opposite {
                    face[cursor] = *vertex;
                    cursor += 1;
                }
            }
            face.sort_unstable();
            boundary.insert(face);
            if boundary.len() as u64 > options.maximum_cavity_boundary_faces {
                return Err(resource("cavity boundary-face limit exceeded"));
            }
        }
    }
    Ok(boundary.into_iter().collect())
}

fn point_in_tetrahedron(
    topology: &DelaunayVolumeTopology,
    vertices: [u32; 4],
    node: DelaunayVolumeNode,
    work: &mut InsertionWork<'_>,
) -> Result<bool, DelaunayInsertionError> {
    for replace in 0..4 {
        work.predicate()?;
        let mut points = vertices.map(|vertex| topology.nodes[vertex as usize].coordinates_m);
        points[replace] = node.coordinates_m;
        let sign = orient3d(points).map_err(predicate_error)?;
        if !matches!(sign, PredicateSign::Positive | PredicateSign::Zero) {
            return Ok(false);
        }
    }
    Ok(true)
}

fn in_circumsphere(
    topology: &DelaunayVolumeTopology,
    vertices: [u32; 4],
    node: DelaunayVolumeNode,
    work: &mut InsertionWork<'_>,
) -> Result<bool, DelaunayInsertionError> {
    work.predicate()?;
    let points = vertices.map(|vertex| predicate_point(topology.nodes[vertex as usize]));
    let query = predicate_point(node);
    insphere3d_symbolic([points[0], points[1], points[2], points[3], query])
        .map(|sign| sign == PredicateSign::Positive)
        .map_err(predicate_error)
}

fn predicate_point(node: DelaunayVolumeNode) -> SpatialPredicatePoint {
    SpatialPredicatePoint {
        identity: node.identity,
        coordinates: node.coordinates_m,
    }
}

fn insert_node_canonically(
    nodes: &[DelaunayVolumeNode],
    node: DelaunayVolumeNode,
) -> (Vec<DelaunayVolumeNode>, Vec<u32>, u32) {
    let insertion = nodes.partition_point(|existing| existing.identity < node.identity);
    let mut result = nodes.to_vec();
    result.insert(insertion, node);
    let remap = (0..nodes.len())
        .map(|index| (index + usize::from(index >= insertion)) as u32)
        .collect();
    (result, remap, insertion as u32)
}

fn validate_new_node(
    topology: &DelaunayVolumeTopology,
    node: DelaunayVolumeNode,
) -> Result<(), DelaunayInsertionError> {
    if node.identity == runmat_meshing_core::StableDigest::ZERO
        || node
            .coordinates_m
            .iter()
            .any(|coordinate| !coordinate.is_finite())
    {
        return Err(error(
            DelaunayInsertionErrorKind::InvalidNode,
            "node identity must be nonzero and coordinates finite",
        ));
    }
    if topology.nodes.iter().any(|existing| {
        existing.identity == node.identity || existing.coordinates_m == node.coordinates_m
    }) {
        return Err(error(
            DelaunayInsertionErrorKind::InvalidNode,
            "node identity and geometric position must both be unique",
        ));
    }
    Ok(())
}

pub(super) fn validate_options(
    options: DelaunayInsertionOptions,
) -> Result<(), DelaunayInsertionError> {
    if options.maximum_cavity_tetrahedra == 0
        || options.maximum_cavity_boundary_faces == 0
        || options.maximum_predicate_evaluations == 0
    {
        return Err(error(
            DelaunayInsertionErrorKind::InvalidOptions,
            "insertion work limits must be nonzero",
        ));
    }
    Ok(())
}

pub(super) fn predicate_error(error_value: SpatialPredicateError) -> DelaunayInsertionError {
    error(
        DelaunayInsertionErrorKind::InvalidNode,
        format!("spatial predicate rejected coordinates or identity: {error_value:?}"),
    )
}

pub(super) fn topology_error(error_value: DelaunayTopologyError) -> DelaunayInsertionError {
    let kind = match error_value.kind {
        super::DelaunayTopologyErrorKind::Cancelled => DelaunayInsertionErrorKind::Cancelled,
        super::DelaunayTopologyErrorKind::ResourceLimit => {
            DelaunayInsertionErrorKind::ResourceLimit
        }
        _ => DelaunayInsertionErrorKind::InvalidTopology,
    };
    error(kind, error_value.to_string())
}

pub(super) fn resource(reason: impl Into<String>) -> DelaunayInsertionError {
    error(DelaunayInsertionErrorKind::ResourceLimit, reason)
}

pub(super) fn error(
    kind: DelaunayInsertionErrorKind,
    reason: impl Into<String>,
) -> DelaunayInsertionError {
    DelaunayInsertionError {
        kind,
        reason: reason.into(),
    }
}

#[cfg(test)]
#[path = "insertion/tests.rs"]
mod tests;
