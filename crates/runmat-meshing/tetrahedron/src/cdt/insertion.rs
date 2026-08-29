use std::collections::BTreeSet;

use runmat_meshing_core::{
    quality::predicate::{orient3d, PredicateSign, SpatialPredicateError},
    MeshingCancellationSignal,
};

use super::{
    mutation::{TopologyMutation, TopologyMutationError, TopologyMutationErrorKind},
    DelaunayTopologyError, DelaunayTopologyOptions, DelaunayVolumeNode, DelaunayVolumeTopology,
};

mod cavity;
mod legalization;
mod location;
mod validation;
mod work;

use cavity::{connected_cavity, node_coplanar_with_face, stable_face};
use legalization::{legalize_delaunay_volume_topology, legalize_delaunay_volume_topology_complete};
use location::TetrahedronPointLocator;
pub(super) use validation::validate_constrained_delaunay_volume_topology;
pub use validation::validate_delaunay_volume_topology;
use work::InsertionWork;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DelaunayInsertionOptions {
    pub topology: DelaunayTopologyOptions,
    pub maximum_protected_faces: u64,
    pub maximum_cavity_tetrahedra: u64,
    pub maximum_cavity_boundary_faces: u64,
    pub maximum_predicate_evaluations: u64,
}

impl Default for DelaunayInsertionOptions {
    fn default() -> Self {
        Self {
            topology: DelaunayTopologyOptions::default(),
            maximum_protected_faces: 2_000_000_000,
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
    insert_delaunay_volume_node_internal(topology, node, &[], false, options, cancellation)
}

pub(super) fn insert_delaunay_volume_node_with_barriers(
    topology: DelaunayVolumeTopology,
    node: DelaunayVolumeNode,
    protected_faces: &[[runmat_meshing_core::StableDigest; 3]],
    options: DelaunayInsertionOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<DelaunayVolumeTopology, DelaunayInsertionError> {
    insert_delaunay_volume_node_internal(
        topology,
        node,
        protected_faces,
        true,
        options,
        cancellation,
    )
}

pub(super) fn legalize_delaunay_volume_topology_with_barriers(
    topology: DelaunayVolumeTopology,
    protected_faces: &[[runmat_meshing_core::StableDigest; 3]],
    options: DelaunayInsertionOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<DelaunayVolumeTopology, DelaunayInsertionError> {
    validate_options(options)?;
    if protected_faces.len() as u64 > options.maximum_protected_faces {
        return Err(resource("protected-face inventory limit exceeded"));
    }
    let protected_faces = protected_faces.iter().copied().collect::<BTreeSet<_>>();
    let mut work = InsertionWork::new(options, cancellation);
    legalize_delaunay_volume_topology(topology, &protected_faces, true, &mut work)
}

pub(super) fn legalize_delaunay_volume_topology_completely_with_barriers(
    topology: DelaunayVolumeTopology,
    protected_faces: &[[runmat_meshing_core::StableDigest; 3]],
    options: DelaunayInsertionOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<DelaunayVolumeTopology, DelaunayInsertionError> {
    validate_options(options)?;
    if protected_faces.len() as u64 > options.maximum_protected_faces {
        return Err(resource("protected-face inventory limit exceeded"));
    }
    let protected_faces = protected_faces.iter().copied().collect::<BTreeSet<_>>();
    let mut work = InsertionWork::new(options, cancellation);
    legalize_delaunay_volume_topology_complete(topology, &protected_faces, true, &mut work)
}

fn insert_delaunay_volume_node_internal(
    topology: DelaunayVolumeTopology,
    node: DelaunayVolumeNode,
    protected_faces: &[[runmat_meshing_core::StableDigest; 3]],
    protect_region_boundaries: bool,
    options: DelaunayInsertionOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<DelaunayVolumeTopology, DelaunayInsertionError> {
    validate_options(options)?;
    validate_new_node(&topology, node)?;
    let protected_faces = protected_faces.iter().copied().collect::<BTreeSet<_>>();

    let mut work = InsertionWork::new(options, cancellation);
    let locator = TetrahedronPointLocator::walking(&topology);
    let seed = locator
        .first_containing_from(node, None, &mut work)?
        .ok_or_else(|| {
        let bounds = topology_bounds(&topology);
        error(
            DelaunayInsertionErrorKind::PointOutsideTopology,
            format!(
                "node {:?} is not contained by the admitted tetrahedron complex (nodes={}, tetrahedra={}, bounds={bounds:?})",
                node.coordinates_m,
                topology.nodes.len(),
                topology.tetrahedra.len()
            ),
        )
        })?;

    let mut cavity = connected_cavity(
        &topology,
        node,
        seed,
        &protected_faces,
        protect_region_boundaries,
        &mut work,
    )?;
    let used_containing_star = match regularize_cavity_boundary(
        &topology,
        &mut cavity,
        node,
        &protected_faces,
        protect_region_boundaries,
        &mut work,
    ) {
        Ok(()) => false,
        Err(failure) if failure.kind == DelaunayInsertionErrorKind::InvalidTopology => {
            cavity = locator.all_containing(node, &mut work)?;
            if cavity.is_empty() {
                return Err(error(
                    DelaunayInsertionErrorKind::PointOutsideTopology,
                    "fallback insertion could not recover the node's containing star",
                ));
            }
            true
        }
        Err(failure) => return Err(failure),
    };
    let boundary = cavity_boundary(&topology, &cavity, node, &mut work)?;
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
    let removed = cavity
        .into_iter()
        .map(|index| u32::try_from(index).map_err(|_| mutation_error("cavity index overflow")))
        .collect::<Result<BTreeSet<_>, _>>()?;
    let mut mutation = TopologyMutation::from_canonical(topology);
    let inserted_index = mutation
        .append_node(node)
        .map_err(mutation_topology_error)?;
    let created = boundary
        .into_iter()
        .map(|face| {
            (
                [face[0], face[1], face[2], inserted_index],
                cavity_region.clone(),
            )
        })
        .collect();
    mutation
        .replace_cavity(&removed, created)
        .map_err(mutation_topology_error)?;
    let rebuilt = mutation
        .freeze(options.topology, cancellation)
        .map_err(mutation_topology_error)?;
    if used_containing_star {
        legalize_delaunay_volume_topology_complete(
            rebuilt,
            &protected_faces,
            protect_region_boundaries,
            &mut work,
        )
    } else {
        legalize_delaunay_volume_topology(
            rebuilt,
            &protected_faces,
            protect_region_boundaries,
            &mut work,
        )
    }
}

fn topology_bounds(topology: &DelaunayVolumeTopology) -> [[f64; 3]; 2] {
    let mut minimum = [f64::INFINITY; 3];
    let mut maximum = [f64::NEG_INFINITY; 3];
    for node in &topology.nodes {
        for axis in 0..3 {
            minimum[axis] = minimum[axis].min(node.coordinates_m[axis]);
            maximum[axis] = maximum[axis].max(node.coordinates_m[axis]);
        }
    }
    [minimum, maximum]
}

fn cavity_boundary(
    topology: &DelaunayVolumeTopology,
    cavity: &BTreeSet<usize>,
    node: DelaunayVolumeNode,
    work: &mut InsertionWork<'_>,
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
            // A point admitted on the convex hull replaces a coplanar boundary
            // face with the triangles induced by the adjacent non-coplanar
            // cavity faces. Coning the old face to the point would create an
            // exactly zero-volume tetrahedron.
            if node_coplanar_with_face(topology, tetrahedron.vertex_indices, opposite, node, work)?
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
            if boundary.len() as u64 > work.options.maximum_cavity_boundary_faces {
                return Err(resource("cavity boundary-face limit exceeded"));
            }
        }
    }
    Ok(boundary.into_iter().collect())
}

fn regularize_cavity_boundary(
    topology: &DelaunayVolumeTopology,
    cavity: &mut BTreeSet<usize>,
    node: DelaunayVolumeNode,
    protected_faces: &BTreeSet<[runmat_meshing_core::StableDigest; 3]>,
    protect_region_boundaries: bool,
    work: &mut InsertionWork<'_>,
) -> Result<(), DelaunayInsertionError> {
    loop {
        let boundary = cavity_boundary(topology, cavity, node, work)?;
        let mut edge_uses = std::collections::BTreeMap::<[u32; 2], usize>::new();
        for face in &boundary {
            for mut edge in [[face[0], face[1]], [face[1], face[2]], [face[2], face[0]]] {
                edge.sort_unstable();
                *edge_uses.entry(edge).or_default() += 1;
            }
        }
        let Some((edge, _)) = edge_uses.into_iter().find(|(_, uses)| *uses > 2) else {
            return Ok(());
        };
        let boundary = boundary.into_iter().collect::<BTreeSet<_>>();
        let cavity_region = topology.tetrahedra[*cavity.iter().next().ok_or_else(|| {
            error(
                DelaunayInsertionErrorKind::InvalidTopology,
                "insertion cavity is empty",
            )
        })?]
        .region_id
        .clone();
        let candidate = topology.incidence.vertex_stars[edge[0] as usize]
            .iter()
            .copied()
            .filter(|index| !cavity.contains(&(*index as usize)))
            .filter(|index| {
                let tetrahedron = &topology.tetrahedra[*index as usize];
                tetrahedron.vertex_indices.contains(&edge[1])
                    && tetrahedron.region_id == cavity_region
            })
            .find(|index| {
                let tetrahedron = &topology.tetrahedra[*index as usize];
                let shared = (0..4)
                    .filter_map(|opposite| {
                        let mut face = [0; 3];
                        let mut cursor = 0;
                        for (slot, vertex) in tetrahedron.vertex_indices.iter().enumerate() {
                            if slot != opposite {
                                face[cursor] = *vertex;
                                cursor += 1;
                            }
                        }
                        face.sort_unstable();
                        boundary.contains(&face).then_some(opposite)
                    })
                    .collect::<Vec<_>>();
                shared.len() >= 2
                    && shared.iter().all(|opposite| {
                        !protected_faces.contains(&stable_face(
                            topology,
                            tetrahedron.vertex_indices,
                            *opposite,
                        ))
                    })
                    && (!protect_region_boundaries || tetrahedron.region_id == cavity_region)
            })
            .map(|index| index as usize)
            .ok_or_else(|| {
                error(
                    DelaunayInsertionErrorKind::InvalidTopology,
                    format!(
                        "insertion cavity boundary is non-manifold at edge {edge:?} and cannot \
                         be regularized without crossing a protected face"
                    ),
                )
            })?;
        cavity.insert(candidate);
        if cavity.len() as u64 > work.options.maximum_cavity_tetrahedra {
            return Err(resource("regularized cavity tetrahedron limit exceeded"));
        }
    }
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

fn two_to_three_topology_is_admissible(
    topology: &DelaunayVolumeTopology,
    left_tetrahedron: usize,
    right_tetrahedron: usize,
    left_apex: u32,
    right_apex: u32,
) -> bool {
    !topology.incidence.vertex_stars[left_apex as usize]
        .iter()
        .copied()
        .any(|tetrahedron| {
            let tetrahedron = tetrahedron as usize;
            tetrahedron != left_tetrahedron
                && tetrahedron != right_tetrahedron
                && topology.tetrahedra[tetrahedron]
                    .vertex_indices
                    .contains(&right_apex)
        })
}

fn validate_new_node(
    topology: &DelaunayVolumeTopology,
    node: DelaunayVolumeNode,
) -> Result<(), DelaunayInsertionError> {
    validate_node_values(node)?;
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

fn validate_node_values(node: DelaunayVolumeNode) -> Result<(), DelaunayInsertionError> {
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
    Ok(())
}

pub(super) fn validate_options(
    options: DelaunayInsertionOptions,
) -> Result<(), DelaunayInsertionError> {
    if options.maximum_cavity_tetrahedra == 0
        || options.maximum_protected_faces == 0
        || options.maximum_cavity_boundary_faces == 0
        || options.maximum_predicate_evaluations == 0
        || options.topology.maximum_nodes == 0
        || options.topology.maximum_tetrahedra == 0
        || options.topology.cancellation_check_interval == 0
    {
        return Err(error(
            DelaunayInsertionErrorKind::InvalidOptions,
            "insertion and topology work limits must be nonzero",
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

fn mutation_topology_error(error_value: TopologyMutationError) -> DelaunayInsertionError {
    let kind = match error_value.kind {
        TopologyMutationErrorKind::InvalidTopology => DelaunayInsertionErrorKind::InvalidTopology,
        TopologyMutationErrorKind::ResourceLimit => DelaunayInsertionErrorKind::ResourceLimit,
        TopologyMutationErrorKind::Cancelled => DelaunayInsertionErrorKind::Cancelled,
    };
    error(kind, error_value.reason)
}

fn mutation_error(reason: impl Into<String>) -> DelaunayInsertionError {
    error(DelaunayInsertionErrorKind::InvalidTopology, reason)
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
