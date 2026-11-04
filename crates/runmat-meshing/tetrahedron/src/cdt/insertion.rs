pub(super) use legalization::local::legalize_delaunay_volume_topology_locally_with_barriers;
use std::collections::BTreeSet;

use runmat_geometry_core::PersistentEntityId;
use runmat_meshing_core::{
    quality::predicate::{PredicateSign, SpatialPredicateError},
    MeshingCancellationSignal,
};

use super::{
    mutation::{TopologyMutation, TopologyMutationError, TopologyMutationErrorKind},
    topology::orient_materialized_volume_nodes as orient_volume_nodes,
    DelaunayTopologyError, DelaunayTopologyOptions, DelaunayVolumeNode, DelaunayVolumeTopology,
};

mod cavity;
mod conforming;
pub(in crate::cdt) use conforming::insert_conforming_volume_node_into_mutation_with_barriers;
pub(in crate::cdt) mod cospherical;
mod exterior;
pub(in crate::cdt) mod local_admission;
pub(in crate::cdt) use exterior::UnconstrainedHull;
mod legalization;
mod location;
mod materialization;
mod mutable;
mod physical_admission;
mod repair;
pub(in crate::cdt) use physical_admission::{restore_admitted_mutation, AdmittedPhysicalTopology};
pub(super) use repair::restore_constrained_delaunay_topology;
pub(super) mod topology_access;
mod validation;
mod work;

use cavity::{
    cavity_is_star_shaped, connected_cavity, node_lies_on_face, restrict_cavity_to_protected_side,
};
use legalization::{legalize_delaunay_volume_topology, legalize_delaunay_volume_topology_complete};
use location::TetrahedronPointLocator;
use materialization::representable_nodes;
pub(in crate::cdt) use mutable::{
    apply_prepared_delaunay_volume_node_insertion, insert_delaunay_volume_node_into_mutation,
    insert_delaunay_volume_node_into_mutation_with_barriers,
    insert_delaunay_volume_node_into_mutation_with_prepared_barriers_from_hint,
    legalize_inserted_node_and_check_star,
};
use topology_access::TetrahedralTopology;
#[cfg(test)]
pub(super) use validation::validate_canonical_conforming_volume_topology;
pub use validation::validate_delaunay_volume_topology;
pub(super) use validation::{
    validate_conforming_volume_topology, validate_constrained_delaunay_volume_topology,
    validate_locally_delaunay_volume_topology,
};
use work::InsertionWork;

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
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
    let rebuilt =
        insert_delaunay_volume_node_internal(topology, node, &[], false, options, cancellation)?;
    validate_delaunay_volume_topology(&rebuilt, options, cancellation)?;
    Ok(rebuilt)
}

/// Inserts into an already constrained, locally Delaunay topology and limits
/// legalization to the star changed by the Bowyer-Watson cavity. This is the
/// insertion path for candidate evaluation: callers independently validate the
/// resulting constrained topology before admitting it as an artifact.
pub(super) fn insert_delaunay_volume_node_locally_with_barriers(
    topology: DelaunayVolumeTopology,
    node: DelaunayVolumeNode,
    protected_faces: &[[runmat_meshing_core::StableDigest; 3]],
    options: DelaunayInsertionOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<DelaunayVolumeTopology, DelaunayInsertionError> {
    let mut mutation = TopologyMutation::from_canonical(topology);
    insert_delaunay_volume_node_into_mutation_with_barriers(
        &mut mutation,
        node,
        protected_faces,
        true,
        options,
        cancellation,
    )?;
    // This provider returns the canonical state and never uses its working
    // indexes again. Consume them before independent output reconstruction.
    mutation
        .freeze(options.topology, cancellation)
        .map_err(mutation_topology_error)
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
    legalize_delaunay_volume_topology(topology, &protected_faces, true, None, &mut work)
}

/// Lawson flips around each focus node's link only. After nodes are inserted into a
/// constrained-Delaunay complex, only faces of their stars can be illegal; callers validate the
/// result and fall back to the complete repair when local flips stall.
pub(super) fn legalize_delaunay_volume_topology_around_nodes_with_barriers(
    mut topology: DelaunayVolumeTopology,
    protected_faces: &[[runmat_meshing_core::StableDigest; 3]],
    focus: &[runmat_meshing_core::StableDigest],
    options: DelaunayInsertionOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<DelaunayVolumeTopology, DelaunayInsertionError> {
    validate_options(options)?;
    if protected_faces.len() as u64 > options.maximum_protected_faces {
        return Err(resource("protected-face inventory limit exceeded"));
    }
    let protected_faces = protected_faces.iter().copied().collect::<BTreeSet<_>>();
    let mut work = InsertionWork::new(options, cancellation);
    for identity in focus {
        topology = legalize_delaunay_volume_topology(
            topology,
            &protected_faces,
            true,
            Some(*identity),
            &mut work,
        )?;
    }
    Ok(topology)
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
    let prepared = prepare_insertion(
        &topology,
        node,
        &protected_faces,
        protect_region_boundaries,
        &mut work,
    )?;
    let mut mutation = TopologyMutation::from_canonical(topology);
    let inserted_index = mutation
        .append_node(prepared.node)
        .map_err(mutation_topology_error)?;
    let created = prepared
        .boundary
        .into_iter()
        .map(|face| {
            (
                [face[0], face[1], face[2], inserted_index],
                prepared.region_id.clone(),
            )
        })
        .collect();
    mutation
        .replace_cavity(&prepared.removed, created)
        .map_err(mutation_topology_error)?;
    let rebuilt = mutation
        .freeze(options.topology, cancellation)
        .map_err(mutation_topology_error)?;
    legalize_delaunay_volume_topology_complete(
        rebuilt,
        &protected_faces,
        protect_region_boundaries,
        &mut work,
    )
}

pub(in crate::cdt) struct PreparedInsertion {
    pub(in crate::cdt) node: DelaunayVolumeNode,
    pub(in crate::cdt) removed: BTreeSet<u32>,
    pub(in crate::cdt) boundary: Vec<[u32; 3]>,
    pub(in crate::cdt) region_id: Option<PersistentEntityId>,
}

pub(in crate::cdt) fn prepare_delaunay_volume_node_insertion_with_prepared_barriers_from_hint(
    topology: &impl TetrahedralTopology,
    node: DelaunayVolumeNode,
    protected_faces: &BTreeSet<[runmat_meshing_core::StableDigest; 3]>,
    protect_region_boundaries: bool,
    hint: Option<u32>,
    options: DelaunayInsertionOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<PreparedInsertion, DelaunayInsertionError> {
    validate_options(options)?;
    validate_new_node(topology, node)?;
    if protected_faces.len() as u64 > options.maximum_protected_faces {
        return Err(resource("protected-face inventory limit exceeded"));
    }
    let mut work = InsertionWork::new(options, cancellation);
    prepare_insertion_from_hint(
        topology,
        node,
        protected_faces,
        protect_region_boundaries,
        hint,
        &mut work,
    )
}

pub(in crate::cdt::insertion) fn prepare_insertion(
    topology: &impl TetrahedralTopology,
    node: DelaunayVolumeNode,
    protected_faces: &BTreeSet<[runmat_meshing_core::StableDigest; 3]>,
    protect_region_boundaries: bool,
    work: &mut InsertionWork<'_>,
) -> Result<PreparedInsertion, DelaunayInsertionError> {
    prepare_insertion_from_hint(
        topology,
        node,
        protected_faces,
        protect_region_boundaries,
        None,
        work,
    )
}

pub(in crate::cdt::insertion) fn prepare_insertion_from_hint(
    topology: &impl TetrahedralTopology,
    node: DelaunayVolumeNode,
    protected_faces: &BTreeSet<[runmat_meshing_core::StableDigest; 3]>,
    protect_region_boundaries: bool,
    hint: Option<u32>,
    work: &mut InsertionWork<'_>,
) -> Result<PreparedInsertion, DelaunayInsertionError> {
    let mut first_geometry_failure = None;
    for candidate in representable_nodes(node) {
        match prepare_insertion_at_coordinates(
            topology,
            candidate,
            protected_faces,
            protect_region_boundaries,
            hint,
            work,
        ) {
            Ok(prepared) => return Ok(prepared),
            Err(failure)
                if matches!(
                    failure.kind,
                    DelaunayInsertionErrorKind::InvalidNode
                        | DelaunayInsertionErrorKind::InvalidTopology
                        | DelaunayInsertionErrorKind::PointOutsideTopology
                ) =>
            {
                first_geometry_failure.get_or_insert(failure);
            }
            Err(failure) => return Err(failure),
        }
    }
    Err(first_geometry_failure.unwrap_or_else(|| {
        error(
            DelaunayInsertionErrorKind::InvalidNode,
            "constructed node has no admissible solver-coordinate materialization",
        )
    }))
}

fn prepare_insertion_at_coordinates(
    topology: &impl TetrahedralTopology,
    node: DelaunayVolumeNode,
    protected_faces: &BTreeSet<[runmat_meshing_core::StableDigest; 3]>,
    protect_region_boundaries: bool,
    hint: Option<u32>,
    work: &mut InsertionWork<'_>,
) -> Result<PreparedInsertion, DelaunayInsertionError> {
    let locator = TetrahedronPointLocator::walking(topology);
    let seed = locator
        .first_containing_from(node, hint.map(|index| index as usize), work)?
        .ok_or_else(|| {
            error(
                DelaunayInsertionErrorKind::PointOutsideTopology,
                format!(
                    "node {:?} is not contained by the admitted tetrahedron complex (nodes={}, slots={}, bounds={:?})",
                    node.coordinates_m.canonical_value(),
                    topology.nodes().len(),
                    topology.tetrahedron_slot_count(),
                    topology_bounds(topology),
                ),
            )
        })?;
    let mut cavity = connected_cavity(
        topology,
        node,
        seed,
        protected_faces,
        protect_region_boundaries,
        work,
    )?;
    restrict_cavity_to_protected_side(topology, &mut cavity, seed, node, protected_faces, work)?;
    regularize_cavity_boundary(topology, &mut cavity, node, work)?;
    if !cavity_is_star_shaped(topology, &cavity, node, work)? {
        return Err(error(
            DelaunayInsertionErrorKind::InvalidTopology,
            "validated insertion cavity is not star-shaped",
        ));
    }
    let boundary = cavity_boundary(topology, &cavity, node, work)?;
    let regions = cavity
        .iter()
        .map(|index| {
            topology
                .tetrahedron(*index)
                .expect("prepared cavity tetrahedron remains active")
                .region_id
                .clone()
        })
        .collect::<BTreeSet<_>>();
    if regions.len() != 1 {
        return Err(error(
            DelaunayInsertionErrorKind::InvalidTopology,
            "insertion cavity crosses assigned region incidence",
        ));
    }
    Ok(PreparedInsertion {
        node,
        removed: cavity,
        boundary,
        region_id: regions.into_iter().next().flatten(),
    })
}

pub(super) fn topology_bounds(topology: &impl TetrahedralTopology) -> [[f64; 3]; 2] {
    let mut minimum = [f64::INFINITY; 3];
    let mut maximum = [f64::NEG_INFINITY; 3];
    for node in topology.nodes() {
        for axis in 0..3 {
            minimum[axis] = minimum[axis].min(node.coordinates_m.canonical_value()[axis]);
            maximum[axis] = maximum[axis].max(node.coordinates_m.canonical_value()[axis]);
        }
    }
    [minimum, maximum]
}

fn cavity_boundary(
    topology: &impl TetrahedralTopology,
    cavity: &BTreeSet<u32>,
    node: DelaunayVolumeNode,
    work: &mut InsertionWork<'_>,
) -> Result<Vec<[u32; 3]>, DelaunayInsertionError> {
    let mut boundary = BTreeSet::new();
    for index in cavity {
        let tetrahedron = topology.tetrahedron(*index).ok_or_else(|| {
            error(
                DelaunayInsertionErrorKind::InvalidTopology,
                "cavity boundary references an inactive tetrahedron",
            )
        })?;
        for opposite in 0..4 {
            let neighbor = tetrahedron.neighbors[opposite];
            if neighbor.is_some_and(|neighbor| cavity.contains(&neighbor)) {
                continue;
            }
            // A point admitted on the convex hull replaces a coplanar boundary
            // face with the triangles induced by the adjacent non-coplanar
            // cavity faces. Coning the old face to the point would create an
            // exactly zero-volume tetrahedron. A merely coplanar face inside
            // the complex is still part of the cavity boundary and must not be
            // discarded: doing so would open a hole in the tetrahedralization.
            if neighbor.is_none()
                && node_lies_on_face(topology, tetrahedron.vertex_indices, opposite, node, work)?
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
    topology: &impl TetrahedralTopology,
    cavity: &mut BTreeSet<u32>,
    node: DelaunayVolumeNode,
    work: &mut InsertionWork<'_>,
) -> Result<(), DelaunayInsertionError> {
    loop {
        let mut cut = None;
        for index in cavity.iter().copied() {
            let tetrahedron = topology.tetrahedron(index).ok_or_else(|| {
                error(
                    DelaunayInsertionErrorKind::InvalidTopology,
                    "insertion cavity references an inactive tetrahedron",
                )
            })?;
            for opposite in 0..4 {
                if tetrahedron.neighbors[opposite]
                    .is_some_and(|neighbor| cavity.contains(&neighbor))
                {
                    continue;
                }
                if tetrahedron.neighbors[opposite].is_none()
                    && node_lies_on_face(
                        topology,
                        tetrahedron.vertex_indices,
                        opposite,
                        node,
                        work,
                    )?
                {
                    continue;
                }
                let mut face = [topology.nodes()[tetrahedron.vertex_indices[0] as usize]; 3];
                let mut cursor = 0;
                for (slot, vertex) in tetrahedron.vertex_indices.iter().enumerate() {
                    if slot != opposite {
                        face[cursor] = topology.nodes()[*vertex as usize];
                        cursor += 1;
                    }
                }
                let interior = topology.nodes()[tetrahedron.vertex_indices[opposite] as usize];
                work.predicate()?;
                let interior_sign = orient_volume_nodes([face[0], face[1], face[2], interior])
                    .map_err(predicate_error)?;
                work.predicate()?;
                let node_sign = orient_volume_nodes([face[0], face[1], face[2], node])
                    .map_err(predicate_error)?;
                if node_sign == PredicateSign::Zero || node_sign != interior_sign {
                    cut = Some(index);
                    break;
                }
            }
            if cut.is_some() {
                break;
            }
        }
        let Some(index) = cut else {
            break;
        };
        cavity.remove(&index);
        if cavity.is_empty() {
            return Err(error(
                DelaunayInsertionErrorKind::InvalidTopology,
                "star-shaped cavity validation removed every conflict tetrahedron",
            ));
        }
    }
    let boundary = cavity_boundary(topology, cavity, node, work)?;
    let mut edge_uses = std::collections::BTreeMap::<[u32; 2], usize>::new();
    for face in boundary {
        for mut edge in [[face[0], face[1]], [face[1], face[2]], [face[2], face[0]]] {
            edge.sort_unstable();
            *edge_uses.entry(edge).or_default() += 1;
        }
    }
    if let Some((edge, _)) = edge_uses.into_iter().find(|(_, uses)| *uses > 2) {
        return Err(error(
            DelaunayInsertionErrorKind::InvalidTopology,
            format!("insertion conflict cavity is non-manifold at edge {edge:?}"),
        ));
    }
    Ok(())
}

fn point_in_tetrahedron(
    topology: &impl TetrahedralTopology,
    vertices: [u32; 4],
    node: DelaunayVolumeNode,
    work: &mut InsertionWork<'_>,
) -> Result<bool, DelaunayInsertionError> {
    for replace in 0..4 {
        work.predicate()?;
        let mut points = vertices.map(|vertex| topology.nodes()[vertex as usize]);
        points[replace] = node;
        let sign = orient_volume_nodes(points).map_err(predicate_error)?;
        if !matches!(sign, PredicateSign::Positive | PredicateSign::Zero) {
            return Ok(false);
        }
    }
    Ok(true)
}

pub(super) fn two_to_three_topology_is_admissible(
    topology: &impl TetrahedralTopology,
    left_tetrahedron: u32,
    right_tetrahedron: u32,
    left_apex: u32,
    right_apex: u32,
) -> bool {
    !topology
        .vertex_star(left_apex)
        .into_iter()
        .any(|tetrahedron| {
            tetrahedron != left_tetrahedron
                && tetrahedron != right_tetrahedron
                && topology
                    .tetrahedron(tetrahedron)
                    .is_some_and(|tetrahedron| tetrahedron.vertex_indices.contains(&right_apex))
        })
}

fn validate_new_node(
    topology: &impl TetrahedralTopology,
    node: DelaunayVolumeNode,
) -> Result<(), DelaunayInsertionError> {
    validate_node_values(node)?;
    if topology.node_index(node.identity).is_some()
        || topology.contains_position(node.coordinates_m.canonical_value())
    {
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
            .canonical_value()
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

pub(in crate::cdt) fn mutation_topology_error(
    error_value: TopologyMutationError,
) -> DelaunayInsertionError {
    let kind = match error_value.kind {
        TopologyMutationErrorKind::InvalidTopology => DelaunayInsertionErrorKind::InvalidTopology,
        TopologyMutationErrorKind::ResourceLimit => DelaunayInsertionErrorKind::ResourceLimit,
        TopologyMutationErrorKind::Cancelled => DelaunayInsertionErrorKind::Cancelled,
    };
    error(kind, error_value.reason)
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
