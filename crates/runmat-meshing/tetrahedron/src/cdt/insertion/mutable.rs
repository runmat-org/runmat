use std::collections::BTreeSet;

use runmat_meshing_core::{
    quality::predicate::PredicateSign, MeshingCancellationSignal, StableDigest,
};

use super::{
    cavity::{circumsphere_relation, stable_face},
    error, mutation_topology_error, prepare_insertion_from_hint,
    two_to_three_topology_is_admissible, validate_new_node, validate_options,
    DelaunayInsertionError, DelaunayInsertionErrorKind, DelaunayInsertionOptions,
    DelaunayVolumeNode, InsertionWork, PreparedInsertion, TopologyMutation,
};
use crate::cdt::topology::orient_materialized_volume_nodes as orient_volume_nodes;
use crate::reconnect::two_to_three_geometry_is_admissible;

pub(in crate::cdt) fn insert_delaunay_volume_node_into_mutation(
    topology: &mut TopologyMutation,
    node: DelaunayVolumeNode,
    options: DelaunayInsertionOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<(), DelaunayInsertionError> {
    insert_delaunay_volume_node_into_mutation_with_barriers(
        topology,
        node,
        &[],
        false,
        options,
        cancellation,
    )
}

pub(in crate::cdt) fn insert_delaunay_volume_node_into_mutation_with_barriers(
    topology: &mut TopologyMutation,
    node: DelaunayVolumeNode,
    protected_faces: &[[StableDigest; 3]],
    protect_region_boundaries: bool,
    options: DelaunayInsertionOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<(), DelaunayInsertionError> {
    insert_delaunay_volume_node_into_mutation_with_barriers_from_hint(
        topology,
        node,
        protected_faces,
        protect_region_boundaries,
        None,
        options,
        cancellation,
    )
}

pub(in crate::cdt) fn insert_delaunay_volume_node_into_mutation_with_barriers_from_hint(
    topology: &mut TopologyMutation,
    node: DelaunayVolumeNode,
    protected_faces: &[[StableDigest; 3]],
    protect_region_boundaries: bool,
    hint: Option<u32>,
    options: DelaunayInsertionOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<(), DelaunayInsertionError> {
    validate_options(options)?;
    validate_new_node(topology, node)?;
    if protected_faces.len() as u64 > options.maximum_protected_faces {
        return Err(super::resource("protected-face inventory limit exceeded"));
    }
    let protected_faces = protected_faces.iter().copied().collect::<BTreeSet<_>>();
    insert_delaunay_volume_node_into_mutation_with_prepared_barriers_from_hint(
        topology,
        node,
        &protected_faces,
        protect_region_boundaries,
        hint,
        options,
        cancellation,
    )
}

pub(in crate::cdt) fn insert_delaunay_volume_node_into_mutation_with_prepared_barriers_from_hint(
    topology: &mut TopologyMutation,
    node: DelaunayVolumeNode,
    protected_faces: &BTreeSet<[StableDigest; 3]>,
    protect_region_boundaries: bool,
    hint: Option<u32>,
    options: DelaunayInsertionOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<(), DelaunayInsertionError> {
    validate_options(options)?;
    validate_new_node(topology, node)?;
    if protected_faces.len() as u64 > options.maximum_protected_faces {
        return Err(super::resource("protected-face inventory limit exceeded"));
    }
    let mut work = InsertionWork::new(options, cancellation);
    let prepared = prepare_insertion_from_hint(
        topology,
        node,
        protected_faces,
        protect_region_boundaries,
        hint,
        &mut work,
    )
    .map_err(|mut failure| {
        failure.reason = format!(
            "{}; insertion preparation coordinates={:?}, nodes={}, work={:?}",
            failure.reason,
            node.coordinates_m.canonical_value(),
            topology.nodes.len(),
            work.inventory()
        );
        failure
    })?;
    apply_prepared_delaunay_volume_node_insertion_with_work(
        topology,
        prepared,
        protected_faces,
        protect_region_boundaries,
        &mut work,
    )
    .map_err(|mut failure| {
        failure.reason = format!(
            "{}; insertion application coordinates={:?}, nodes={}, work={:?}",
            failure.reason,
            node.coordinates_m.canonical_value(),
            topology.nodes.len(),
            work.inventory()
        );
        failure
    })
}

pub(in crate::cdt) fn apply_prepared_delaunay_volume_node_insertion(
    topology: &mut TopologyMutation,
    prepared: PreparedInsertion,
    protected_faces: &BTreeSet<[StableDigest; 3]>,
    protect_region_boundaries: bool,
    options: DelaunayInsertionOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<(), DelaunayInsertionError> {
    validate_options(options)?;
    validate_new_node(topology, prepared.node)?;
    if protected_faces.len() as u64 > options.maximum_protected_faces {
        return Err(super::resource("protected-face inventory limit exceeded"));
    }
    let mut work = InsertionWork::new(options, cancellation);
    apply_prepared_delaunay_volume_node_insertion_with_work(
        topology,
        prepared,
        protected_faces,
        protect_region_boundaries,
        &mut work,
    )
}

pub(super) fn apply_prepared_delaunay_volume_node_insertion_with_work(
    topology: &mut TopologyMutation,
    prepared: PreparedInsertion,
    protected_faces: &BTreeSet<[StableDigest; 3]>,
    protect_region_boundaries: bool,
    work: &mut InsertionWork<'_>,
) -> Result<(), DelaunayInsertionError> {
    require_mutation_capacity(
        topology,
        prepared.removed.len(),
        prepared.boundary.len(),
        work.options,
    )?;
    let inserted = topology
        .append_node(prepared.node)
        .map_err(mutation_topology_error)?;
    let created = prepared
        .boundary
        .into_iter()
        .map(|face| {
            (
                [face[0], face[1], face[2], inserted],
                prepared.region_id.clone(),
            )
        })
        .collect();
    topology
        .replace_cavity(&prepared.removed, created)
        .map_err(mutation_topology_error)?;
    legalize_inserted_node(
        topology,
        inserted,
        &mut Protection::new(protected_faces),
        protect_region_boundaries,
        LinkFlips::TwoToThree,
        work,
    )?;
    // A construction-verified physical prefix must carry forward the changed
    // neighborhood, rather than accumulating every earlier insertion until a
    // later recovery operation asks for admission. The shared repair owner
    // discharges only exactly verified faces. An unrepairable neighborhood
    // retains its dirty inventory for the independent complete fallback.
    if protected_faces.is_empty() && topology.physical_dirty_faces().is_some() {
        super::legalization::mutable_repair::restore_admitted(topology, work)?;
    }
    Ok(())
}

pub(super) fn require_mutation_capacity(
    topology: &TopologyMutation,
    removed: usize,
    created: usize,
    options: DelaunayInsertionOptions,
) -> Result<(), DelaunayInsertionError> {
    if topology.nodes.len() as u64 >= options.topology.maximum_nodes
        || topology.nodes.len() >= u32::MAX as usize
    {
        return Err(super::resource(
            "mutable insertion node inventory limit exceeded",
        ));
    }
    let result = topology
        .active_tetrahedron_count()
        .checked_sub(removed)
        .and_then(|count| count.checked_add(created))
        .ok_or_else(|| super::resource("mutable insertion tetrahedron count overflow"))?;
    if result as u64 > options.topology.maximum_tetrahedra || result > u32::MAX as usize {
        return Err(super::resource(
            "mutable insertion tetrahedron inventory limit exceeded",
        ));
    }
    Ok(())
}

/// Restores the constrained Delaunay property after `inserted` was placed by a local split of a
/// constrained-Delaunay complex whose created cells all contain it. Link flips run on the
/// working topology; afterwards every face of the inserted node's star is checked with the
/// validator's exact strict empty-sphere test. Every cell the split or a link flip created
/// contains `inserted`, so faces outside that star are unchanged and stay legal. Returns
/// `false` when an illegal face remains (the caller falls back to complete legalization).
pub(in crate::cdt) fn legalize_inserted_node_and_check_star(
    topology: &mut TopologyMutation,
    inserted: u32,
    protected_faces: &BTreeSet<[StableDigest; 3]>,
    kept_edges: &BTreeSet<[StableDigest; 2]>,
    options: DelaunayInsertionOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<bool, DelaunayInsertionError> {
    legalize_inserted_node_and_check_star_with_facet_flips(
        topology,
        inserted,
        protected_faces,
        kept_edges,
        false,
        options,
        cancellation,
    )
    .map(|(legal, _)| legal)
}

/// [`legalize_inserted_node_and_check_star`]; with `facet_flips`, a 4-4 flip may also replace
/// two coplanar protected triangles that share a non-kept edge (see [`FacetFlip`]). The star
/// check then uses the flipped protection, and the flips are returned for the caller's records.
pub(in crate::cdt) fn legalize_inserted_node_and_check_star_with_facet_flips(
    topology: &mut TopologyMutation,
    inserted: u32,
    protected_faces: &BTreeSet<[StableDigest; 3]>,
    kept_edges: &BTreeSet<[StableDigest; 2]>,
    facet_flips: bool,
    options: DelaunayInsertionOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<(bool, Vec<FacetFlip>), DelaunayInsertionError> {
    validate_options(options)?;
    let mut work = InsertionWork::new(options, cancellation);
    let mut protection = Protection::new(protected_faces);
    legalize_inserted_node(
        topology,
        inserted,
        &mut protection,
        true,
        LinkFlips::Complete {
            kept_edges,
            facet_flips,
        },
        &mut work,
    )?;
    let legal = star_is_legal(topology, inserted, &protection, &mut work)?;
    Ok((legal, protection.flips))
}

fn star_is_legal(
    topology: &TopologyMutation,
    inserted: u32,
    protection: &Protection<'_>,
    work: &mut InsertionWork<'_>,
) -> Result<bool, DelaunayInsertionError> {
    use super::topology_access::TetrahedralTopology;
    for left_index in topology.vertex_stars[inserted as usize].iter().copied() {
        let left = topology
            .tetrahedron(left_index)
            .expect("inserted-node star references an active tetrahedron");
        for (opposite, neighbor) in left.neighbors.iter().enumerate() {
            let Some(right_index) = *neighbor else {
                continue;
            };
            work.checkpoint()?;
            let right = topology.tetrahedron(right_index).ok_or_else(|| {
                error(
                    DelaunayInsertionErrorKind::InvalidTopology,
                    "star check neighbor is inactive",
                )
            })?;
            if left.region_id != right.region_id
                || protection.contains(&stable_face(topology, left.vertex_indices, opposite))
            {
                continue;
            }
            let Some(apex) = right
                .vertex_indices
                .iter()
                .copied()
                .find(|vertex| !left.vertex_indices.contains(vertex))
            else {
                return Ok(false);
            };
            work.predicate()?;
            let points = left
                .vertex_indices
                .map(|vertex| topology.nodes[vertex as usize]);
            if topology
                .insphere_nodes([
                    points[0],
                    points[1],
                    points[2],
                    points[3],
                    topology.nodes[apex as usize],
                ])
                .map_err(super::predicate_error)?
                == PredicateSign::Positive
            {
                return Ok(false);
            }
        }
    }
    Ok(true)
}

/// Lawson flips (2-3, 3-2 and 4-4) in the link of each listed node, in order, on the working
/// topology. No flip removes an edge in `kept_edges` (sorted node identities).
pub(in crate::cdt) fn legalize_inserted_nodes(
    topology: &mut TopologyMutation,
    inserted: &[u32],
    protected_faces: &BTreeSet<[StableDigest; 3]>,
    kept_edges: &BTreeSet<[StableDigest; 2]>,
    options: DelaunayInsertionOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<(), DelaunayInsertionError> {
    legalize_inserted_nodes_with_facet_flips(
        topology,
        inserted,
        protected_faces,
        kept_edges,
        false,
        options,
        cancellation,
    )
    .map(|_| ())
}

/// [`legalize_inserted_nodes`]; with `facet_flips`, a 4-4 flip may also replace two coplanar
/// protected triangles that share a non-kept edge. Returns those flips in order.
pub(in crate::cdt) fn legalize_inserted_nodes_with_facet_flips(
    topology: &mut TopologyMutation,
    inserted: &[u32],
    protected_faces: &BTreeSet<[StableDigest; 3]>,
    kept_edges: &BTreeSet<[StableDigest; 2]>,
    facet_flips: bool,
    options: DelaunayInsertionOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<Vec<FacetFlip>, DelaunayInsertionError> {
    validate_options(options)?;
    let mut work = InsertionWork::new(options, cancellation);
    let mut protection = Protection::new(protected_faces);
    for node in inserted {
        legalize_inserted_node(
            topology,
            *node,
            &mut protection,
            true,
            LinkFlips::Complete {
                kept_edges,
                facet_flips,
            },
            &mut work,
        )?;
    }
    Ok(protection.flips)
}

/// A 2-2 or 4-4 flip inside a facet plane: the coplanar protected triangles `removed` (sharing the
/// flipped edge) became `created` (sharing the new edge), as sorted node identities.
/// `off_plane` is a node of a removed cell off that plane, for orienting the new triangles.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::cdt) struct FacetFlip {
    pub(in crate::cdt) removed: [[StableDigest; 3]; 2],
    pub(in crate::cdt) created: [[StableDigest; 3]; 2],
    pub(in crate::cdt) off_plane: StableDigest,
}

/// Protected faces as the caller gave them, with the facet flips made so far applied.
struct Protection<'a> {
    base: &'a BTreeSet<[StableDigest; 3]>,
    removed: BTreeSet<[StableDigest; 3]>,
    added: BTreeSet<[StableDigest; 3]>,
    flips: Vec<FacetFlip>,
}

impl<'a> Protection<'a> {
    fn new(base: &'a BTreeSet<[StableDigest; 3]>) -> Self {
        Self {
            base,
            removed: BTreeSet::new(),
            added: BTreeSet::new(),
            flips: Vec::new(),
        }
    }

    fn contains(&self, face: &[StableDigest; 3]) -> bool {
        self.added.contains(face) || self.base.contains(face) && !self.removed.contains(face)
    }

    fn apply(&mut self, flip: FacetFlip) {
        for face in flip.removed {
            if !self.added.remove(&face) {
                self.removed.insert(face);
            }
        }
        for face in flip.created {
            if !self.removed.remove(&face) {
                self.added.insert(face);
            }
        }
        self.flips.push(flip);
    }
}

/// Admits a local edit of a constrained-Delaunay complex with the validator's exact rules,
/// without a whole-mesh pass. A face that no created cell owns is a face of two unchanged
/// cells, so it kept its legality; checking every face of every created cell therefore covers
/// each face the edit can affect. A protected face can only disappear with a removed cell, so
/// the protected faces of removed cells must still exist. Returns the first violation.
pub(in crate::cdt) fn observed_change_violation(
    topology: &TopologyMutation,
    changes: &crate::cdt::mutation::ObservedCellChanges,
    protected_faces: &BTreeSet<[StableDigest; 3]>,
    options: DelaunayInsertionOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<Option<String>, DelaunayInsertionError> {
    use super::topology_access::TetrahedralTopology;
    validate_options(options)?;
    let mut work = InsertionWork::new(options, cancellation);
    for corners in &changes.removed {
        for opposite in 0..4 {
            work.checkpoint()?;
            if !protected_faces.contains(&stable_face(topology, *corners, opposite)) {
                continue;
            }
            let mut face = [0_u32; 3];
            for (cursor, slot) in (0..4).filter(|slot| *slot != opposite).enumerate() {
                face[cursor] = corners[slot];
            }
            if topology.face_incidence(face).is_empty() {
                return Ok(Some(format!(
                    "the edit removed protected face {:?}",
                    face.map(|node| topology.nodes[node as usize]
                        .coordinates_m
                        .canonical_value())
                )));
            }
        }
    }
    for left_index in &changes.created {
        let left = topology
            .tetrahedron(*left_index)
            .expect("created cells are active");
        for (opposite, neighbor) in left.neighbors.iter().enumerate() {
            let Some(right_index) = *neighbor else {
                continue;
            };
            work.checkpoint()?;
            let right = topology.tetrahedron(right_index).ok_or_else(|| {
                error(
                    DelaunayInsertionErrorKind::InvalidTopology,
                    "local admission neighbor is inactive",
                )
            })?;
            let face = stable_face(topology, left.vertex_indices, opposite);
            if protected_faces.contains(&face) {
                continue;
            }
            if left.region_id != right.region_id {
                return Ok(Some(
                    "assigned region boundary is missing protected facet provenance".to_owned(),
                ));
            }
            let Some(apex) = right
                .vertex_indices
                .iter()
                .copied()
                .find(|vertex| !left.vertex_indices.contains(vertex))
            else {
                return Ok(Some(
                    "neighbors do not share exactly one triangular face".to_owned(),
                ));
            };
            work.predicate()?;
            let points = left
                .vertex_indices
                .map(|vertex| topology.nodes[vertex as usize]);
            if topology
                .insphere_nodes([
                    points[0],
                    points[1],
                    points[2],
                    points[3],
                    topology.nodes[apex as usize],
                ])
                .map_err(super::predicate_error)?
                == PredicateSign::Positive
            {
                return Ok(Some(format!(
                    "tetrahedron {:?} has neighbor apex {:?} strictly inside its circumsphere",
                    points.map(|node| node.coordinates_m.canonical_value()),
                    topology.nodes[apex as usize]
                        .coordinates_m
                        .canonical_value(),
                )));
            }
        }
    }
    Ok(None)
}

/// Which link flips restore the Delaunay property around an inserted node.
#[derive(Clone, Copy)]
pub(in crate::cdt) enum LinkFlips<'a> {
    /// 2-3 flips only. A Bowyer-Watson cavity leaves its link faces legal except across
    /// barriers, where no flip may cross, so this suffices after a cavity insertion.
    TwoToThree,
    /// 2-3, 3-2 and 4-4 flips (Joe's flip set), for a node placed by a local split whose link
    /// can be non-convex at an illegal face. A 3-2 or 4-4 flip removes the edge of the link
    /// face that the inserted node's segment to the opposite apex passes; it never removes a
    /// listed edge (sorted node identities).
    Complete {
        kept_edges: &'a BTreeSet<[StableDigest; 2]>,
        /// Allow the 2-2 and 4-4 flips that replace the two coplanar protected triangles on the
        /// flipped edge (their only protected faces); the caller then owns the reported
        /// [`FacetFlip`]s.
        facet_flips: bool,
    },
}

fn legalize_inserted_node(
    topology: &mut TopologyMutation,
    inserted: u32,
    protected_faces: &mut Protection<'_>,
    protect_region_boundaries: bool,
    flips: LinkFlips<'_>,
    work: &mut InsertionWork<'_>,
) -> Result<(), DelaunayInsertionError> {
    let mut flip_count = 0_u64;
    while let Some(flip) = first_illegal_inserted_face(
        topology,
        inserted,
        &*protected_faces,
        protect_region_boundaries,
        flips,
        work,
    )? {
        flip_count = flip_count.checked_add(1).ok_or_else(|| {
            error(
                DelaunayInsertionErrorKind::ResourceLimit,
                "mutable legalization flip count overflow",
            )
        })?;
        if flip_count > work.options.maximum_cavity_tetrahedra {
            return Err(error(
                DelaunayInsertionErrorKind::ResourceLimit,
                "mutable legalization flip limit exceeded",
            ));
        }
        if topology.active_tetrahedron_count() as u64 >= work.options.topology.maximum_tetrahedra {
            return Err(super::resource(
                "mutable legalization tetrahedron inventory limit exceeded",
            ));
        }
        match flip {
            LinkFlip::TwoToThree(flip) => {
                apply_two_to_three_flip(topology, flip).map_err(mutation_topology_error)?
            }
            LinkFlip::Edge {
                removed,
                created,
                facet,
            } => {
                topology
                    .replace_cavity(&removed, created)
                    .map_err(mutation_topology_error)?;
                if let Some(facet) = facet {
                    protected_faces.apply(facet);
                }
            }
        }
    }
    Ok(())
}

struct TwoToThreeFlip {
    left: u32,
    right: u32,
    face: [u32; 3],
}

enum LinkFlip {
    TwoToThree(TwoToThreeFlip),
    /// A 2-2, 3-2 or 4-4 flip that removes one edge of the illegal link face. Each created
    /// cell carries its region.
    Edge {
        removed: BTreeSet<u32>,
        created: Vec<([u32; 4], Option<runmat_geometry_core::PersistentEntityId>)>,
        facet: Option<FacetFlip>,
    },
}

fn first_illegal_inserted_face(
    topology: &TopologyMutation,
    inserted: u32,
    protected_faces: &Protection<'_>,
    protect_region_boundaries: bool,
    flips: LinkFlips<'_>,
    work: &mut InsertionWork<'_>,
) -> Result<Option<LinkFlip>, DelaunayInsertionError> {
    let mut edge_flip = None;
    for left_index in topology.vertex_stars[inserted as usize].iter().copied() {
        let left = topology
            .tetrahedron(left_index)
            .expect("inserted-node star references an active tetrahedron");
        let opposite = left
            .vertex_indices
            .iter()
            .position(|vertex| *vertex == inserted)
            .expect("inserted-node star tetrahedron contains the inserted node");
        let Some(right_index) = left.neighbors[opposite] else {
            continue;
        };
        work.checkpoint()?;
        let right = topology.tetrahedron(right_index).ok_or_else(|| {
            error(
                DelaunayInsertionErrorKind::InvalidTopology,
                "mutable legalization neighbor is inactive",
            )
        })?;
        let stable = stable_face(topology, left.vertex_indices, opposite);
        if protected_faces.contains(&stable)
            || protect_region_boundaries && left.region_id != right.region_id
        {
            continue;
        }
        let right_apex = right
            .vertex_indices
            .iter()
            .copied()
            .find(|vertex| !left.vertex_indices.contains(vertex))
            .ok_or_else(|| {
                error(
                    DelaunayInsertionErrorKind::InvalidTopology,
                    "mutable legalization neighbors do not share one face",
                )
            })?;
        let mut face = [0; 3];
        let mut cursor = 0;
        for vertex in left.vertex_indices {
            if right.vertex_indices.contains(&vertex) {
                face[cursor] = vertex;
                cursor += 1;
            }
        }
        face.sort_unstable();
        if circumsphere_relation(
            topology,
            left.vertex_indices,
            topology.nodes[right_apex as usize],
            work,
        )? != PredicateSign::Positive
        {
            continue;
        }
        if two_to_three_topology_is_admissible(
            topology,
            left_index,
            right_index,
            inserted,
            right_apex,
        ) && two_to_three_geometry_is_admissible(
            left.vertex_indices,
            right.vertex_indices,
            |node| {
                let node = topology.nodes[node as usize];
                DelaunayVolumeNode::explicit(node.identity, node.coordinates_m.canonical_value())
                    .predicate_point()
            },
        )
        .map_err(super::predicate_error)?
            && created_tetrahedra_are_exactly_nondegenerate(topology, inserted, right_apex, face)?
        {
            return Ok(Some(LinkFlip::TwoToThree(TwoToThreeFlip {
                left: left_index,
                right: right_index,
                face,
            })));
        }
        // Prefer a 2-3 flip anywhere in the link; remember the first edge flip.
        if edge_flip.is_none() {
            if let LinkFlips::Complete {
                kept_edges,
                facet_flips,
            } = flips
            {
                edge_flip = edge_link_flip(
                    topology,
                    [left_index, right_index],
                    [inserted, right_apex],
                    face,
                    protected_faces,
                    kept_edges,
                    facet_flips,
                    work,
                )?;
            }
        }
    }
    if edge_flip.is_some() {
        return Ok(edge_flip);
    }
    let LinkFlips::Complete {
        kept_edges,
        facet_flips: true,
    } = flips
    else {
        return Ok(None);
    };
    // A protected facet triangle t with a coplanar neighbor apex inside its circumcircle makes
    // every cell on t illegal, because each sphere through t meets the plane in that circle.
    // When the inserted node is off the plane, the illegal face between its cells on t and on
    // the neighbor triangle runs through the inserted node, so no link flip reaches it. Only a
    // planar flip on such a face (the 2D Delaunay flip of the two triangles, protected as a facet
    // flip or unprotected on the hull) repairs it.
    use super::topology_access::TetrahedralTopology;
    for left_index in topology.vertex_stars[inserted as usize].iter().copied() {
        let left = topology
            .tetrahedron(left_index)
            .expect("inserted-node star references an active tetrahedron");
        for (opposite, neighbor) in left.neighbors.iter().enumerate() {
            let left_apex = left.vertex_indices[opposite];
            let Some(right_index) = *neighbor else {
                continue;
            };
            if left_apex == inserted {
                continue;
            }
            work.checkpoint()?;
            let right = topology.tetrahedron(right_index).ok_or_else(|| {
                error(
                    DelaunayInsertionErrorKind::InvalidTopology,
                    "mutable legalization neighbor is inactive",
                )
            })?;
            if left.region_id != right.region_id
                || protected_faces.contains(&stable_face(topology, left.vertex_indices, opposite))
            {
                continue;
            }
            let Some(right_apex) = right
                .vertex_indices
                .iter()
                .copied()
                .find(|vertex| !left.vertex_indices.contains(vertex))
            else {
                continue;
            };
            work.predicate()?;
            let points = left
                .vertex_indices
                .map(|vertex| topology.nodes[vertex as usize]);
            if topology
                .insphere_nodes([
                    points[0],
                    points[1],
                    points[2],
                    points[3],
                    topology.nodes[right_apex as usize],
                ])
                .map_err(super::predicate_error)?
                != PredicateSign::Positive
            {
                continue;
            }
            let mut face = [0; 3];
            let mut cursor = 0;
            for (slot, vertex) in left.vertex_indices.iter().enumerate() {
                if slot != opposite {
                    face[cursor] = *vertex;
                    cursor += 1;
                }
            }
            face.sort_unstable();
            // A non-cavity split (an edge-star split or a pair split) can leave an illegal face
            // through the inserted node whose 2-3 flip is admissible, for example a face in a
            // facet plane not yet recovered with apexes on both sides of it.
            if two_to_three_topology_is_admissible(
                topology,
                left_index,
                right_index,
                left_apex,
                right_apex,
            ) && two_to_three_geometry_is_admissible(
                left.vertex_indices,
                right.vertex_indices,
                |node| {
                    let node = topology.nodes[node as usize];
                    DelaunayVolumeNode::explicit(
                        node.identity,
                        node.coordinates_m.canonical_value(),
                    )
                    .predicate_point()
                },
            )
            .map_err(super::predicate_error)?
                && created_tetrahedra_are_exactly_nondegenerate(
                    topology, left_apex, right_apex, face,
                )?
            {
                return Ok(Some(LinkFlip::TwoToThree(TwoToThreeFlip {
                    left: left_index,
                    right: right_index,
                    face,
                })));
            }
            // Otherwise only planar flips: a facet flip, or a 2-2 flip of two unprotected hull
            // triangles.
            if let Some(flip @ LinkFlip::Edge { .. }) = edge_link_flip(
                topology,
                [left_index, right_index],
                [left_apex, right_apex],
                face,
                protected_faces,
                kept_edges,
                true,
                work,
            )? {
                if let LinkFlip::Edge { removed, facet, .. } = &flip {
                    if facet.is_some() || removed.len() == 2 {
                        return Ok(Some(flip));
                    }
                }
            }
        }
    }
    Ok(None)
}

/// The 2-2, 3-2 or 4-4 flip that removes an edge `xy` of the illegal link face `xyz` between
/// `cells = [(p,x,y,z), (x,y,z,d)]`, where `ends = [p, d]`:
/// - 2-2 (facet flips only): `xy` has only the cells `pxyz` and `xyzd`, so `xyp` and `xyd`
///   are boundary faces; both are protected, `p`, `d`, `x`, `y` are coplanar and `pd` crosses
///   `xy`; the cells become `pdxz` and `pdzy`.
/// - 3-2: `xy` has exactly the three cells `pxyz`, `xyzd`, `xypd`, and `xy` crosses the
///   interior of triangle `pzd`; the cells become `xpzd` and `ypzd`.
/// - 4-4: `p`, `d`, `x`, `y` are coplanar, `pd` and `xy` cross at interior points of both, and
///   `xy` has exactly the four cells `pxyz`, `xyzd`, `xyde`, `xyep`; the cells become
///   `pdxz`, `pdzy`, `pdye`, `pdex`.
///
/// Each flip retriangulates the same polytope, so the change is local. Every predicate is
/// exact. Edges in `kept_edges` and protected faces among the removed cells' shared faces
/// block the flip, except the protected pair `xyp`, `xyd` of a facet flip. The cells must share
/// one region, except that the cells `xyde`, `xyep` of a 4-4 facet flip may form a second
/// region across the facet plane; each created cell keeps the region of its side.
#[allow(clippy::too_many_arguments)]
fn edge_link_flip(
    topology: &TopologyMutation,
    cells: [u32; 2],
    ends: [u32; 2],
    face: [u32; 3],
    protected_faces: &Protection<'_>,
    kept_edges: &BTreeSet<[StableDigest; 2]>,
    facet_flips: bool,
    work: &mut InsertionWork<'_>,
) -> Result<Option<LinkFlip>, DelaunayInsertionError> {
    let [p, d] = ends;
    let region = topology
        .tetrahedron(cells[0])
        .expect("link flip cell is active")
        .region_id
        .clone();
    let orient = |vertices: [u32; 4]| {
        orient_volume_nodes(vertices.map(|vertex| topology.nodes[vertex as usize]))
            .map_err(super::predicate_error)
    };
    for (x, y, z) in [
        (face[0], face[1], face[2]),
        (face[1], face[2], face[0]),
        (face[0], face[2], face[1]),
    ] {
        work.checkpoint()?;
        let mut identities = [
            topology.nodes[x as usize].identity,
            topology.nodes[y as usize].identity,
        ];
        identities.sort_unstable();
        if kept_edges.contains(&identities) {
            continue;
        }
        let star = topology.vertex_stars[x as usize]
            .iter()
            .copied()
            .filter(|index| {
                topology
                    .tetrahedron(*index)
                    .is_some_and(|tetrahedron| tetrahedron.vertex_indices.contains(&y))
            })
            .collect::<BTreeSet<_>>();
        if !(2..=4).contains(&star.len()) || !cells.iter().all(|cell| star.contains(cell)) {
            continue;
        }
        // Every face shared by two cells of the edge star contains xy. A 3-2 flip needs one
        // region and no protected shared face; the 2-2 and 4-4 facet cases below admit exactly
        // the two coplanar protected triangles on xy, which may separate two regions.
        let region_of = |index: u32| {
            topology
                .tetrahedron(index)
                .expect("edge star cell is active")
                .region_id
                .clone()
        };
        let mut ring = BTreeSet::new();
        let mut protected_shared = BTreeSet::new();
        for index in &star {
            let tetrahedron = topology
                .tetrahedron(*index)
                .expect("edge star cell is active");
            for vertex in tetrahedron.vertex_indices {
                if vertex != x && vertex != y {
                    ring.insert(vertex);
                }
            }
            for (slot, neighbor) in tetrahedron.neighbors.iter().enumerate() {
                let shared = stable_face(topology, tetrahedron.vertex_indices, slot);
                if neighbor.is_some_and(|neighbor| star.contains(&neighbor))
                    && protected_faces.contains(&shared)
                {
                    protected_shared.insert(shared);
                }
            }
        }
        if cells.iter().any(|cell| region_of(*cell) != region) {
            continue;
        }
        let stable = |vertices: [u32; 3]| {
            let mut identities = vertices.map(|vertex| topology.nodes[vertex as usize].identity);
            identities.sort_unstable();
            identities
        };
        let facet_record = || FacetFlip {
            removed: [stable([x, y, p]), stable([x, y, d])],
            created: [stable([p, d, x]), stable([p, d, y])],
            off_plane: topology.nodes[z as usize].identity,
        };
        let opposite = |first: PredicateSign, second: PredicateSign| {
            first != PredicateSign::Zero && second != PredicateSign::Zero && first != second
        };
        // The new edge pd must not exist outside the removed cells.
        let pd_outside_star = || {
            topology.vertex_stars[p as usize].iter().any(|index| {
                !star.contains(index)
                    && topology
                        .tetrahedron(*index)
                        .is_some_and(|tetrahedron| tetrahedron.vertex_indices.contains(&d))
            })
        };
        work.predicate()?;
        if star.len() == 2 {
            // 2-2 on a boundary facet: xy has only the cells pxyz and xyzd, so x-y-p and x-y-d
            // are boundary faces. With p, d, x, y coplanar and pd crossing xy, the cells become
            // pdxz and pdzy over the same polytope, and the two boundary triangles become
            // p-d-x and p-d-y. Either both boundary triangles are protected (a facet flip, only
            // with `facet_flips`) or neither is (an ordinary hull flip).
            let boundary_protection =
                [stable([x, y, p]), stable([x, y, d])].map(|face| protected_faces.contains(&face));
            let facet = match boundary_protection {
                [true, true] if facet_flips => Some(facet_record()),
                [false, false] => None,
                _ => continue,
            };
            if !protected_shared.is_empty() {
                continue;
            }
            if orient([p, d, x, y])? != PredicateSign::Zero
                || !opposite(orient([p, d, z, x])?, orient([p, d, z, y])?)
                || !opposite(orient([x, y, z, p])?, orient([x, y, z, d])?)
                || pd_outside_star()
            {
                continue;
            }
            return Ok(Some(LinkFlip::Edge {
                removed: star,
                created: vec![([p, d, x, z], region.clone()), ([p, d, z, y], region)],
                facet,
            }));
        }
        if ring.len() != star.len() {
            continue;
        }
        if star.len() == 3 {
            if !protected_shared.is_empty() || star.iter().any(|cell| region_of(*cell) != region) {
                continue;
            }
            if ring != BTreeSet::from([p, z, d]) {
                continue;
            }
            let created = [[x, p, z, d], [y, p, z, d]];
            if !two_to_three_geometry_is_admissible(created[0], created[1], |vertex| {
                topology.nodes[vertex as usize].materialized_predicate_point()
            })
            .map_err(super::predicate_error)?
            {
                continue;
            }
            return Ok(Some(LinkFlip::Edge {
                removed: star,
                created: created
                    .into_iter()
                    .map(|cell| (cell, region.clone()))
                    .collect(),
                facet: None,
            }));
        }
        // 4-4: the fourth ring vertex e closes the ring p, z, d, e around xy.
        let Some(e) = ring
            .iter()
            .copied()
            .find(|vertex| ![p, z, d].contains(vertex))
        else {
            continue;
        };
        if !ring.contains(&p) || !ring.contains(&d) || ring.len() != 4 {
            continue;
        }
        let cell_with = |vertices: [u32; 4]| {
            star.iter().copied().find(|index| {
                let corners = topology
                    .tetrahedron(*index)
                    .expect("edge star cell is active")
                    .vertex_indices;
                vertices.iter().all(|vertex| corners.contains(vertex))
            })
        };
        let (Some(xyde), Some(xyep)) = (cell_with([x, y, d, e]), cell_with([x, y, e, p])) else {
            continue;
        };
        if orient([p, d, x, y])? != PredicateSign::Zero {
            continue;
        }
        if !opposite(orient([p, d, z, x])?, orient([p, d, z, y])?)
            || !opposite(orient([x, y, z, p])?, orient([x, y, z, d])?)
            || !opposite(orient([p, d, x, z])?, orient([p, d, x, e])?)
        {
            continue;
        }
        if pd_outside_star() {
            continue;
        }
        // A 4-4 flip may replace the two coplanar protected triangles x-y-p and x-y-d of one
        // facet by p-d-x and p-d-y: the same planar region, now triangulated through p-d. Any
        // other protected shared face blocks it. The cells on the e side of the plane may form
        // a second region only when that plane is the protected facet; each side keeps its
        // region.
        let far_region = region_of(xyde);
        if region_of(xyep) != far_region {
            continue;
        }
        let facet = if protected_shared.is_empty() {
            None
        } else if facet_flips
            && protected_shared == BTreeSet::from([stable([x, y, p]), stable([x, y, d])])
        {
            Some(facet_record())
        } else {
            continue;
        };
        if far_region != region && facet.is_none() {
            continue;
        }
        return Ok(Some(LinkFlip::Edge {
            removed: star,
            created: vec![
                ([p, d, x, z], region.clone()),
                ([p, d, z, y], region),
                ([p, d, y, e], far_region.clone()),
                ([p, d, e, x], far_region),
            ],
            facet,
        }));
    }
    Ok(None)
}

fn created_tetrahedra_are_exactly_nondegenerate(
    topology: &TopologyMutation,
    left_apex: u32,
    right_apex: u32,
    face: [u32; 3],
) -> Result<bool, DelaunayInsertionError> {
    for edge in [[face[0], face[1]], [face[1], face[2]], [face[2], face[0]]] {
        let sign = orient_volume_nodes([
            topology.nodes[left_apex as usize],
            topology.nodes[right_apex as usize],
            topology.nodes[edge[0] as usize],
            topology.nodes[edge[1] as usize],
        ])
        .map_err(super::predicate_error)?;
        if sign == PredicateSign::Zero {
            return Ok(false);
        }
    }
    Ok(true)
}

fn apply_two_to_three_flip(
    topology: &mut TopologyMutation,
    flip: TwoToThreeFlip,
) -> Result<(), super::TopologyMutationError> {
    let left = topology
        .tetrahedron(flip.left)
        .expect("legalization left tetrahedron remains active")
        .clone();
    let right = topology
        .tetrahedron(flip.right)
        .expect("legalization right tetrahedron remains active")
        .clone();
    if left.region_id != right.region_id {
        return Err(crate::cdt::mutation::TopologyMutationError {
            kind: crate::cdt::mutation::TopologyMutationErrorKind::InvalidTopology,
            reason: "mutable legalization cannot cross a region boundary".to_owned(),
        });
    }
    let left_apex = left
        .vertex_indices
        .iter()
        .copied()
        .find(|vertex| !flip.face.contains(vertex))
        .expect("legalization face leaves one left apex");
    let right_apex = right
        .vertex_indices
        .iter()
        .copied()
        .find(|vertex| !flip.face.contains(vertex))
        .expect("legalization face leaves one right apex");
    let removed = [flip.left, flip.right].into_iter().collect();
    let created = [
        [left_apex, right_apex, flip.face[0], flip.face[1]],
        [left_apex, right_apex, flip.face[1], flip.face[2]],
        [left_apex, right_apex, flip.face[2], flip.face[0]],
    ]
    .into_iter()
    .map(|tetrahedron| (tetrahedron, left.region_id.clone()))
    .collect();
    topology.replace_cavity(&removed, created)?;
    Ok(())
}

#[cfg(test)]
mod tests;
