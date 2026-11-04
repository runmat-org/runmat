use std::collections::BTreeSet;

use runmat_meshing_core::{
    quality::predicate::{
        points_collinear_constructed, projected_orient2d_constructed,
        triangle_projection_axes_constructed, PredicateSign,
    },
    StableDigest,
};

use super::{
    cavity::recovered_segments_exist, error, node_index, DelaunayFacetRecoveryError,
    DelaunayFacetRecoveryErrorKind, DelaunayRecoveredFacetTriangle, DelaunaySegmentRecovery,
    FacetRecoveryWork,
};
use crate::cdt::{
    insertion::{
        legalize_delaunay_volume_topology_with_barriers, legalize_inserted_node_and_check_star,
        topology_access::TetrahedralTopology, validate_constrained_delaunay_volume_topology,
    },
    mutation::{TopologyMutation, TopologyMutationErrorKind},
    segment_recovery::{record_recovered_segment_split, recovered_segment_midpoint},
    topology::{orient_materialized_volume_nodes, orient_volume_nodes},
    DelaunayInsertionErrorKind, DelaunayVolumeNode, DelaunayVolumeTopology,
};

pub(super) struct ProtectedFacetSplit {
    pub(super) source: DelaunayRecoveredFacetTriangle,
    pub(super) children: Vec<DelaunayRecoveredFacetTriangle>,
}

pub(super) struct ConformingInsertion {
    pub(super) recovery: DelaunaySegmentRecovery,
    pub(super) splits: Vec<ProtectedFacetSplit>,
}

pub(super) fn apply_protected_facet_splits(
    facets: &mut [super::DelaunayRecoveredFacet],
    protected: &mut Vec<DelaunayRecoveredFacetTriangle>,
    splits: Vec<ProtectedFacetSplit>,
) {
    for split in splits {
        for facet in facets.iter_mut() {
            if let Some(index) = facet
                .triangles
                .iter()
                .position(|triangle| *triangle == split.source)
            {
                facet
                    .triangles
                    .splice(index..=index, split.children.clone());
            }
        }
        if let Some(index) = protected
            .iter()
            .position(|triangle| *triangle == split.source)
        {
            protected.splice(index..=index, split.children);
        }
    }
}

enum PointLocation {
    Interior,
    Edge([StableDigest; 2]),
}

pub(super) fn try_insert_on_protected_facets(
    current: &DelaunaySegmentRecovery,
    node: DelaunayVolumeNode,
    protected: &[DelaunayRecoveredFacetTriangle],
    constraint_index: u32,
    work: &mut FacetRecoveryWork<'_>,
) -> Result<Option<ConformingInsertion>, DelaunayFacetRecoveryError> {
    let mut hits = Vec::new();
    for triangle in protected {
        work.support_step(constraint_index)?;
        if let Some(location) = point_location(current, node, *triangle, constraint_index)? {
            hits.push((*triangle, location));
        }
    }
    if hits.is_empty() {
        return Ok(None);
    }
    let edge = hits.iter().find_map(|(_, location)| match location {
        PointLocation::Interior => None,
        PointLocation::Edge(edge) => Some(*edge),
    });
    insert_conformingly(
        current,
        node,
        protected,
        edge,
        &hits,
        constraint_index,
        work,
    )
}

pub(super) fn try_split_recovered_segment_edge(
    current: &DelaunaySegmentRecovery,
    edge: [StableDigest; 2],
    protected: &[DelaunayRecoveredFacetTriangle],
    constraint_index: u32,
    work: &mut FacetRecoveryWork<'_>,
) -> Result<Option<ConformingInsertion>, DelaunayFacetRecoveryError> {
    work.support_step(constraint_index)?;
    let Some((segment_index, interval_index)) = recovered_segment_edge_location(current, edge)
    else {
        return Ok(None);
    };
    let (node, _) = recovered_segment_midpoint(current, segment_index, interval_index)
        .map_err(super::segment_error)?;
    if current.topology.nodes.iter().any(|existing| {
        crate::cdt::segment_recovery::coordinates_share_representable_location(
            existing.coordinates_m.canonical_value(),
            node.coordinates_m.canonical_value(),
        )
    }) {
        work.reject("conflicting segment midpoint duplicates an existing node position");
        return Ok(None);
    }
    insert_conformingly(
        current,
        node,
        protected,
        Some(edge),
        &[],
        constraint_index,
        work,
    )
}

/// How a recovered segment edge is split before legalization.
#[derive(Clone, Copy, PartialEq, Eq)]
enum SplitStrategy {
    /// The constrained Bowyer-Watson cavity of the new node, corrected to stay star-shaped
    /// from it; the edge star when the corrected cavity is not star-shaped.
    ConstrainedCavity,
    /// The tetrahedra around the edge only.
    EdgeStar,
}

/// Splits with the constrained cavity first. In thin layers between parallel interfaces,
/// splitting only the edge star can leave an illegal link face that no flip removes; the
/// cavity removes those cells before refilling. The edge-star split remains the fallback.
fn insert_conformingly(
    current: &DelaunaySegmentRecovery,
    node: DelaunayVolumeNode,
    protected: &[DelaunayRecoveredFacetTriangle],
    selected_edge: Option<[StableDigest; 2]>,
    hits: &[(DelaunayRecoveredFacetTriangle, PointLocation)],
    constraint_index: u32,
    work: &mut FacetRecoveryWork<'_>,
) -> Result<Option<ConformingInsertion>, DelaunayFacetRecoveryError> {
    if selected_edge.is_some() {
        if let Some(insertion) = insert_conformingly_with(
            current,
            node,
            protected,
            selected_edge,
            hits,
            SplitStrategy::ConstrainedCavity,
            constraint_index,
            work,
        )? {
            return Ok(Some(insertion));
        }
    }
    let cavity_rejection = work.last_rejection().map(str::to_owned);
    let insertion = insert_conformingly_with(
        current,
        node,
        protected,
        selected_edge,
        hits,
        SplitStrategy::EdgeStar,
        constraint_index,
        work,
    )?;
    if insertion.is_none() {
        if let (Some(cavity), Some(star)) = (cavity_rejection, work.last_rejection()) {
            let combined = format!("constrained cavity: {cavity}; edge star: {star}");
            work.reject(combined);
        }
    }
    Ok(insertion)
}

#[allow(clippy::too_many_arguments)]
fn insert_conformingly_with(
    current: &DelaunaySegmentRecovery,
    node: DelaunayVolumeNode,
    protected: &[DelaunayRecoveredFacetTriangle],
    selected_edge: Option<[StableDigest; 2]>,
    hits: &[(DelaunayRecoveredFacetTriangle, PointLocation)],
    strategy: SplitStrategy,
    constraint_index: u32,
    work: &mut FacetRecoveryWork<'_>,
) -> Result<Option<ConformingInsertion>, DelaunayFacetRecoveryError> {
    let mut mutation = TopologyMutation::from_canonical(current.topology.clone());
    let inserted = mutation.append_node(node).map_err(|failure| {
        error(
            DelaunayFacetRecoveryErrorKind::InvalidTopology,
            Some(constraint_index),
            failure.to_string(),
        )
    })?;
    let (removed, created, splits) = if let Some(edge) = selected_edge {
        split_edge(&mutation, inserted, edge, protected, strategy, constraint_index)?
    } else {
        split_faces(&mutation, inserted, hits, constraint_index)?
    };
    if removed.is_empty() {
        work.reject("the constrained cavity of the split node is not star-shaped from it");
        return Ok(None);
    }
    if let Some((tetrahedron, _)) = created.iter().find(|(tetrahedron, _)| {
        orient_materialized_volume_nodes(tetrahedron.map(|index| mutation.nodes[index as usize]))
            .is_ok_and(|sign| sign == PredicateSign::Zero)
    }) {
        work.reject(format!(
            "conforming facet {:?} split produced exact-zero tetrahedron {:?} at coordinates {:?}; inserted construction {:?}",
            selected_edge,
            tetrahedron,
            tetrahedron.map(|index| mutation.nodes[index as usize].coordinates_m.canonical_value()),
            node.construction,
        ));
        return Ok(None);
    }
    if let Err(failure) = mutation.replace_cavity(&removed, created) {
        work.reject(format!(
            "conforming facet cavity replacement was rejected: {failure}"
        ));
        return Ok(None);
    }
    let protected_faces = updated_protected_faces(protected, &splits);
    // Every created cell contains the inserted node, so link flips around it and an exact check
    // of its star restore and prove the constrained Delaunay property locally. When an illegal
    // face remains, the trial rolls back and the complete repair below runs instead.
    let locally_legal = {
        let protected_set = protected_faces.iter().copied().collect::<BTreeSet<_>>();
        let mut trial = mutation.cavity_trial();
        match legalize_inserted_node_and_check_star(
            &mut trial,
            inserted,
            &protected_set,
            work.options.segment_recovery.insertion,
            work.cancellation,
        ) {
            Ok(true) => {
                trial.commit();
                true
            }
            Ok(false) => false,
            Err(failure)
                if matches!(
                    failure.kind,
                    DelaunayInsertionErrorKind::ResourceLimit | DelaunayInsertionErrorKind::Cancelled
                ) =>
            {
                return Err(insertion_error(
                    failure.kind,
                    failure.to_string(),
                    constraint_index,
                ))
            }
            Err(_) => false,
        }
    };
    let topology = match mutation.freeze(
        work.options.segment_recovery.insertion.topology,
        work.cancellation,
    ) {
        Ok(topology) => topology,
        Err(failure) if failure.kind == TopologyMutationErrorKind::InvalidTopology => {
            work.reject(format!("conforming facet topology was rejected: {failure}"));
            return Ok(None);
        }
        Err(failure) => {
            return Err(error(
                match failure.kind {
                    TopologyMutationErrorKind::ResourceLimit => {
                        DelaunayFacetRecoveryErrorKind::ResourceLimit
                    }
                    TopologyMutationErrorKind::Cancelled => {
                        DelaunayFacetRecoveryErrorKind::Cancelled
                    }
                    TopologyMutationErrorKind::InvalidTopology => unreachable!(),
                },
                Some(constraint_index),
                failure.to_string(),
            ))
        }
    };
    if locally_legal {
        return finish_conforming_insertion(current, topology, selected_edge, splits, work);
    }
    let topology = match legalize_delaunay_volume_topology_with_barriers(
        topology,
        &protected_faces,
        work.options.segment_recovery.insertion,
        work.cancellation,
    ) {
        Ok(topology) => topology,
        Err(failure)
            if matches!(
                failure.kind,
                DelaunayInsertionErrorKind::InvalidTopology
                    | DelaunayInsertionErrorKind::InvalidNode
                    | DelaunayInsertionErrorKind::PointOutsideTopology
            ) =>
        {
            work.reject(format!(
                "conforming facet legalization was rejected: {failure}"
            ));
            return Ok(None);
        }
        Err(failure) => {
            return Err(insertion_error(
                failure.kind,
                failure.to_string(),
                constraint_index,
            ))
        }
    };
    if let Err(failure) = validate_constrained_delaunay_volume_topology(
        &topology,
        &protected_faces,
        work.options.segment_recovery.insertion,
        work.cancellation,
    ) {
        return match failure.kind {
            DelaunayInsertionErrorKind::ResourceLimit | DelaunayInsertionErrorKind::Cancelled => {
                Err(insertion_error(
                    failure.kind,
                    failure.to_string(),
                    constraint_index,
                ))
            }
            _ => {
                work.reject(format!(
                    "conforming facet validation was rejected: {failure}"
                ));
                Ok(None)
            }
        };
    }
    finish_conforming_insertion(current, topology, selected_edge, splits, work)
}

fn finish_conforming_insertion(
    current: &DelaunaySegmentRecovery,
    topology: DelaunayVolumeTopology,
    selected_edge: Option<[StableDigest; 2]>,
    splits: Vec<ProtectedFacetSplit>,
    work: &mut FacetRecoveryWork<'_>,
) -> Result<Option<ConformingInsertion>, DelaunayFacetRecoveryError> {
    let mut recovery = current.clone();
    recovery.topology = topology;
    if let Some(edge) = selected_edge {
        if let Some((segment_index, interval_index)) =
            recovered_segment_edge_location(current, edge)
        {
            let (_, recovered) = recovered_segment_midpoint(current, segment_index, interval_index)
                .map_err(super::segment_error)?;
            record_recovered_segment_split(&mut recovery, segment_index, interval_index, recovered)
                .map_err(super::segment_error)?;
        }
    }
    if !recovered_segments_exist(&recovery) {
        work.reject("conforming facet insertion removed a recovered segment edge");
        return Ok(None);
    }
    Ok(Some(ConformingInsertion { recovery, splits }))
}

fn recovered_segment_edge_location(
    recovery: &DelaunaySegmentRecovery,
    mut edge: [StableDigest; 2],
) -> Option<(usize, usize)> {
    edge.sort_unstable();
    recovery
        .segments
        .iter()
        .enumerate()
        .find_map(|(segment_index, segment)| {
            segment
                .nodes
                .windows(2)
                .enumerate()
                .find_map(|(interval_index, interval)| {
                    let mut candidate = [interval[0].identity, interval[1].identity];
                    candidate.sort_unstable();
                    (candidate == edge).then_some((segment_index, interval_index))
                })
        })
}

fn point_location(
    recovery: &DelaunaySegmentRecovery,
    node: DelaunayVolumeNode,
    triangle: DelaunayRecoveredFacetTriangle,
    constraint_index: u32,
) -> Result<Option<PointLocation>, DelaunayFacetRecoveryError> {
    let vertices = triangle.node_identities.map(|identity| {
        node_index(&recovery.topology, identity)
            .map(|index| recovery.topology.nodes[index])
            .ok_or_else(|| invalid(constraint_index, "protected facet node is missing"))
    });
    let [first, second, third] = vertices;
    let vertices = [first?, second?, third?];
    if orient_volume_nodes([vertices[0], vertices[1], vertices[2], node]).map_err(|failure| {
        invalid(
            constraint_index,
            format!("facet split predicate failed: {failure:?}"),
        )
    })? != PredicateSign::Zero
    {
        return Ok(None);
    }
    let points = vertices.map(DelaunayVolumeNode::predicate_point);
    let axes = triangle_projection_axes_constructed(points)
        .map_err(|failure| {
            invalid(
                constraint_index,
                format!("facet split projection failed: {failure:?}"),
            )
        })?
        .ok_or_else(|| invalid(constraint_index, "protected facet is collinear"))?;
    let query = node.predicate_point();
    let signs = [
        projected_orient2d_constructed([points[0], points[1], query], axes),
        projected_orient2d_constructed([points[1], points[2], query], axes),
        projected_orient2d_constructed([points[2], points[0], query], axes),
    ];
    if signs.iter().any(Result::is_err) {
        return Err(invalid(
            constraint_index,
            "facet split projection predicate failed",
        ));
    }
    let signs = signs.map(Result::unwrap);
    if signs.contains(&PredicateSign::Positive) && signs.contains(&PredicateSign::Negative) {
        return Ok(None);
    }
    if let Some(slot) = signs.iter().position(|sign| *sign == PredicateSign::Zero) {
        if !points_collinear_constructed([points[slot], points[(slot + 1) % 3], query]).map_err(
            |failure| {
                invalid(
                    constraint_index,
                    format!("facet edge classification failed: {failure:?}"),
                )
            },
        )? {
            return Ok(Some(PointLocation::Interior));
        }
        let mut edge = [
            triangle.node_identities[slot],
            triangle.node_identities[(slot + 1) % 3],
        ];
        edge.sort_unstable();
        return Ok(Some(PointLocation::Edge(edge)));
    }
    Ok(Some(PointLocation::Interior))
}

type Replacement = (
    BTreeSet<u32>,
    Vec<([u32; 4], Option<runmat_geometry_core::PersistentEntityId>)>,
    Vec<ProtectedFacetSplit>,
);

fn split_edge(
    topology: &TopologyMutation,
    inserted: u32,
    edge: [StableDigest; 2],
    protected: &[DelaunayRecoveredFacetTriangle],
    strategy: SplitStrategy,
    constraint_index: u32,
) -> Result<Replacement, DelaunayFacetRecoveryError> {
    let left = node_index_mutation(topology, edge[0])
        .ok_or_else(|| invalid(constraint_index, "facet split edge node is missing"))?;
    let right = node_index_mutation(topology, edge[1])
        .ok_or_else(|| invalid(constraint_index, "facet split edge node is missing"))?;
    let star = topology.vertex_stars[left as usize]
        .iter()
        .copied()
        .filter(|index| {
            topology
                .tetrahedron(*index)
                .is_some_and(|tet| tet.vertex_indices.contains(&right))
        })
        .collect::<BTreeSet<_>>();
    let mut cavity = constrained_cavity(
        topology,
        inserted,
        star.clone(),
        edge,
        protected,
        constraint_index,
    )?;
    let barriers = barrier_faces(protected, edge);
    // Opposite vertices of the protected triangles split at the edge: their edges to each
    // edge endpoint must stay on the cavity boundary to carry the split children.
    let split_apexes = protected
        .iter()
        .filter(|triangle| edge.iter().all(|node| triangle.node_identities.contains(node)))
        .filter_map(|triangle| {
            triangle
                .node_identities
                .into_iter()
                .find(|node| !edge.contains(node))
                .and_then(|apex| node_index_mutation(topology, apex))
        })
        .collect::<Vec<_>>();
    // Cavity correction: drop the tetrahedra that would hide a boundary face from the node,
    // remove a protected face from both sides, or surround a split child's edge.
    for _ in 0..64 {
        let mut dropped = BTreeSet::new();
        for index in &cavity {
            let tet = topology.tetrahedron(*index).expect("cavity tetrahedra remain active");
            for (slot, neighbor) in tet.neighbors.iter().enumerate() {
                let inside = neighbor.is_some_and(|neighbor| cavity.contains(&neighbor));
                if inside {
                    let neighbor = neighbor.expect("inside neighbor");
                    if barriers.contains(&canonical_mutation_face(topology, tet.vertex_indices, slot)) {
                        dropped.extend([*index, neighbor].into_iter().filter(|t| !star.contains(t)));
                    }
                    continue;
                }
                let face = (0..4)
                    .filter(|vertex| *vertex != slot)
                    .map(|vertex| tet.vertex_indices[vertex])
                    .collect::<Vec<_>>();
                if face.contains(&left) && face.contains(&right) {
                    continue;
                }
                let mut vertices = tet.vertex_indices;
                vertices[slot] = inserted;
                let visible = cone_is_visible(topology, vertices);
                if !visible && !star.contains(index) {
                    dropped.insert(*index);
                }
            }
        }
        for apex in &split_apexes {
            for endpoint in [left, right] {
                let around = topology.vertex_stars[endpoint as usize]
                    .iter()
                    .copied()
                    .filter(|index| {
                        topology
                            .tetrahedron(*index)
                            .is_some_and(|tet| tet.vertex_indices.contains(apex))
                    })
                    .collect::<Vec<_>>();
                if around.iter().all(|index| cavity.contains(index)) {
                    dropped.extend(around.into_iter().filter(|index| !star.contains(index)));
                }
            }
        }
        if dropped.is_empty() {
            break;
        }
        cavity.retain(|index| !dropped.contains(index));
        cavity = connected_from_star(topology, &cavity, &star, &barriers);
    }
    let fill = |removed: &BTreeSet<u32>| {
        let mut created = Vec::new();
        for index in removed {
            let tet = topology
                .tetrahedron(*index)
                .expect("cavity tetrahedra remain active");
            for (slot, neighbor) in tet.neighbors.iter().enumerate() {
                if neighbor.is_some_and(|neighbor| removed.contains(&neighbor)) {
                    continue;
                }
                let face = (0..4)
                    .filter(|vertex| *vertex != slot)
                    .map(|vertex| tet.vertex_indices[vertex])
                    .collect::<Vec<_>>();
                // The inserted node lies on the split edge, so a cavity face through that
                // edge is coplanar with it; the faces on either side of the node close it.
                if face.contains(&left) && face.contains(&right) {
                    continue;
                }
                let mut vertices = tet.vertex_indices;
                vertices[slot] = inserted;
                created.push((vertices, tet.region_id.clone()));
            }
        }
        created
    };
    // A cavity that is still not star-shaped from the node falls back to the edge star,
    // whose split is always valid; flips then legalize it.
    let star_shaped = |created: &Vec<([u32; 4], Option<runmat_geometry_core::PersistentEntityId>)>| {
        created
            .iter()
            .all(|(tetrahedron, _)| cone_is_visible(topology, *tetrahedron))
    };
    let created = fill(&cavity);
    let (removed, created) = if strategy == SplitStrategy::ConstrainedCavity
        && cavity.len() > star.len()
        && star_shaped(&created)
    {
        (cavity, created)
    } else if strategy == SplitStrategy::ConstrainedCavity {
        // The edge-star strategy performs this split; an empty replacement declines it here.
        (BTreeSet::new(), Vec::new())
    } else {
        let created = fill(&star);
        (star, created)
    };
    let splits = protected
        .iter()
        .filter(|triangle| {
            edge.iter()
                .all(|node| triangle.node_identities.contains(node))
        })
        .map(|triangle| {
            let slot = (0..3)
                .find(|slot| {
                    let candidate = [
                        triangle.node_identities[*slot],
                        triangle.node_identities[(*slot + 1) % 3],
                    ];
                    edge.contains(&candidate[0]) && edge.contains(&candidate[1])
                })
                .expect("protected edge belongs to triangle");
            let a = triangle.node_identities[slot];
            let b = triangle.node_identities[(slot + 1) % 3];
            let c = triangle.node_identities[(slot + 2) % 3];
            ProtectedFacetSplit {
                source: *triangle,
                children: vec![
                    DelaunayRecoveredFacetTriangle {
                        node_identities: [a, topology.nodes[inserted as usize].identity, c],
                    },
                    DelaunayRecoveredFacetTriangle {
                        node_identities: [topology.nodes[inserted as usize].identity, b, c],
                    },
                ],
            }
        })
        .collect();
    Ok((removed, created, splits))
}

/// The Bowyer-Watson cavity of a node inserted on a recovered segment edge: the edge star plus
/// every tetrahedron reachable from it whose circumsphere strictly contains the node. Protected
/// faces other than the triangles split at the edge, and region interfaces, are barriers.
fn constrained_cavity(
    topology: &TopologyMutation,
    inserted: u32,
    star: BTreeSet<u32>,
    edge: [StableDigest; 2],
    protected: &[DelaunayRecoveredFacetTriangle],
    constraint_index: u32,
) -> Result<BTreeSet<u32>, DelaunayFacetRecoveryError> {
    const MAXIMUM_CAVITY_TETRAHEDRA: usize = 1 << 16;
    let barriers = barrier_faces(protected, edge);
    let node = topology.nodes[inserted as usize];
    let mut cavity = star.clone();
    let mut frontier = star.into_iter().collect::<Vec<_>>();
    while let Some(index) = frontier.pop() {
        let tet = topology
            .tetrahedron(index)
            .expect("cavity tetrahedra remain active");
        for (slot, neighbor) in tet.neighbors.iter().enumerate() {
            let Some(neighbor) = *neighbor else { continue };
            if cavity.contains(&neighbor) {
                continue;
            }
            let other = topology
                .tetrahedron(neighbor)
                .expect("neighbor tetrahedra remain active");
            if other.region_id != tet.region_id {
                continue;
            }
            if barriers.contains(&canonical_mutation_face(topology, tet.vertex_indices, slot)) {
                continue;
            }
            let points = other.vertex_indices.map(|vertex| topology.nodes[vertex as usize]);
            let inside = crate::cdt::topology::insphere_materialized_volume_nodes([
                points[0], points[1], points[2], points[3], node,
            ])
            .map_err(|failure| {
                invalid(
                    constraint_index,
                    format!("conforming segment cavity predicate failed: {failure:?}"),
                )
            })? == PredicateSign::Positive;
            if inside {
                if cavity.len() >= MAXIMUM_CAVITY_TETRAHEDRA {
                    return Err(error(
                        DelaunayFacetRecoveryErrorKind::ResourceLimit,
                        Some(constraint_index),
                        "conforming segment cavity exceeded its tetrahedron limit",
                    ));
                }
                cavity.insert(neighbor);
                frontier.push(neighbor);
            }
        }
    }
    Ok(cavity)
}

/// A cavity face is visible from the inserted node when the cone over it is positively
/// oriented and not flat: a node constructed on a facet plane sits off that plane only by
/// rounding, and a cone over another triangle of the same facet would be a zero-volume sliver.
fn cone_is_visible(topology: &TopologyMutation, vertices: [u32; 4]) -> bool {
    const MINIMUM_RELATIVE_VOLUME: f64 = 1.0e-10;
    let nodes = vertices.map(|vertex| topology.nodes[vertex as usize]);
    if !orient_materialized_volume_nodes(nodes).is_ok_and(|sign| sign == PredicateSign::Positive) {
        return false;
    }
    let points = nodes.map(|node| node.coordinates_m.canonical_value());
    let edge = |a: usize, b: usize| -> [f64; 3] {
        std::array::from_fn(|axis| points[b][axis] - points[a][axis])
    };
    let (u, v, w) = (edge(0, 1), edge(0, 2), edge(0, 3));
    let volume = (u[0] * (v[1] * w[2] - v[2] * w[1]) - u[1] * (v[0] * w[2] - v[2] * w[0])
        + u[2] * (v[0] * w[1] - v[1] * w[0]))
        .abs();
    let longest = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        .into_iter()
        .map(|(a, b)| edge(a, b).iter().map(|value| value * value).sum::<f64>().sqrt())
        .fold(0.0, f64::max);
    volume >= MINIMUM_RELATIVE_VOLUME * longest.powi(3)
}

fn barrier_faces(
    protected: &[DelaunayRecoveredFacetTriangle],
    edge: [StableDigest; 2],
) -> BTreeSet<[StableDigest; 3]> {
    protected
        .iter()
        .filter(|triangle| !edge.iter().all(|node| triangle.node_identities.contains(node)))
        .map(|triangle| {
            let mut face = triangle.node_identities;
            face.sort_unstable();
            face
        })
        .collect()
}

fn canonical_mutation_face(
    topology: &TopologyMutation,
    vertices: [u32; 4],
    opposite: usize,
) -> [StableDigest; 3] {
    let mut face = [StableDigest::ZERO; 3];
    for (cursor, vertex) in (0..4).filter(|vertex| *vertex != opposite).enumerate() {
        face[cursor] = topology.nodes[vertices[vertex] as usize].identity;
    }
    face.sort_unstable();
    face
}

/// The cavity tetrahedra reachable from the edge star across non-barrier faces.
fn connected_from_star(
    topology: &TopologyMutation,
    cavity: &BTreeSet<u32>,
    star: &BTreeSet<u32>,
    barriers: &BTreeSet<[StableDigest; 3]>,
) -> BTreeSet<u32> {
    let mut connected = star.clone();
    let mut frontier = star.iter().copied().collect::<Vec<_>>();
    while let Some(index) = frontier.pop() {
        let tet = topology.tetrahedron(index).expect("cavity tetrahedra remain active");
        for (slot, neighbor) in tet.neighbors.iter().enumerate() {
            let Some(neighbor) = *neighbor else { continue };
            if !cavity.contains(&neighbor)
                || connected.contains(&neighbor)
                || barriers.contains(&canonical_mutation_face(topology, tet.vertex_indices, slot))
            {
                continue;
            }
            connected.insert(neighbor);
            frontier.push(neighbor);
        }
    }
    connected
}

fn split_faces(
    topology: &TopologyMutation,
    inserted: u32,
    hits: &[(DelaunayRecoveredFacetTriangle, PointLocation)],
    constraint_index: u32,
) -> Result<Replacement, DelaunayFacetRecoveryError> {
    let mut removed = BTreeSet::new();
    let mut created = Vec::new();
    let mut splits = Vec::new();
    for (triangle, _) in hits {
        let face = triangle.node_identities.map(|identity| {
            node_index_mutation(topology, identity)
                .ok_or_else(|| invalid(constraint_index, "facet split node is missing"))
        });
        let [a, b, c] = face;
        let face = [a?, b?, c?];
        for index in 0..topology.tetrahedron_slot_count() {
            let Some(tet) = topology.tetrahedron(index as u32) else {
                continue;
            };
            if !face.iter().all(|node| tet.vertex_indices.contains(node)) {
                continue;
            }
            removed.insert(index as u32);
            let apex = *tet
                .vertex_indices
                .iter()
                .find(|node| !face.contains(node))
                .expect("face tet has apex");
            created.extend([
                ([face[0], face[1], inserted, apex], tet.region_id.clone()),
                ([face[1], face[2], inserted, apex], tet.region_id.clone()),
                ([face[2], face[0], inserted, apex], tet.region_id.clone()),
            ]);
        }
        let identity = topology.nodes[inserted as usize].identity;
        splits.push(ProtectedFacetSplit {
            source: *triangle,
            children: vec![
                DelaunayRecoveredFacetTriangle {
                    node_identities: [
                        triangle.node_identities[0],
                        triangle.node_identities[1],
                        identity,
                    ],
                },
                DelaunayRecoveredFacetTriangle {
                    node_identities: [
                        triangle.node_identities[1],
                        triangle.node_identities[2],
                        identity,
                    ],
                },
                DelaunayRecoveredFacetTriangle {
                    node_identities: [
                        triangle.node_identities[2],
                        triangle.node_identities[0],
                        identity,
                    ],
                },
            ],
        });
    }
    Ok((removed, created, splits))
}

fn updated_protected_faces(
    protected: &[DelaunayRecoveredFacetTriangle],
    splits: &[ProtectedFacetSplit],
) -> Vec<[StableDigest; 3]> {
    protected
        .iter()
        .filter(|triangle| !splits.iter().any(|split| split.source == **triangle))
        .flat_map(|triangle| vec![*triangle])
        .chain(splits.iter().flat_map(|split| split.children.clone()))
        .map(|triangle| {
            let mut face = triangle.node_identities;
            face.sort_unstable();
            face
        })
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn node_index_mutation(topology: &TopologyMutation, identity: StableDigest) -> Option<u32> {
    topology
        .nodes
        .iter()
        .position(|node| node.identity == identity)
        .map(|index| index as u32)
}

fn insertion_error(
    kind: DelaunayInsertionErrorKind,
    reason: String,
    constraint_index: u32,
) -> DelaunayFacetRecoveryError {
    let kind = match kind {
        DelaunayInsertionErrorKind::ResourceLimit => DelaunayFacetRecoveryErrorKind::ResourceLimit,
        DelaunayInsertionErrorKind::Cancelled => DelaunayFacetRecoveryErrorKind::Cancelled,
        _ => DelaunayFacetRecoveryErrorKind::InvalidTopology,
    };
    error(kind, Some(constraint_index), reason)
}

fn invalid(constraint_index: u32, reason: impl Into<String>) -> DelaunayFacetRecoveryError {
    error(
        DelaunayFacetRecoveryErrorKind::InvalidTopology,
        Some(constraint_index),
        reason,
    )
}
