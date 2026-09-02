use std::collections::{BTreeMap, BTreeSet, VecDeque};

use runmat_meshing_core::{quality::predicate::PredicateSign, StableDigest};

use super::{
    cavity::{circumsphere_relation, stable_face},
    error, topology_error, two_to_three_topology_is_admissible, DelaunayInsertionError,
    DelaunayInsertionErrorKind, DelaunayVolumeNode, DelaunayVolumeTopology, InsertionWork,
};
use crate::{
    cdt::topology::{
        build_delaunay_volume_topology_with_regions, orient_materialized_volume_nodes,
    },
    reconnect::{
        local_tetrahedron_boundary_faces, three_to_two_edge_flip_candidate,
        two_to_three_geometry_is_admissible, LocalTetrahedron,
    },
};

use crate::cdt::topology::insphere_materialized_volume_nodes as insphere_volume_nodes;
use crate::cdt::topology::orient_materialized_volume_nodes as orient_volume_nodes;

type CavityReplacement = (Vec<usize>, Vec<[u32; 4]>);
type ScoredCavityReplacement = (Vec<usize>, Vec<[u32; 4]>, u64, u64);

pub(super) fn legalize_delaunay_volume_topology(
    mut topology: DelaunayVolumeTopology,
    protected_faces: &BTreeSet<[StableDigest; 3]>,
    protect_region_boundaries: bool,
    focus_identity: Option<StableDigest>,
    work: &mut InsertionWork<'_>,
) -> Result<DelaunayVolumeTopology, DelaunayInsertionError> {
    let mut flip_count = 0_u64;
    let focus = focus_identity
        .map(|identity| {
            topology
                .nodes
                .binary_search_by_key(&identity, |node| node.identity)
                .map(|index| index as u32)
                .map_err(|_| {
                    error(
                        DelaunayInsertionErrorKind::InvalidTopology,
                        "local legalization focus node is absent",
                    )
                })
        })
        .transpose()?;
    loop {
        let Some((left, right, face)) = first_illegal_face(
            &topology,
            protected_faces,
            protect_region_boundaries,
            focus,
            work,
        )?
        else {
            return Ok(topology);
        };
        record_flip(&mut flip_count, work)?;
        topology = apply_two_to_three_flip(topology, left, right, face, work)?;
    }
}

pub(super) fn legalize_delaunay_volume_topology_complete(
    mut topology: DelaunayVolumeTopology,
    protected_faces: &BTreeSet<[StableDigest; 3]>,
    protect_region_boundaries: bool,
    work: &mut InsertionWork<'_>,
) -> Result<DelaunayVolumeTopology, DelaunayInsertionError> {
    let mut flip_count = 0_u64;
    let mut visited_global_states = BTreeSet::new();
    let globally_unconstrained =
        protected_faces.is_empty() && topology.incidence.regions.is_empty();
    loop {
        visited_global_states.insert(topology_key(&topology));
        if let Some((removed, created)) =
            first_illegal_boundary_edge(&topology, protected_faces, work)?
        {
            record_flip(&mut flip_count, work)?;
            topology = apply_reconnection(topology, &removed, created, work)?;
            continue;
        }
        if globally_unconstrained {
            if let Some((removed, created)) = first_global_vertex_reinsertion(
                &topology,
                protected_faces,
                &visited_global_states,
                work,
            )? {
                record_flip(&mut flip_count, work)?;
                topology = apply_reconnection(topology, &removed, created, work)?;
                continue;
            }
            if let Some((removed, created)) = first_globally_delaunay_edge_rotation(
                &topology,
                protected_faces,
                &visited_global_states,
                work,
            )? {
                record_flip(&mut flip_count, work)?;
                topology = apply_reconnection(topology, &removed, created, work)?;
                continue;
            }
        }
        if let Some((removed, created)) =
            first_reducible_edge(&topology, protected_faces, protect_region_boundaries, work)?
        {
            record_flip(&mut flip_count, work)?;
            topology = apply_reconnection(topology, &removed, created, work)?;
            continue;
        }
        let Some((left, right, face)) = first_illegal_face(
            &topology,
            protected_faces,
            protect_region_boundaries,
            None,
            work,
        )?
        else {
            return Ok(topology);
        };
        record_flip(&mut flip_count, work)?;
        topology = apply_two_to_three_flip(topology, left, right, face, work)?;
    }
}

fn first_global_vertex_reinsertion(
    topology: &DelaunayVolumeTopology,
    protected_faces: &BTreeSet<[StableDigest; 3]>,
    visited_states: &BTreeSet<Vec<[u32; 4]>>,
    work: &mut InsertionWork<'_>,
) -> Result<Option<CavityReplacement>, DelaunayInsertionError> {
    let mut plateau_candidate = None;
    for query in global_delaunay_violation_queries(topology, work)? {
        let Some((removed, created, removed_violations, created_violations)) =
            vertex_reinsertion_candidate(topology, query, protected_faces, work)?
        else {
            continue;
        };
        if created_violations < removed_violations {
            return Ok(Some((removed, created)));
        }
        if created_violations == removed_violations
            && plateau_candidate.is_none()
            && !visited_states.contains(&replacement_topology_key(topology, &removed, &created))
        {
            plateau_candidate = Some((removed, created));
        }
    }
    Ok(plateau_candidate)
}

fn vertex_reinsertion_candidate(
    topology: &DelaunayVolumeTopology,
    query: usize,
    protected_faces: &BTreeSet<[StableDigest; 3]>,
    work: &mut InsertionWork<'_>,
) -> Result<Option<ScoredCavityReplacement>, DelaunayInsertionError> {
    let mut removed = topology.incidence.vertex_stars[query]
        .iter()
        .map(|index| *index as usize)
        .collect::<BTreeSet<_>>();
    if removed.is_empty() {
        return Ok(None);
    }
    let query_node = topology.nodes[query];
    let mut queue = removed
        .iter()
        .flat_map(|index| topology.tetrahedra[*index].neighbors)
        .flatten()
        .map(|index| index as usize)
        .collect::<VecDeque<_>>();
    while let Some(index) = queue.pop_front() {
        if removed.contains(&index) {
            continue;
        }
        let tetrahedron = &topology.tetrahedra[index];
        work.predicate()?;
        let points = tetrahedron
            .vertex_indices
            .map(|node| topology.nodes[node as usize]);
        if matches!(
            insphere_volume_nodes([points[0], points[1], points[2], points[3], query_node])
                .map_err(super::predicate_error)?,
            PredicateSign::Positive | PredicateSign::Zero
        ) {
            removed.insert(index);
            queue.extend(
                tetrahedron
                    .neighbors
                    .iter()
                    .flatten()
                    .map(|index| *index as usize),
            );
        }
    }
    shrink_reinsertion_cavity(topology, &mut removed, query as u32, work)?;
    let removed = removed.into_iter().collect::<Vec<_>>();
    if removed_internal_face_is_protected(topology, &removed, protected_faces) {
        return Ok(None);
    }
    let region = &topology.tetrahedra[removed[0]].region_id;
    if removed
        .iter()
        .any(|index| &topology.tetrahedra[*index].region_id != region)
    {
        return Ok(None);
    }
    let removed_tetrahedra = removed
        .iter()
        .map(|index| topology.tetrahedra[*index].vertex_indices)
        .collect::<Vec<_>>();
    let boundary = local_tetrahedron_boundary_faces(&removed_tetrahedra);
    let mut created = Vec::new();
    for face in boundary
        .iter()
        .filter(|face| !face.contains(&(query as u32)))
    {
        let tetrahedron = [face[0], face[1], face[2], query as u32];
        if orient_volume_nodes(tetrahedron.map(|node| topology.nodes[node as usize]))
            .map_err(super::predicate_error)?
            != PredicateSign::Zero
        {
            created.push(tetrahedron);
        }
    }
    created.sort_unstable();
    created.dedup();
    let boundary_matches = local_tetrahedron_boundary_faces(&created) == boundary;
    let duplicates = replacement_duplicates_retained_tetrahedron(topology, &removed, &created);
    let embedded = prospective_tetrahedra_form_embedded_manifold(topology, &removed, &created)?;
    if created.is_empty()
        || !boundary_matches
        || duplicates
        || !embedded
        || !tetrahedra_are_materializable(topology, &created)?
    {
        return Ok(None);
    }
    let removed_violations = indexed_tetrahedra_global_violation_count(topology, &removed, work)?;
    let created_violations = tetrahedra_global_violation_count(topology, &created, work)?;
    if created_violations > removed_violations {
        return Ok(None);
    }
    Ok(Some((
        removed,
        created,
        removed_violations,
        created_violations,
    )))
}

fn shrink_reinsertion_cavity(
    topology: &DelaunayVolumeTopology,
    cavity: &mut BTreeSet<usize>,
    query: u32,
    work: &mut InsertionWork<'_>,
) -> Result<(), DelaunayInsertionError> {
    loop {
        let mut cut = None;
        for index in cavity.iter().copied() {
            let tetrahedron = &topology.tetrahedra[index];
            for opposite in 0..4 {
                if tetrahedron.neighbors[opposite]
                    .is_some_and(|neighbor| cavity.contains(&(neighbor as usize)))
                {
                    continue;
                }
                let mut face = [0; 3];
                let mut cursor = 0;
                for (slot, vertex) in tetrahedron.vertex_indices.iter().enumerate() {
                    if slot != opposite {
                        face[cursor] = *vertex;
                        cursor += 1;
                    }
                }
                if face.contains(&query) {
                    continue;
                }
                let face_nodes = face.map(|node| topology.nodes[node as usize]);
                let interior = topology.nodes[tetrahedron.vertex_indices[opposite] as usize];
                work.predicate()?;
                let interior_sign =
                    orient_volume_nodes([face_nodes[0], face_nodes[1], face_nodes[2], interior])
                        .map_err(super::predicate_error)?;
                work.predicate()?;
                let query_sign = orient_volume_nodes([
                    face_nodes[0],
                    face_nodes[1],
                    face_nodes[2],
                    topology.nodes[query as usize],
                ])
                .map_err(super::predicate_error)?;
                if query_sign == PredicateSign::Zero || query_sign != interior_sign {
                    cut = Some(index);
                    break;
                }
            }
            if cut.is_some() {
                break;
            }
        }
        let Some(index) = cut else {
            return Ok(());
        };
        cavity.remove(&index);
        if cavity.is_empty() {
            return Err(error(
                DelaunayInsertionErrorKind::InvalidTopology,
                "global vertex reinsertion cavity validation removed every tetrahedron",
            ));
        }
    }
}

fn first_globally_delaunay_edge_rotation(
    topology: &DelaunayVolumeTopology,
    protected_faces: &BTreeSet<[StableDigest; 3]>,
    visited_states: &BTreeSet<Vec<[u32; 4]>>,
    work: &mut InsertionWork<'_>,
) -> Result<Option<CavityReplacement>, DelaunayInsertionError> {
    let Some((violating_tetrahedron, query_node)) =
        first_global_delaunay_violation(topology, work)?
    else {
        return Ok(None);
    };
    let violating_vertices = topology.tetrahedra[violating_tetrahedron].vertex_indices;
    let mut candidate_edges = BTreeSet::new();
    for tetrahedron_index in std::iter::once(violating_tetrahedron).chain(
        topology.incidence.vertex_stars[query_node]
            .iter()
            .map(|index| *index as usize),
    ) {
        let vertices = topology.tetrahedra[tetrahedron_index].vertex_indices;
        for left in 0..4 {
            for right in (left + 1)..4 {
                let mut edge = [vertices[left], vertices[right]];
                edge.sort_unstable();
                candidate_edges.insert(edge);
            }
        }
    }
    let mut plateau_candidate = None;
    for removed_edge in candidate_edges {
        let star = topology.incidence.vertex_stars[removed_edge[0] as usize]
            .iter()
            .copied()
            .map(|index| index as usize)
            .filter(|index| {
                topology.tetrahedra[*index]
                    .vertex_indices
                    .contains(&removed_edge[1])
            })
            .collect::<Vec<_>>();
        if star.len() < 3 {
            continue;
        }
        let removed_violations = indexed_tetrahedra_global_violation_count(topology, &star, work)?;
        if removed_violations == 0
            && !removed_edge
                .iter()
                .any(|vertex| violating_vertices.contains(vertex))
        {
            continue;
        }
        let region = &topology.tetrahedra[star[0]].region_id;
        if star
            .iter()
            .any(|index| &topology.tetrahedra[*index].region_id != region)
            || removed_internal_face_is_protected(topology, &star, protected_faces)
        {
            continue;
        }
        let removed_tetrahedra = star
            .iter()
            .map(|index| topology.tetrahedra[*index].vertex_indices)
            .collect::<Vec<_>>();
        let boundary = local_tetrahedron_boundary_faces(&removed_tetrahedra);
        let cavity_nodes = removed_tetrahedra
            .iter()
            .flatten()
            .copied()
            .collect::<BTreeSet<_>>();
        let nodes = cavity_nodes.iter().copied().collect::<Vec<_>>();
        for left in 0..nodes.len() {
            for right in (left + 1)..nodes.len() {
                let mut target = [nodes[left], nodes[right]];
                target.sort_unstable();
                if target == removed_edge || topology_has_edge(topology, target) {
                    continue;
                }
                let Some(created) = edge_star_replacement(topology, &boundary, target)? else {
                    continue;
                };
                if !prospective_tetrahedra_form_embedded_manifold(topology, &star, &created)?
                    || replacement_duplicates_retained_tetrahedron(topology, &star, &created)
                    || !tetrahedra_are_materializable(topology, &created)?
                {
                    continue;
                }
                let created_violations =
                    tetrahedra_global_violation_count(topology, &created, work)?;
                if created_violations > removed_violations {
                    continue;
                }
                if created_violations < removed_violations {
                    return Ok(Some((star, created)));
                }
                if plateau_candidate.is_none()
                    && target.contains(&(query_node as u32))
                    && target
                        .iter()
                        .any(|vertex| violating_vertices.contains(vertex))
                    && !visited_states
                        .contains(&replacement_topology_key(topology, &star, &created))
                {
                    plateau_candidate = Some((star.clone(), created));
                }
            }
        }
    }
    Ok(plateau_candidate)
}

fn topology_key(topology: &DelaunayVolumeTopology) -> Vec<[u32; 4]> {
    topology
        .tetrahedra
        .iter()
        .map(|tetrahedron| {
            let mut key = tetrahedron.vertex_indices;
            key.sort_unstable();
            key
        })
        .collect()
}

fn replacement_topology_key(
    topology: &DelaunayVolumeTopology,
    removed: &[usize],
    created: &[[u32; 4]],
) -> Vec<[u32; 4]> {
    let mut key = topology
        .tetrahedra
        .iter()
        .enumerate()
        .filter(|(index, _)| !removed.contains(index))
        .map(|(_, tetrahedron)| {
            let mut key = tetrahedron.vertex_indices;
            key.sort_unstable();
            key
        })
        .chain(created.iter().map(|tetrahedron| {
            let mut key = *tetrahedron;
            key.sort_unstable();
            key
        }))
        .collect::<Vec<_>>();
    key.sort_unstable();
    key
}

fn first_global_delaunay_violation(
    topology: &DelaunayVolumeTopology,
    work: &mut InsertionWork<'_>,
) -> Result<Option<(usize, usize)>, DelaunayInsertionError> {
    for (tetrahedron_index, tetrahedron) in topology.tetrahedra.iter().enumerate() {
        let points = tetrahedron
            .vertex_indices
            .map(|node| topology.nodes[node as usize]);
        for (node_index, node) in topology.nodes.iter().copied().enumerate() {
            if tetrahedron.vertex_indices.contains(&(node_index as u32))
                || topology.incidence.vertex_stars[node_index].is_empty()
            {
                continue;
            }
            work.predicate()?;
            if insphere_volume_nodes([points[0], points[1], points[2], points[3], node])
                .map_err(super::predicate_error)?
                == PredicateSign::Positive
            {
                return Ok(Some((tetrahedron_index, node_index)));
            }
        }
    }
    Ok(None)
}

fn global_delaunay_violation_queries(
    topology: &DelaunayVolumeTopology,
    work: &mut InsertionWork<'_>,
) -> Result<Vec<usize>, DelaunayInsertionError> {
    let mut queries = BTreeSet::new();
    for tetrahedron in &topology.tetrahedra {
        let points = tetrahedron
            .vertex_indices
            .map(|node| topology.nodes[node as usize]);
        for (node_index, node) in topology.nodes.iter().copied().enumerate() {
            if tetrahedron.vertex_indices.contains(&(node_index as u32))
                || topology.incidence.vertex_stars[node_index].is_empty()
                || queries.contains(&node_index)
            {
                continue;
            }
            work.predicate()?;
            if insphere_volume_nodes([points[0], points[1], points[2], points[3], node])
                .map_err(super::predicate_error)?
                == PredicateSign::Positive
            {
                queries.insert(node_index);
            }
        }
    }
    Ok(queries.into_iter().collect())
}

fn indexed_tetrahedra_global_violation_count(
    topology: &DelaunayVolumeTopology,
    tetrahedra: &[usize],
    work: &mut InsertionWork<'_>,
) -> Result<u64, DelaunayInsertionError> {
    tetrahedra_global_violation_count(
        topology,
        &tetrahedra
            .iter()
            .map(|index| topology.tetrahedra[*index].vertex_indices)
            .collect::<Vec<_>>(),
        work,
    )
}

fn tetrahedra_global_violation_count(
    topology: &DelaunayVolumeTopology,
    tetrahedra: &[[u32; 4]],
    work: &mut InsertionWork<'_>,
) -> Result<u64, DelaunayInsertionError> {
    let mut violations = 0_u64;
    for tetrahedron in tetrahedra {
        let mut tetrahedron = *tetrahedron;
        match orient_volume_nodes(tetrahedron.map(|node| topology.nodes[node as usize]))
            .map_err(super::predicate_error)?
        {
            PredicateSign::Positive => {}
            PredicateSign::Negative => tetrahedron.swap(0, 1),
            PredicateSign::Zero => return Ok(u64::MAX),
        }
        let points = tetrahedron.map(|node| topology.nodes[node as usize]);
        for (node_index, node) in topology.nodes.iter().copied().enumerate() {
            if tetrahedron.contains(&(node_index as u32)) {
                continue;
            }
            work.predicate()?;
            if insphere_volume_nodes([points[0], points[1], points[2], points[3], node])
                .map_err(super::predicate_error)?
                == PredicateSign::Positive
            {
                violations = violations.checked_add(1).ok_or_else(|| {
                    error(
                        DelaunayInsertionErrorKind::ResourceLimit,
                        "global Delaunay violation count overflow",
                    )
                })?;
            }
        }
    }
    Ok(violations)
}

fn edge_star_replacement(
    topology: &DelaunayVolumeTopology,
    boundary: &BTreeSet<[u32; 3]>,
    target: [u32; 2],
) -> Result<Option<Vec<[u32; 4]>>, DelaunayInsertionError> {
    let other = boundary
        .iter()
        .flatten()
        .copied()
        .filter(|node| !target.contains(node))
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let mut created = Vec::new();
    for first in 0..other.len() {
        for second in (first + 1)..other.len() {
            let mut left = [target[0], other[first], other[second]];
            let mut right = [target[1], other[first], other[second]];
            left.sort_unstable();
            right.sort_unstable();
            if boundary.contains(&left) && boundary.contains(&right) {
                let tetrahedron = [target[0], target[1], other[first], other[second]];
                if orient_volume_nodes(tetrahedron.map(|node| topology.nodes[node as usize]))
                    .map_err(super::predicate_error)?
                    != PredicateSign::Zero
                {
                    created.push(tetrahedron);
                }
            }
        }
    }
    Ok(
        (created.len() >= 2 && local_tetrahedron_boundary_faces(&created) == *boundary)
            .then_some(created),
    )
}

fn topology_has_edge(topology: &DelaunayVolumeTopology, edge: [u32; 2]) -> bool {
    topology.incidence.vertex_stars[edge[0] as usize]
        .iter()
        .any(|index| {
            topology.tetrahedra[*index as usize]
                .vertex_indices
                .contains(&edge[1])
        })
}

fn removed_internal_face_is_protected(
    topology: &DelaunayVolumeTopology,
    removed: &[usize],
    protected_faces: &BTreeSet<[StableDigest; 3]>,
) -> bool {
    removed.iter().any(|index| {
        let tetrahedron = &topology.tetrahedra[*index];
        tetrahedron
            .neighbors
            .iter()
            .enumerate()
            .any(|(opposite, neighbor)| {
                neighbor.is_some_and(|neighbor| removed.contains(&(neighbor as usize)))
                    && protected_faces.contains(&stable_face(
                        topology,
                        tetrahedron.vertex_indices,
                        opposite,
                    ))
            })
    })
}

fn replacement_duplicates_retained_tetrahedron(
    topology: &DelaunayVolumeTopology,
    removed: &[usize],
    created: &[[u32; 4]],
) -> bool {
    let created_count = created.len();
    let created = created
        .iter()
        .map(|tetrahedron| {
            let mut key = *tetrahedron;
            key.sort_unstable();
            key
        })
        .collect::<BTreeSet<_>>();
    created.len() != created_count
        || topology
            .tetrahedra
            .iter()
            .enumerate()
            .filter(|(index, _)| !removed.contains(index))
            .any(|(_, tetrahedron)| {
                let mut key = tetrahedron.vertex_indices;
                key.sort_unstable();
                created.contains(&key)
            })
}

fn first_illegal_boundary_edge(
    topology: &DelaunayVolumeTopology,
    protected_faces: &BTreeSet<[StableDigest; 3]>,
    work: &mut InsertionWork<'_>,
) -> Result<Option<CavityReplacement>, DelaunayInsertionError> {
    let mut edge_uses = BTreeMap::<[u32; 2], Vec<(usize, [u32; 3], u32)>>::new();
    for (tetrahedron_index, tetrahedron) in topology.tetrahedra.iter().enumerate() {
        for opposite in 0..4 {
            if tetrahedron.neighbors[opposite].is_some() {
                continue;
            }
            let mut face = [0; 3];
            let mut cursor = 0;
            for (slot, vertex) in tetrahedron.vertex_indices.iter().enumerate() {
                if slot != opposite {
                    face[cursor] = *vertex;
                    cursor += 1;
                }
            }
            face.sort_unstable();
            for mut edge in [[face[0], face[1]], [face[1], face[2]], [face[0], face[2]]] {
                edge.sort_unstable();
                edge_uses.entry(edge).or_default().push((
                    tetrahedron_index,
                    face,
                    tetrahedron.vertex_indices[opposite],
                ));
            }
        }
    }
    for (edge, uses) in edge_uses {
        let [(left_index, left_face, left_apex), (right_index, right_face, right_apex)] =
            uses.as_slice()
        else {
            continue;
        };
        work.checkpoint()?;
        if topology.tetrahedra[*left_index].region_id != topology.tetrahedra[*right_index].region_id
            || protected_faces.contains(&stable_face(
                topology,
                topology.tetrahedra[*left_index].vertex_indices,
                topology.tetrahedra[*left_index]
                    .vertex_indices
                    .iter()
                    .position(|vertex| *vertex == *left_apex)
                    .expect("boundary apex belongs to tetrahedron"),
            ))
            || protected_faces.contains(&stable_face(
                topology,
                topology.tetrahedra[*right_index].vertex_indices,
                topology.tetrahedra[*right_index]
                    .vertex_indices
                    .iter()
                    .position(|vertex| *vertex == *right_apex)
                    .expect("boundary apex belongs to tetrahedron"),
            ))
        {
            continue;
        }
        let left_boundary_apex = left_face
            .iter()
            .copied()
            .find(|vertex| !edge.contains(vertex))
            .expect("boundary face has one apex opposite its edge");
        let right_boundary_apex = right_face
            .iter()
            .copied()
            .find(|vertex| !edge.contains(vertex))
            .expect("boundary face has one apex opposite its edge");
        let coplanar = orient_volume_nodes([
            topology.nodes[edge[0] as usize],
            topology.nodes[edge[1] as usize],
            topology.nodes[left_boundary_apex as usize],
            topology.nodes[right_boundary_apex as usize],
        ])
        .map_err(super::predicate_error)?
            == PredicateSign::Zero;
        if !coplanar
            || circumsphere_relation(
                topology,
                topology.tetrahedra[*left_index].vertex_indices,
                topology.nodes[right_boundary_apex as usize],
                work,
            )? != PredicateSign::Positive
        {
            continue;
        }
        if left_apex == right_apex {
            let created = vec![
                [left_boundary_apex, right_boundary_apex, edge[0], *left_apex],
                [left_boundary_apex, right_boundary_apex, *left_apex, edge[1]],
            ];
            if prospective_tetrahedra_form_embedded_manifold(
                topology,
                &[*left_index, *right_index],
                &created,
            )? && tetrahedra_are_materializable(topology, &created)?
            {
                return Ok(Some((vec![*left_index, *right_index], created)));
            }
            continue;
        }
        let star = topology.incidence.vertex_stars[edge[0] as usize]
            .iter()
            .copied()
            .map(|index| index as usize)
            .filter(|index| {
                topology.tetrahedra[*index]
                    .vertex_indices
                    .contains(&edge[1])
            })
            .collect::<Vec<_>>();
        if star.len() < 2 {
            continue;
        }
        let region = &topology.tetrahedra[star[0]].region_id;
        if star
            .iter()
            .any(|index| &topology.tetrahedra[*index].region_id != region)
        {
            continue;
        }
        let removed_tetrahedra = star
            .iter()
            .map(|index| topology.tetrahedra[*index].vertex_indices)
            .collect::<Vec<_>>();
        let boundary = local_tetrahedron_boundary_faces(&removed_tetrahedra);
        let mut removed_internal_faces = BTreeSet::new();
        for index in &star {
            let tetrahedron = &topology.tetrahedra[*index];
            for opposite in 0..4 {
                if tetrahedron.neighbors[opposite]
                    .is_some_and(|neighbor| star.contains(&(neighbor as usize)))
                {
                    removed_internal_faces.insert(stable_face(
                        topology,
                        tetrahedron.vertex_indices,
                        opposite,
                    ));
                }
            }
        }
        if removed_internal_faces
            .iter()
            .any(|face| protected_faces.contains(face))
        {
            continue;
        }
        let cavity_nodes = removed_tetrahedra
            .iter()
            .flatten()
            .copied()
            .collect::<BTreeSet<_>>();
        let other = cavity_nodes
            .iter()
            .copied()
            .filter(|node| *node != left_boundary_apex && *node != right_boundary_apex)
            .collect::<Vec<_>>();
        let mut created = Vec::new();
        for first in 0..other.len() {
            for second in (first + 1)..other.len() {
                let mut left = [left_boundary_apex, other[first], other[second]];
                let mut right = [right_boundary_apex, other[first], other[second]];
                left.sort_unstable();
                right.sort_unstable();
                if boundary.contains(&left) && boundary.contains(&right) {
                    let tetrahedron = [
                        left_boundary_apex,
                        right_boundary_apex,
                        other[first],
                        other[second],
                    ];
                    if orient_volume_nodes(tetrahedron.map(|node| topology.nodes[node as usize]))
                        .map_err(super::predicate_error)?
                        != PredicateSign::Zero
                    {
                        created.push(tetrahedron);
                    }
                }
            }
        }
        if created.len() < 2 || local_tetrahedron_boundary_faces(&created) != boundary {
            continue;
        }
        if !prospective_tetrahedra_form_embedded_manifold(topology, &star, &created)?
            || !tetrahedra_are_materializable(topology, &created)?
        {
            continue;
        }
        return Ok(Some((star, created)));
    }
    Ok(None)
}

fn record_flip(
    flip_count: &mut u64,
    work: &InsertionWork<'_>,
) -> Result<(), DelaunayInsertionError> {
    *flip_count = flip_count.checked_add(1).ok_or_else(|| {
        error(
            DelaunayInsertionErrorKind::ResourceLimit,
            "Delaunay legalization flip count overflow",
        )
    })?;
    if *flip_count > work.options.maximum_cavity_tetrahedra {
        return Err(error(
            DelaunayInsertionErrorKind::ResourceLimit,
            "Delaunay legalization flip limit exceeded",
        ));
    }
    Ok(())
}

pub(super) fn first_reducible_edge(
    topology: &DelaunayVolumeTopology,
    protected_faces: &BTreeSet<[StableDigest; 3]>,
    _protect_region_boundaries: bool,
    work: &mut InsertionWork<'_>,
) -> Result<Option<CavityReplacement>, DelaunayInsertionError> {
    let mut edge_stars = BTreeMap::<[u32; 2], Vec<usize>>::new();
    for (tetrahedron_index, tetrahedron) in topology.tetrahedra.iter().enumerate() {
        for left in 0..4 {
            for right in (left + 1)..4 {
                let mut edge = [
                    tetrahedron.vertex_indices[left],
                    tetrahedron.vertex_indices[right],
                ];
                edge.sort_unstable();
                edge_stars.entry(edge).or_default().push(tetrahedron_index);
            }
        }
    }
    for (edge, star) in edge_stars {
        if star.len() != 3 {
            continue;
        }
        work.checkpoint()?;
        let region = &topology.tetrahedra[star[0]].region_id;
        if star
            .iter()
            .any(|index| &topology.tetrahedra[*index].region_id != region)
        {
            continue;
        }
        let local = std::array::from_fn(|slot| LocalTetrahedron {
            tetrahedron_id: star[slot] as u32,
            node_ids: topology.tetrahedra[star[slot]].vertex_indices,
        });
        let Ok(candidate) = three_to_two_edge_flip_candidate(local, edge) else {
            continue;
        };
        let mut removed_faces = BTreeSet::new();
        for index in &star {
            let tetrahedron = &topology.tetrahedra[*index];
            for opposite in 0..4 {
                if tetrahedron.neighbors[opposite]
                    .is_some_and(|neighbor| star.contains(&(neighbor as usize)))
                {
                    removed_faces.insert(stable_face(
                        topology,
                        tetrahedron.vertex_indices,
                        opposite,
                    ));
                }
            }
        }
        if removed_faces
            .iter()
            .any(|face| protected_faces.contains(face))
        {
            continue;
        }
        let mut created: [[u32; 4]; 2] = candidate
            .created_tetrahedra
            .try_into()
            .expect("three-to-two reconnection creates two tetrahedra");
        let mut nondegenerate = true;
        for tetrahedron in &mut created {
            match orient_volume_nodes(tetrahedron.map(|node| topology.nodes[node as usize]))
                .map_err(super::predicate_error)?
            {
                PredicateSign::Positive => {}
                PredicateSign::Negative => tetrahedron.swap(0, 1),
                PredicateSign::Zero => nondegenerate = false,
            }
        }
        if !nondegenerate {
            continue;
        }
        let created_keys = created.map(|mut tetrahedron| {
            tetrahedron.sort_unstable();
            tetrahedron
        });
        if created_keys[0] == created_keys[1]
            || topology
                .tetrahedra
                .iter()
                .enumerate()
                .filter(|(index, _)| !star.contains(index))
                .any(|(_, tetrahedron)| {
                    let mut key = tetrahedron.vertex_indices;
                    key.sort_unstable();
                    created_keys.contains(&key)
                })
        {
            continue;
        }
        if !prospective_tetrahedra_form_embedded_manifold(topology, &star, &created)? {
            continue;
        }
        if !tetrahedra_are_materializable(topology, &created)? {
            continue;
        }
        if !two_to_three_geometry_is_admissible(created[0], created[1], |node| {
            topology.nodes[node as usize].predicate_point()
        })
        .map_err(super::predicate_error)?
        {
            continue;
        }
        let opposite = created[1]
            .iter()
            .copied()
            .find(|node| !created[0].contains(node))
            .expect("inverse face flip has an opposite node");
        if circumsphere_relation(
            topology,
            created[0],
            topology.nodes[opposite as usize],
            work,
        )? == PredicateSign::Positive
        {
            continue;
        }
        return Ok(Some((star, created.into_iter().collect())));
    }
    Ok(None)
}

fn prospective_tetrahedra_form_embedded_manifold(
    topology: &DelaunayVolumeTopology,
    removed: &[usize],
    created: &[[u32; 4]],
) -> Result<bool, DelaunayInsertionError> {
    let removed = removed.iter().copied().collect::<BTreeSet<_>>();
    let mut created_keys = BTreeSet::new();
    let mut face_uses = BTreeMap::<[u32; 3], Vec<u32>>::new();
    for tetrahedron in created.iter().copied() {
        let mut key = tetrahedron;
        key.sort_unstable();
        if !created_keys.insert(key)
            || topology.incidence.vertex_stars[key[0] as usize]
                .iter()
                .copied()
                .map(|index| index as usize)
                .filter(|index| !removed.contains(index))
                .any(|index| {
                    let mut retained = topology.tetrahedra[index].vertex_indices;
                    retained.sort_unstable();
                    retained == key
                })
        {
            return Ok(false);
        }
        for opposite in 0..4 {
            let mut face = [0u32; 3];
            let mut cursor = 0;
            for (slot, vertex) in tetrahedron.iter().enumerate() {
                if slot != opposite {
                    face[cursor] = *vertex;
                    cursor += 1;
                }
            }
            face.sort_unstable();
            let uses = face_uses.entry(face).or_default();
            uses.push(tetrahedron[opposite]);
            if uses.len() > 2 {
                return Ok(false);
            }
        }
    }
    for (face, mut apices) in face_uses {
        for index in topology.incidence.vertex_stars[face[0] as usize]
            .iter()
            .copied()
            .map(|index| index as usize)
            .filter(|index| !removed.contains(index))
        {
            let tetrahedron = &topology.tetrahedra[index];
            if tetrahedron.vertex_indices.contains(&face[1])
                && tetrahedron.vertex_indices.contains(&face[2])
            {
                apices.push(
                    tetrahedron
                        .vertex_indices
                        .iter()
                        .copied()
                        .find(|vertex| !face.contains(vertex))
                        .expect("retained face use has one opposite vertex"),
                );
            }
        }
        let [left, right] = apices.as_slice() else {
            if apices.len() > 2 {
                return Ok(false);
            }
            continue;
        };
        let face_nodes = face.map(|node| topology.nodes[node as usize]);
        let left_sign = orient_volume_nodes([
            face_nodes[0],
            face_nodes[1],
            face_nodes[2],
            topology.nodes[*left as usize],
        ])
        .map_err(super::predicate_error)?;
        let right_sign = orient_volume_nodes([
            face_nodes[0],
            face_nodes[1],
            face_nodes[2],
            topology.nodes[*right as usize],
        ])
        .map_err(super::predicate_error)?;
        if left_sign == PredicateSign::Zero
            || right_sign == PredicateSign::Zero
            || left_sign == right_sign
        {
            return Ok(false);
        }
    }
    Ok(true)
}

fn apply_reconnection(
    topology: DelaunayVolumeTopology,
    removed: &[usize],
    created: Vec<[u32; 4]>,
    work: &InsertionWork<'_>,
) -> Result<DelaunayVolumeTopology, DelaunayInsertionError> {
    let region = topology.tetrahedra[removed[0]].region_id.clone();
    let removed = removed.iter().copied().collect::<BTreeSet<_>>();
    let mut tetrahedra = topology
        .tetrahedra
        .iter()
        .enumerate()
        .filter(|(index, _)| !removed.contains(index))
        .map(|(_, tetrahedron)| (tetrahedron.vertex_indices, tetrahedron.region_id.clone()))
        .collect::<Vec<_>>();
    tetrahedra.extend(
        created
            .into_iter()
            .map(|tetrahedron| (tetrahedron, region.clone())),
    );
    build_delaunay_volume_topology_with_regions(
        topology.nodes,
        tetrahedra,
        work.options.topology,
        work.cancellation(),
    )
    .map_err(topology_error)
}

fn first_illegal_face(
    topology: &DelaunayVolumeTopology,
    protected_faces: &BTreeSet<[StableDigest; 3]>,
    protect_region_boundaries: bool,
    focus: Option<u32>,
    work: &mut InsertionWork<'_>,
) -> Result<Option<(usize, usize, [u32; 3])>, DelaunayInsertionError> {
    for (left_index, left) in topology.tetrahedra.iter().enumerate() {
        if focus.is_some_and(|focus| !left.vertex_indices.contains(&focus)) {
            continue;
        }
        for (opposite, neighbor) in left.neighbors.iter().enumerate() {
            if focus.is_some_and(|focus| left.vertex_indices[opposite] != focus) {
                continue;
            }
            let Some(right_index) = neighbor.map(|index| index as usize) else {
                continue;
            };
            if focus.is_none() && right_index <= left_index {
                continue;
            }
            work.checkpoint()?;
            let right = &topology.tetrahedra[right_index];
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
                        "legalization neighbors do not share exactly one triangular face",
                    )
                })?;
            let left_apex = left
                .vertex_indices
                .iter()
                .copied()
                .find(|vertex| !right.vertex_indices.contains(vertex))
                .ok_or_else(|| {
                    error(
                        DelaunayInsertionErrorKind::InvalidTopology,
                        "legalization neighbors do not have distinct opposite vertices",
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
                || !two_to_three_topology_is_admissible(
                    topology,
                    left_index as u32,
                    right_index as u32,
                    left_apex,
                    right_apex,
                )
                || !two_to_three_geometry_is_admissible(
                    left.vertex_indices,
                    right.vertex_indices,
                    |node| {
                        let node = topology.nodes[node as usize];
                        DelaunayVolumeNode::explicit(node.identity, node.coordinates_m)
                            .predicate_point()
                    },
                )
                .map_err(super::predicate_error)?
            {
                continue;
            }
            return Ok(Some((left_index, right_index, face)));
        }
    }
    Ok(None)
}

fn tetrahedra_are_materializable(
    topology: &DelaunayVolumeTopology,
    tetrahedra: &[[u32; 4]],
) -> Result<bool, DelaunayInsertionError> {
    for tetrahedron in tetrahedra {
        let nodes = tetrahedron.map(|node| topology.nodes[node as usize]);
        let constructed =
            crate::cdt::topology::orient_volume_nodes(nodes).map_err(super::predicate_error)?;
        let materialized =
            orient_materialized_volume_nodes(nodes).map_err(super::predicate_error)?;
        if constructed == PredicateSign::Zero || materialized == PredicateSign::Zero {
            return Ok(false);
        }
    }
    Ok(true)
}

fn apply_two_to_three_flip(
    topology: DelaunayVolumeTopology,
    left_index: usize,
    right_index: usize,
    face: [u32; 3],
    work: &mut InsertionWork<'_>,
) -> Result<DelaunayVolumeTopology, DelaunayInsertionError> {
    let left = &topology.tetrahedra[left_index];
    let right = &topology.tetrahedra[right_index];
    if left.region_id != right.region_id {
        return Err(error(
            DelaunayInsertionErrorKind::InvalidTopology,
            "Delaunay legalization cannot cross a region boundary",
        ));
    }
    let left_apex = left
        .vertex_indices
        .iter()
        .copied()
        .find(|vertex| !face.contains(vertex))
        .ok_or_else(|| {
            error(
                DelaunayInsertionErrorKind::InvalidTopology,
                "missing left apex",
            )
        })?;
    let right_apex = right
        .vertex_indices
        .iter()
        .copied()
        .find(|vertex| !face.contains(vertex))
        .ok_or_else(|| {
            error(
                DelaunayInsertionErrorKind::InvalidTopology,
                "missing right apex",
            )
        })?;
    let mut tetrahedra = topology
        .tetrahedra
        .iter()
        .enumerate()
        .filter(|(index, _)| *index != left_index && *index != right_index)
        .map(|(_, tetrahedron)| (tetrahedron.vertex_indices, tetrahedron.region_id.clone()))
        .collect::<Vec<_>>();
    for edge in [[face[0], face[1]], [face[1], face[2]], [face[2], face[0]]] {
        tetrahedra.push((
            [left_apex, right_apex, edge[0], edge[1]],
            left.region_id.clone(),
        ));
    }
    build_delaunay_volume_topology_with_regions(
        topology.nodes,
        tetrahedra,
        work.options.topology,
        work.cancellation(),
    )
    .map_err(topology_error)
}
