//! Recover a segment crossing several coplanar hull triangles by checked boundary flips.
//! The transaction is admitted by the caller's independent Delaunay and edge checks.
use super::*;
use runmat_meshing_core::quality::predicate::orient2d;
use std::collections::BTreeMap;

pub(super) fn recover(
    topology: &mut TopologyMutation,
    left: u32,
    right: u32,
    protected_edges: &BTreeSet<[u32; 2]>,
    protected_faces: &[[StableDigest; 3]],
    constraint_index: u32,
    work: &mut RecoveryWork<'_>,
) -> Result<bool, DelaunaySegmentRecoveryError> {
    // A hull-triangle corridor cannot recover an endpoint strictly inside the
    // volume. Resolve this necessary condition through local stars before
    // allocating a transaction or inspecting the boundary inventory.
    for endpoint in [left, right] {
        let mut on_boundary = false;
        for index in topology.vertex_star(endpoint) {
            work.search_flip_step(constraint_index)?;
            let cell = topology.tetrahedron(index).expect("live endpoint star");
            if (0..4).any(|opposite| {
                cell.vertex_indices[opposite] != endpoint && cell.neighbors[opposite].is_none()
            }) {
                on_boundary = true;
                break;
            }
        }
        if !on_boundary {
            return Ok(false);
        }
    }
    let mut current = topology.cavity_trial();
    let mut visited = BTreeSet::new();
    loop {
        let key = topology_state(&current, &mut || work.search_flip_step(constraint_index))?;
        if !visited.insert(key) {
            return Ok(false);
        }
        let uses = boundary_edges(&current, [left, right], constraint_index, work)?;
        let mut advanced = false;
        for (edge, faces) in uses {
            work.search_flip_step(constraint_index)?;
            if faces.len() != 2
                || protected_edges.contains(&edge)
                || edge.contains(&left)
                || edge.contains(&right)
            {
                continue;
            }
            let a = current.nodes()[left as usize];
            let b = current.nodes()[right as usize];
            let c = current.nodes()[edge[0] as usize];
            let d = current.nodes()[edge[1] as usize];
            if orient_volume_nodes([a, b, c, d])
                .map_err(|failure| invalid_topology(constraint_index, format!("{failure:?}")))?
                != PredicateSign::Zero
            {
                continue;
            }
            let axes = match projection(
                [
                    a.coordinates_m.canonical_value(),
                    b.coordinates_m.canonical_value(),
                    c.coordinates_m.canonical_value(),
                ],
                constraint_index,
            )? {
                Some(axes) => axes,
                None => continue,
            };
            if !crosses(
                [
                    a.coordinates_m.canonical_value(),
                    b.coordinates_m.canonical_value(),
                ],
                [
                    c.coordinates_m.canonical_value(),
                    d.coordinates_m.canonical_value(),
                ],
                axes,
                constraint_index,
            )? {
                continue;
            }
            let [first, second] = [faces[0], faces[1]];
            if protected_faces.contains(&stable_face(&*current, first.1))
                || protected_faces.contains(&stable_face(&*current, second.1))
            {
                continue;
            }
            let first_tet = current.tetrahedron(first.0).expect("live boundary owner");
            let second_tet = current.tetrahedron(second.0).expect("live boundary owner");
            let apex = first_tet.vertex_indices[first.2];
            if apex != second_tet.vertex_indices[second.2]
                || first_tet.region_id != second_tet.region_id
            {
                continue;
            }
            let x = *first
                .1
                .iter()
                .find(|vertex| !edge.contains(vertex))
                .expect("triangle leaves one vertex");
            let y = *second
                .1
                .iter()
                .find(|vertex| !edge.contains(vertex))
                .expect("triangle leaves one vertex");
            let mut regions_coplanar = true;
            for vertex in [x, y] {
                if [left, right].contains(&vertex) {
                    continue;
                }
                if orient_volume_nodes([a, b, c, current.nodes()[vertex as usize]])
                    .map_err(|failure| invalid_topology(constraint_index, format!("{failure:?}")))?
                    != PredicateSign::Zero
                {
                    regions_coplanar = false;
                }
            }
            if !regions_coplanar {
                continue;
            }
            // Both diagonals must meet strictly: this is a convex quadrilateral,
            // not a boundary change across a concavity or an overlapping patch.
            if !crosses(
                [
                    c.coordinates_m.canonical_value(),
                    d.coordinates_m.canonical_value(),
                ],
                [
                    current.nodes()[x as usize].coordinates_m.canonical_value(),
                    current.nodes()[y as usize].coordinates_m.canonical_value(),
                ],
                axes,
                constraint_index,
            )? {
                continue;
            }
            work.flip_attempt(constraint_index)?;
            let region = first_tet.region_id.clone();
            let mut trial = current.cavity_trial();
            match trial.replace_cavity(
                &BTreeSet::from([first.0, second.0]),
                vec![
                    ([x, y, edge[0], apex], region.clone()),
                    ([x, y, edge[1], apex], region),
                ],
            ) {
                Ok(_) => {}
                Err(failure)
                    if failure.kind
                        == crate::cdt::mutation::TopologyMutationErrorKind::InvalidTopology =>
                {
                    continue
                }
                Err(failure) => return Err(super::super::mutation_error(failure)),
            };
            let key = topology_state(&trial, &mut || work.search_flip_step(constraint_index))?;
            if visited.contains(&key) {
                continue;
            }
            let recovered = edge_exists(&*trial, a.identity, b.identity, constraint_index, work)?;
            trial.commit();
            if recovered {
                current.commit();
                return Ok(true);
            }
            advanced = true;
            break;
        }
        if !advanced {
            return Ok(false);
        }
    }
}

type FaceUse = (u32, [u32; 3], usize);
fn boundary_edges(
    topology: &TopologyMutation,
    endpoints: [u32; 2],
    constraint_index: u32,
    work: &mut RecoveryWork<'_>,
) -> Result<BTreeMap<[u32; 2], Vec<FaceUse>>, DelaunaySegmentRecoveryError> {
    let mut uses = BTreeMap::<[u32; 2], Vec<FaceUse>>::new();
    // Both faces of a crossing hull edge intersect the closed segment bounds.
    // Query cells conservatively, then use maintained hull ownership. Interior
    // cells and noncrossing edges are harmless candidates; exact tests follow.
    for owner in super::spatial_candidates::candidate_cells(
        topology,
        endpoints[0],
        endpoints[1],
        constraint_index,
        work,
    )? {
        let cell = topology.tetrahedron(owner).expect("live spatial candidate");
        for opposite in 0..4 {
            work.search_flip_step(constraint_index)?;
            if cell.neighbors[opposite].is_some() {
                continue;
            }
            let mut face = [0; 3];
            let mut slot = 0;
            for (corner, vertex) in cell.vertex_indices.into_iter().enumerate() {
                if corner != opposite {
                    face[slot] = vertex;
                    slot += 1;
                }
            }
            face.sort_unstable();
            let [a, b, c] = face;
            for mut edge in [[a, b], [b, c], [c, a]] {
                edge.sort_unstable();
                uses.entry(edge).or_default().push((owner, face, opposite));
            }
        }
    }
    for faces in uses.values_mut() {
        // Order owners by stable corner keys without rebasing mutable cell slots.
        faces.sort_by_key(|(owner, _, _)| {
            topology
                .tetrahedron(*owner)
                .expect("live boundary owner")
                .vertex_indices
                .map(|vertex| topology.nodes()[vertex as usize].identity)
        });
    }
    Ok(uses)
}

fn projection(
    points: [[f64; 3]; 3],
    index: u32,
) -> Result<Option<[usize; 2]>, DelaunaySegmentRecoveryError> {
    for axes in [[0, 1], [0, 2], [1, 2]] {
        if sign(points, axes, index)? != PredicateSign::Zero {
            return Ok(Some(axes));
        }
    }
    Ok(None)
}
fn sign(
    points: [[f64; 3]; 3],
    axes: [usize; 2],
    index: u32,
) -> Result<PredicateSign, DelaunaySegmentRecoveryError> {
    orient2d(points.map(|point| [point[axes[0]], point[axes[1]]]))
        .map_err(|failure| invalid_topology(index, format!("{failure:?}")))
}
fn crosses(
    a: [[f64; 3]; 2],
    b: [[f64; 3]; 2],
    axes: [usize; 2],
    index: u32,
) -> Result<bool, DelaunaySegmentRecoveryError> {
    let signs = [
        sign([a[0], a[1], b[0]], axes, index)?,
        sign([a[0], a[1], b[1]], axes, index)?,
        sign([b[0], b[1], a[0]], axes, index)?,
        sign([b[0], b[1], a[1]], axes, index)?,
    ];
    Ok(signs.iter().all(|sign| *sign != PredicateSign::Zero)
        && signs[0] != signs[1]
        && signs[2] != signs[3])
}

#[cfg(test)]
mod tests;
