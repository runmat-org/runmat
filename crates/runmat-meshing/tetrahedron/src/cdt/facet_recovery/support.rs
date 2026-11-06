use std::collections::{BTreeMap, BTreeSet};

use runmat_meshing_core::{
    quality::predicate::{
        projected_orient2d_constructed, triangle_projection_axes_constructed, PredicateSign,
        SpatialPredicatePoint,
    },
    StableDigest,
};

use super::{
    error, node_index, DelaunayConstraints, DelaunayFacetRecoveryError,
    DelaunayFacetRecoveryErrorKind, DelaunayRecoveredFacetTriangle, DelaunaySegmentRecovery,
    FacetRecoveryWork,
};

use crate::cdt::insertion::topology_access::TetrahedralTopology;

mod delaunay;
mod topology;
mod witness;
use topology::triangulate_with_existing_topology_edges;

fn triangle_edges(triangle: [StableDigest; 3]) -> [[StableDigest; 2]; 3] {
    [
        [triangle[0], triangle[1]],
        [triangle[1], triangle[2]],
        [triangle[2], triangle[0]],
    ]
}

fn topology_edge_exists(topology: &impl TetrahedralTopology, endpoints: [StableDigest; 2]) -> bool {
    let [Some(first), Some(second)] = endpoints.map(|identity| topology.node_index(identity))
    else {
        return false;
    };
    topology.has_edge([first, second])
}

pub(super) struct FacetSupportView<'a, T> {
    pub(super) topology: &'a T,
    segments: &'a [crate::cdt::DelaunayRecoveredSegment],
}

pub(super) fn facet_support(
    recovery: &DelaunaySegmentRecovery,
    constraints: &DelaunayConstraints,
    constraint_index: u32,
    work: &mut FacetRecoveryWork<'_>,
) -> Result<Vec<DelaunayRecoveredFacetTriangle>, DelaunayFacetRecoveryError> {
    facet_support_on(
        &recovery.topology,
        &recovery.segments,
        constraints,
        constraint_index,
        work,
    )
}

pub(super) fn facet_support_on(
    topology: &impl TetrahedralTopology,
    segments: &[crate::cdt::DelaunayRecoveredSegment],
    constraints: &DelaunayConstraints,
    constraint_index: u32,
    work: &mut FacetRecoveryWork<'_>,
) -> Result<Vec<DelaunayRecoveredFacetTriangle>, DelaunayFacetRecoveryError> {
    construct_support(
        &FacetSupportView { topology, segments },
        constraints,
        constraint_index,
        work,
    )
}

fn construct_support(
    recovery: &FacetSupportView<'_, impl TetrahedralTopology>,
    constraints: &DelaunayConstraints,
    constraint_index: u32,
    work: &mut FacetRecoveryWork<'_>,
) -> Result<Vec<DelaunayRecoveredFacetTriangle>, DelaunayFacetRecoveryError> {
    let facet = constraints
        .facets
        .get(constraint_index as usize)
        .ok_or_else(|| {
            invalid(
                constraint_index,
                "facet constraint index is outside the inventory",
            )
        })?;
    let mut boundary = Vec::new();
    let mut source_edge_membership = BTreeMap::<StableDigest, BTreeSet<usize>>::new();
    for (source_edge, edge) in [
        [facet.vertex_indices[0], facet.vertex_indices[1]],
        [facet.vertex_indices[1], facet.vertex_indices[2]],
        [facet.vertex_indices[2], facet.vertex_indices[0]],
    ]
    .into_iter()
    .enumerate()
    {
        let chain = edge_chain(recovery, constraints, edge, constraint_index, work)?;
        for identity in &chain {
            source_edge_membership
                .entry(*identity)
                .or_default()
                .insert(source_edge);
        }
        boundary.extend(chain.into_iter().skip(usize::from(!boundary.is_empty())));
    }
    if boundary.last() == boundary.first() {
        boundary.pop();
    }
    if boundary.len() < 3
        || boundary.iter().copied().collect::<BTreeSet<_>>().len() != boundary.len()
    {
        return Err(invalid(
            constraint_index,
            "facet boundary is collapsed or repeats a node",
        ));
    }
    let source = facet
        .vertex_indices
        .map(|index| constraints.nodes[index as usize].identity);
    triangulate_boundary(
        recovery,
        boundary,
        source,
        &source_edge_membership,
        constraint_index,
        work,
    )
}

fn edge_chain(
    recovery: &FacetSupportView<'_, impl TetrahedralTopology>,
    constraints: &DelaunayConstraints,
    edge: [u32; 2],
    constraint_index: u32,
    work: &mut FacetRecoveryWork<'_>,
) -> Result<Vec<StableDigest>, DelaunayFacetRecoveryError> {
    work.support_step(constraint_index)?;
    let first = constraints.nodes[edge[0] as usize].identity;
    let last = constraints.nodes[edge[1] as usize].identity;
    let mut key = edge;
    key.sort_unstable();
    let segment_index = constraints
        .segments
        .binary_search_by_key(&key, |segment| segment.vertex_indices)
        .map_err(|_| invalid(constraint_index, "facet edge has no segment constraint"))?;
    let recovered = recovery.segments.get(segment_index).ok_or_else(|| {
        invalid(
            constraint_index,
            "facet edge has no recovered segment evidence",
        )
    })?;
    let mut chain = recovered
        .nodes
        .iter()
        .map(|node| node.identity)
        .collect::<Vec<_>>();
    if chain.first() == Some(&last) && chain.last() == Some(&first) {
        chain.reverse();
    }
    if chain.first() != Some(&first) || chain.last() != Some(&last) {
        return Err(invalid(
            constraint_index,
            "recovered facet-edge chain does not bind to its oriented endpoints",
        ));
    }
    Ok(chain)
}

fn triangulate_boundary(
    recovery: &FacetSupportView<'_, impl TetrahedralTopology>,
    mut boundary: Vec<StableDigest>,
    source: [StableDigest; 3],
    source_edge_membership: &BTreeMap<StableDigest, BTreeSet<usize>>,
    constraint_index: u32,
    work: &mut FacetRecoveryWork<'_>,
) -> Result<Vec<DelaunayRecoveredFacetTriangle>, DelaunayFacetRecoveryError> {
    let projection = projection_axes(recovery, source, constraint_index, work)?;
    // Source-edge traversal fixes winding. A rounded split-point corner cannot
    // replace the original facet's orientation authority.
    let polygon_sign = orientation(recovery, source, projection, constraint_index, work)?;
    if let Some(triangles) = triangulate_with_existing_topology_edges(
        recovery,
        &boundary,
        projection,
        polygon_sign,
        source_edge_membership,
        constraint_index,
        work,
    )? {
        return Ok(triangles);
    }
    if let Some(triangles) = delaunay::triangulate_delaunay(
        recovery,
        &boundary,
        source,
        projection,
        polygon_sign,
        source_edge_membership,
        constraint_index,
        work,
    )? {
        return Ok(triangles);
    }
    let mut triangles = Vec::with_capacity(boundary.len() - 2);
    while boundary.len() > 3 {
        let mut ear = None;
        let mut recoverable_ear = None;
        for index in 0..boundary.len() {
            let triangle = [
                boundary[(index + boundary.len() - 1) % boundary.len()],
                boundary[index],
                boundary[(index + 1) % boundary.len()],
            ];
            if shares_source_edge(triangle, source_edge_membership) {
                continue;
            }
            if orientation(recovery, triangle, projection, constraint_index, work)? != polygon_sign
            {
                continue;
            }
            if triangle_contains_other_boundary_node(
                recovery,
                &boundary,
                triangle,
                projection,
                polygon_sign,
                constraint_index,
                work,
            )? {
                continue;
            }
            let mut remaining = boundary.clone();
            remaining.remove(index);
            if polygon_has_area(
                recovery,
                &remaining,
                projection,
                polygon_sign,
                source_edge_membership,
                constraint_index,
                work,
            )? {
                if topology_edge_exists(recovery.topology, [triangle[0], triangle[2]]) {
                    ear = Some((index, triangle));
                    break;
                }
                recoverable_ear.get_or_insert((index, triangle));
            }
        }
        let ear = ear.or(recoverable_ear);
        let Some((index, triangle)) = ear else {
            witness::report(
                recovery,
                &boundary,
                source,
                source_edge_membership,
                (projection, polygon_sign),
                constraint_index,
                work,
            )?;
            return Err(invalid(
                constraint_index,
                "facet boundary has no deterministic nondegenerate ear",
            ));
        };
        triangles.push(DelaunayRecoveredFacetTriangle {
            node_identities: triangle,
        });
        boundary.remove(index);
    }
    let final_triangle = [boundary[0], boundary[1], boundary[2]];
    if shares_source_edge(final_triangle, source_edge_membership)
        || orientation(recovery, final_triangle, projection, constraint_index, work)?
            != polygon_sign
    {
        return Err(invalid(
            constraint_index,
            "facet support leaves a degenerate final triangle",
        ));
    }
    triangles.push(DelaunayRecoveredFacetTriangle {
        node_identities: final_triangle,
    });
    align_support_with_topology(
        recovery,
        &mut triangles,
        projection,
        polygon_sign,
        constraint_index,
        work,
    )?;
    Ok(triangles)
}

fn align_support_with_topology(
    recovery: &FacetSupportView<'_, impl TetrahedralTopology>,
    triangles: &mut [DelaunayRecoveredFacetTriangle],
    axes: [usize; 2],
    polygon_sign: PredicateSign,
    constraint_index: u32,
    work: &mut FacetRecoveryWork<'_>,
) -> Result<(), DelaunayFacetRecoveryError> {
    loop {
        let mut owners = BTreeMap::<[StableDigest; 2], Vec<usize>>::new();
        for (triangle_index, triangle) in triangles.iter().enumerate() {
            for mut edge in triangle_edges(triangle.node_identities) {
                edge.sort_unstable();
                owners.entry(edge).or_default().push(triangle_index);
            }
        }
        let mut replacement = None;
        for (edge, owner_indices) in owners {
            work.support_step(constraint_index)?;
            if owner_indices.len() != 2 || topology_edge_exists(recovery.topology, edge) {
                continue;
            }
            let opposite = [owner_indices[0], owner_indices[1]].map(|triangle_index| {
                triangles[triangle_index]
                    .node_identities
                    .into_iter()
                    .find(|identity| !edge.contains(identity))
            });
            let [Some(first), Some(second)] = opposite else {
                continue;
            };
            if first == second || !topology_edge_exists(recovery.topology, [first, second]) {
                continue;
            }
            let mut candidates = [[first, second, edge[0]], [second, first, edge[1]]];
            let mut valid = true;
            for candidate in &mut candidates {
                let sign = orientation(recovery, *candidate, axes, constraint_index, work)?;
                if sign != polygon_sign {
                    candidate.swap(0, 1);
                }
                if orientation(recovery, *candidate, axes, constraint_index, work)? != polygon_sign
                {
                    valid = false;
                    break;
                }
            }
            if valid {
                replacement = Some((owner_indices, candidates));
                break;
            }
        }
        let Some((owner_indices, candidates)) = replacement else {
            return Ok(());
        };
        triangles[owner_indices[0]].node_identities = candidates[0];
        triangles[owner_indices[1]].node_identities = candidates[1];
    }
}

fn shares_source_edge(
    triangle: [StableDigest; 3],
    memberships: &BTreeMap<StableDigest, BTreeSet<usize>>,
) -> bool {
    (0..3).any(|source_edge| {
        triangle.iter().all(|identity| {
            memberships
                .get(identity)
                .is_some_and(|edges| edges.contains(&source_edge))
        })
    })
}

fn triangle_contains_other_boundary_node(
    recovery: &FacetSupportView<'_, impl TetrahedralTopology>,
    boundary: &[StableDigest],
    triangle: [StableDigest; 3],
    axes: [usize; 2],
    triangle_sign: PredicateSign,
    constraint_index: u32,
    work: &mut FacetRecoveryWork<'_>,
) -> Result<bool, DelaunayFacetRecoveryError> {
    for point in boundary
        .iter()
        .copied()
        .filter(|point| !triangle.contains(point))
    {
        let signs = [
            orientation(
                recovery,
                [triangle[0], triangle[1], point],
                axes,
                constraint_index,
                work,
            )?,
            orientation(
                recovery,
                [triangle[1], triangle[2], point],
                axes,
                constraint_index,
                work,
            )?,
            orientation(
                recovery,
                [triangle[2], triangle[0], point],
                axes,
                constraint_index,
                work,
            )?,
        ];
        if signs
            .iter()
            .all(|sign| *sign == triangle_sign || *sign == PredicateSign::Zero)
        {
            return Ok(true);
        }
    }
    Ok(false)
}

fn polygon_has_area(
    recovery: &FacetSupportView<'_, impl TetrahedralTopology>,
    boundary: &[StableDigest],
    axes: [usize; 2],
    polygon_sign: PredicateSign,
    source_edge_membership: &BTreeMap<StableDigest, BTreeSet<usize>>,
    constraint_index: u32,
    work: &mut FacetRecoveryWork<'_>,
) -> Result<bool, DelaunayFacetRecoveryError> {
    for index in 0..boundary.len() {
        let triangle = [
            boundary[(index + boundary.len() - 1) % boundary.len()],
            boundary[index],
            boundary[(index + 1) % boundary.len()],
        ];
        if shares_source_edge(triangle, source_edge_membership) {
            continue;
        }
        let sign = orientation(recovery, triangle, axes, constraint_index, work)?;
        if sign == polygon_sign {
            return Ok(true);
        }
    }
    Ok(false)
}

fn points(
    recovery: &FacetSupportView<'_, impl TetrahedralTopology>,
    triangle: [StableDigest; 3],
    constraint_index: u32,
) -> Result<[SpatialPredicatePoint; 3], DelaunayFacetRecoveryError> {
    let points = triangle.map(|identity| {
        node_index(recovery.topology, identity)
            .map(|index| recovery.topology.nodes()[index].predicate_point())
            .ok_or_else(|| {
                invalid(
                    constraint_index,
                    "facet support node is missing from topology",
                )
            })
    });
    let [first, second, third] = points;
    Ok([first?, second?, third?])
}
fn projection_axes(
    recovery: &FacetSupportView<'_, impl TetrahedralTopology>,
    triangle: [StableDigest; 3],
    constraint_index: u32,
    work: &mut FacetRecoveryWork<'_>,
) -> Result<[usize; 2], DelaunayFacetRecoveryError> {
    work.support_step(constraint_index)?;
    triangle_projection_axes_constructed(points(recovery, triangle, constraint_index)?)
        .map_err(|failure| {
            error(
                DelaunayFacetRecoveryErrorKind::InvalidTopology,
                Some(constraint_index),
                format!("facet projection predicate failed: {failure:?}"),
            )
        })?
        .ok_or_else(|| {
            invalid(
                constraint_index,
                "facet source vertices are exactly collinear",
            )
        })
}
fn orientation(
    recovery: &FacetSupportView<'_, impl TetrahedralTopology>,
    triangle: [StableDigest; 3],
    axes: [usize; 2],
    constraint_index: u32,
    work: &mut FacetRecoveryWork<'_>,
) -> Result<PredicateSign, DelaunayFacetRecoveryError> {
    work.support_step(constraint_index)?;
    projected_orient2d_constructed(points(recovery, triangle, constraint_index)?, axes).map_err(
        |failure| {
            error(
                DelaunayFacetRecoveryErrorKind::InvalidTopology,
                Some(constraint_index),
                format!("facet support predicate failed: {failure:?}"),
            )
        },
    )
}

fn invalid(constraint_index: u32, reason: &'static str) -> DelaunayFacetRecoveryError {
    error(
        DelaunayFacetRecoveryErrorKind::InvalidConstraints,
        Some(constraint_index),
        reason,
    )
}

#[cfg(test)]
mod tests;
