use std::collections::BTreeSet;

use runmat_meshing_core::{
    quality::predicate::{orient2d, PredicateSign},
    StableDigest,
};

use super::{
    error, node_index, DelaunayConstraints, DelaunayFacetRecoveryError,
    DelaunayFacetRecoveryErrorKind, DelaunayRecoveredFacetTriangle, DelaunaySegmentRecovery,
    FacetRecoveryWork,
};

pub(super) fn facet_support(
    recovery: &DelaunaySegmentRecovery,
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
    for edge in [
        [facet.vertex_indices[0], facet.vertex_indices[1]],
        [facet.vertex_indices[1], facet.vertex_indices[2]],
        [facet.vertex_indices[2], facet.vertex_indices[0]],
    ] {
        append_edge_chain(
            recovery,
            constraints,
            edge,
            constraint_index,
            &mut boundary,
            work,
        )?;
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
    triangulate_boundary(recovery, boundary, source, constraint_index, work)
}

fn append_edge_chain(
    recovery: &DelaunaySegmentRecovery,
    constraints: &DelaunayConstraints,
    edge: [u32; 2],
    constraint_index: u32,
    boundary: &mut Vec<StableDigest>,
    work: &mut FacetRecoveryWork<'_>,
) -> Result<(), DelaunayFacetRecoveryError> {
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
    boundary.extend(chain.into_iter().skip(usize::from(!boundary.is_empty())));
    Ok(())
}

fn triangulate_boundary(
    recovery: &DelaunaySegmentRecovery,
    mut boundary: Vec<StableDigest>,
    source: [StableDigest; 3],
    constraint_index: u32,
    work: &mut FacetRecoveryWork<'_>,
) -> Result<Vec<DelaunayRecoveredFacetTriangle>, DelaunayFacetRecoveryError> {
    let projection = projection_axes(recovery, source, constraint_index, work)?;
    let polygon_sign = polygon_sign(recovery, &boundary, projection, constraint_index, work)?;
    let mut triangles = Vec::with_capacity(boundary.len() - 2);
    while boundary.len() > 3 {
        let mut ear = None;
        for index in 0..boundary.len() {
            let triangle = [
                boundary[(index + boundary.len() - 1) % boundary.len()],
                boundary[index],
                boundary[(index + 1) % boundary.len()],
            ];
            if orientation(recovery, triangle, projection, constraint_index, work)? == polygon_sign
            {
                ear = Some((index, triangle));
                break;
            }
        }
        let Some((index, triangle)) = ear else {
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
    if orientation(recovery, final_triangle, projection, constraint_index, work)? != polygon_sign {
        return Err(invalid(
            constraint_index,
            "facet support leaves a degenerate final triangle",
        ));
    }
    triangles.push(DelaunayRecoveredFacetTriangle {
        node_identities: final_triangle,
    });
    Ok(triangles)
}

fn projection_axes(
    recovery: &DelaunaySegmentRecovery,
    triangle: [StableDigest; 3],
    constraint_index: u32,
    work: &mut FacetRecoveryWork<'_>,
) -> Result<[usize; 2], DelaunayFacetRecoveryError> {
    for axes in [[0, 1], [0, 2], [1, 2]] {
        if orientation(recovery, triangle, axes, constraint_index, work)? != PredicateSign::Zero {
            return Ok(axes);
        }
    }
    Err(invalid(
        constraint_index,
        "facet source vertices are exactly collinear",
    ))
}

fn polygon_sign(
    recovery: &DelaunaySegmentRecovery,
    boundary: &[StableDigest],
    axes: [usize; 2],
    constraint_index: u32,
    work: &mut FacetRecoveryWork<'_>,
) -> Result<PredicateSign, DelaunayFacetRecoveryError> {
    for index in 0..boundary.len() {
        let sign = orientation(
            recovery,
            [
                boundary[(index + boundary.len() - 1) % boundary.len()],
                boundary[index],
                boundary[(index + 1) % boundary.len()],
            ],
            axes,
            constraint_index,
            work,
        )?;
        if sign != PredicateSign::Zero {
            return Ok(sign);
        }
    }
    Err(invalid(
        constraint_index,
        "facet boundary is exactly collinear",
    ))
}

fn orientation(
    recovery: &DelaunaySegmentRecovery,
    triangle: [StableDigest; 3],
    axes: [usize; 2],
    constraint_index: u32,
    work: &mut FacetRecoveryWork<'_>,
) -> Result<PredicateSign, DelaunayFacetRecoveryError> {
    work.support_step(constraint_index)?;
    let points = triangle.map(|identity| {
        node_index(&recovery.topology, identity)
            .map(|index| recovery.topology.nodes[index].coordinates_m)
            .ok_or_else(|| {
                invalid(
                    constraint_index,
                    "facet support node is missing from topology",
                )
            })
    });
    let [first, second, third] = points;
    let points = [first?, second?, third?];
    orient2d(points.map(|point| [point[axes[0]], point[axes[1]]])).map_err(|predicate| {
        error(
            DelaunayFacetRecoveryErrorKind::InvalidTopology,
            Some(constraint_index),
            format!("facet support predicate failed: {predicate:?}"),
        )
    })
}

fn invalid(constraint_index: u32, reason: &'static str) -> DelaunayFacetRecoveryError {
    error(
        DelaunayFacetRecoveryErrorKind::InvalidConstraints,
        Some(constraint_index),
        reason,
    )
}
