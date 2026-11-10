use std::collections::{BTreeMap, BTreeSet, VecDeque};

use runmat_meshing_core::{
    quality::predicate::{
        closest_point_on_triangle, triangle_normal_alignment_constructed, PredicateSign,
    },
    MeshingCancellationSignal, StableDigest,
};

use super::super::{
    error, face_exists, node_index, DelaunayConstraints, DelaunayFacetRecovery,
    DelaunayFacetRecoveryError, DelaunayFacetRecoveryErrorKind, DelaunayFacetRecoveryOptions,
    DelaunayRecoveredFacetTriangle, FacetRecoveryWork,
};
use crate::cdt::segment_recovery::coordinates_share_representable_location;
use crate::cdt::topology::orient_volume_nodes;

pub(super) fn validate_recovered_facets(
    recovery: &DelaunayFacetRecovery,
    effective_segments: &super::super::DelaunaySegmentRecovery,
    constraints: &DelaunayConstraints,
    options: DelaunayFacetRecoveryOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<(), DelaunayFacetRecoveryError> {
    let mut work = FacetRecoveryWork::new(options, cancellation);
    let mut globally_recovered = BTreeSet::new();
    for (constraint_index, recovered) in recovery.facets.iter().enumerate() {
        work.begin_constraint();
        let constraint_index = constraint_index as u32;
        if recovered.constraint_index != constraint_index || recovered.triangles.is_empty() {
            return Err(invalid(
                constraint_index,
                "recovered facet does not bind to its ordered nonempty constraint evidence",
            ));
        }
        let expected_support = super::super::support::facet_support(
            effective_segments,
            constraints,
            constraint_index,
            &mut work,
        )?;
        let expected_boundary = boundary_edges(&expected_support, constraint_index)?;
        let source = constraints.facets[constraint_index as usize]
            .vertex_indices
            .map(|index| constraints.nodes[index as usize].identity);
        let source_nodes = source.map(|identity| {
            node_index(&recovery.topology, identity)
                .map(|index| recovery.topology.nodes[index])
                .ok_or_else(|| invalid(constraint_index, "source facet node is absent"))
        });
        let [first, second, third] = source_nodes;
        let source_nodes = [first?, second?, third?];
        let mut uses = BTreeMap::<[StableDigest; 2], Vec<usize>>::new();
        let mut vertices = BTreeSet::new();
        for (triangle_index, triangle) in recovered.triangles.iter().enumerate() {
            work.support_step(constraint_index)?;
            let identities = triangle.node_identities;
            let mut canonical = identities;
            canonical.sort_unstable();
            if canonical.contains(&StableDigest::ZERO)
                || canonical.windows(2).any(|pair| pair[0] == pair[1])
                || !globally_recovered.insert(canonical)
            {
                return Err(invalid(
                    constraint_index,
                    "recovered facet triangles must be nondegenerate and globally unique",
                ));
            }
            if !face_exists(&recovery.topology, identities, constraint_index, &mut work)? {
                return Err(invalid(
                    constraint_index,
                    "recovered facet triangle is absent from the final topology",
                ));
            }
            let triangle_nodes = identities.map(|identity| {
                node_index(&recovery.topology, identity)
                    .map(|index| recovery.topology.nodes[index].predicate_point())
                    .ok_or_else(|| invalid(constraint_index, "recovered facet node is absent"))
            });
            let [triangle_first, triangle_second, triangle_third] = triangle_nodes;
            if triangle_normal_alignment_constructed(
                source_nodes.map(|node| node.predicate_point()),
                [triangle_first?, triangle_second?, triangle_third?],
            )
            .map_err(|failure| {
                invalid(
                    constraint_index,
                    format!("facet orientation predicate failed: {failure:?}"),
                )
            })? != PredicateSign::Positive
            {
                return Err(invalid(
                    constraint_index,
                    "recovered facet triangle orientation disagrees with its source facet",
                ));
            }
            for identity in identities {
                vertices.insert(identity);
                if source.contains(&identity) {
                    continue;
                }
                let node = node_index(&recovery.topology, identity)
                    .map(|index| recovery.topology.nodes[index])
                    .ok_or_else(|| invalid(constraint_index, "recovered facet node is absent"))?;
                let exact_coplanarity =
                    orient_volume_nodes([source_nodes[0], source_nodes[1], source_nodes[2], node])
                        .map_err(|failure| {
                            invalid(
                                constraint_index,
                                format!("facet planarity predicate failed: {failure:?}"),
                            )
                        })?;
                let source_coordinates =
                    source_nodes.map(|source| source.coordinates_m.canonical_value());
                let coincident_at_resolution = coordinates_share_representable_location(
                    node.coordinates_m.canonical_value(),
                    closest_point_on_triangle(
                        node.coordinates_m.canonical_value(),
                        source_coordinates,
                    ),
                );
                if exact_coplanarity != PredicateSign::Zero && !coincident_at_resolution {
                    let coordinates = node.coordinates_m.canonical_value();
                    let closest = closest_point_on_triangle(coordinates, source_coordinates);
                    return Err(invalid(
                        constraint_index,
                        format!(
                            "recovered facet node is neither exactly coplanar nor coincident with \
                             its source facet at coordinate resolution: node {coordinates:?} \
                             ({:?}) is {:.3e} m from source facet {source_coordinates:?}",
                            node.construction,
                            (0..3)
                                .map(|axis| (coordinates[axis] - closest[axis]).powi(2))
                                .sum::<f64>()
                                .sqrt(),
                        ),
                    ));
                }
            }
            for edge in triangle_edges(identities) {
                uses.entry(edge).or_default().push(triangle_index);
            }
        }
        if uses.values().any(|uses| uses.len() > 2) {
            return Err(invalid(
                constraint_index,
                "recovered facet has a nonmanifold triangle edge",
            ));
        }
        let actual_boundary = uses
            .iter()
            .filter_map(|(edge, uses)| (uses.len() == 1).then_some(*edge))
            .collect::<BTreeSet<_>>();
        if actual_boundary != expected_boundary {
            return Err(invalid(
                constraint_index,
                "recovered facet boundary differs from its recovered segment chains",
            ));
        }
        validate_connected_disk(
            recovered.triangles.len(),
            vertices.len(),
            &uses,
            constraint_index,
        )?;
    }
    Ok(())
}

fn boundary_edges(
    triangles: &[DelaunayRecoveredFacetTriangle],
    constraint_index: u32,
) -> Result<BTreeSet<[StableDigest; 2]>, DelaunayFacetRecoveryError> {
    let mut uses = BTreeMap::<[StableDigest; 2], u32>::new();
    for triangle in triangles {
        for edge in triangle_edges(triangle.node_identities) {
            *uses.entry(edge).or_default() += 1;
        }
    }
    if uses.values().any(|count| *count > 2) {
        return Err(invalid(
            constraint_index,
            "derived facet support has a nonmanifold edge",
        ));
    }
    Ok(uses
        .into_iter()
        .filter_map(|(edge, count)| (count == 1).then_some(edge))
        .collect())
}

fn validate_connected_disk(
    triangle_count: usize,
    vertex_count: usize,
    edge_uses: &BTreeMap<[StableDigest; 2], Vec<usize>>,
    constraint_index: u32,
) -> Result<(), DelaunayFacetRecoveryError> {
    let mut adjacency = vec![Vec::new(); triangle_count];
    for uses in edge_uses.values().filter(|uses| uses.len() == 2) {
        adjacency[uses[0]].push(uses[1]);
        adjacency[uses[1]].push(uses[0]);
    }
    let mut visited = vec![false; triangle_count];
    let mut queue = VecDeque::from([0]);
    while let Some(index) = queue.pop_front() {
        if std::mem::replace(&mut visited[index], true) {
            continue;
        }
        queue.extend(adjacency[index].iter().copied());
    }
    let euler_characteristic = vertex_count as i64 - edge_uses.len() as i64 + triangle_count as i64;
    if visited.iter().any(|visited| !visited) || euler_characteristic != 1 {
        return Err(invalid(
            constraint_index,
            "recovered facet triangles do not form one connected topological disk",
        ));
    }
    Ok(())
}

fn triangle_edges(triangle: [StableDigest; 3]) -> [[StableDigest; 2]; 3] {
    [
        [triangle[0], triangle[1]],
        [triangle[1], triangle[2]],
        [triangle[2], triangle[0]],
    ]
    .map(|mut edge| {
        edge.sort_unstable();
        edge
    })
}

fn invalid(constraint_index: u32, reason: impl Into<String>) -> DelaunayFacetRecoveryError {
    error(
        DelaunayFacetRecoveryErrorKind::InvalidTopology,
        Some(constraint_index),
        reason,
    )
}
