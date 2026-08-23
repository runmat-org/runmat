use std::collections::BTreeSet;

use runmat_meshing_core::quality::predicate::{orient3d, PredicateSign};

use super::{flood, Classification};
use crate::cdt::carving::{
    error, CarvingWork, DelaunayCarvingError, DelaunayCarvingErrorKind,
    DelaunayConstraintFacetSide, DelaunayConstraints, DelaunayFacetRecovery,
};

pub(super) fn classify_facet_sides(
    recovery: &DelaunayFacetRecovery,
    constraints: &DelaunayConstraints,
    blocked: &BTreeSet<[u32; 3]>,
    classifications: &mut [Option<Classification>],
    work: &mut CarvingWork<'_>,
) -> Result<(), DelaunayCarvingError> {
    let topology = &recovery.topology;
    for recovered in &recovery.facets {
        let constraint = &constraints.facets[recovered.constraint_index as usize];
        for triangle in &recovered.triangles {
            let indices = triangle.node_identities.map(|identity| {
                topology
                    .nodes
                    .binary_search_by_key(&identity, |node| node.identity)
                    .map(|index| index as u32)
                    .map_err(|_| {
                        error(
                            DelaunayCarvingErrorKind::InvalidTopology,
                            Some(recovered.constraint_index),
                            "recovered facet node is missing from carving topology",
                        )
                    })
            });
            let [first, second, third] = indices;
            let oriented = [first?, second?, third?];
            let mut face = oriented;
            face.sort_unstable();
            if !blocked.contains(&face) {
                return Err(error(
                    DelaunayCarvingErrorKind::InvalidTopology,
                    Some(recovered.constraint_index),
                    "recovered facet support is absent from the blocked-face inventory",
                ));
            }
            let uses = topology.incidence.vertex_stars[face[0] as usize]
                .iter()
                .copied()
                .filter(|tetrahedron| {
                    let vertices = topology.tetrahedra[*tetrahedron as usize].vertex_indices;
                    vertices.contains(&face[1]) && vertices.contains(&face[2])
                })
                .collect::<Vec<_>>();
            if uses.is_empty() || uses.len() > 2 {
                return Err(error(
                    DelaunayCarvingErrorKind::InvalidTopology,
                    Some(recovered.constraint_index),
                    "recovered facet support has invalid tetrahedron incidence",
                ));
            }
            let points = oriented.map(|index| topology.nodes[index as usize].coordinates_m);
            for tetrahedron in uses {
                let opposite = topology.tetrahedra[tetrahedron as usize]
                    .vertex_indices
                    .into_iter()
                    .find(|vertex| !face.contains(vertex))
                    .ok_or_else(|| {
                        error(
                            DelaunayCarvingErrorKind::InvalidTopology,
                            Some(recovered.constraint_index),
                            "facet incident tetrahedron has no opposite vertex",
                        )
                    })?;
                let sign = orient3d([
                    points[0],
                    points[1],
                    points[2],
                    topology.nodes[opposite as usize].coordinates_m,
                ])
                .map_err(|predicate| {
                    error(
                        DelaunayCarvingErrorKind::InvalidTopology,
                        Some(recovered.constraint_index),
                        format!("facet-side predicate failed: {predicate:?}"),
                    )
                })?;
                // `orient3d([a, b, c, p])` has the opposite sign of the half-space reached
                // along the oriented triangle normal: negative is the positive side.
                let side = match sign {
                    PredicateSign::Negative => &constraint.positive_side,
                    PredicateSign::Positive => &constraint.negative_side,
                    PredicateSign::Zero => {
                        return Err(error(
                            DelaunayCarvingErrorKind::InvalidTopology,
                            Some(recovered.constraint_index),
                            "facet incident tetrahedron is exactly coplanar",
                        ));
                    }
                };
                flood(
                    topology,
                    blocked,
                    BTreeSet::from([tetrahedron]),
                    classification(side),
                    Some(recovered.constraint_index),
                    classifications,
                    work,
                )?;
            }
        }
    }
    Ok(())
}

fn classification(side: &DelaunayConstraintFacetSide) -> Classification {
    match side {
        DelaunayConstraintFacetSide::Region(region) => Classification::Region(region.clone()),
        DelaunayConstraintFacetSide::Exterior => Classification::Exterior,
        DelaunayConstraintFacetSide::Void => Classification::Void,
    }
}
