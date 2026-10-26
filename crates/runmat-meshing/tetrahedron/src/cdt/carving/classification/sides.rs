use std::collections::BTreeSet;

use runmat_meshing_core::quality::predicate::PredicateSign;

use super::{flood, Classification};
use crate::cdt::carving::{
    error, CarvingWork, DelaunayCarvingError, DelaunayCarvingErrorKind,
    DelaunayConstraintFacetSide, DelaunayConstraints, DelaunayFacetRecovery,
};
use crate::cdt::topology::orient_materialized_volume_nodes;

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
            let use_count = uses.len();
            let points = oriented.map(|index| topology.nodes[index as usize]);
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
                let sign = orient_materialized_volume_nodes([
                    points[0],
                    points[1],
                    points[2],
                    topology.nodes[opposite as usize],
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
                            "facet incident tetrahedron is coplanar at solver coordinates",
                        ));
                    }
                };
                let intended = classification(side);
                flood(
                    topology,
                    blocked,
                    BTreeSet::from([tetrahedron]),
                    intended.clone(),
                    Some(recovered.constraint_index),
                    classifications,
                    work,
                )
                .map_err(|mut failure| {
                    failure.reason = format!(
                        "support {:?} with {} incident tetrahedra reached tetrahedron {tetrahedron} across exact side {sign:?}, intended {intended:?}: {}",
                        triangle.node_identities,
                        use_count,
                        failure.reason,
                    );
                    failure
                })?;
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

#[cfg(test)]
mod tests {
    use runmat_meshing_core::{
        quality::predicate::{orient3d, Point3Construction},
        StableDigest,
    };

    use super::*;
    use crate::cdt::DelaunayVolumeNode;

    #[test]
    fn facet_support_side_uses_solver_coordinates() {
        let vertices = [
            [2.380896857755243, 0.7533057313016712, 0.03495121082094581],
            [2.38085681630771, 0.7534792408036042, 0.034809126648409974],
            [2.380885189085703, 0.7533447406117164, 0.03493475896938903],
        ];
        let explicit = |identity, coordinates_m| {
            DelaunayVolumeNode::explicit(StableDigest::from_bytes([identity; 32]), coordinates_m)
        };
        let interior = DelaunayVolumeNode {
            identity: StableDigest::from_bytes([4; 32]),
            coordinates_m: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Position,
                _,
            >::new(std::array::from_fn(|axis| {
                (2.0 * vertices[0][axis] + vertices[1][axis] + vertices[2][axis]) * 0.25
            })),
            construction: Point3Construction::TriangleDyadic {
                vertices,
                weights_numerator: [2, 1, 1],
                weight_exponent: 2,
            },
        };
        let support = [
            explicit(1, vertices[0]),
            explicit(2, vertices[1]),
            interior,
            explicit(3, vertices[2]),
        ];

        assert_ne!(
            orient3d(support.map(|node| node.coordinates_m.canonical_value())).unwrap(),
            PredicateSign::Zero
        );
        assert_ne!(
            orient_materialized_volume_nodes(support).unwrap(),
            PredicateSign::Zero
        );
    }
}
