use std::collections::{BTreeMap, BTreeSet};

use runmat_meshing_core::quality::predicate::{
    distance_squared, orient3d, point_in_closed_triangle_surface, Point3, PointInClosedSurface,
    PredicateSign, Triangle3,
};
use runmat_meshing_core::quality::tolerance::MeshingTolerance;

use super::{
    ConstrainedCavity, ConstrainedCavityNode, ConstrainedCavityRefillError,
    ConstrainedCavityRefillOptions,
};

pub(super) fn boundary_node_coordinates(
    cavity: &ConstrainedCavity,
    nodes: &[ConstrainedCavityNode],
) -> Result<BTreeMap<u32, Point3>, ConstrainedCavityRefillError> {
    let coordinates = nodes
        .iter()
        .map(|node| (node.node_id, node.coordinates_m.canonical_value()))
        .collect::<BTreeMap<_, _>>();
    for face in &cavity.boundary_faces {
        for node_id in face.node_ids {
            if !coordinates.contains_key(&node_id) {
                return Err(ConstrainedCavityRefillError::MissingBoundaryNode { node_id });
            }
        }
    }
    Ok(coordinates)
}

pub(super) fn cavity_boundary_node_ids(cavity: &ConstrainedCavity) -> BTreeSet<u32> {
    cavity
        .boundary_faces
        .iter()
        .flat_map(|face| face.node_ids)
        .collect()
}

pub(super) fn candidate_respects_protected_boundary_distance(
    cavity: &ConstrainedCavity,
    boundary_nodes: &BTreeMap<u32, Point3>,
    point: Point3,
    options: ConstrainedCavityRefillOptions,
) -> bool {
    if options.min_protected_node_distance_m <= 0.0 || cavity.protected_node_ids.is_empty() {
        return true;
    }
    let min_distance_squared = options.min_protected_node_distance_m.powi(2);
    cavity.protected_node_ids.iter().all(|node_id| {
        boundary_nodes.get(node_id).is_none_or(|protected_point| {
            distance_squared(point, *protected_point) > min_distance_squared
        })
    })
}

pub(super) fn cavity_boundary_triangles(
    cavity: &ConstrainedCavity,
    nodes: &BTreeMap<u32, Point3>,
) -> Result<Vec<Triangle3>, ConstrainedCavityRefillError> {
    cavity
        .boundary_faces
        .iter()
        .map(|face| {
            Ok([
                *nodes.get(&face.node_ids[0]).ok_or(
                    ConstrainedCavityRefillError::MissingBoundaryNode {
                        node_id: face.node_ids[0],
                    },
                )?,
                *nodes.get(&face.node_ids[1]).ok_or(
                    ConstrainedCavityRefillError::MissingBoundaryNode {
                        node_id: face.node_ids[1],
                    },
                )?,
                *nodes.get(&face.node_ids[2]).ok_or(
                    ConstrainedCavityRefillError::MissingBoundaryNode {
                        node_id: face.node_ids[2],
                    },
                )?,
            ])
        })
        .collect()
}

pub(super) fn resolved_cavity_tolerance(
    boundary_nodes: &BTreeMap<u32, Point3>,
    configured: MeshingTolerance,
) -> MeshingTolerance {
    let scale_m = boundary_nodes
        .values()
        .flat_map(|point| point.iter())
        .map(|coordinate| coordinate.abs())
        .fold(1.0_f64, f64::max);
    MeshingTolerance {
        absolute_m: runmat_types::physical::Length::new(configured.length_epsilon(scale_m)),
        relative: configured.relative,
    }
}

pub(super) fn classify_cavity_point(
    cavity: &ConstrainedCavity,
    boundary_nodes: &BTreeMap<u32, Point3>,
    boundary_triangles: &[Triangle3],
    point: Point3,
    tolerance: MeshingTolerance,
) -> PointInClosedSurface {
    if let Some(classification) = classify_convex_cavity_point(cavity, boundary_nodes, point) {
        return classification;
    }
    point_in_closed_triangle_surface(point, boundary_triangles, tolerance)
}

fn classify_convex_cavity_point(
    cavity: &ConstrainedCavity,
    boundary_nodes: &BTreeMap<u32, Point3>,
    point: Point3,
) -> Option<PointInClosedSurface> {
    let node_ids = cavity_boundary_node_ids(cavity);
    if node_ids.len() < 4 || cavity.boundary_faces.len() < 4 {
        return None;
    }
    let mut on_boundary = false;
    for face in &cavity.boundary_faces {
        let triangle = face
            .node_ids
            .map(|node_id| boundary_nodes.get(&node_id).copied())
            .into_iter()
            .collect::<Option<Vec<_>>>()?;
        let mut interior_sign = None;
        for node_id in node_ids
            .iter()
            .filter(|node_id| !face.node_ids.contains(node_id))
        {
            let other = *boundary_nodes.get(node_id)?;
            let sign = orient3d([triangle[0], triangle[1], triangle[2], other]).ok()?;
            if sign == PredicateSign::Zero {
                continue;
            }
            if interior_sign.is_some_and(|interior| interior != sign) {
                return None;
            }
            interior_sign = Some(sign);
        }
        let interior_sign = interior_sign?;
        let query_sign = orient3d([triangle[0], triangle[1], triangle[2], point]).ok()?;
        if query_sign == PredicateSign::Zero {
            on_boundary = true;
        } else if query_sign != interior_sign {
            return Some(PointInClosedSurface::Outside);
        }
    }
    Some(if on_boundary {
        PointInClosedSurface::OnBoundary
    } else {
        PointInClosedSurface::Inside
    })
}

pub(super) fn cavity_boundary_node_centroid(
    cavity: &ConstrainedCavity,
    boundary_nodes: &BTreeMap<u32, Point3>,
) -> Option<Point3> {
    let node_ids = cavity_boundary_node_ids(cavity);
    if node_ids.is_empty() {
        return None;
    }
    let mut centroid = [0.0_f64; 3];
    for node_id in &node_ids {
        let point = boundary_nodes.get(node_id)?;
        centroid[0] += point[0];
        centroid[1] += point[1];
        centroid[2] += point[2];
    }
    let scale = 1.0 / node_ids.len() as f64;
    Some([
        centroid[0] * scale,
        centroid[1] * scale,
        centroid[2] * scale,
    ])
}

pub(super) fn next_cavity_node_id(cavity: &ConstrainedCavity) -> u32 {
    cavity_boundary_node_ids(cavity)
        .into_iter()
        .max()
        .unwrap_or(0)
        .saturating_add(1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cavity::constrained::{
        generate_constrained_cavity_interior_steiner_candidates,
        retriangulate_constrained_cavity_from_nodes_with_diagnostics,
        ConstrainedCavityBoundaryFace, ConstrainedCavityNode, ConstrainedCavityRefillBudget,
        ConstrainedCavityRefillOptions, ConstrainedCavitySteinerCandidateBudget,
    };
    use runmat_meshing_core::{quality::predicate::tetrahedron_volume, NeverCancelled};

    fn face(node_ids: [u32; 3]) -> ConstrainedCavityBoundaryFace {
        ConstrainedCavityBoundaryFace {
            node_ids,
            outside_tetrahedron_ids: Vec::new(),
            source_face_id: None,
            source_edge_ids: [None; 3],
            region_ids: Vec::new(),
        }
    }

    #[test]
    fn exact_convex_classification_handles_a_triangular_bipyramid() {
        let cavity = ConstrainedCavity {
            removed_tetrahedron_ids: vec![0, 1],
            boundary_faces: vec![
                face([0, 1, 3]),
                face([1, 2, 3]),
                face([2, 0, 3]),
                face([0, 4, 1]),
                face([1, 4, 2]),
                face([2, 4, 0]),
            ],
            protected_node_ids: Vec::new(),
            target_volume_m3: 1.0 / 3.0,
        };
        let nodes = BTreeMap::from([
            (0, [0.0, 0.0, 0.0]),
            (1, [1.0, 0.0, 0.0]),
            (2, [0.0, 1.0, 0.0]),
            (3, [0.0, 0.0, 1.0]),
            (4, [0.0, 0.0, -1.0]),
        ]);

        assert_eq!(
            classify_convex_cavity_point(&cavity, &nodes, [0.25, 0.25, 0.0]),
            Some(PointInClosedSurface::Inside)
        );
        assert_eq!(
            classify_convex_cavity_point(&cavity, &nodes, [0.25, 0.25, 0.5]),
            Some(PointInClosedSurface::OnBoundary)
        );
        assert_eq!(
            classify_convex_cavity_point(&cavity, &nodes, [0.5, 0.5, 0.5]),
            Some(PointInClosedSurface::Outside)
        );
    }

    #[test]
    fn torus_sliver_classification_is_invariant_to_node_remapping() {
        let faces = [
            [0, 1, 3],
            [0, 1, 4],
            [0, 3, 4],
            [1, 2, 3],
            [1, 2, 4],
            [2, 3, 4],
        ];
        let mut cavity = ConstrainedCavity {
            removed_tetrahedron_ids: vec![0, 1],
            boundary_faces: faces.into_iter().map(face).collect(),
            protected_node_ids: vec![2, 3, 4],
            target_volume_m3: 1.0,
        };
        let nodes = BTreeMap::from([
            (
                0,
                [
                    -0.41447178586538996,
                    -2.372786459114944,
                    -0.2831125732812946,
                ],
            ),
            (
                1,
                [
                    -0.4144913511363605,
                    -2.3730164987601645,
                    -0.2829423906344274,
                ],
            ),
            (
                2,
                [
                    -0.41453983898180924,
                    -2.3735865970113634,
                    -0.2825206336400174,
                ],
            ),
            (
                3,
                [
                    -0.4109888296816472,
                    -2.3726248366903917,
                    -0.28370451292257165,
                ],
            ),
            (
                4,
                [
                    -0.41795474204913274,
                    -2.3729480815394957,
                    -0.2825206336400174,
                ],
            ),
        ]);
        cavity.target_volume_m3 = tetrahedron_volume([nodes[&0], nodes[&1], nodes[&3], nodes[&4]])
            + tetrahedron_volume([nodes[&1], nodes[&2], nodes[&3], nodes[&4]]);
        let candidates = [
            [
                -0.4156925211004131,
                -2.373347079389859,
                -0.28288553950062456,
            ],
            [
                -0.41567024764048954,
                -2.3730825643949722,
                -0.28308587055029294,
            ],
        ];
        let tolerance = MeshingTolerance {
            absolute_m: runmat_types::physical::Length::new(0.0),
            relative: 0.0,
        };
        let triangles = cavity_boundary_triangles(&cavity, &nodes).unwrap();
        let classifications = candidates.map(|candidate| {
            classify_cavity_point(&cavity, &nodes, &triangles, candidate, tolerance)
        });

        let remapped = ConstrainedCavity {
            removed_tetrahedron_ids: cavity.removed_tetrahedron_ids.clone(),
            boundary_faces: cavity
                .boundary_faces
                .iter()
                .map(|boundary| face(boundary.node_ids.map(|node| node + 1)))
                .collect(),
            protected_node_ids: cavity
                .protected_node_ids
                .iter()
                .map(|node| node + 1)
                .collect(),
            target_volume_m3: cavity.target_volume_m3,
        };
        let remapped_nodes = nodes
            .into_iter()
            .map(|(node, coordinates)| (node + 1, coordinates))
            .collect::<BTreeMap<_, _>>();
        let remapped_triangles = cavity_boundary_triangles(&remapped, &remapped_nodes).unwrap();
        let remapped_classifications = candidates.map(|candidate| {
            classify_cavity_point(
                &remapped,
                &remapped_nodes,
                &remapped_triangles,
                candidate,
                tolerance,
            )
        });

        assert_eq!(classifications, remapped_classifications);
        assert_eq!(
            classifications,
            [PointInClosedSurface::Outside, PointInClosedSurface::Outside]
        );
        let centroid = cavity_boundary_node_centroid(&remapped, &remapped_nodes)
            .expect("the cavity has boundary nodes");
        assert_eq!(
            classify_cavity_point(
                &remapped,
                &remapped_nodes,
                &remapped_triangles,
                centroid,
                tolerance,
            ),
            PointInClosedSurface::Outside
        );
        let points = remapped_nodes.values().copied().collect::<Vec<_>>();
        let signs = (0..points.len())
            .map(|omitted| {
                let tetrahedron = points
                    .iter()
                    .enumerate()
                    .filter_map(|(index, point)| (index != omitted).then_some(*point))
                    .collect::<Vec<_>>();
                orient3d([
                    tetrahedron[0],
                    tetrahedron[1],
                    tetrahedron[2],
                    tetrahedron[3],
                ])
                .unwrap()
            })
            .collect::<Vec<_>>();
        assert_eq!(
            signs,
            vec![
                PredicateSign::Negative,
                PredicateSign::Negative,
                PredicateSign::Negative,
                PredicateSign::Positive,
                PredicateSign::Negative,
            ]
        );
        let mut interior_witnesses = Vec::new();
        for first in 0..points.len() {
            for second in (first + 1)..points.len() {
                let midpoint =
                    std::array::from_fn(|axis| (points[first][axis] + points[second][axis]) * 0.5);
                if classify_cavity_point(
                    &remapped,
                    &remapped_nodes,
                    &remapped_triangles,
                    midpoint,
                    tolerance,
                ) == PointInClosedSurface::Inside
                {
                    interior_witnesses.push(midpoint);
                }
                for third in (second + 1)..points.len() {
                    let centroid = std::array::from_fn(|axis| {
                        (points[first][axis] + points[second][axis] + points[third][axis]) / 3.0
                    });
                    if classify_cavity_point(
                        &remapped,
                        &remapped_nodes,
                        &remapped_triangles,
                        centroid,
                        tolerance,
                    ) == PointInClosedSurface::Inside
                    {
                        interior_witnesses.push(centroid);
                    }
                }
            }
        }
        assert!(!interior_witnesses.is_empty());

        let options = ConstrainedCavityRefillOptions {
            min_volume_m3: f64::MIN_POSITIVE,
            max_aspect_ratio: f64::MAX,
            min_scaled_jacobian: 0.0,
            classification_tolerance: tolerance,
            ..ConstrainedCavityRefillOptions::default()
        };
        let mut refill_nodes = remapped_nodes
            .iter()
            .map(|(node_id, coordinates_m)| ConstrainedCavityNode {
                node_id: *node_id,
                coordinates_m: runmat_types::physical::Quantity::<
                    runmat_types::physical::kind::Position,
                    _,
                >::new(*coordinates_m),
            })
            .collect::<Vec<_>>();
        let generated = generate_constrained_cavity_interior_steiner_candidates(
            &remapped,
            &refill_nodes,
            options,
            ConstrainedCavitySteinerCandidateBudget {
                maximum_candidates: 2_048,
                maximum_evaluations: 1_000_000,
                cancellation_check_interval: 1_024,
            },
            &NeverCancelled,
        )
        .unwrap();
        let mut next_node_id = 6u32;
        for coordinates_m in generated {
            refill_nodes.push(ConstrainedCavityNode {
                node_id: next_node_id,
                coordinates_m: runmat_types::physical::Quantity::<
                    runmat_types::physical::kind::Position,
                    _,
                >::new(coordinates_m),
            });
            next_node_id += 1;
        }
        let result = retriangulate_constrained_cavity_from_nodes_with_diagnostics(
            &remapped,
            &refill_nodes,
            options,
            ConstrainedCavityRefillBudget::default(),
            &NeverCancelled,
        )
        .unwrap();
        assert!(result.refill.is_some(), "{result:#?}");
    }
}
