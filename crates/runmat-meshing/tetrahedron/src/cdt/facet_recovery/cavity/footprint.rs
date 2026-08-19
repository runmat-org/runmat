//! Exact-sign tests that keep a facet cavity bounded to the authored triangular footprint.

use runmat_meshing_core::quality::predicate::{orient2d, orient3d, PredicateSign};

use super::{super::FacetRecoveryWork, invalid_topology};
use crate::cdt::{DelaunayFacetRecoveryError, DelaunayVolumeTopology};

pub(super) fn facet_footprint_crosses_face(
    topology: &DelaunayVolumeTopology,
    facet: [u32; 3],
    face: [u32; 3],
    constraint_index: u32,
    work: &mut FacetRecoveryWork<'_>,
) -> Result<bool, DelaunayFacetRecoveryError> {
    let coordinates = |node: u32| topology.nodes[node as usize].coordinates_m;
    let facet_points = facet.map(coordinates);
    let face_points = face.map(coordinates);
    let face_plane_signs = [
        orientation(
            [
                facet_points[0],
                facet_points[1],
                facet_points[2],
                face_points[0],
            ],
            constraint_index,
            work,
        )?,
        orientation(
            [
                facet_points[0],
                facet_points[1],
                facet_points[2],
                face_points[1],
            ],
            constraint_index,
            work,
        )?,
        orientation(
            [
                facet_points[0],
                facet_points[1],
                facet_points[2],
                face_points[2],
            ],
            constraint_index,
            work,
        )?,
    ];
    if face_plane_signs
        .iter()
        .all(|sign| *sign == PredicateSign::Zero)
    {
        return coplanar_triangles_overlap_interior(
            facet_points,
            face_points,
            constraint_index,
            work,
        );
    }
    for edge in triangle_edges(face) {
        if segment_hits_triangle_interior(topology, edge, facet, constraint_index, work)? {
            return Ok(true);
        }
    }
    for edge in triangle_edges(facet) {
        if segment_hits_triangle_interior(topology, edge, face, constraint_index, work)? {
            return Ok(true);
        }
    }
    Ok(false)
}

fn coplanar_triangles_overlap_interior(
    left: [[f64; 3]; 3],
    right: [[f64; 3]; 3],
    constraint_index: u32,
    work: &mut FacetRecoveryWork<'_>,
) -> Result<bool, DelaunayFacetRecoveryError> {
    let axes = projection_axes(left, constraint_index, work)?;
    let project = |point: [f64; 3]| [point[axes[0]], point[axes[1]]];
    let left = left.map(project);
    let right = right.map(project);
    triangles_overlap_across_every_edge(left, right, constraint_index, work).and_then(|overlap| {
        if overlap {
            triangles_overlap_across_every_edge(right, left, constraint_index, work)
        } else {
            Ok(false)
        }
    })
}

fn triangles_overlap_across_every_edge(
    reference: [[f64; 2]; 3],
    candidate: [[f64; 2]; 3],
    constraint_index: u32,
    work: &mut FacetRecoveryWork<'_>,
) -> Result<bool, DelaunayFacetRecoveryError> {
    let winding = planar_orientation(reference, constraint_index, work)?;
    if winding == PredicateSign::Zero {
        return Err(invalid_topology(
            constraint_index,
            "facet-footprint triangle is exactly collinear",
        ));
    }
    for edge in triangle_edges([0, 1, 2]) {
        let mut reaches_interior = false;
        for point in candidate {
            if planar_orientation(
                [
                    reference[edge[0] as usize],
                    reference[edge[1] as usize],
                    point,
                ],
                constraint_index,
                work,
            )? == winding
            {
                reaches_interior = true;
                break;
            }
        }
        if !reaches_interior {
            return Ok(false);
        }
    }
    Ok(true)
}

fn segment_hits_triangle_interior(
    topology: &DelaunayVolumeTopology,
    segment: [u32; 2],
    triangle: [u32; 3],
    constraint_index: u32,
    work: &mut FacetRecoveryWork<'_>,
) -> Result<bool, DelaunayFacetRecoveryError> {
    let point = |node: u32| topology.nodes[node as usize].coordinates_m;
    let plane_signs = [
        orientation(
            [
                point(triangle[0]),
                point(triangle[1]),
                point(triangle[2]),
                point(segment[0]),
            ],
            constraint_index,
            work,
        )?,
        orientation(
            [
                point(triangle[0]),
                point(triangle[1]),
                point(triangle[2]),
                point(segment[1]),
            ],
            constraint_index,
            work,
        )?,
    ];
    for (endpoint, sign) in segment.into_iter().zip(plane_signs) {
        if sign == PredicateSign::Zero
            && point_strictly_inside_triangle(topology, endpoint, triangle, constraint_index, work)?
        {
            return Ok(true);
        }
    }
    if !opposite_nonzero(plane_signs[0], plane_signs[1]) {
        return Ok(false);
    }

    let around = [
        orientation(
            [
                point(segment[0]),
                point(segment[1]),
                point(triangle[0]),
                point(triangle[1]),
            ],
            constraint_index,
            work,
        )?,
        orientation(
            [
                point(segment[0]),
                point(segment[1]),
                point(triangle[1]),
                point(triangle[2]),
            ],
            constraint_index,
            work,
        )?,
        orientation(
            [
                point(segment[0]),
                point(segment[1]),
                point(triangle[2]),
                point(triangle[0]),
            ],
            constraint_index,
            work,
        )?,
    ];
    Ok(around.iter().all(|sign| *sign == PredicateSign::Positive)
        || around.iter().all(|sign| *sign == PredicateSign::Negative))
}

fn point_strictly_inside_triangle(
    topology: &DelaunayVolumeTopology,
    node: u32,
    triangle: [u32; 3],
    constraint_index: u32,
    work: &mut FacetRecoveryWork<'_>,
) -> Result<bool, DelaunayFacetRecoveryError> {
    let coordinates = |node: u32| topology.nodes[node as usize].coordinates_m;
    let points = triangle.map(coordinates);
    let axes = projection_axes(points, constraint_index, work)?;
    let projected = |point: [f64; 3]| [point[axes[0]], point[axes[1]]];
    let query = projected(coordinates(node));
    let signs = [
        planar_orientation(
            [projected(points[0]), projected(points[1]), query],
            constraint_index,
            work,
        )?,
        planar_orientation(
            [projected(points[1]), projected(points[2]), query],
            constraint_index,
            work,
        )?,
        planar_orientation(
            [projected(points[2]), projected(points[0]), query],
            constraint_index,
            work,
        )?,
    ];
    Ok(signs.iter().all(|sign| *sign == PredicateSign::Positive)
        || signs.iter().all(|sign| *sign == PredicateSign::Negative))
}

fn projection_axes(
    triangle: [[f64; 3]; 3],
    constraint_index: u32,
    work: &mut FacetRecoveryWork<'_>,
) -> Result<[usize; 2], DelaunayFacetRecoveryError> {
    for axes in [[0, 1], [0, 2], [1, 2]] {
        if planar_orientation(
            triangle.map(|point| [point[axes[0]], point[axes[1]]]),
            constraint_index,
            work,
        )? != PredicateSign::Zero
        {
            return Ok(axes);
        }
    }
    Err(invalid_topology(
        constraint_index,
        "facet-footprint triangle is exactly collinear",
    ))
}

fn orientation(
    points: [[f64; 3]; 4],
    constraint_index: u32,
    work: &mut FacetRecoveryWork<'_>,
) -> Result<PredicateSign, DelaunayFacetRecoveryError> {
    work.search_step(constraint_index)?;
    orient3d(points).map_err(|failure| {
        invalid_topology(
            constraint_index,
            format!("facet-footprint spatial predicate failed: {failure:?}"),
        )
    })
}

fn planar_orientation(
    points: [[f64; 2]; 3],
    constraint_index: u32,
    work: &mut FacetRecoveryWork<'_>,
) -> Result<PredicateSign, DelaunayFacetRecoveryError> {
    work.search_step(constraint_index)?;
    orient2d(points).map_err(|failure| {
        invalid_topology(
            constraint_index,
            format!("facet-footprint planar predicate failed: {failure:?}"),
        )
    })
}

fn opposite_nonzero(left: PredicateSign, right: PredicateSign) -> bool {
    matches!(
        (left, right),
        (PredicateSign::Positive, PredicateSign::Negative)
            | (PredicateSign::Negative, PredicateSign::Positive)
    )
}

fn triangle_edges(triangle: [u32; 3]) -> [[u32; 2]; 3] {
    [
        [triangle[0], triangle[1]],
        [triangle[1], triangle[2]],
        [triangle[2], triangle[0]],
    ]
}

#[cfg(test)]
mod tests {
    use runmat_meshing_core::NeverCancelled;

    use super::*;
    use crate::cdt::DelaunayFacetRecoveryOptions;

    fn overlaps(left: [[f64; 3]; 3], right: [[f64; 3]; 3]) -> bool {
        let mut work =
            FacetRecoveryWork::new(DelaunayFacetRecoveryOptions::default(), &NeverCancelled);
        coplanar_triangles_overlap_interior(left, right, 0, &mut work).unwrap()
    }

    #[test]
    fn coplanar_footprints_require_positive_area_overlap() {
        let target = [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [0.0, 4.0, 0.0]];
        assert!(overlaps(
            target,
            [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [2.0, 2.0, 0.0]],
        ));
        assert!(!overlaps(
            target,
            [[0.0, 0.0, 0.0], [-2.0, 0.0, 0.0], [0.0, -2.0, 0.0]],
        ));
        assert!(!overlaps(
            target,
            [[1.0, 0.0, 0.0], [3.0, 0.0, 0.0], [2.0, -2.0, 0.0]],
        ));
    }
}
