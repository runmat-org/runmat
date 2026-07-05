use std::collections::BTreeMap;

use runmat_meshing_core::{
    contracts::{PlcFacet, TetrahedronBoundaryFace, TetrahedronMesh, TopologyEntityId},
    quality::{
        predicate::{point_triangle_distance, triangle_area, Triangle3},
        tolerance::MeshingTolerance,
    },
};

use super::topology::sorted_topology_ids;

pub(super) fn boundary_source_face_area_coverage_complete(
    exterior_boundary_faces: &[&TetrahedronBoundaryFace],
    facet: &PlcFacet,
    node_coordinates: &BTreeMap<TopologyEntityId, [f64; 3]>,
    tolerance: MeshingTolerance,
) -> bool {
    let facet_node_ids = sorted_topology_ids(facet.node_ids.clone());
    if exterior_boundary_faces
        .iter()
        .any(|face| sorted_topology_ids(face.node_ids.clone()) == facet_node_ids)
    {
        return false;
    }
    let Some(facet_triangle) = triangle_for_node_ids(facet.node_ids.clone(), node_coordinates)
    else {
        return false;
    };
    let facet_area = triangle_area(facet_triangle);
    if !facet_area.is_finite() || facet_area <= tolerance.absolute_m * tolerance.absolute_m {
        return false;
    }

    let mut covered_area = 0.0_f64;
    let mut covered_face_count = 0_usize;
    for boundary_face in exterior_boundary_faces
        .iter()
        .filter(|face| face.source_face_id == facet.source_face_id)
    {
        let Some(child_triangle) =
            triangle_for_node_ids(boundary_face.node_ids.clone(), node_coordinates)
        else {
            continue;
        };
        if !child_triangle
            .iter()
            .all(|point| point_triangle_distance(*point, facet_triangle) <= tolerance.absolute_m)
        {
            continue;
        }
        let child_area = triangle_area(child_triangle);
        if !child_area.is_finite() || child_area <= tolerance.absolute_m * tolerance.absolute_m {
            continue;
        }
        covered_area += child_area;
        covered_face_count += 1;
    }

    covered_face_count > 0 && (covered_area - facet_area).abs() <= facet_area.max(1.0) * 1.0e-8
}

fn triangle_for_node_ids(
    node_ids: [TopologyEntityId; 3],
    node_coordinates: &BTreeMap<TopologyEntityId, [f64; 3]>,
) -> Option<Triangle3> {
    Some([
        *node_coordinates.get(&node_ids[0])?,
        *node_coordinates.get(&node_ids[1])?,
        *node_coordinates.get(&node_ids[2])?,
    ])
}

pub(super) fn tetrahedron_node_coordinates(
    tetrahedron_mesh: &TetrahedronMesh,
) -> BTreeMap<TopologyEntityId, [f64; 3]> {
    tetrahedron_mesh
        .nodes
        .iter()
        .map(|node| (node.node_id.clone(), node.coordinates_m))
        .collect()
}

pub(super) fn recovery_geometry_tolerance(
    node_coordinates: &BTreeMap<TopologyEntityId, [f64; 3]>,
) -> MeshingTolerance {
    let mut bounds_min = [f64::INFINITY; 3];
    let mut bounds_max = [f64::NEG_INFINITY; 3];
    for coordinates in node_coordinates.values() {
        for axis in 0..3 {
            bounds_min[axis] = bounds_min[axis].min(coordinates[axis]);
            bounds_max[axis] = bounds_max[axis].max(coordinates[axis]);
        }
    }
    if bounds_min
        .iter()
        .chain(bounds_max.iter())
        .all(|coordinate| coordinate.is_finite())
    {
        MeshingTolerance::from_bounds(bounds_min, bounds_max)
    } else {
        MeshingTolerance::default()
    }
}
