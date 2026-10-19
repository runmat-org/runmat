use super::*;
use runmat_meshing_core::CanonicalMeshingContract;

#[test]
fn canonical_quadratic_mesh_keeps_node_element_and_region_bindings() {
    let original = runmat_meshing_core::fixtures::canonical_tetrahedron_solver_mesh(
        runmat_meshing_core::ElementOrder::Tet10,
    );
    original.validate_canonical().unwrap();
    let topology = render_topology_from_solver_mesh(Some(&original)).unwrap();
    let boundary = &topology.meshes[0];
    assert_eq!(topology.volume_element_count, 1);
    assert_eq!(
        boundary.triangles.len(),
        original.topology.boundary_faces.len() * 4
    );
    assert_eq!(boundary.vertices.len(), original.topology.nodes.len());
    assert!(boundary
        .triangle_volume_element_indices
        .iter()
        .all(|index| *index == Some(0)));
    let mut used = vec![false; original.topology.nodes.len()];
    for triangle in &boundary.triangles {
        for index in triangle {
            used[*index as usize] = true;
            assert_eq!(
                boundary.vertex_volume_node_indices[*index as usize],
                Some(*index as usize)
            );
            assert_eq!(
                boundary.vertices[*index as usize],
                original.topology.nodes[*index as usize].coordinates_m
            );
        }
    }
    assert!(used.into_iter().all(|used| used));
    for face in &original.topology.boundary_faces {
        for entity in &face.provenance {
            let region = boundary
                .regions
                .iter()
                .find(|region| region.region_id == entity.source_topology_id)
                .unwrap();
            assert_eq!(
                region
                    .triangle_ranges
                    .iter()
                    .map(|range| range.count)
                    .sum::<u32>(),
                4
            );
            for range in &region.triangle_ranges {
                for triangle in
                    &boundary.triangles[range.start as usize..(range.start + range.count) as usize]
                {
                    assert!(triangle.iter().all(|index| face
                        .node_ids
                        .contains(&original.topology.nodes[*index as usize].node_id)));
                }
            }
        }
    }
}

#[test]
fn incomplete_boundary_is_rejected_instead_of_rendered_with_missing_faces() {
    let mut mesh = runmat_meshing_core::fixtures::canonical_tetrahedron_solver_mesh(
        runmat_meshing_core::ElementOrder::Tet10,
    );
    mesh.topology.boundary_faces[0].node_ids[3] = u64::MAX;
    assert!(render_topology_from_solver_mesh(Some(&mesh)).is_none());
    mesh.topology.boundary_faces[0].node_ids.pop();
    assert!(render_topology_from_solver_mesh(Some(&mesh)).is_none());
}

#[test]
fn quadratic_boundary_uses_original_midside_values_and_preserves_winding() {
    let ids = [11, 23, 37, 41, 53, 67];
    let indices = ids
        .into_iter()
        .enumerate()
        .map(|(index, id)| (id, index as u32))
        .collect();
    let triangles = boundary_triangles(&ids, &indices).unwrap();
    let positions = [
        [0., 0.],
        [1., 0.],
        [0., 1.],
        [0.5, 0.],
        [0.5, 0.5],
        [0., 0.5],
    ];
    // A quadratic mode can vanish at all corners and move at midside nodes.
    // The rendering must retain those original degrees of freedom.
    let original_values = [0., 0., 0., 1., 2., 3.];
    assert_eq!(triangles.len(), 4);
    let mut used = [false; 6];
    let mut area = 0.;
    for triangle in triangles {
        let [a, b, c] = triangle.map(|index| positions[index as usize]);
        let twice_area = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
        assert!(twice_area > 0.);
        area += twice_area / 2.;
        assert!(triangle
            .iter()
            .any(|index| original_values[*index as usize] != 0.));
        for index in triangle {
            used[index as usize] = true;
        }
    }
    assert_eq!(used, [true; 6]);
    assert_eq!(area, 0.5);
}

#[test]
fn linear_boundary_preserves_the_original_triangle_and_missing_nodes_fail_closed() {
    let indices = HashMap::from([(11, 5), (23, 8), (37, 2)]);
    assert_eq!(
        boundary_triangles(&[11, 23, 37], &indices).unwrap(),
        [[5, 8, 2]]
    );
    assert!(boundary_triangles(&[11, 23, 37, 41, 53, 67], &indices).is_none());
    assert!(boundary_triangles(&[11, 23], &indices).is_none());
}

#[test]
fn all_quadratic_pieces_retain_one_original_face_region() {
    let regions = render_regions_from_triangle_indices(BTreeMap::from([
        ("support".into(), vec![0, 1, 2, 3]),
        ("loaded".into(), vec![4, 5, 6, 7]),
    ]));
    assert_eq!(regions.len(), 2);
    assert_eq!(regions[0].region_id, "loaded");
    assert_eq!(
        regions[0].triangle_ranges,
        [AnalysisRenderTriangleRange { start: 4, count: 4 }]
    );
    assert_eq!(
        regions[1].triangle_ranges,
        [AnalysisRenderTriangleRange { start: 0, count: 4 }]
    );
}
