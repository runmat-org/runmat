use super::*;

#[test]
fn occupied_cells_keep_largest_connected_component() {
    let grid = StructuredGrid {
        x: vec![0.0, 1.0, 2.0, 3.0, 4.0],
        y: vec![0.0, 1.0],
        z: vec![0.0, 1.0],
    };
    let occupied = vec![true, true, false, true];

    let retained = largest_connected_occupied_component(&grid, occupied);

    assert_eq!(retained, vec![true, true, false, false]);
}
#[test]
fn boundary_triangle_centroids_mark_intersected_cells_occupied() {
    let input = BoundaryMeshInput {
        mesh_id: "surface".to_string(),
        source_geometry_id: "geo_surface".to_string(),
        source_geometry_revision: 1,
        source_geometry_sha256: None,
        vertices: vec![[1.2, 0.1, 0.1], [1.8, 0.1, 0.1], [1.5, 0.8, 0.1]],
        triangles: vec![BoundaryMeshTriangle {
            triangle_id: 0,
            node_ids: [0, 1, 2],
            region_ids: vec!["region".to_string()],
            material_region_ids: Vec::new(),
            provenance: Vec::new(),
        }],
        region_ids: vec!["region".to_string()],
        material_region_ids: Vec::new(),
        bounds_min_m: [0.0, 0.0, 0.0],
        bounds_max_m: [2.0, 1.0, 1.0],
    };
    let grid = StructuredGrid {
        x: vec![0.0, 1.0, 2.0],
        y: vec![0.0, 1.0],
        z: vec![0.0, 1.0],
    };

    let cells = boundary_triangle_centroid_cells(&input, &grid);

    assert_eq!(cells, vec![false, true]);
}
