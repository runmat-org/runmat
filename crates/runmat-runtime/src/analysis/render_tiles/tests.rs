use super::selection::view_score;
use super::*;

#[test]
fn tiles_round_trip_exact_topology_including_interior_nodes() {
    let topology = AnalysisRenderTopology {
        schema_version: "source/v1".into(),
        source: AnalysisRenderTopologySource::AnalysisMesh,
        // Only element zero touches this face; the other six are interior.
        volume_element_count: 7,
        meshes: vec![AnalysisRenderMesh {
            mesh_id: "mesh".into(),
            vertices: vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            triangles: vec![[0, 1, 2]],
            regions: vec![],
            vertex_volume_node_indices: vec![Some(0), Some(1), Some(2), Some(3)],
            triangle_volume_element_indices: vec![Some(0)],
        }],
    };
    let (index, objects) = partition(&topology).unwrap();
    assert_eq!(index.volume_element_count, 7);
    assert_eq!(index.meshes[0].tiles.len(), 2);
    let restored = reconstruct(&index, |digest| {
        Ok(objects
            .iter()
            .find(|(id, _)| *id == digest)
            .map(|(_, bytes)| bytes.clone()))
    })
    .unwrap();
    assert_eq!(restored, topology);
}

#[test]
fn tile_digest_rejects_corruption() {
    let descriptor = RenderTileDescriptor {
        digest: Digest::sha256(b"good"),
        bounds: [[0.0; 3]; 2],
        triangle_count: 0,
    };
    assert!(decode_tile(&descriptor, b"bad").is_err());
}

#[test]
fn tile_budget_preserves_global_triangle_identity() {
    let count = MAX_TRIANGLES_PER_TILE + 1;
    let topology = AnalysisRenderTopology {
        schema_version: "source/v1".into(),
        source: AnalysisRenderTopologySource::AnalysisMesh,
        volume_element_count: count,
        meshes: vec![AnalysisRenderMesh {
            mesh_id: "mesh".into(),
            vertices: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            triangles: vec![[0, 1, 2]; count],
            regions: vec![],
            vertex_volume_node_indices: vec![Some(0), Some(1), Some(2)],
            triangle_volume_element_indices: (0..count).map(Some).collect(),
        }],
    };
    let (index, objects) = partition(&topology).unwrap();
    assert_eq!(index.meshes[0].tiles.len(), 2);
    let root = index.meshes[0].lod_root.expect("LOD root");
    assert!(index.meshes[0].lod_nodes[root].tile.triangle_count <= MAX_TRIANGLES_PER_LOD_NODE);
    let overview = index
        .plan_view(Some(RenderViewRequest {
            camera_position: [0.0, 0.0, 2.0],
            camera_target: [0.0, 0.0, 0.0],
            camera_up: [0.0, 1.0, 0.0],
            fov_y_radians: 0.8,
            orthographic_height: None,
            aspect_ratio: 1.0,
            viewport_height_px: 800.0,
            triangle_budget: MAX_TRIANGLES_PER_LOD_NODE,
            refine_above_px: 10_000.0,
        }))
        .unwrap();
    assert_eq!(overview.nodes, vec![(0, root)]);
    assert!(!overview.exact);
    let exact = index
        .plan_view(Some(RenderViewRequest {
            camera_position: [0.0, 0.0, 2.0],
            camera_target: [0.0, 0.0, 0.0],
            camera_up: [0.0, 1.0, 0.0],
            fov_y_radians: 0.8,
            orthographic_height: None,
            aspect_ratio: 1.0,
            viewport_height_px: 800.0,
            triangle_budget: count,
            refine_above_px: 1.0,
        }))
        .unwrap();
    assert!(exact.exact);
    assert_eq!(exact.triangle_count, count);
    let request = RenderViewRequest {
        camera_position: [0.0, 0.0, 2.0],
        camera_target: [0.0, 0.0, 0.0],
        camera_up: [0.0, 1.0, 0.0],
        fov_y_radians: 0.8,
        orthographic_height: None,
        aspect_ratio: 1.0,
        viewport_height_px: 800.0,
        triangle_budget: count,
        refine_above_px: 1.0,
    };
    let mut behind = index.meshes[0].lod_nodes[root].clone();
    behind.source_bounds = [[-0.5, -0.5, 4.0], [0.5, 0.5, 5.0]];
    assert_eq!(view_score(&behind, Some(request)), 0);
    let wide_orthographic = RenderViewRequest {
        orthographic_height: Some(10.0),
        ..request
    };
    let close_orthographic = RenderViewRequest {
        orthographic_height: Some(1.0),
        ..request
    };
    assert!(
        view_score(&index.meshes[0].lod_nodes[root], Some(close_orthographic))
            > view_score(&index.meshes[0].lod_nodes[root], Some(wide_orthographic))
    );
    let restored = reconstruct(&index, |digest| {
        Ok(objects
            .iter()
            .find(|(id, _)| *id == digest)
            .map(|(_, bytes)| bytes.clone()))
    })
    .unwrap();
    assert_eq!(restored, topology);
}

#[test]
fn clustered_vertices_keep_their_solver_node_positions() {
    let source = RenderTile {
        global_vertex_indices: vec![0, 1, 2, 3],
        vertices: vec![
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        vertex_volume_node_indices: vec![Some(0), Some(1), Some(2), Some(3)],
        global_triangle_indices: vec![0, 1],
        triangles: vec![[0, 2, 3], [1, 2, 3]],
        triangle_volume_element_indices: vec![Some(0), Some(1)],
    };
    let coarse = cluster_tile(&source, bounds(&source.vertices).unwrap(), 2).unwrap();
    for (local, global) in coarse.global_vertex_indices.iter().enumerate() {
        assert_eq!(coarse.vertices[local], source.vertices[*global as usize]);
        assert_eq!(
            coarse.vertex_volume_node_indices[local],
            source.vertex_volume_node_indices[*global as usize]
        );
    }
}

#[test]
fn one_large_exact_tile_has_a_bounded_overview() {
    let topology = AnalysisRenderTopology {
        schema_version: "source/v1".into(),
        source: AnalysisRenderTopologySource::AnalysisMesh,
        volume_element_count: 1,
        meshes: vec![AnalysisRenderMesh {
            mesh_id: "mesh".into(),
            vertices: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            triangles: vec![[0, 1, 2]; MAX_TRIANGLES_PER_TILE],
            regions: vec![],
            vertex_volume_node_indices: vec![Some(0), Some(1), Some(2)],
            triangle_volume_element_indices: vec![Some(0); MAX_TRIANGLES_PER_TILE],
        }],
    };
    let (index, _) = partition(&topology).unwrap();
    let root = index.meshes[0].lod_root.unwrap();
    assert_eq!(index.meshes[0].lod_nodes[root].children.len(), 1);
    assert!(index.meshes[0].lod_nodes[root].tile.triangle_count <= MAX_TRIANGLES_PER_LOD_NODE);
    let overview = index
        .plan_view(Some(RenderViewRequest {
            camera_position: [0.0, 0.0, 2.0],
            camera_target: [0.0, 0.0, 0.0],
            camera_up: [0.0, 1.0, 0.0],
            fov_y_radians: 0.8,
            orthographic_height: None,
            aspect_ratio: 1.0,
            viewport_height_px: 800.0,
            triangle_budget: MAX_TRIANGLES_PER_LOD_NODE,
            refine_above_px: 1.0,
        }))
        .unwrap();
    assert_eq!(overview.nodes, vec![(0, root)]);
}
