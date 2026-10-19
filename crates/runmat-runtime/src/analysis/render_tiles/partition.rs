//! Exact bounded tile partitioning from original topology.
use super::*;
pub fn partition(topology: &AnalysisRenderTopology) -> Result<RenderTilePartition, String> {
    let mut objects = Vec::new();
    let mut meshes = Vec::with_capacity(topology.meshes.len());
    for mesh in &topology.meshes {
        if mesh
            .triangle_volume_element_indices
            .iter()
            .flatten()
            .any(|index| *index >= topology.volume_element_count)
        {
            return Err(
                "render topology refers to a volume element outside the solver mesh".into(),
            );
        }
        if mesh.vertices.len() > u32::MAX as usize || mesh.triangles.len() > u32::MAX as usize {
            return Err("render mesh exceeds u32 index capacity".into());
        }
        let mesh_bounds = bounds(&mesh.vertices)?;
        let mut order = Vec::with_capacity(mesh.triangles.len());
        for (index, triangle) in mesh.triangles.iter().enumerate() {
            let positions = triangle.map(|vertex| mesh.vertices.get(vertex as usize));
            let [Some(a), Some(b), Some(c)] = positions else {
                return Err(format!(
                    "render mesh '{}' contains an invalid triangle",
                    mesh.mesh_id
                ));
            };
            let center = [0, 1, 2].map(|axis| (a[axis] + b[axis] + c[axis]) / 3.0);
            order.push((morton_key(center, mesh_bounds), index));
        }
        order.sort_unstable();
        let mut tiles = Vec::new();
        let mut referenced_vertices = vec![false; mesh.vertices.len()];
        for batch in order.chunks(MAX_TRIANGLES_PER_TILE) {
            let mut remap = HashMap::<u32, u32>::new();
            let mut tile = RenderTile {
                global_vertex_indices: Vec::new(),
                vertices: Vec::new(),
                vertex_volume_node_indices: Vec::new(),
                global_triangle_indices: Vec::new(),
                triangles: Vec::new(),
                triangle_volume_element_indices: Vec::new(),
            };
            for (_, triangle_index) in batch {
                let source = mesh.triangles[*triangle_index];
                let mut local = [0_u32; 3];
                for (corner, global) in source.into_iter().enumerate() {
                    local[corner] = if let Some(local) = remap.get(&global) {
                        *local
                    } else {
                        let local = u32::try_from(tile.vertices.len())
                            .map_err(|_| "render tile has too many vertices")?;
                        tile.global_vertex_indices.push(global);
                        tile.vertices.push(mesh.vertices[global as usize]);
                        tile.vertex_volume_node_indices.push(
                            mesh.vertex_volume_node_indices
                                .get(global as usize)
                                .copied()
                                .flatten(),
                        );
                        referenced_vertices[global as usize] = true;
                        remap.insert(global, local);
                        local
                    };
                }
                tile.global_triangle_indices.push(
                    u32::try_from(*triangle_index)
                        .map_err(|_| "render mesh has too many triangles")?,
                );
                tile.triangles.push(local);
                tile.triangle_volume_element_indices.push(
                    mesh.triangle_volume_element_indices
                        .get(*triangle_index)
                        .copied()
                        .flatten(),
                );
            }
            let tile_bounds = bounds(&tile.vertices)?;
            let bytes = serde_json::to_vec(&tile)
                .map_err(|error| format!("failed to encode render tile: {error}"))?;
            let digest = Digest::sha256(&bytes);
            tiles.push(RenderTileDescriptor {
                digest,
                bounds: tile_bounds,
                triangle_count: tile.triangles.len(),
            });
            objects.push((digest, bytes));
        }
        // Interior solver nodes are not referenced by boundary triangles, but
        // remain part of the exact topology needed for sections and exports.
        let unreferenced = referenced_vertices
            .iter()
            .enumerate()
            .filter_map(|(index, referenced)| (!referenced).then_some(index))
            .collect::<Vec<_>>();
        for batch in unreferenced.chunks(MAX_TRIANGLES_PER_TILE) {
            let tile = RenderTile {
                global_vertex_indices: batch.iter().map(|index| *index as u32).collect(),
                vertices: batch.iter().map(|index| mesh.vertices[*index]).collect(),
                vertex_volume_node_indices: batch
                    .iter()
                    .map(|index| {
                        mesh.vertex_volume_node_indices
                            .get(*index)
                            .copied()
                            .flatten()
                    })
                    .collect(),
                global_triangle_indices: Vec::new(),
                triangles: Vec::new(),
                triangle_volume_element_indices: Vec::new(),
            };
            let tile_bounds = bounds(&tile.vertices)?;
            let bytes = serde_json::to_vec(&tile)
                .map_err(|error| format!("failed to encode render tile: {error}"))?;
            let digest = Digest::sha256(&bytes);
            tiles.push(RenderTileDescriptor {
                digest,
                bounds: tile_bounds,
                triangle_count: 0,
            });
            objects.push((digest, bytes));
        }
        let root_budget = (DEFAULT_VIEW_TRIANGLE_BUDGET / topology.meshes.len().max(1))
            .clamp(1, MAX_TRIANGLES_PER_LOD_NODE);
        let (lod_nodes, lod_root) = build_lod_hierarchy(&tiles, &mut objects, root_budget)?;
        meshes.push(RenderTileMeshIndex {
            mesh_id: mesh.mesh_id.clone(),
            vertex_count: mesh.vertices.len(),
            triangle_count: mesh.triangles.len(),
            regions: mesh.regions.clone(),
            tiles,
            lod_nodes,
            lod_root,
        });
    }
    Ok((
        RenderTileIndex {
            schema_version: "analysis_render_tiles/v2".into(),
            topology_schema_version: topology.schema_version.clone(),
            source: topology.source,
            meshes,
            volume_element_count: topology.volume_element_count,
            field_summaries: Vec::new(),
        },
        objects,
    ))
}
