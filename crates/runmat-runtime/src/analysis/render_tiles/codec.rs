//! Authenticate tile payloads and reconstruct exact original topology.
use super::*;
pub fn decode_tile(descriptor: &RenderTileDescriptor, bytes: &[u8]) -> Result<RenderTile, String> {
    if Digest::sha256(bytes) != descriptor.digest {
        return Err("render tile differs from its content address".into());
    }
    let tile = serde_json::from_slice::<RenderTile>(bytes)
        .map_err(|error| format!("failed to decode render tile: {error}"))?;
    if tile.triangles.len() != descriptor.triangle_count
        || tile.global_triangle_indices.len() != tile.triangles.len()
        || tile.triangle_volume_element_indices.len() != tile.triangles.len()
        || tile.global_vertex_indices.len() != tile.vertices.len()
        || tile.vertex_volume_node_indices.len() != tile.vertices.len()
        || bounds(&tile.vertices)? != descriptor.bounds
        || tile
            .triangles
            .iter()
            .flatten()
            .any(|index| *index as usize >= tile.vertices.len())
    {
        return Err("render tile differs from its index descriptor".into());
    }
    Ok(tile)
}

pub fn reconstruct(
    index: &RenderTileIndex,
    mut read: impl FnMut(Digest) -> Result<Option<Vec<u8>>, String>,
) -> Result<AnalysisRenderTopology, String> {
    index.validate()?;
    let mut meshes = Vec::with_capacity(index.meshes.len());
    for mesh in &index.meshes {
        let mut vertices = vec![[0.0; 3]; mesh.vertex_count];
        let mut vertex_volume_node_indices = vec![None; mesh.vertex_count];
        let mut triangles = vec![[0; 3]; mesh.triangle_count];
        let mut triangle_volume_element_indices = vec![None; mesh.triangle_count];
        let mut seen_vertices = vec![false; mesh.vertex_count];
        let mut seen_triangles = vec![false; mesh.triangle_count];
        for descriptor in &mesh.tiles {
            let bytes = read(descriptor.digest)?.ok_or("render tile object is missing")?;
            let tile = decode_tile(descriptor, &bytes)?;
            for (local, global) in tile.global_vertex_indices.iter().enumerate() {
                let global = *global as usize;
                if global >= vertices.len() {
                    return Err("render tile vertex index is invalid".into());
                }
                if seen_vertices[global]
                    && (vertices[global] != tile.vertices[local]
                        || vertex_volume_node_indices[global]
                            != tile.vertex_volume_node_indices[local])
                {
                    return Err("render tile vertex is inconsistent".into());
                }
                seen_vertices[global] = true;
                vertices[global] = tile.vertices[local];
                vertex_volume_node_indices[global] = tile.vertex_volume_node_indices[local];
            }
            for (local, global) in tile.global_triangle_indices.iter().enumerate() {
                let global = *global as usize;
                if global >= triangles.len() || seen_triangles[global] {
                    return Err("render tile triangle index is invalid".into());
                }
                seen_triangles[global] = true;
                triangles[global] =
                    tile.triangles[local].map(|index| tile.global_vertex_indices[index as usize]);
                triangle_volume_element_indices[global] =
                    tile.triangle_volume_element_indices[local];
            }
        }
        if seen_vertices.iter().any(|seen| !seen) || seen_triangles.iter().any(|seen| !seen) {
            return Err("render tile index does not cover the source mesh".into());
        }
        meshes.push(AnalysisRenderMesh {
            mesh_id: mesh.mesh_id.clone(),
            vertices,
            triangles,
            regions: mesh.regions.clone(),
            vertex_volume_node_indices,
            triangle_volume_element_indices,
        });
    }
    Ok(AnalysisRenderTopology {
        schema_version: index.topology_schema_version.clone(),
        source: index.source,
        volume_element_count: index.volume_element_count,
        meshes,
    })
}

pub(super) fn bounds(vertices: &[[f64; 3]]) -> Result<[[f64; 3]; 2], String> {
    if vertices.is_empty() {
        return Ok([[0.0; 3]; 2]);
    }
    let mut min = [f64::INFINITY; 3];
    let mut max = [f64::NEG_INFINITY; 3];
    for vertex in vertices {
        for axis in 0..3 {
            if !vertex[axis].is_finite() {
                return Err("render topology contains a non-finite coordinate".into());
            }
            min[axis] = min[axis].min(vertex[axis]);
            max[axis] = max[axis].max(vertex[axis]);
        }
    }
    Ok([min, max])
}

pub(super) fn morton_key(point: [f64; 3], bounds: [[f64; 3]; 2]) -> u64 {
    let coordinate = [0, 1, 2].map(|axis| {
        let width = bounds[1][axis] - bounds[0][axis];
        if width == 0.0 {
            0_u32
        } else {
            (((point[axis] - bounds[0][axis]) / width).clamp(0.0, 1.0) * 1023.0) as u32
        }
    });
    let mut key = 0_u64;
    for bit in 0..10 {
        for axis in 0..3 {
            key |= (((coordinate[axis] >> bit) & 1) as u64) << (3 * bit + axis);
        }
    }
    key
}
