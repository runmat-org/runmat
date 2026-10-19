//! Construct conservative bounds and disposable LOD hierarchy.
use super::*;
pub(super) fn build_lod_hierarchy(
    tiles: &[RenderTileDescriptor],
    objects: &mut Vec<(Digest, Vec<u8>)>,
    root_budget: usize,
) -> Result<(Vec<RenderLodNode>, Option<usize>), String> {
    let mut object_positions = objects
        .iter()
        .enumerate()
        .map(|(index, (digest, _))| (*digest, index))
        .collect::<HashMap<_, _>>();
    let mut nodes = Vec::new();
    let mut level = Vec::new();
    for (tile_index, tile) in tiles.iter().enumerate() {
        if tile.triangle_count == 0 {
            continue;
        }
        level.push(nodes.len());
        nodes.push(RenderLodNode {
            tile: tile.clone(),
            source_bounds: tile.bounds,
            children: Vec::new(),
            exact_tile_index: Some(tile_index),
        });
    }
    while level.len() > 1
        || level
            .first()
            .is_some_and(|root| nodes[*root].tile.triangle_count > root_budget)
    {
        let mut parents = Vec::with_capacity(level.len().div_ceil(LOD_FANOUT));
        for group in level.chunks(LOD_FANOUT) {
            let mut child_tiles = Vec::with_capacity(group.len());
            for child in group {
                let descriptor = &nodes[*child].tile;
                let position = object_positions
                    .get(&descriptor.digest)
                    .ok_or("render LOD child object is missing")?;
                child_tiles.push(decode_tile(descriptor, &objects[*position].1)?);
            }
            let parent_tile = coarsen(&child_tiles, root_budget)?;
            let parent_bounds = bounds(&parent_tile.vertices)?;
            let bytes = serde_json::to_vec(&parent_tile)
                .map_err(|error| format!("failed to encode render LOD tile: {error}"))?;
            let digest = Digest::sha256(&bytes);
            let descriptor = RenderTileDescriptor {
                digest,
                bounds: parent_bounds,
                triangle_count: parent_tile.triangles.len(),
            };
            object_positions.entry(digest).or_insert_with(|| {
                let position = objects.len();
                objects.push((digest, bytes));
                position
            });
            let source_bounds = union_bounds(group.iter().map(|child| nodes[*child].source_bounds));
            parents.push(nodes.len());
            nodes.push(RenderLodNode {
                tile: descriptor,
                source_bounds,
                children: group.to_vec(),
                exact_tile_index: None,
            });
        }
        level = parents;
    }
    Ok((nodes, level.first().copied()))
}

pub(super) fn union_bounds(bounds: impl Iterator<Item = [[f64; 3]; 2]>) -> [[f64; 3]; 2] {
    let mut result = [[f64::INFINITY; 3], [f64::NEG_INFINITY; 3]];
    for bounds in bounds {
        for axis in 0..3 {
            result[0][axis] = result[0][axis].min(bounds[0][axis]);
            result[1][axis] = result[1][axis].max(bounds[1][axis]);
        }
    }
    result
}

pub(super) fn coarsen(
    children: &[RenderTile],
    target_triangles: usize,
) -> Result<RenderTile, String> {
    let mut merged = RenderTile {
        global_vertex_indices: Vec::new(),
        vertices: Vec::new(),
        vertex_volume_node_indices: Vec::new(),
        global_triangle_indices: Vec::new(),
        triangles: Vec::new(),
        triangle_volume_element_indices: Vec::new(),
    };
    for child in children {
        for (triangle_index, triangle) in child.triangles.iter().enumerate() {
            let mut indices = [0; 3];
            for (corner, vertex) in triangle.iter().enumerate() {
                let vertex = *vertex as usize;
                indices[corner] = u32::try_from(merged.vertices.len())
                    .map_err(|_| "render LOD tile exceeds u32 vertex capacity")?;
                merged.vertices.push(child.vertices[vertex]);
                merged
                    .global_vertex_indices
                    .push(child.global_vertex_indices[vertex]);
                merged
                    .vertex_volume_node_indices
                    .push(child.vertex_volume_node_indices[vertex]);
            }
            merged.triangles.push(indices);
            merged
                .global_triangle_indices
                .push(child.global_triangle_indices[triangle_index]);
            merged
                .triangle_volume_element_indices
                .push(child.triangle_volume_element_indices[triangle_index]);
        }
    }
    if merged.triangles.len() <= target_triangles {
        return Ok(merged);
    }
    let merged_bounds = bounds(&merged.vertices)?;
    let mut best = None;
    for resolution in 2..=32 {
        let candidate = cluster_tile(&merged, merged_bounds, resolution)?;
        if candidate.triangles.len() <= target_triangles && !candidate.triangles.is_empty() {
            best = Some(candidate);
        } else if candidate.triangles.len() > target_triangles {
            break;
        }
    }
    if let Some(best) = best {
        return Ok(best);
    }
    // Disconnected fine features can exceed the budget even at the coarsest
    // grid. Keep representative faces so the parent remains bounded.
    let stride = merged.triangles.len().div_ceil(target_triangles);
    let mut sampled = RenderTile {
        global_vertex_indices: Vec::new(),
        vertices: Vec::new(),
        vertex_volume_node_indices: Vec::new(),
        global_triangle_indices: Vec::new(),
        triangles: Vec::new(),
        triangle_volume_element_indices: Vec::new(),
    };
    for source in (0..merged.triangles.len()).step_by(stride) {
        let mut triangle = [0; 3];
        for (corner, vertex) in merged.triangles[source].iter().enumerate() {
            let vertex = *vertex as usize;
            triangle[corner] = sampled.vertices.len() as u32;
            sampled.vertices.push(merged.vertices[vertex]);
            sampled
                .global_vertex_indices
                .push(merged.global_vertex_indices[vertex]);
            sampled
                .vertex_volume_node_indices
                .push(merged.vertex_volume_node_indices[vertex]);
        }
        sampled.triangles.push(triangle);
        sampled
            .global_triangle_indices
            .push(merged.global_triangle_indices[source]);
        sampled
            .triangle_volume_element_indices
            .push(merged.triangle_volume_element_indices[source]);
    }
    Ok(sampled)
}

pub(super) fn cluster_tile(
    source: &RenderTile,
    source_bounds: [[f64; 3]; 2],
    resolution: u16,
) -> Result<RenderTile, String> {
    let mut cells = HashMap::<[u16; 3], u32>::new();
    let mut tile = RenderTile {
        global_vertex_indices: Vec::new(),
        vertices: Vec::new(),
        vertex_volume_node_indices: Vec::new(),
        global_triangle_indices: Vec::new(),
        triangles: Vec::new(),
        triangle_volume_element_indices: Vec::new(),
    };
    let mut remap = Vec::with_capacity(source.vertices.len());
    for (vertex, position) in source.vertices.iter().enumerate() {
        let key = [0, 1, 2].map(|axis| {
            let width = source_bounds[1][axis] - source_bounds[0][axis];
            if width == 0.0 {
                0
            } else {
                (((position[axis] - source_bounds[0][axis]) / width).clamp(0.0, 1.0)
                    * f64::from(resolution - 1)) as u16
            }
        });
        let local = if let Some(local) = cells.get(&key) {
            *local
        } else {
            let local = u32::try_from(tile.vertices.len())
                .map_err(|_| "render LOD tile exceeds u32 vertex capacity")?;
            cells.insert(key, local);
            // The representative keeps its original solver-node position and
            // identity. Averaging positions while retaining one node index
            // would plot that node's field at a different physical point.
            tile.vertices.push(*position);
            tile.global_vertex_indices
                .push(source.global_vertex_indices[vertex]);
            tile.vertex_volume_node_indices
                .push(source.vertex_volume_node_indices[vertex]);
            local
        };
        remap.push(local);
    }
    let mut seen = HashSet::new();
    for (triangle_index, triangle) in source.triangles.iter().enumerate() {
        let mapped = triangle.map(|vertex| remap[vertex as usize]);
        if mapped[0] == mapped[1] || mapped[1] == mapped[2] || mapped[0] == mapped[2] {
            continue;
        }
        let mut key = mapped;
        key.sort_unstable();
        if !seen.insert(key) {
            continue;
        }
        tile.triangles.push(mapped);
        tile.global_triangle_indices
            .push(source.global_triangle_indices[triangle_index]);
        tile.triangle_volume_element_indices
            .push(source.triangle_volume_element_indices[triangle_index]);
    }
    Ok(tile)
}
