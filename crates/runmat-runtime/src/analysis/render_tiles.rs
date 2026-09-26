//! Immutable, spatially ordered surface pieces for out-of-core result views.
//! The source topology remains exact; these pieces retain original node and
//! triangle identities so fields can be resolved without serializing a plot.

use std::collections::{BinaryHeap, HashMap, HashSet};

use runmat_execution::Digest;
use serde::{Deserialize, Serialize};

use super::{
    AnalysisRenderMesh, AnalysisRenderRegion, AnalysisRenderTopology, AnalysisRenderTopologySource,
};

pub const MAX_TRIANGLES_PER_TILE: usize = 8192;
pub const MAX_TRIANGLES_PER_LOD_NODE: usize = 1024;
pub const DEFAULT_VIEW_TRIANGLE_BUDGET: usize = 250_000;
const LOD_FANOUT: usize = 4;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RenderTileIndex {
    pub schema_version: String,
    pub topology_schema_version: String,
    pub source: AnalysisRenderTopologySource,
    pub meshes: Vec<RenderTileMeshIndex>,
    pub volume_element_count: usize,
    pub field_summaries: Vec<RenderFieldSummary>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RenderFieldSummary {
    pub field_id: String,
    pub minimum: f64,
    pub maximum: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RenderTileMeshIndex {
    pub mesh_id: String,
    pub vertex_count: usize,
    pub triangle_count: usize,
    pub regions: Vec<AnalysisRenderRegion>,
    pub tiles: Vec<RenderTileDescriptor>,
    pub lod_nodes: Vec<RenderLodNode>,
    pub lod_root: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RenderLodNode {
    pub tile: RenderTileDescriptor,
    pub source_bounds: [[f64; 3]; 2],
    pub children: Vec<usize>,
    pub exact_tile_index: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RenderTileDescriptor {
    pub digest: Digest,
    pub bounds: [[f64; 3]; 2],
    pub triangle_count: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RenderTile {
    pub global_vertex_indices: Vec<u32>,
    pub vertices: Vec<[f64; 3]>,
    pub vertex_volume_node_indices: Vec<Option<usize>>,
    pub global_triangle_indices: Vec<u32>,
    pub triangles: Vec<[u32; 3]>,
    pub triangle_volume_element_indices: Vec<Option<usize>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RenderViewRequest {
    pub camera_position: [f64; 3],
    pub camera_target: [f64; 3],
    pub camera_up: [f64; 3],
    pub fov_y_radians: f64,
    #[serde(default)]
    pub orthographic_height: Option<f64>,
    pub aspect_ratio: f64,
    pub viewport_height_px: f64,
    pub triangle_budget: usize,
    pub refine_above_px: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RenderViewPlan {
    pub nodes: Vec<(usize, usize)>,
    pub triangle_count: usize,
    pub exact: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RenderViewSelection {
    pub topology_schema_version: String,
    pub source: AnalysisRenderTopologySource,
    pub volume_element_count: usize,
    pub mesh_count: usize,
    pub field_summaries: Vec<RenderFieldSummary>,
    pub model_bounds: [[f64; 3]; 2],
    pub nodes: Vec<RenderSelectedNode>,
    pub exact: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RenderSelectedNode {
    pub mesh_index: usize,
    pub node_index: usize,
    pub mesh_id: String,
}

impl RenderTileIndex {
    pub fn plan_view(&self, request: Option<RenderViewRequest>) -> Result<RenderViewPlan, String> {
        self.validate()?;
        self.plan_view_validated(request)
    }

    // Persisted indices are validated once when read into the artifact cache.
    // Camera updates can then plan against the cached index without walking
    // the entire hierarchy for every pan or zoom.
    pub(crate) fn plan_view_validated(
        &self,
        request: Option<RenderViewRequest>,
    ) -> Result<RenderViewPlan, String> {
        let budget = request.map_or(DEFAULT_VIEW_TRIANGLE_BUDGET, |request| {
            request.triangle_budget
        });
        if budget == 0 {
            return Err("render view triangle budget must be positive".into());
        }
        if let Some(request) = request {
            if request
                .camera_position
                .iter()
                .any(|value| !value.is_finite())
                || request.camera_target.iter().any(|value| !value.is_finite())
                || request.camera_up.iter().any(|value| !value.is_finite())
                || !request.fov_y_radians.is_finite()
                || !(0.01..=3.13).contains(&request.fov_y_radians)
                || request
                    .orthographic_height
                    .is_some_and(|height| !height.is_finite() || height <= 0.0)
                || !request.aspect_ratio.is_finite()
                || request.aspect_ratio <= 0.0
                || !request.viewport_height_px.is_finite()
                || request.viewport_height_px <= 0.0
                || !request.refine_above_px.is_finite()
                || request.refine_above_px <= 0.0
            {
                return Err("render view camera or refinement threshold is invalid".into());
            }
        }
        let exact_fits = self
            .meshes
            .iter()
            .fold(0_usize, |sum, mesh| sum.saturating_add(mesh.triangle_count))
            <= budget;
        let mut selected = HashSet::new();
        let mut candidates = BinaryHeap::new();
        let mut triangle_count = 0_usize;
        for (mesh_index, mesh) in self.meshes.iter().enumerate() {
            if let Some(root) = mesh.lod_root {
                selected.insert((mesh_index, root));
                triangle_count =
                    triangle_count.saturating_add(mesh.lod_nodes[root].tile.triangle_count);
                candidates.push((view_score(&mesh.lod_nodes[root], request), mesh_index, root));
            }
        }
        if triangle_count > budget {
            return Err("render view budget cannot hold the model overview".into());
        }
        while let Some((score, mesh_index, node_index)) = candidates.pop() {
            let mesh = &self.meshes[mesh_index];
            let node = &mesh.lod_nodes[node_index];
            if node.children.is_empty()
                || (!exact_fits
                    && (request.is_none()
                        || f64::from_bits(score) < request.unwrap().refine_above_px))
            {
                continue;
            }
            let child_triangles = node
                .children
                .iter()
                .map(|child| mesh.lod_nodes[*child].tile.triangle_count)
                .fold(0_usize, usize::saturating_add);
            let proposed = triangle_count
                .saturating_sub(node.tile.triangle_count)
                .saturating_add(child_triangles);
            if proposed > budget {
                continue;
            }
            triangle_count = proposed;
            selected.remove(&(mesh_index, node_index));
            for child in &node.children {
                selected.insert((mesh_index, *child));
                candidates.push((
                    view_score(&mesh.lod_nodes[*child], request),
                    mesh_index,
                    *child,
                ));
            }
        }
        let mut nodes = selected.into_iter().collect::<Vec<_>>();
        nodes.sort_unstable();
        let exact = nodes.iter().all(|(mesh, node)| {
            self.meshes[*mesh].lod_nodes[*node]
                .exact_tile_index
                .is_some()
        });
        Ok(RenderViewPlan {
            nodes,
            triangle_count,
            exact,
        })
    }

    pub(crate) fn select_view_validated(
        &self,
        request: Option<RenderViewRequest>,
    ) -> Result<RenderViewSelection, String> {
        let plan = self.plan_view_validated(request)?;
        let model_bounds = union_bounds(
            self.meshes
                .iter()
                .filter_map(|mesh| mesh.lod_root.map(|root| mesh.lod_nodes[root].source_bounds)),
        );
        Ok(RenderViewSelection {
            topology_schema_version: self.topology_schema_version.clone(),
            source: self.source,
            volume_element_count: self.volume_element_count,
            mesh_count: self.meshes.len(),
            field_summaries: self.field_summaries.clone(),
            model_bounds,
            nodes: plan
                .nodes
                .into_iter()
                .map(|(mesh_index, node_index)| RenderSelectedNode {
                    mesh_index,
                    node_index,
                    mesh_id: self.meshes[mesh_index].mesh_id.clone(),
                })
                .collect(),
            exact: plan.exact,
        })
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != "analysis_render_tiles/v2" {
            return Err("unsupported render tile index schema".into());
        }
        if self.field_summaries.iter().any(|summary| {
            summary.field_id.is_empty()
                || !summary.minimum.is_finite()
                || !summary.maximum.is_finite()
                || summary.minimum > summary.maximum
        }) {
            return Err("render field summary is invalid".into());
        }
        for mesh in &self.meshes {
            let mut covered = 0_usize;
            for tile in &mesh.tiles {
                if tile.triangle_count > MAX_TRIANGLES_PER_TILE {
                    return Err("render tile exceeds triangle budget".into());
                }
                for axis in 0..3 {
                    if !tile.bounds[0][axis].is_finite()
                        || !tile.bounds[1][axis].is_finite()
                        || tile.bounds[0][axis] > tile.bounds[1][axis]
                    {
                        return Err("render tile has invalid bounds".into());
                    }
                }
                covered = covered
                    .checked_add(tile.triangle_count)
                    .ok_or("render tile triangle count overflow")?;
            }
            if covered != mesh.triangle_count || (mesh.vertex_count > 0 && mesh.tiles.is_empty()) {
                return Err("render tile index does not cover the mesh".into());
            }
            for (index, node) in mesh.lod_nodes.iter().enumerate() {
                if node.tile.triangle_count > MAX_TRIANGLES_PER_TILE
                    || node.children.len() > LOD_FANOUT
                    || node.children.iter().any(|child| *child >= index)
                    || (node.children.is_empty() == node.exact_tile_index.is_none())
                    || node
                        .exact_tile_index
                        .is_some_and(|tile| tile >= mesh.tiles.len())
                    || node
                        .exact_tile_index
                        .is_some_and(|tile| node.tile != mesh.tiles[tile])
                {
                    return Err("render LOD hierarchy is invalid".into());
                }
                for axis in 0..3 {
                    if !node.source_bounds[0][axis].is_finite()
                        || !node.source_bounds[1][axis].is_finite()
                        || node.source_bounds[0][axis] > node.source_bounds[1][axis]
                    {
                        return Err("render LOD source bounds are invalid".into());
                    }
                }
            }
            if mesh.triangle_count > 0
                && mesh
                    .lod_root
                    .and_then(|root| mesh.lod_nodes.get(root))
                    .is_none()
            {
                return Err("render LOD hierarchy has no root".into());
            }
            if let Some(root) = mesh.lod_root {
                let mut stack = vec![root];
                let mut seen = HashSet::new();
                let mut leaves = HashSet::new();
                while let Some(node_index) = stack.pop() {
                    if !seen.insert(node_index) {
                        return Err("render LOD hierarchy repeats a node".into());
                    }
                    let node = &mesh.lod_nodes[node_index];
                    if let Some(tile) = node.exact_tile_index {
                        leaves.insert(tile);
                    }
                    for child in &node.children {
                        let child_bounds = mesh.lod_nodes[*child].source_bounds;
                        if (0..3).any(|axis| {
                            child_bounds[0][axis] < node.source_bounds[0][axis]
                                || child_bounds[1][axis] > node.source_bounds[1][axis]
                        }) {
                            return Err("render LOD parent does not cover its children".into());
                        }
                        stack.push(*child);
                    }
                }
                if mesh
                    .tiles
                    .iter()
                    .enumerate()
                    .any(|(index, tile)| tile.triangle_count > 0 && !leaves.contains(&index))
                {
                    return Err("render LOD hierarchy omits an exact tile".into());
                }
            }
        }
        Ok(())
    }
}

fn view_score(node: &RenderLodNode, request: Option<RenderViewRequest>) -> u64 {
    let Some(request) = request else { return 0 };
    let center =
        [0, 1, 2].map(|axis| (node.source_bounds[0][axis] + node.source_bounds[1][axis]) / 2.0);
    let diagonal = (0..3)
        .map(|axis| (node.source_bounds[1][axis] - node.source_bounds[0][axis]).powi(2))
        .sum::<f64>()
        .sqrt();
    let radius = diagonal / 2.0;
    let direction =
        [0, 1, 2].map(|axis| request.camera_target[axis] - request.camera_position[axis]);
    let length = dot(direction, direction).sqrt();
    if length > f64::EPSILON {
        let forward = direction.map(|value| value / length);
        let right = cross(forward, request.camera_up);
        let right_length = dot(right, right).sqrt();
        if right_length > f64::EPSILON {
            let right = right.map(|value| value / right_length);
            let up = cross(right, forward);
            let to_center = [0, 1, 2].map(|axis| center[axis] - request.camera_position[axis]);
            let depth = dot(to_center, forward);
            let half_height = request
                .orthographic_height
                .map(|height| height / 2.0)
                .unwrap_or_else(|| depth.max(0.0) * (request.fov_y_radians / 2.0).tan());
            if depth + radius <= 0.0
                || dot(to_center, right).abs() - radius > half_height * request.aspect_ratio
                || dot(to_center, up).abs() - radius > half_height
            {
                return 0;
            }
        }
    }
    let distance = (0..3)
        .map(|axis| (center[axis] - request.camera_position[axis]).powi(2))
        .sum::<f64>()
        .sqrt()
        .max(diagonal / 2.0)
        .max(f64::EPSILON);
    let visible_height = request
        .orthographic_height
        .unwrap_or_else(|| 2.0 * (request.fov_y_radians / 2.0).tan() * distance);
    let pixels = diagonal * request.viewport_height_px / visible_height;
    pixels.max(0.0).to_bits()
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    (0..3).map(|axis| a[axis] * b[axis]).sum()
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

pub fn partition(
    topology: &AnalysisRenderTopology,
) -> Result<(RenderTileIndex, Vec<(Digest, Vec<u8>)>), String> {
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
            .max(1)
            .min(MAX_TRIANGLES_PER_LOD_NODE);
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

fn build_lod_hierarchy(
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

fn union_bounds(bounds: impl Iterator<Item = [[f64; 3]; 2]>) -> [[f64; 3]; 2] {
    let mut result = [[f64::INFINITY; 3], [f64::NEG_INFINITY; 3]];
    for bounds in bounds {
        for axis in 0..3 {
            result[0][axis] = result[0][axis].min(bounds[0][axis]);
            result[1][axis] = result[1][axis].max(bounds[1][axis]);
        }
    }
    result
}

fn coarsen(children: &[RenderTile], target_triangles: usize) -> Result<RenderTile, String> {
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

fn cluster_tile(
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

fn bounds(vertices: &[[f64; 3]]) -> Result<[[f64; 3]; 2], String> {
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

fn morton_key(point: [f64; 3], bounds: [[f64; 3]; 2]) -> u64 {
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

#[cfg(test)]
mod tests {
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
}
