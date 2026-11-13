//! Governed view selection over immutable LOD nodes.
use super::*;
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

pub(super) fn view_score(node: &RenderLodNode, request: Option<RenderViewRequest>) -> u64 {
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

pub(super) fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    (0..3).map(|axis| a[axis] * b[axis]).sum()
}

pub(super) fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
