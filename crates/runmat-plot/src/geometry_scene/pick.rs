//! Picking in a geometry scene: the pick request and result, and the index (a BVH over the
//! scene's triangles plus its feature edges) that answers face, edge and point picks.

use super::*;

#[derive(Debug, Clone)]
pub struct GeometryScenePickRequest {
    pub camera: Camera,
    pub surface_size: [f32; 2],
    pub position: [f32; 2],
    pub mode: GeometryScenePickMode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GeometryScenePickMode {
    #[default]
    Face,
    Edge,
    Point,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GeometryScenePickResult {
    pub mesh_id: Option<String>,
    pub chunk_id: String,
    pub triangle_index: usize,
    pub region_id: Option<String>,
    pub region_label: Option<String>,
    pub region_tag: Option<String>,
    pub distance: f32,
    pub position: [f32; 3],
    /// Vertex indices of the hit tessellation triangle. These preserve stable mesh topology for
    /// point/node and edge selection without pretending that every click is a face selection.
    pub vertex_indices: [u32; 3],
    /// Barycentric coordinates of `position` within `vertex_indices`.
    pub barycentric_coordinates: [f32; 3],
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub topology_selection: Option<GeometrySceneTopologySelection>,
    /// The owner nodes (bodies, assembly components) of the chunk hit, outermost first, as the
    /// scene's `hidden_owner_node_ids` names them. Empty when the scene names none.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub owner_node_ids: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct GeometryScenePickIndex {
    scene_key: GeometrySceneCacheKey,
    triangles: Vec<IndexedTriangle>,
    nodes: Vec<PickBvhNode>,
    root: Option<usize>,
    feature_edges: HashMap<PickEdgeKey, FeatureEdgeInfo>,
    feature_edge_neighbors: HashMap<PickVertexKey, Vec<PickEdgeKey>>,
    scene_diagonal: f32,
    /// Instanced chunks, picked per placement in the prototype's coordinates.
    instanced: Vec<super::pick_instances::InstancedPick>,
    /// Regions and owner nodes the view hides: picks pass through them.
    hidden_regions: HashSet<String>,
    hidden_owners: HashSet<String>,
}

impl GeometryScenePickIndex {
    pub fn build(scene: &GeometryScene) -> Self {
        let mut triangles = Vec::with_capacity(scene.triangle_count());
        for (chunk_index, chunk) in scene.chunks.iter().enumerate() {
            if !chunk.visible
                || chunk.render_data.pipeline_type != PipelineType::Triangles
                || chunk.is_instanced()
            {
                continue;
            }
            let Some(indices) = chunk.indices.as_ref() else {
                continue;
            };
            for (triangle_index, triangle) in indices.chunks_exact(3).enumerate() {
                let a = chunk.vertices.get(triangle[0] as usize);
                let b = chunk.vertices.get(triangle[1] as usize);
                let c = chunk.vertices.get(triangle[2] as usize);
                let (Some(a), Some(b), Some(c)) = (a, b, c) else {
                    continue;
                };
                let a = Vec3::from_array(a.position);
                let b = Vec3::from_array(b.position);
                let c = Vec3::from_array(c.position);
                let bounds = triangle_bounds(a, b, c);
                let region = chunk.region_for_triangle(triangle_index as u32);
                triangles.push(IndexedTriangle {
                    a,
                    b,
                    c,
                    bounds,
                    centroid: (a + b + c) / 3.0,
                    mesh_id: chunk.mesh_id.clone(),
                    chunk_index,
                    chunk_id: chunk.chunk_id.clone(),
                    triangle_index,
                    vertex_indices: [triangle[0], triangle[1], triangle[2]],
                    region_id: region.map(|item| item.region_id.clone()),
                    region_label: region.and_then(|item| item.label.clone()),
                    region_tag: region.and_then(|item| item.tag.clone()),
                    owner_node_ids: chunk.owner_node_ids.clone(),
                });
            }
        }

        let mut nodes = Vec::new();
        let triangle_count = triangles.len();
        let root = build_pick_bvh_node(&mut nodes, &mut triangles, 0, triangle_count);
        let (feature_edges, feature_edge_neighbors) = collect_feature_edges(&triangles);
        Self {
            scene_key: scene.cache_key(),
            triangles,
            nodes,
            root,
            feature_edges,
            feature_edge_neighbors,
            scene_diagonal: scene.bounds.size().length().max(1.0e-6),
            instanced: super::pick_instances::build(scene),
            hidden_regions: HashSet::new(),
            hidden_owners: HashSet::new(),
        }
    }

    /// Follows the view's hidden regions and owner nodes, so picks pass through what isn't drawn.
    /// Instanced placements follow it too: a hidden placement region or owner node is skipped.
    pub fn set_hidden(&mut self, presentation: &GeometryScenePresentation) {
        let regions: HashSet<String> = presentation.hidden_region_ids.iter().cloned().collect();
        let owners: BTreeSet<String> = self
            .owner_ids()
            .chain(super::pick_instances::owner_ids(&self.instanced))
            .cloned()
            .collect();
        let owners: HashSet<String> = presentation
            .resolved_hidden_owner_node_ids(owners.iter().map(String::as_str), &BTreeSet::new())
            .into_iter()
            .collect();
        super::pick_instances::set_hidden(&mut self.instanced, &regions, &owners);
        self.set_hidden_sets(regions, owners);
    }

    pub(super) fn set_hidden_sets(&mut self, regions: HashSet<String>, owners: HashSet<String>) {
        self.hidden_regions = regions;
        self.hidden_owners = owners;
    }

    pub(super) fn region_hidden(&self, region_id: &str) -> bool {
        self.hidden_regions.contains(region_id)
    }

    pub(super) fn owner_ids(&self) -> impl Iterator<Item = &String> {
        self.triangles.iter().flat_map(|t| t.owner_node_ids.iter())
    }

    fn hidden(&self, triangle: &IndexedTriangle) -> bool {
        triangle.region_id.as_ref().is_some_and(|id| self.hidden_regions.contains(id))
            || triangle.owner_node_ids.iter().any(|id| self.hidden_owners.contains(id))
    }

    #[cfg(test)]
    pub(super) fn instanced_sets(&self) -> &[super::pick_instances::InstancedPick] {
        &self.instanced
    }

    pub fn scene_key(&self) -> &GeometrySceneCacheKey {
        &self.scene_key
    }

    pub fn triangle_count(&self) -> usize {
        self.triangles.len()
    }

    pub fn is_empty(&self) -> bool {
        self.triangles.is_empty()
    }

    pub fn pick(&self, request: GeometryScenePickRequest) -> Option<GeometryScenePickResult> {
        if request.surface_size[0] <= 0.0 || request.surface_size[1] <= 0.0 {
            return None;
        }
        let mut camera = request.camera;
        let screen_pos = Vec2::from_array(request.position);
        let screen_size = Vec2::from_array(request.surface_size);
        match request.mode {
            GeometryScenePickMode::Face => {
                let ray = pick_ray(&mut camera, screen_size, screen_pos)?;
                let hit = self.hit_ray(&ray);
                let placed = super::pick_instances::pick(
                    &self.instanced,
                    &ray,
                    hit.map_or(f32::INFINITY, |(_, d)| d),
                );
                if placed.is_some() {
                    return placed;
                }
                let (triangle_index, distance) = hit?;
                let position = ray.origin + ray.direction * distance;
                self.result_for_triangle(triangle_index, position, distance, None)
            }
            GeometryScenePickMode::Edge | GeometryScenePickMode::Point => {
                self.pick_screen_topology(&mut camera, screen_size, screen_pos, request.mode)
            }
        }
    }

    /// The nearest triangle the ray hits: its index and distance.
    pub(super) fn hit_ray(&self, ray: &PickRay) -> Option<(usize, f32)> {
        let mut best = None;
        if let Some(root) = self.root {
            self.pick_node(root, ray, &mut best);
        }
        best.map(|hit: PickHit| (hit.triangle_index, hit.distance))
    }

    pub(super) fn result_for_triangle(
        &self,
        triangle_index: usize,
        position: Vec3,
        distance: f32,
        topology_selection: Option<GeometrySceneTopologySelection>,
    ) -> Option<GeometryScenePickResult> {
        let triangle = self.triangles.get(triangle_index)?;
        Some(GeometryScenePickResult {
            mesh_id: triangle.mesh_id.clone(),
            chunk_id: triangle.chunk_id.clone(),
            triangle_index: triangle.triangle_index,
            region_id: triangle.region_id.clone(),
            region_label: triangle.region_label.clone(),
            region_tag: triangle.region_tag.clone(),
            distance,
            position: position.to_array(),
            vertex_indices: triangle.vertex_indices,
            barycentric_coordinates: triangle_barycentric_coordinates(
                position, triangle.a, triangle.b, triangle.c,
            ),
            topology_selection,
            owner_node_ids: triangle.owner_node_ids.clone(),
        })
    }

    fn pick_at(
        &self,
        camera: &mut Camera,
        size: Vec2,
        position: Vec2,
    ) -> Option<(PickHit, PickRay)> {
        let ray = pick_ray(camera, size, position)?;
        let mut best = None;
        if let Some(root) = self.root {
            self.pick_node(root, &ray, &mut best);
        }
        best.map(|hit| (hit, ray))
    }

    fn pick_screen_topology(
        &self,
        camera: &mut Camera,
        size: Vec2,
        cursor: Vec2,
        mode: GeometryScenePickMode,
    ) -> Option<GeometryScenePickResult> {
        const OFFSETS: [[f32; 2]; 25] = [
            [0.0, 0.0],
            [3.0, 0.0],
            [-3.0, 0.0],
            [0.0, 3.0],
            [0.0, -3.0],
            [3.0, 3.0],
            [3.0, -3.0],
            [-3.0, 3.0],
            [-3.0, -3.0],
            [6.0, 0.0],
            [-6.0, 0.0],
            [0.0, 6.0],
            [0.0, -6.0],
            [6.0, 6.0],
            [6.0, -6.0],
            [-6.0, 6.0],
            [-6.0, -6.0],
            [12.0, 0.0],
            [-12.0, 0.0],
            [0.0, 12.0],
            [0.0, -12.0],
            [12.0, 12.0],
            [12.0, -12.0],
            [-12.0, 12.0],
            [-12.0, -12.0],
        ];
        let view_proj = camera.view_proj_matrix();
        let radius_px = match mode {
            GeometryScenePickMode::Edge => 10.0,
            GeometryScenePickMode::Point => 13.0,
            GeometryScenePickMode::Face => return None,
        };
        let mut visited_triangles = HashSet::new();
        let mut nearby_edges = HashSet::new();
        let mut best: Option<PickCandidate> = None;
        for offset in OFFSETS {
            let sample = cursor + Vec2::from_array(offset);
            if sample.x < 0.0 || sample.y < 0.0 || sample.x > size.x || sample.y > size.y {
                continue;
            }
            let Some((hit, _)) = self.pick_at(camera, size, sample) else {
                continue;
            };
            if !visited_triangles.insert(hit.triangle_index) {
                continue;
            }
            let triangle = &self.triangles[hit.triangle_index];
            let corners = [triangle.a, triangle.b, triangle.c];
            match mode {
                GeometryScenePickMode::Edge => {
                    for (a, b) in [(0, 1), (1, 2), (2, 0)] {
                        let edge_key = pick_edge_key(triangle, a, b);
                        if self.feature_edges.contains_key(&edge_key) {
                            nearby_edges.insert(edge_key);
                        }
                    }
                    // Rays just inside a face may never hit its neighbor across
                    // a crease. Include feature edges incident to its vertices.
                    for corner in corners {
                        let vertex_key = PickVertexKey {
                            chunk_index: triangle.chunk_index,
                            position: corner.to_array().map(f32::to_bits),
                        };
                        if let Some(neighbors) = self.feature_edge_neighbors.get(&vertex_key) {
                            nearby_edges.extend(neighbors.iter().cloned());
                        }
                    }
                }
                GeometryScenePickMode::Point => {
                    for (corner, position) in corners.into_iter().enumerate() {
                        let Some((screen_point, _)) = project_pick_point(view_proj, position, size)
                        else {
                            continue;
                        };
                        let distance_px_sq = cursor.distance_squared(screen_point);
                        if distance_px_sq > radius_px * radius_px {
                            continue;
                        }
                        let selection = GeometrySceneTopologySelection::Point {
                            chunk_id: triangle.chunk_id.clone(),
                            vertex_index: triangle.vertex_indices[corner],
                        };
                        self.consider_pick_candidate(
                            camera,
                            size,
                            screen_point,
                            PickCandidate {
                                triangle_index: hit.triangle_index,
                                position,
                                distance_px_sq,
                                endpoint_margin_px: 0.0,
                                projected_length_px: 0.0,
                                depth: camera.position.distance(position),
                                selection,
                                edge_key: None,
                            },
                            &mut best,
                        );
                    }
                }
                GeometryScenePickMode::Face => unreachable!(),
            }
        }
        if mode == GeometryScenePickMode::Edge {
            for edge_key in nearby_edges {
                let info = &self.feature_edges[&edge_key];
                let (Some(pa), Some(pb)) = (
                    project_pick_point(view_proj, info.positions[0], size),
                    project_pick_point(view_proj, info.positions[1], size),
                ) else {
                    continue;
                };
                let ab = pb.0 - pa.0;
                let length_px = ab.length();
                let t = ((cursor - pa.0).dot(ab) / ab.length_squared().max(1.0e-8)).clamp(0.0, 1.0);
                let screen_point = pa.0 + ab * t;
                let distance_px_sq = cursor.distance_squared(screen_point);
                if distance_px_sq > radius_px * radius_px {
                    continue;
                }
                let world_t = t * pa.1 / ((1.0 - t) * pb.1 + t * pa.1);
                let world_point = info.positions[0].lerp(info.positions[1], world_t);
                let triangle = &self.triangles[info.triangle_index];
                let selection = GeometrySceneTopologySelection::Edge {
                    chunk_id: triangle.chunk_id.clone(),
                    vertex_indices: info.vertex_indices,
                    segments: Vec::new(),
                };
                self.consider_pick_candidate(
                    camera,
                    size,
                    screen_point,
                    PickCandidate {
                        triangle_index: info.triangle_index,
                        position: world_point,
                        distance_px_sq,
                        endpoint_margin_px: t.min(1.0 - t) * length_px,
                        projected_length_px: length_px,
                        depth: camera.position.distance(world_point),
                        selection,
                        edge_key: Some(edge_key),
                    },
                    &mut best,
                );
            }
        }
        let mut best = best?;
        if let (Some(key), GeometrySceneTopologySelection::Edge { segments, .. }) =
            (&best.edge_key, &mut best.selection)
        {
            *segments = self.edge_chain(key);
        }
        self.result_for_triangle(
            best.triangle_index,
            best.position,
            best.depth,
            Some(best.selection),
        )
    }

    fn consider_pick_candidate(
        &self,
        camera: &mut Camera,
        size: Vec2,
        screen_point: Vec2,
        candidate: PickCandidate,
        best: &mut Option<PickCandidate>,
    ) {
        if let Some((front, ray)) = self.pick_at(camera, size, screen_point) {
            let candidate_distance = ray.origin.distance(candidate.position);
            let tolerance = self.scene_diagonal * 1.0e-3;
            if front.distance + tolerance < candidate_distance {
                return;
            }
        }
        if let Some(current) = best {
            // A line ending at the cursor is a weaker match than one running
            // under it. This resolves shared CAD corners without favoring a
            // short, foreshortened edge whose endpoint happens to be nearby.
            let score = |item: &PickCandidate| {
                let endpoint_penalty = if item.edge_key.is_some() {
                    (6.0 - item.endpoint_margin_px).max(0.0).powi(2)
                } else {
                    0.0
                };
                item.distance_px_sq + endpoint_penalty
            };
            let candidate_score = score(&candidate);
            let current_score = score(current);
            let tied_score = (candidate_score - current_score).abs() <= 1.0;
            let tied_depth =
                (candidate.depth - current.depth).abs() <= self.scene_diagonal * 1.0e-4;
            if candidate_score > current_score + 1.0
                || (tied_score && candidate.depth > current.depth && !tied_depth)
                || (tied_score
                    && tied_depth
                    && candidate.projected_length_px <= current.projected_length_px)
            {
                return;
            }
        }
        *best = Some(candidate);
    }

    fn edge_chain(&self, selected: &PickEdgeKey) -> Vec<[u32; 2]> {
        let Some(info) = self.feature_edges.get(selected) else {
            return Vec::new();
        };
        let mut segments = vec![info.vertex_indices];
        // Untagged mesh edges are individual finite elements. CAD face boundaries
        // can span many tessellation segments and should highlight as one edge.
        if info.regions.is_empty() {
            return segments;
        }
        let mut visited = HashSet::from([selected.clone()]);
        for (endpoint, other) in [(0, 1), (1, 0)] {
            let mut current = info.positions[endpoint];
            let mut direction = (current - info.positions[other]).normalize_or_zero();
            while segments.len() < 8192 {
                let vertex_key = PickVertexKey {
                    chunk_index: selected.chunk_index,
                    position: current.to_array().map(f32::to_bits),
                };
                let Some(neighbors) = self.feature_edge_neighbors.get(&vertex_key) else {
                    break;
                };
                let mut continuation: Option<(&PickEdgeKey, Vec3, Vec3, f32)> = None;
                for key in neighbors {
                    if visited.contains(key) {
                        continue;
                    }
                    let Some(candidate) = self.feature_edges.get(key) else {
                        continue;
                    };
                    if candidate.regions != info.regions {
                        continue;
                    }
                    let next = if candidate.positions[0].to_array().map(f32::to_bits)
                        == vertex_key.position
                    {
                        candidate.positions[1]
                    } else {
                        candidate.positions[0]
                    };
                    let next_direction = (next - current).normalize_or_zero();
                    let alignment = direction.dot(next_direction);
                    if alignment < 0.70 {
                        continue;
                    }
                    if continuation.as_ref().is_none_or(|item| alignment > item.3) {
                        continuation = Some((key, next, next_direction, alignment));
                    }
                }
                let Some((key, next, next_direction, _)) = continuation else {
                    break;
                };
                segments.push(self.feature_edges[key].vertex_indices);
                visited.insert(key.clone());
                current = next;
                direction = next_direction;
            }
        }
        segments
    }

    fn pick_node(&self, node_index: usize, ray: &PickRay, best: &mut Option<PickHit>) {
        let Some(node) = self.nodes.get(node_index) else {
            return;
        };
        let max_distance = best
            .as_ref()
            .map(|hit| hit.distance)
            .unwrap_or(f32::INFINITY);
        let Some(bounds_distance) = ray_intersects_bounds(ray, node.bounds) else {
            return;
        };
        if bounds_distance > max_distance {
            return;
        }
        match node.kind {
            PickBvhNodeKind::Leaf { start, end } => {
                for triangle_index in start..end {
                    let Some(triangle) = self.triangles.get(triangle_index) else {
                        continue;
                    };
                    if self.hidden(triangle) {
                        continue;
                    }
                    if let Some(hit_distance) =
                        ray_intersects_triangle(ray, triangle.a, triangle.b, triangle.c)
                    {
                        if hit_distance > 0.0
                            && hit_distance
                                < best
                                    .as_ref()
                                    .map(|hit| hit.distance)
                                    .unwrap_or(f32::INFINITY)
                        {
                            *best = Some(PickHit {
                                triangle_index,
                                distance: hit_distance,
                            });
                        }
                    }
                }
            }
            PickBvhNodeKind::Branch { left, right } => {
                self.pick_node(left, ray, best);
                self.pick_node(right, ray, best);
            }
        }
    }
}


/// The ray under a screen position.
fn pick_ray(camera: &mut Camera, size: Vec2, position: Vec2) -> Option<PickRay> {
    let origin = camera.screen_to_world(position, size, 0.0);
    let far = camera.screen_to_world(position, size, 1.0);
    let direction = (far - origin).normalize_or_zero();
    (direction.length_squared() > f32::EPSILON).then_some(PickRay { origin, direction })
}

#[derive(Debug, Clone)]
struct IndexedTriangle {
    a: Vec3,
    b: Vec3,
    c: Vec3,
    bounds: BoundingBox,
    centroid: Vec3,
    mesh_id: Option<String>,
    chunk_index: usize,
    chunk_id: String,
    triangle_index: usize,
    vertex_indices: [u32; 3],
    region_id: Option<String>,
    region_label: Option<String>,
    region_tag: Option<String>,
    owner_node_ids: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct PickEdgeKey {
    chunk_index: usize,
    endpoints: [[u32; 3]; 2],
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct PickVertexKey {
    chunk_index: usize,
    position: [u32; 3],
}

#[derive(Debug, Clone)]
struct FeatureEdgeInfo {
    positions: [Vec3; 2],
    vertex_indices: [u32; 2],
    regions: Vec<String>,
    triangle_index: usize,
}

#[derive(Debug, Clone)]
struct PickCandidate {
    triangle_index: usize,
    position: Vec3,
    distance_px_sq: f32,
    endpoint_margin_px: f32,
    projected_length_px: f32,
    depth: f32,
    selection: GeometrySceneTopologySelection,
    edge_key: Option<PickEdgeKey>,
}

fn pick_edge_key(triangle: &IndexedTriangle, a: usize, b: usize) -> PickEdgeKey {
    let positions = [triangle.a, triangle.b, triangle.c];
    let first = positions[a].to_array().map(f32::to_bits);
    let second = positions[b].to_array().map(f32::to_bits);
    PickEdgeKey {
        chunk_index: triangle.chunk_index,
        endpoints: if first <= second {
            [first, second]
        } else {
            [second, first]
        },
    }
}

fn collect_feature_edges(
    triangles: &[IndexedTriangle],
) -> (
    HashMap<PickEdgeKey, FeatureEdgeInfo>,
    HashMap<PickVertexKey, Vec<PickEdgeKey>>,
) {
    struct Incidence {
        count: usize,
        region_id: Option<String>,
        regions: BTreeSet<String>,
        normal: Vec3,
        is_feature: bool,
        positions: [Vec3; 2],
        vertex_indices: [u32; 2],
        triangle_index: usize,
    }
    let mut edges: HashMap<PickEdgeKey, Incidence> = HashMap::new();
    for (triangle_index, triangle) in triangles.iter().enumerate() {
        let normal = (triangle.b - triangle.a)
            .cross(triangle.c - triangle.a)
            .normalize_or_zero();
        for (a, b) in [(0, 1), (1, 2), (2, 0)] {
            let key = pick_edge_key(triangle, a, b);
            if let Some(existing) = edges.get_mut(&key) {
                existing.count += 1;
                existing.is_feature |=
                    existing.region_id != triangle.region_id || existing.normal.dot(normal) < 0.966;
                if let Some(region_id) = &triangle.region_id {
                    existing.regions.insert(region_id.clone());
                }
            } else {
                let positions = [triangle.a, triangle.b, triangle.c];
                edges.insert(
                    key,
                    Incidence {
                        count: 1,
                        region_id: triangle.region_id.clone(),
                        regions: triangle.region_id.iter().cloned().collect(),
                        normal,
                        is_feature: false,
                        positions: [positions[a], positions[b]],
                        vertex_indices: [triangle.vertex_indices[a], triangle.vertex_indices[b]],
                        triangle_index,
                    },
                );
            }
        }
    }
    let mut features = HashMap::new();
    let mut neighbors: HashMap<PickVertexKey, Vec<PickEdgeKey>> = HashMap::new();
    for (key, incidence) in edges {
        if incidence.count > 1 && !incidence.is_feature && incidence.region_id.is_some() {
            continue;
        }
        let info = FeatureEdgeInfo {
            positions: incidence.positions,
            vertex_indices: incidence.vertex_indices,
            regions: incidence.regions.into_iter().collect(),
            triangle_index: incidence.triangle_index,
        };
        for position in info.positions {
            neighbors
                .entry(PickVertexKey {
                    chunk_index: key.chunk_index,
                    position: position.to_array().map(f32::to_bits),
                })
                .or_default()
                .push(key.clone());
        }
        features.insert(key, info);
    }
    (features, neighbors)
}

pub(super) fn project_pick_point(view_proj: Mat4, position: Vec3, screen_size: Vec2) -> Option<(Vec2, f32)> {
    let clip = view_proj * position.extend(1.0);
    if !clip.w.is_finite() || clip.w <= 0.0 {
        return None;
    }
    let ndc = clip.truncate() / clip.w;
    if !ndc.is_finite() {
        return None;
    }
    Some((
        Vec2::new(
            (ndc.x + 1.0) * screen_size.x * 0.5,
            (1.0 - ndc.y) * screen_size.y * 0.5,
        ),
        clip.w,
    ))
}

fn triangle_barycentric_coordinates(point: Vec3, a: Vec3, b: Vec3, c: Vec3) -> [f32; 3] {
    let v0 = b - a;
    let v1 = c - a;
    let v2 = point - a;
    let d00 = v0.dot(v0);
    let d01 = v0.dot(v1);
    let d11 = v1.dot(v1);
    let d20 = v2.dot(v0);
    let d21 = v2.dot(v1);
    let denominator = d00 * d11 - d01 * d01;
    if denominator.abs() <= f32::EPSILON {
        return [1.0, 0.0, 0.0];
    }
    let v = (d11 * d20 - d01 * d21) / denominator;
    let w = (d00 * d21 - d01 * d20) / denominator;
    [1.0 - v - w, v, w]
}

#[derive(Debug, Clone)]
struct PickBvhNode {
    bounds: BoundingBox,
    kind: PickBvhNodeKind,
}

#[derive(Debug, Clone, Copy)]
enum PickBvhNodeKind {
    Leaf { start: usize, end: usize },
    Branch { left: usize, right: usize },
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct PickRay {
    pub(super) origin: Vec3,
    pub(super) direction: Vec3,
}

#[derive(Debug, Clone, Copy)]
struct PickHit {
    triangle_index: usize,
    distance: f32,
}

fn build_pick_bvh_node(
    nodes: &mut Vec<PickBvhNode>,
    triangles: &mut [IndexedTriangle],
    start: usize,
    end: usize,
) -> Option<usize> {
    if start >= end {
        return None;
    }
    let bounds = combined_triangle_bounds(&triangles[start..end]);
    let node_index = nodes.len();
    nodes.push(PickBvhNode {
        bounds,
        kind: PickBvhNodeKind::Leaf { start, end },
    });
    const LEAF_TRIANGLES: usize = 32;
    if end - start <= LEAF_TRIANGLES {
        return Some(node_index);
    }
    let centroid_bounds = combined_centroid_bounds(&triangles[start..end]);
    let extent = centroid_bounds.max - centroid_bounds.min;
    let axis = if extent.x >= extent.y && extent.x >= extent.z {
        0
    } else if extent.y >= extent.z {
        1
    } else {
        2
    };
    triangles[start..end].sort_by(|a, b| {
        a.centroid[axis]
            .partial_cmp(&b.centroid[axis])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mid = start + (end - start) / 2;
    let left = build_pick_bvh_node(nodes, triangles, start, mid);
    let right = build_pick_bvh_node(nodes, triangles, mid, end);
    if let (Some(left), Some(right)) = (left, right) {
        nodes[node_index].kind = PickBvhNodeKind::Branch { left, right };
    }
    Some(node_index)
}

fn triangle_bounds(a: Vec3, b: Vec3, c: Vec3) -> BoundingBox {
    let mut bounds = BoundingBox::new(a, a);
    bounds.expand(b);
    bounds.expand(c);
    bounds
}

fn combined_triangle_bounds(triangles: &[IndexedTriangle]) -> BoundingBox {
    let mut bounds = BoundingBox::default();
    for triangle in triangles {
        bounds.expand_by_box(&triangle.bounds);
    }
    bounds
}

fn combined_centroid_bounds(triangles: &[IndexedTriangle]) -> BoundingBox {
    let Some(first) = triangles.first() else {
        return BoundingBox::default();
    };
    let mut bounds = BoundingBox::new(first.centroid, first.centroid);
    for triangle in triangles.iter().skip(1) {
        bounds.expand(triangle.centroid);
    }
    bounds
}

pub(super) fn ray_intersects_bounds(ray: &PickRay, bounds: BoundingBox) -> Option<f32> {
    let mut t_min: f32 = 0.0;
    let mut t_max = f32::INFINITY;
    for axis in 0..3 {
        let origin = ray.origin[axis];
        let direction = ray.direction[axis];
        let min = bounds.min[axis];
        let max = bounds.max[axis];
        if direction.abs() < 1e-8 {
            if origin < min || origin > max {
                return None;
            }
            continue;
        }
        let inv_direction = 1.0 / direction;
        let mut t0 = (min - origin) * inv_direction;
        let mut t1 = (max - origin) * inv_direction;
        if t0 > t1 {
            std::mem::swap(&mut t0, &mut t1);
        }
        t_min = t_min.max(t0);
        t_max = t_max.min(t1);
        if t_max < t_min {
            return None;
        }
    }
    Some(t_min.max(0.0))
}

pub(super) fn ray_intersects_triangle(ray: &PickRay, a: Vec3, b: Vec3, c: Vec3) -> Option<f32> {
    // A fixed determinant cutoff depends on coordinate units and silently drops
    // small CAD faces. Use f64 and a dimensionless relative parallelism test.
    let a = a.as_dvec3();
    let edge1 = b.as_dvec3() - a;
    let edge2 = c.as_dvec3() - a;
    let direction = ray.direction.as_dvec3();
    let pvec = direction.cross(edge2);
    let det = edge1.dot(pvec);
    let scale = edge1.length() * pvec.length();
    if !det.is_finite() || !scale.is_finite() || det.abs() <= scale * (16.0 * f64::EPSILON) {
        return None;
    }
    let inv_det = 1.0 / det;
    let tvec = ray.origin.as_dvec3() - a;
    let u = tvec.dot(pvec) * inv_det;
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let qvec = tvec.cross(edge1);
    let v = direction.dot(qvec) * inv_det;
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let t = (edge2.dot(qvec) * inv_det) as f32;
    (t.is_finite() && t > 0.0).then_some(t)
}
