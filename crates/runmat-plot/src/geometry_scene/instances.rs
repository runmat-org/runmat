//! Instanced chunks: one prototype mesh (a CAD part's faces, or its edges) placed many times.
//!
//! A chunk with [`GeometrySceneChunk::instances`] keeps its vertices in the prototype's own
//! coordinates. Each placement is one region (an assembly occurrence): selection, hover and role
//! highlights paint the placement through its instance tint, and picking tests each placement's
//! ray in the prototype's coordinates. A section view draws the chunk flattened into world
//! coordinates, where the CPU clipper works.
use super::*;
use crate::core::{InstanceRaw, NodeInstances};
use std::sync::Arc;

/// The upper end of the finest level's band: any size (JSON has no infinity, so a scene says
/// `f32::MAX`).
pub const LOD_OPEN: f32 = f32::MAX;

/// One placement of an instanced chunk.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GeometrySceneInstance {
    /// The rotation's rows, then the translation (the CAD display's occurrence transform).
    pub transform: [f32; 12],
    pub region_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tag: Option<String>,
}

impl GeometrySceneInstance {
    pub fn raw(&self) -> InstanceRaw {
        InstanceRaw::from_rows_translation(&self.transform)
    }
}

impl GeometrySceneChunk {
    /// Places this chunk's mesh at `instances`, drawn when a placement's projected diameter is
    /// within `lod_pixels` (device px). The chunk's bounds become the placements' world bounds.
    pub fn with_instances(
        mut self,
        instances: Vec<GeometrySceneInstance>,
        lod_pixels: Option<[f32; 2]>,
    ) -> Self {
        if instances.is_empty() {
            return self;
        }
        let local = self.bounds;
        let mut world: Option<BoundingBox> = None;
        for instance in &instances {
            let raw = instance.raw();
            let placed = box_corners(local).map(|c| raw.apply(c));
            let b = BoundingBox::from_points(&placed);
            world = Some(world.map_or(b, |w| w.union(&b)));
        }
        if let Some(world) = world {
            self.bounds = world;
            self.render_data.bounds = Some(world);
        }
        self.instances = instances;
        self.lod_pixels = lod_pixels;
        self
    }

    pub fn is_instanced(&self) -> bool {
        !self.instances.is_empty()
    }

    /// Whether the chunk is a coarser level of detail of its prototype: its band ends below
    /// [`LOD_OPEN`] pixels. The finest level's band is open above.
    pub fn is_coarse_level(&self) -> bool {
        self.lod_pixels.is_some_and(|[_, max]| max.is_finite() && max < LOD_OPEN * 0.5)
    }

    /// The mesh's own bounding sphere (center, radius), in the prototype's coordinates.
    pub(super) fn local_sphere(&self) -> (Vec3, f32) {
        let b = bounds_from_vertices(&self.vertices);
        ((b.min + b.max) * 0.5, (b.max - b.min).length() * 0.5)
    }

    /// The placements with the presentation's paints: role highlights fill, hover and selection
    /// tint a placement's faces and color (and thicken) its edges.
    pub(super) fn instance_paints(&self, presentation: &GeometryScenePresentation) -> Vec<InstanceRaw> {
        let mut raw: Vec<InstanceRaw> = self.instances.iter().map(GeometrySceneInstance::raw).collect();
        let index: HashMap<&str, usize> = self
            .instances
            .iter()
            .enumerate()
            .map(|(n, i)| (i.region_id.as_str(), n))
            .collect();
        let edges = self.is_edge_chunk();
        let mut paint = |region_id: &str, paint: RegionPaint| {
            let Some(&n) = index.get(region_id) else {
                return;
            };
            let r = &mut raw[n];
            match (edges, paint) {
                (_, RegionPaint::Fill(c)) => r.tint = [c[0], c[1], c[2], 1.0],
                (false, RegionPaint::Tint { face, amount, .. }) => r.tint = [face[0], face[1], face[2], amount],
                (true, RegionPaint::Tint { edge, .. }) => {
                    r.tint = [edge[0], edge[1], edge[2], 1.0];
                    r.extra[0] = paint.edge_weight();
                }
            }
        };
        for highlight in &presentation.region_highlights {
            let color = highlight
                .color
                .unwrap_or_else(|| geometry_region_role_color(highlight.role.as_deref()));
            paint(&highlight.region_id, RegionPaint::Fill(color));
        }
        if let Some(region_id) = presentation.hovered_region_id.as_deref() {
            paint(region_id, HOVER_PAINT);
        }
        for region_id in presentation
            .selected_region_ids
            .iter()
            .map(String::as_str)
            .chain(presentation.selected_region_id.as_deref())
        {
            paint(region_id, SELECTED_PAINT);
        }
        raw
    }

    /// The scene node's placements for `presentation`.
    pub(super) fn node_instances(&self, presentation: &GeometryScenePresentation) -> Arc<NodeInstances> {
        Arc::new(NodeInstances::new(
            self.instance_paints(presentation),
            self.local_sphere(),
            self.lod_pixels,
        ))
    }

    /// The chunk with every placement written out in world coordinates, one region per
    /// placement: what a section view clips. Only the finest level of detail (a band open
    /// above) flattens; coarser levels of the same prototype flatten to nothing, so a section
    /// shows every part once.
    pub fn flattened(&self) -> GeometrySceneChunk {
        let mut chunk = self.clone();
        chunk.instances = Vec::new();
        chunk.lod_pixels = None;
        if self.is_coarse_level() {
            chunk.vertices.clear();
            chunk.indices = chunk.indices.as_ref().map(|_| Vec::new());
            chunk.render_data.vertices.clear();
            chunk.render_data.indices = chunk.indices.clone();
            chunk.render_data.draw_calls.clear();
            return chunk;
        }
        let per = self.vertices.len();
        let mut vertices = Vec::with_capacity(per * self.instances.len());
        let mut indices = self.indices.as_ref().map(|i| Vec::with_capacity(i.len() * self.instances.len()));
        let primitive = if self.is_edge_chunk() { 2 } else { 3 };
        let per_primitives = self.indices.as_ref().map_or(per, Vec::len) / primitive;
        let mut regions = Vec::with_capacity(self.instances.len());
        for (n, instance) in self.instances.iter().enumerate() {
            let raw = instance.raw();
            let m = raw.matrix();
            let base = vertices.len() as u32;
            for v in &self.vertices {
                let mut v = *v;
                v.position = raw.apply(Vec3::from_array(v.position)).to_array();
                v.normal = m.transform_vector3(Vec3::from_array(v.normal)).to_array();
                vertices.push(v);
            }
            if let (Some(out), Some(src)) = (indices.as_mut(), self.indices.as_ref()) {
                out.extend(src.iter().map(|i| i + base));
            }
            regions.push(GeometrySceneRegion::new(
                instance.region_id.clone(),
                instance.label.clone(),
                instance.tag.clone(),
                vec![GeometrySceneTriangleRange::new(
                    (n * per_primitives) as u32,
                    per_primitives as u32,
                )],
            ));
        }
        let vertex_count = vertices.len();
        let index_count = indices.as_ref().map(Vec::len);
        chunk.render_data.vertices = vertices.clone();
        chunk.render_data.indices = indices.clone();
        chunk.render_data.draw_calls = vec![DrawCall {
            vertex_offset: 0,
            vertex_count,
            index_offset: index_count.map(|_| 0),
            index_count,
            instance_count: 1,
        }];
        chunk.vertices = vertices;
        chunk.indices = indices;
        if self.is_edge_chunk() {
            chunk.segment_regions = regions;
        } else {
            chunk.regions = regions;
        }
        chunk
    }
}

fn box_corners(b: BoundingBox) -> [Vec3; 8] {
    std::array::from_fn(|k| {
        Vec3::new(
            if k & 1 == 0 { b.min.x } else { b.max.x },
            if k & 2 == 0 { b.min.y } else { b.max.y },
            if k & 4 == 0 { b.min.z } else { b.max.z },
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_triangle() -> GeometrySceneChunk {
        let vertices = vec![
            vertex([0.0, 0.0, 0.0], [0.5, 0.5, 0.5, 1.0], [0.0, 0.0, 1.0]),
            vertex([1.0, 0.0, 0.0], [0.5, 0.5, 0.5, 1.0], [0.0, 0.0, 1.0]),
            vertex([0.0, 1.0, 0.0], [0.5, 0.5, 0.5, 1.0], [0.0, 0.0, 1.0]),
        ];
        GeometrySceneChunk::indexed_triangles("proto", vertices, vec![0, 1, 2], cad_default_material())
    }

    fn placed(x: f32, id: &str) -> GeometrySceneInstance {
        GeometrySceneInstance {
            transform: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, x, 0.0, 0.0],
            region_id: id.into(),
            label: None,
            tag: Some("occurrence".into()),
        }
    }

    #[test]
    fn placements_set_world_bounds_and_paint_their_own_region() {
        let chunk = unit_triangle().with_instances(vec![placed(0.0, "occ:0"), placed(10.0, "occ:1")], None);
        assert!((chunk.bounds.max.x - 11.0).abs() < 1e-6, "{:?}", chunk.bounds);
        let presentation = GeometryScenePresentation {
            selected_region_ids: vec!["occ:1".into()],
            ..Default::default()
        };
        let paints = chunk.instance_paints(&presentation);
        assert_eq!(paints[0].tint[3], 0.0, "the other placement keeps its color");
        assert!(paints[1].tint[3] > 0.0, "the selected placement is tinted");
    }

    #[test]
    fn a_flattened_chunk_has_each_placement_as_a_region_in_world_coordinates() {
        let flat = unit_triangle().with_instances(vec![placed(0.0, "occ:0"), placed(10.0, "occ:1")], None).flattened();
        assert!(!flat.is_instanced());
        assert_eq!(flat.triangle_count(), 2);
        assert_eq!(flat.vertices[3].position, [10.0, 0.0, 0.0]);
        assert_eq!(flat.regions[1].region_id, "occ:1");
        assert_eq!(flat.regions[1].triangle_ranges[0].start, 1);
    }
}
