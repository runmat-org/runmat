//! Portable geometry line segments, kept separately from scientific mesh topology.
use crate::{
    core::{DrawCall, Material, PipelineType, RenderData, Vertex},
    GeometrySceneChunk,
};
use glam::{Vec3, Vec4};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GeometryLines {
    #[serde(default)]
    pub positions: Vec<[f32; 3]>,
    #[serde(default)]
    pub colors: Vec<[f32; 4]>,
    /// The positions packed (base64 of little-endian `f32` x, y, z), in place of `positions`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub packed_positions: Option<String>,
    /// One color for every segment, in place of `colors`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub color: Option<[f32; 4]>,
    pub indices: Option<Vec<u32>>,
    pub mesh_id: Option<String>,
    pub label: Option<String>,
    pub visible: bool,
    pub wireframe_only: bool,
    /// Named groups of segments (ranges count index pairs): a CAD face's boundary edges under the
    /// face's region id, so the presentation's selection and hover color them with the face.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub segment_regions: Vec<SerializedSegmentRegion>,
    /// Placements of the segments (a CAD prototype's edges at each occurrence); positions are
    /// then the prototype's own coordinates. See `geometry_scene::instances`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub instances: Vec<crate::geometry_scene::GeometrySceneInstance>,
    /// The projected diameters (device px) placements draw at.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lod_pixels: Option<[f32; 2]>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SerializedSegmentRegion {
    pub region_id: String,
    pub segment_ranges: Vec<super::SerializedMeshTriangleRange>,
}
impl GeometryLines {
    pub(super) fn capture(
        chunk: &GeometrySceneChunk,
        grid: super::compact_mesh::PositionGrid,
    ) -> Option<Self> {
        if chunk.render_data.pipeline_type != PipelineType::Lines {
            return None;
        }
        let (positions, colors, indices) =
            super::compact_mesh::compact_lines(&chunk.vertices, chunk.indices.as_deref(), grid);
        Some(Self {
            positions,
            colors,
            packed_positions: None,
            color: None,
            indices: Some(indices),
            mesh_id: chunk.mesh_id.clone(),
            label: chunk.label.clone(),
            visible: chunk.visible,
            wireframe_only: chunk.wireframe_only,
            segment_regions: chunk
                .segment_regions
                .iter()
                .map(|region| SerializedSegmentRegion {
                    region_id: region.region_id.clone(),
                    segment_ranges: region.triangle_ranges.iter().copied().map(Into::into).collect(),
                })
                .collect(),
            instances: chunk.instances.clone(),
            lod_pixels: chunk.lod_pixels,
        })
    }
    pub(crate) fn restore(mut self, id: String) -> Result<GeometrySceneChunk, String> {
        if let Some(packed) = self.packed_positions.take() {
            let values = super::packed::f32s(&packed)?;
            if !values.len().is_multiple_of(3) {
                return Err("packed line positions are not whole triples".into());
            }
            self.positions = values.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect();
        }
        if let (Some(color), true) = (self.color, self.colors.is_empty()) {
            self.colors = vec![color; self.positions.len()];
        }
        let count = self.indices.as_ref().map_or(self.positions.len(), Vec::len);
        if self.positions.len() != self.colors.len()
            || !count.is_multiple_of(2)
            || self
                .positions
                .iter()
                .flatten()
                .chain(self.colors.iter().flatten())
                .any(|v| !v.is_finite())
            || self
                .indices
                .as_ref()
                .is_some_and(|values| values.iter().any(|&i| i as usize >= self.positions.len()))
        {
            return Err("invalid saved geometry line segments".into());
        }
        let vertices = self
            .positions
            .into_iter()
            .zip(self.colors)
            .map(|(p, c)| Vertex::new(Vec3::from_array(p), Vec4::from_array(c)))
            .collect::<Vec<_>>();
        let indices = self.indices;
        let indexed = indices.is_some();
        let mut chunk = GeometrySceneChunk::from_render_data(
            id,
            RenderData {
                pipeline_type: PipelineType::Lines,
                draw_calls: vec![DrawCall {
                    vertex_offset: 0,
                    vertex_count: vertices.len(),
                    index_offset: indexed.then_some(0),
                    index_count: indexed.then_some(count),
                    instance_count: 1,
                }],
                vertices,
                indices,
                gpu_vertices: None,
                bounds: None,
                material: Material::default(),
                image: None,
            },
        );
        chunk.mesh_id = self.mesh_id;
        chunk.label = self.label;
        chunk.visible = self.visible;
        chunk.wireframe_only = self.wireframe_only;
        let segments = (count / 2) as u64;
        for region in self.segment_regions {
            if region
                .segment_ranges
                .iter()
                .any(|range| u64::from(range.start) + u64::from(range.count) > segments)
            {
                return Err("geometry line region runs past its segments".into());
            }
            chunk.segment_regions.push(crate::geometry_scene::GeometrySceneRegion::new(
                region.region_id,
                None,
                None,
                region.segment_ranges.into_iter().map(Into::into).collect(),
            ));
        }
        Ok(chunk.with_instances(self.instances, self.lod_pixels))
    }
}
