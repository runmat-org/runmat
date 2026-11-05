//! An axes' drawn data as points, segments and triangles, for a legend placed "best": a line is
//! its segments and a bar or patch its filled triangles, so the legend avoids the whole mark
//! and not only its vertices.

use super::*;
use crate::core::renderer::PipelineType;

/// One drawn primitive in an axes' camera world space (as `view_bounds_for_axes` frames it).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DataPrimitive {
    Point([f32; 2]),
    Segment([[f32; 2]; 2]),
    Triangle([[f32; 2]; 3]),
}

impl PlotRenderer {
    /// Up to about `limit` of an axes' drawn primitives, evenly strided across every visible
    /// node. Line pipelines give segments, triangle pipelines triangles, and the others points.
    /// Nodes whose vertices live only on the GPU add nothing.
    pub fn overlay_data_primitives_for_axes(&self, axes_index: usize, limit: usize) -> Vec<DataPrimitive> {
        let mut nodes = Vec::new();
        let mut total = 0usize;
        for node in self.scene.get_visible_nodes() {
            if node.axes_index != axes_index {
                continue;
            }
            let Some(data) = node.render_data.as_ref() else { continue };
            if data.vertices.is_empty() {
                continue;
            }
            let order: Vec<u32> = match &data.indices {
                Some(indices) => indices.clone(),
                None => (0..data.vertices.len() as u32).collect(),
            };
            let per = match data.pipeline_type {
                PipelineType::Lines | PipelineType::LinesNoDepth => 2,
                PipelineType::Triangles => 3,
                _ => 1,
            };
            total += order.len() / per;
            nodes.push((node.transform, data, order, per));
        }
        let stride = total.div_ceil(limit.max(1)).max(1);
        let mut out = Vec::with_capacity(total.min(limit));
        for (transform, data, order, per) in nodes {
            let at = |i: u32| -> Option<[f32; 2]> {
                let v = data.vertices.get(i as usize)?;
                let p = transform.transform_point3(Vec3::from(v.position));
                (p.x.is_finite() && p.y.is_finite()).then_some([p.x, p.y])
            };
            for group in order.chunks_exact(per).step_by(stride) {
                let primitive = match *group {
                    [a] => at(a).map(DataPrimitive::Point),
                    [a, b] => at(a).zip(at(b)).map(|(a, b)| DataPrimitive::Segment([a, b])),
                    [a, b, c] => match (at(a), at(b), at(c)) {
                        (Some(a), Some(b), Some(c)) => Some(DataPrimitive::Triangle([a, b, c])),
                        _ => None,
                    },
                    _ => None,
                };
                out.extend(primitive);
            }
        }
        out
    }
}
