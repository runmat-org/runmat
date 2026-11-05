//! Geometry edges drawn as screen-space strokes of a set width.
//!
//! GPU line primitives are one device pixel wide, so edges thin out on high-density screens and
//! lose contrast against dark faces. When the presentation asks for an edge width above 1 CSS px,
//! line chunks are drawn as two-triangle strokes that the edge stroke shader widens in screen
//! space. The chunk keeps its line data for picking, clipping and export; only the drawn node
//! changes.
use crate::core::{DrawCall, PipelineType, RenderData, Vertex};

/// Edge widths at or below this many CSS px draw as plain GPU lines.
pub const HAIRLINE_WIDTH_PX: f32 = 1.0;

/// Whether `render_data` is a line item that draws as a stroke at `width_px`.
pub fn widens(render_data: &RenderData, width_px: Option<f32>) -> bool {
    (width_px.is_some_and(|w| w.is_finite() && w > HAIRLINE_WIDTH_PX)
        || render_data.vertices.iter().any(|v| weight(v) > 1.0))
        && render_data.gpu_vertices.is_none()
        && matches!(
            render_data.pipeline_type,
            PipelineType::Lines | PipelineType::LinesNoDepth
        )
}

/// Line render data turned into edge strokes `width_px` CSS px wide. Depth-tested lines become
/// [`PipelineType::EdgeStroke`] and depth-free lines [`PipelineType::EdgeStrokeNoDepth`]. Other
/// render data is returned unchanged.
pub fn widen(render_data: RenderData, width_px: Option<f32>) -> RenderData {
    if !widens(&render_data, width_px) {
        return render_data;
    }
    let width_px = width_px.unwrap_or(HAIRLINE_WIDTH_PX);
    let pipeline_type = if render_data.pipeline_type == PipelineType::LinesNoDepth {
        PipelineType::EdgeStrokeNoDepth
    } else {
        PipelineType::EdgeStroke
    };
    let mut vertices = Vec::new();
    for segment in segments(&render_data) {
        append_segment(&mut vertices, segment, width_px);
    }
    let vertex_count = vertices.len();
    RenderData {
        pipeline_type,
        vertices,
        indices: None,
        gpu_vertices: None,
        bounds: render_data.bounds,
        material: render_data.material,
        draw_calls: if vertex_count == 0 {
            Vec::new()
        } else {
            vec![DrawCall {
                vertex_offset: 0,
                vertex_count,
                index_offset: None,
                index_count: None,
                instance_count: 1,
            }]
        },
        image: None,
    }
}

/// The vertex pairs the line item's draw calls draw.
fn segments(render_data: &RenderData) -> Vec<[&Vertex; 2]> {
    let vertices = &render_data.vertices;
    let mut out = Vec::new();
    for call in &render_data.draw_calls {
        match (&render_data.indices, call.index_offset, call.index_count) {
            (Some(indices), Some(offset), Some(count)) => {
                let end = offset.saturating_add(count).min(indices.len());
                for pair in indices[offset.min(end)..end].chunks_exact(2) {
                    let (Some(a), Some(b)) = (
                        vertices.get(pair[0] as usize + call.vertex_offset),
                        vertices.get(pair[1] as usize + call.vertex_offset),
                    ) else {
                        continue;
                    };
                    out.push([a, b]);
                }
            }
            _ => {
                let start = call.vertex_offset.min(vertices.len());
                let end = start.saturating_add(call.vertex_count).min(vertices.len());
                for pair in vertices[start..end].chunks_exact(2) {
                    out.push([&pair[0], &pair[1]]);
                }
            }
        }
    }
    out
}

/// A line vertex's stroke weight: above 1 where selection or hover painted its segment (the
/// paint stores it in the vertex's second UV), else 1.
fn weight(vertex: &Vertex) -> f32 {
    let w = vertex.tex_coords[1];
    if w.is_finite() && w > 1.0 {
        w
    } else {
        1.0
    }
}

fn append_segment(out: &mut Vec<Vertex>, [a, b]: [&Vertex; 2], width_px: f32) {
    if a.position == b.position || (a.color[3] <= 0.0 && b.color[3] <= 0.0) {
        return;
    }
    let width_px = width_px * weight(a).max(weight(b));
    let corner = |own: &Vertex, other: &Vertex, side: f32| {
        let mut v = *own;
        v.normal = other.position;
        v.tex_coords = [side, width_px];
        v
    };
    // The tangent reverses at the far endpoint, so its side signs do too.
    let a_minus = corner(a, b, -1.0);
    let a_plus = corner(a, b, 1.0);
    let b_minus = corner(b, a, 1.0);
    let b_plus = corner(b, a, -1.0);
    out.extend_from_slice(&[a_minus, a_plus, b_plus, a_minus, b_plus, b_minus]);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::Material;
    use glam::{Vec3, Vec4};

    fn lines(pipeline_type: PipelineType, indexed: bool) -> RenderData {
        let vertices = vec![
            Vertex::new(Vec3::ZERO, Vec4::ONE),
            Vertex::new(Vec3::X, Vec4::ONE),
            Vertex::new(Vec3::Y, Vec4::ONE),
        ];
        let indices = indexed.then(|| vec![0, 1, 1, 2]);
        RenderData {
            pipeline_type,
            draw_calls: vec![DrawCall {
                vertex_offset: 0,
                vertex_count: vertices.len(),
                index_offset: indexed.then_some(0),
                index_count: indexed.then_some(4),
                instance_count: 1,
            }],
            vertices,
            indices,
            gpu_vertices: None,
            bounds: None,
            material: Material::default(),
            image: None,
        }
    }

    #[test]
    fn hairline_widths_keep_gpu_lines() {
        for width in [None, Some(1.0), Some(0.5), Some(f32::NAN)] {
            let out = widen(lines(PipelineType::Lines, true), width);
            assert_eq!(out.pipeline_type, PipelineType::Lines);
            assert_eq!(out.vertices.len(), 3);
        }
    }

    #[test]
    fn indexed_lines_become_two_triangle_strokes_with_width() {
        let out = widen(lines(PipelineType::Lines, true), Some(1.25));
        assert_eq!(out.pipeline_type, PipelineType::EdgeStroke);
        assert_eq!(out.vertices.len(), 12, "two segments, six vertices each");
        assert_eq!(out.draw_calls[0].vertex_count, 12);
        let first = &out.vertices[0];
        assert_eq!(first.position, [0.0, 0.0, 0.0]);
        assert_eq!(first.normal, [1.0, 0.0, 0.0], "other endpoint");
        assert_eq!(first.tex_coords, [-1.0, 1.25]);
        let sides: Vec<f32> = out.vertices[..6].iter().map(|v| v.tex_coords[0]).collect();
        assert_eq!(sides, vec![-1.0, 1.0, -1.0, -1.0, -1.0, 1.0]);
    }

    #[test]
    fn depth_free_lines_become_depth_free_strokes() {
        let out = widen(lines(PipelineType::LinesNoDepth, false), Some(2.0));
        assert_eq!(out.pipeline_type, PipelineType::EdgeStrokeNoDepth);
        assert_eq!(
            out.vertices.len(),
            6,
            "one pair in three unindexed vertices"
        );
    }

    #[test]
    fn cleared_draw_calls_and_invisible_segments_draw_nothing() {
        let mut hidden = lines(PipelineType::Lines, true);
        hidden.draw_calls.clear();
        assert!(widen(hidden, Some(1.25)).draw_calls.is_empty());
        let mut clear = lines(PipelineType::Lines, true);
        for v in &mut clear.vertices {
            v.color[3] = 0.0;
        }
        assert!(widen(clear, Some(1.25)).vertices.is_empty());
    }

    #[test]
    fn triangles_are_left_alone() {
        let out = widen(lines(PipelineType::Triangles, true), Some(1.25));
        assert_eq!(out.pipeline_type, PipelineType::Triangles);
    }
}
