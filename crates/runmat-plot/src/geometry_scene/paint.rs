//! How selection, hover and role highlights color a geometry scene's regions: triangle regions
//! filled or tinted, and a line chunk's segment regions (a face's boundary edges) drawn in the
//! selection or hover color.
use super::{GeometrySceneChunk, GEOMETRY_HOVER_REGION_COLOR, GEOMETRY_SELECTED_REGION_COLOR};
use crate::core::{PipelineType, RenderData, Vertex};

/// Selection keeps the face's shading: the face color moves this far toward the selection color,
/// and the face's boundary edges draw in the full selection color (as SolidWorks shows it).
pub(super) const GEOMETRY_SELECTED_TINT: f32 = 0.4;
/// The color a selected face is tinted toward: a lighter orange than its edges, so the face
/// reads as the base material under a highlight.
pub(super) const GEOMETRY_SELECTED_FACE_TINT: [f32; 4] = [1.0, 0.62, 0.28, 1.0];
/// Hover is a lighter version of the same treatment.
pub(super) const GEOMETRY_HOVER_TINT: f32 = 0.3;
pub(super) const GEOMETRY_HOVER_EDGE_COLOR: [f32; 4] = [0.16, 0.55, 0.96, 1.0];
pub(super) const SELECTED_PAINT: RegionPaint = RegionPaint::Tint {
    face: GEOMETRY_SELECTED_FACE_TINT,
    amount: GEOMETRY_SELECTED_TINT,
    edge: GEOMETRY_SELECTED_REGION_COLOR,
};
pub(super) const HOVER_PAINT: RegionPaint = RegionPaint::Tint {
    face: GEOMETRY_HOVER_REGION_COLOR,
    amount: GEOMETRY_HOVER_TINT,
    edge: GEOMETRY_HOVER_EDGE_COLOR,
};

/// How a region is colored: filled with one color (role highlights, staged and preview
/// geometry), or tinted so the shading still reads (selection and hover).
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) enum RegionPaint {
    Fill([f32; 4]),
    /// An analysis color: the faces take it at their own alpha (so display modes still apply)
    /// and the edges stay as drawn.
    Face([f32; 4]),
    Tint {
        face: [f32; 4],
        amount: f32,
        edge: [f32; 4],
    },
}

impl RegionPaint {
    pub(super) fn face(self, base: [f32; 4]) -> [f32; 4] {
        match self {
            Self::Fill(color) => color,
            Self::Face(color) => [color[0], color[1], color[2], base[3].min(color[3])],
            Self::Tint { face, amount, .. } => {
                let amount = tint_on(base, amount);
                [
                    base[0] + (face[0] - base[0]) * amount,
                    base[1] + (face[1] - base[1]) * amount,
                    base[2] + (face[2] - base[2]) * amount,
                    base[3],
                ]
            }
        }
    }

    /// A segment's color: the paint's edge color at the line's own alpha, so hidden edge
    /// overlays stay hidden.
    pub(super) fn edge(self, base: [f32; 4]) -> [f32; 4] {
        let color = match self {
            Self::Fill(color) => color,
            Self::Face(_) => return base,
            Self::Tint { edge, .. } => edge,
        };
        [color[0], color[1], color[2], base[3].min(color[3])]
    }

    /// How much wider a painted segment draws than the scene's edge width: selected edges
    /// stand out by weight as well as color, as SolidWorks draws them; hover a little less.
    pub(super) fn edge_weight(self) -> f32 {
        match self {
            Self::Fill(_) | Self::Face(_) => 1.0,
            Self::Tint { amount, .. } if amount >= GEOMETRY_SELECTED_TINT => SELECTED_EDGE_WEIGHT,
            Self::Tint { .. } => HOVER_EDGE_WEIGHT,
        }
    }
}

/// Below this luminance a material counts as dark: a tint toward the selection color, then
/// lit, would read as black on carbon or black nylon.
pub(crate) const DARK_LUMINANCE: f32 = 0.25;
/// How much stronger a tint gets on a fully dark material (selection 0.4 → 0.76, hover 0.3 →
/// 0.57), so a selected black part reads orange and hover stays lighter than selection.
pub(crate) const DARK_TINT_BOOST: f32 = 0.9;

/// The tint amount on a face of color `base`: `amount` on light materials, raised on dark ones
/// so the highlight shows. The instanced triangle shader computes the same.
pub(crate) fn tint_on(base: [f32; 4], amount: f32) -> f32 {
    let luminance = 0.2126 * base[0] + 0.7152 * base[1] + 0.0722 * base[2];
    let darkness = (1.0 - luminance / DARK_LUMINANCE).clamp(0.0, 1.0);
    (amount * (1.0 + DARK_TINT_BOOST * darkness)).min(1.0)
}

/// Selected edges draw at twice the scene's edge width (1.25 px edges select at 2.5 px).
pub(super) const SELECTED_EDGE_WEIGHT: f32 = 2.0;
/// Hovered edges draw at 1.6 times the edge width.
pub(super) const HOVER_EDGE_WEIGHT: f32 = 1.6;

impl GeometrySceneChunk {
    pub(super) fn apply_region_color(
        &self,
        render_data: &mut RenderData,
        region_id: &str,
        paint: RegionPaint,
    ) -> usize {
        if matches!(
            self.render_data.pipeline_type,
            PipelineType::Lines | PipelineType::LinesNoDepth
        ) {
            if matches!(paint, RegionPaint::Face(_)) {
                return 0;
            }
            return self.apply_segment_region_color(render_data, region_id, paint);
        }
        if self.render_data.pipeline_type != PipelineType::Triangles {
            return 0;
        }
        let Some(region) = self.regions.iter().find(|item| item.region_id == region_id) else {
            log::debug!(
                target: "runmat_plot",
                "geometry_scene.selection_color_missing chunk_id={} region_id={} available_regions={}",
                self.chunk_id,
                region_id,
                self.regions.len()
            );
            return 0;
        };
        let mut colored = 0usize;
        for range in &region.triangle_ranges {
            let start = range.start as usize;
            let end = start.saturating_add(range.count as usize);
            for primitive in start..end {
                if recolor_primitive(
                    self.indices.as_deref(),
                    render_data,
                    primitive,
                    3,
                    |vertex| vertex.color = paint.face(vertex.color),
                ) {
                    colored += 1;
                }
            }
        }
        colored
    }

    /// Colors the segments of a line chunk's region (a selected face's boundary edges).
    fn apply_segment_region_color(
        &self,
        render_data: &mut RenderData,
        region_id: &str,
        paint: RegionPaint,
    ) -> usize {
        let Some(region) = self
            .segment_regions
            .iter()
            .find(|item| item.region_id == region_id)
        else {
            return 0;
        };
        let mut colored = 0usize;
        for range in &region.triangle_ranges {
            let start = range.start as usize;
            let end = start.saturating_add(range.count as usize);
            for primitive in start..end {
                if recolor_primitive(
                    self.indices.as_deref(),
                    render_data,
                    primitive,
                    2,
                    |vertex| {
                        vertex.color = paint.edge(vertex.color);
                        // The stroke weight rides on the line vertex's spare UV until
                        // `edge_stroke::widen` turns segments into strokes.
                        vertex.tex_coords[1] = paint.edge_weight();
                    },
                ) {
                    colored += 1;
                }
            }
        }
        colored
    }

    pub(super) fn apply_triangle_color(
        &self,
        render_data: &mut RenderData,
        triangle_index: usize,
        paint: RegionPaint,
    ) {
        if self.render_data.pipeline_type != PipelineType::Triangles {
            return;
        }
        recolor_primitive(
            self.indices.as_deref(),
            render_data,
            triangle_index,
            3,
            |vertex| vertex.color = paint.face(vertex.color),
        );
    }
}

/// Recolors one primitive (`corners` vertices: 3 for a triangle, 2 for a segment). Indexed data
/// gets its own copies of the primitive's vertices first, so a shared vertex doesn't carry the
/// color into neighbouring primitives. Returns whether the primitive was found.
fn recolor_primitive(
    source_indices: Option<&[u32]>,
    render_data: &mut RenderData,
    primitive: usize,
    corners: usize,
    paint: impl Fn(&mut Vertex),
) -> bool {
    let base = primitive.saturating_mul(corners);
    let (Some(source_indices), Some(render_indices)) = (source_indices, render_data.indices.as_mut())
    else {
        if render_data.indices.is_some() {
            return false;
        }
        let Some(vertices) = render_data.vertices.get_mut(base..base + corners) else {
            return false;
        };
        for vertex in vertices {
            paint(vertex);
        }
        return true;
    };
    // The primitive's current corners: an earlier paint (an analysis color under a selection)
    // may already have isolated them, and this paint composes over its result.
    if source_indices.get(base..base + corners).is_none() {
        return false;
    }
    let Some(target) = render_indices.get_mut(base..base + corners) else {
        return false;
    };
    let current: [u32; 3] = std::array::from_fn(|k| target.get(k).copied().unwrap_or(0));
    let mut isolated = [0_u32; 3];
    for (slot, index) in current[..corners].iter().copied().enumerate() {
        let Some(mut vertex) = render_data.vertices.get(index as usize).copied() else {
            return false;
        };
        paint(&mut vertex);
        let Ok(next_index) = u32::try_from(render_data.vertices.len()) else {
            return false;
        };
        render_data.vertices.push(vertex);
        isolated[slot] = next_index;
    }
    target.copy_from_slice(&isolated[..corners]);
    true
}

