use super::{Figure, PlotElement};
use crate::core::BoundingBox;

/// Domain-owned extents for axes resolution, without creating render buffers.
/// Contour decorations and reference ranges retain their existing limit policy.
pub(crate) struct PreparedAxesExtents {
    pub bounds: Vec<Option<BoundingBox>>,
}

impl Figure {
    pub(crate) fn prepare_axes_extents(&mut self, viewports: &[(u32, u32)]) -> PreparedAxesExtents {
        let base = self.reference_base_bounds_by_axes();
        let mut bounds: Vec<Option<BoundingBox>> = vec![None; self.total_axes().max(1)];
        for (index, plot) in self.plots.iter_mut().enumerate() {
            if !plot.is_visible() {
                continue;
            }
            let axes = self
                .plot_axes_indices
                .get(index)
                .copied()
                .unwrap_or(0)
                .min(bounds.len() - 1);
            let extent = plot.prepared_axes_extent(
                viewports.get(axes).copied(),
                ReferenceExtentContext {
                    x_limits: self.x_limits,
                    y_limits: self.y_limits,
                    metadata: self.axes_metadata.get(axes),
                    base: base[axes],
                },
            );
            let Some(extent) = extent else {
                continue;
            };
            bounds[axes] = Some(bounds[axes].map_or(extent, |old| old.union(&extent)));
        }
        PreparedAxesExtents { bounds }
    }
}

struct ReferenceExtentContext<'a> {
    x_limits: Option<(f64, f64)>,
    y_limits: Option<(f64, f64)>,
    metadata: Option<&'a super::AxesMetadata>,
    base: Option<BoundingBox>,
}

impl PlotElement {
    /// Exhaustive render-extent policy, distinct from general data bounds.
    /// New plot kinds must explicitly declare how their render packets affect limits.
    fn prepared_axes_extent(
        &mut self,
        viewport: Option<(u32, u32)>,
        reference: ReferenceExtentContext<'_>,
    ) -> Option<BoundingBox> {
        match self {
            Self::Line(plot) => Some(plot.bounds()),
            Self::Scatter(plot) => Some(plot.bounds()),
            Self::Bar(plot) => Some(plot.bounds()),
            // Bounds already include domain cap/base/arrow/shape policies. Marker,
            // mesh edge/vector and patch edge packets add no different extents.
            Self::ErrorBar(plot) => Some(plot.bounds()),
            Self::Stairs(plot) => Some(plot.bounds()),
            Self::Stem(plot) => Some(plot.bounds()),
            Self::Area(plot) => Some(plot.bounds()),
            Self::Quiver(plot) => Some(plot.bounds()),
            // Pie render packets intentionally do not contribute auto-limit extents.
            Self::Pie(_) => None,
            // Includes image-mode/flattened surfaces; their render packet uses bounds().
            Self::Surface(plot) => Some(plot.bounds()),
            Self::Mesh(plot) => Some(plot.bounds()),
            Self::Patch(plot) => Some(plot.bounds()),
            Self::Line3(plot) => Some(plot.bounds()),
            Self::Scatter3(plot) => Some(plot.bounds()),
            Self::Contour(plot) => Some(plot.prepared_bounds(viewport)),
            Self::ContourFill(plot) => Some(plot.bounds()),
            Self::ReferenceLine(line) => {
                let (x, y) = Figure::reference_line_ranges(
                    reference.x_limits,
                    reference.y_limits,
                    reference.metadata,
                    reference.base,
                    line,
                );
                Some(line.range_bounds(x, y))
            }
        }
    }
}

#[cfg(test)]
#[path = "preparation/tests.rs"]
mod tests;
