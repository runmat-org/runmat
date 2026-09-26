//! Shared direction mapping for camera-based data and direct/overlay paths.
use super::*;
use crate::plots::figure::YDirection;

impl PlotRenderer {
    pub fn overlay_y_direction_for_axes(&self, axes_index: usize) -> YDirection {
        self.last_figure
            .as_ref()
            .and_then(|figure| figure.axes_metadata(axes_index))
            .map(|meta| meta.y_direction)
            .unwrap_or_default()
    }

    pub fn axes_model_matrix(&self, axes_index: usize) -> Mat4 {
        let direction = self.overlay_y_direction_for_axes(axes_index);
        let limits = if self.axes_has_3d_content(axes_index) {
            self.display_bounds_3d_for_axes(axes_index)
                .map(|bounds| (bounds.min.y, bounds.max.y))
        } else {
            self.view_bounds_for_axes(axes_index)
                .map(|(_, _, lower, upper)| (lower as f32, upper as f32))
        }
        .unwrap_or((0., 1.));
        direction.model_matrix(limits.0, limits.1)
    }
}
