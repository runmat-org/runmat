//! Shared scene dimensionality, including the renderer's patch precision rule.
use super::*;

impl PlotElement {
    pub fn is_3d_content(&self) -> bool {
        match self {
            Self::Surface(surface) => !surface.image_mode,
            Self::Mesh(_) | Self::Line3(_) | Self::Scatter3(_) => true,
            Self::Patch(patch) => {
                if patch.force_3d() {
                    return true;
                }
                let mut max_xy = 0.0_f32;
                let mut max_z = 0.0_f32;
                for point in patch.vertices() {
                    max_xy = max_xy.max(point.x.abs().max(point.y.abs()));
                    max_z = max_z.max(point.z.abs());
                }
                max_z > 1e-9_f32.max(max_xy * 1e-6)
            }
            Self::Contour(contour) => contour.is_3d(),
            _ => false,
        }
    }
}

impl Figure {
    pub fn axes_has_3d_content(&self, axes_index: usize) -> bool {
        self.plots()
            .zip(self.plot_axes_indices().iter().copied())
            .any(|(plot, axes)| axes == axes_index && plot.is_3d_content())
    }
}
