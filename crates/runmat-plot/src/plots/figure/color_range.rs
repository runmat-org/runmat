use super::{Figure, PlotElement};
use crate::plots::surface::ScalarColorRange;

impl Figure {
    /// Automatic range for the scalar plots governed by axes color limits.
    /// Scatter and mesh use their own local color policies; contours retain
    /// precolored vertices and are not recolored by set_axes_color_limits.
    pub fn axes_scalar_color_range(&self, axes_index: usize) -> ScalarColorRange {
        let mut range: Option<(f64, f64)> = None;
        for (plot, axes) in self.plots().zip(self.plot_axes_indices().iter().copied()) {
            if axes != axes_index || !plot.is_visible() {
                continue;
            }
            if let PlotElement::Surface(surface) = plot {
                match surface.scalar_color_range() {
                    ScalarColorRange::Known(lower, upper) => {
                        range = Some(
                            range.map_or((lower, upper), |(lo, hi)| (lo.min(lower), hi.max(upper))),
                        );
                    }
                    ScalarColorRange::Unavailable => return ScalarColorRange::Unavailable,
                    ScalarColorRange::NotMapped | ScalarColorRange::Empty => {}
                }
            }
        }
        range.map_or(ScalarColorRange::Empty, |(lo, hi)| {
            ScalarColorRange::Known(lo, hi)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plots::surface::{SurfaceColorMapping, SurfacePlot};

    fn surface(low: f64, high: f64) -> SurfacePlot {
        SurfacePlot::new(vec![100., 200.], vec![300.], vec![vec![low], vec![high]]).unwrap()
    }

    #[test]
    fn historical_scene_without_mapping_defaults_to_scaled() {
        let mut figure = Figure::new();
        figure.add_surface_plot(surface(20., 40.));
        let mut json = serde_json::to_value(crate::event::FigureScene::capture(&figure)).unwrap();
        json["plots"][0]
            .as_object_mut()
            .unwrap()
            .remove("color_mapping");
        let scene: crate::event::FigureScene = serde_json::from_value(json).unwrap();
        let restored = scene.into_figure().unwrap();
        assert_eq!(
            restored.axes_scalar_color_range(0),
            ScalarColorRange::Known(20., 40.)
        );
    }

    #[test]
    fn axes_color_range_unions_scaled_contributors_and_preserves_direct_limits() {
        let mut figure = Figure::new();
        figure.add_surface_plot(surface(20., 30.));
        figure.add_surface_plot(surface(10., 40.));
        let direct = figure.add_surface_plot(
            surface(-100., 100.)
                .with_color_mapping(SurfaceColorMapping::Direct)
                .with_color_limits(Some((0., 255.))),
        );
        assert_eq!(
            figure.axes_scalar_color_range(0),
            ScalarColorRange::Known(10., 40.)
        );
        figure.set_axes_color_limits(0, Some((10., 40.)));
        let PlotElement::Surface(surface) = figure.plots().nth(direct).unwrap() else {
            panic!("surface");
        };
        assert_eq!(surface.color_limits, Some((0., 255.)));
        let scene = crate::event::FigureScene::capture(&figure);
        let restored = scene.into_figure().unwrap();
        assert_eq!(
            restored.axes_scalar_color_range(0),
            ScalarColorRange::Known(10., 40.)
        );
        let PlotElement::Surface(direct) = restored.plots().nth(direct).unwrap() else {
            panic!("surface");
        };
        assert_eq!(direct.color_mapping, SurfaceColorMapping::Direct);
    }
}
