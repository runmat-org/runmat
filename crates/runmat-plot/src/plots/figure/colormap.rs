//! Figure defaults and explicit axes palettes share the physical figure owner.
use super::*;

impl Figure {
    pub(super) fn inherited_axes_metadata(&self) -> AxesMetadata {
        AxesMetadata {
            colormap: self.default_colormap.clone(),
            colormap_length: self.default_colormap_length,
            ..Self::default_axes_metadata()
        }
    }

    pub fn set_axes_colormap_with_length(&mut self, axes: usize, map: ColorMap, length: usize) {
        self.set_axes_colormap(axes, map);
        self.axes_metadata[axes].colormap_length = length;
    }

    /// Set existing axes and the default inherited by subsequently created/reset axes.
    /// Does not select a different axes or replace axes identity.
    pub fn set_figure_colormap(&mut self, map: ColorMap, length: usize) {
        self.default_colormap = map.clone();
        self.default_colormap_length = length;
        for axes in 0..self.axes_metadata.len() {
            self.set_axes_colormap_with_length(axes, map.clone(), length);
        }
        self.dirty = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn palette_defaults_inherit_without_selecting_or_erasing_axes_overrides() {
        let mut figure = Figure::new();
        figure.set_subplot_grid(1, 2);
        figure.set_active_axes_index(1);
        figure.set_figure_colormap(ColorMap::Hot, 7);
        figure.set_axes_colormap_with_length(0, ColorMap::Jet, 3);
        figure.set_subplot_grid(1, 3);
        assert_eq!(figure.active_axes_index, 1);
        assert!(matches!(figure.axes_metadata[0].colormap, ColorMap::Jet));
        assert_eq!(figure.axes_metadata[0].colormap_length, 3);
        assert!(matches!(figure.axes_metadata[2].colormap, ColorMap::Hot));
        assert_eq!(figure.axes_metadata[2].colormap_length, 7);
        figure.reset_axes_properties(0);
        assert!(matches!(figure.axes_metadata[0].colormap, ColorMap::Hot));
        assert_eq!(figure.axes_metadata[0].colormap_length, 7);
    }

    #[test]
    fn scene_round_trip_preserves_defaults_and_per_axes_palette_lengths() {
        let mut figure = Figure::new();
        figure.set_figure_colormap(ColorMap::Hot, 7);
        figure.set_axes_colormap_with_length(0, ColorMap::Jet, 3);
        let scene = crate::event::FigureScene::capture(&figure);
        let mut restored = scene.into_figure().unwrap();
        assert_eq!(restored.axes_metadata[0].colormap_length, 3);
        restored.set_subplot_grid(1, 2);
        assert!(matches!(restored.axes_metadata[1].colormap, ColorMap::Hot));
        assert_eq!(restored.axes_metadata[1].colormap_length, 7);
    }

    #[test]
    fn old_scenes_without_palette_length_fields_keep_default_length() {
        let scene = crate::event::FigureScene::capture(&Figure::new());
        let mut json = serde_json::to_value(scene).unwrap();
        let metadata = json["metadata"].as_object_mut().unwrap();
        metadata.remove("defaultColormap");
        metadata.remove("defaultColormapLength");
        for axes in metadata["axesMetadata"].as_array_mut().unwrap() {
            axes.as_object_mut().unwrap().remove("colormapLength");
        }
        let scene: crate::event::FigureScene = serde_json::from_value(json).unwrap();
        let restored = scene.into_figure().unwrap();
        assert_eq!(restored.default_colormap_length, 256);
        assert_eq!(restored.axes_metadata[0].colormap_length, 256);
    }
}
