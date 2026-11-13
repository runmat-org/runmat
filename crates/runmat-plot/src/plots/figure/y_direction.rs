//! Y orientation transforms never reverse the stored numeric limits.
use glam::{Mat4, Vec3};
use serde::{Deserialize, Serialize};

impl super::Figure {
    pub fn set_axes_y_direction(&mut self, axes_index: usize, direction: YDirection) {
        self.ensure_axes_metadata_capacity(axes_index + 1);
        self.axes_metadata[axes_index].y_direction = direction;
        self.dirty = true;
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum YDirection {
    #[default]
    Normal,
    Reverse,
}

impl YDirection {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Normal => "normal",
            Self::Reverse => "reverse",
        }
    }

    /// Reflect a coordinate in a display interval (pixels or normalized units).
    pub fn orient(self, coordinate: f32, lower: f32, upper: f32) -> f32 {
        match self {
            Self::Normal => coordinate,
            Self::Reverse => lower + upper - coordinate,
        }
    }

    /// Data-space reflection works with a perspective or orthographic camera.
    pub fn model_matrix(self, lower: f32, upper: f32) -> Mat4 {
        match self {
            Self::Normal => Mat4::IDENTITY,
            Self::Reverse => {
                Mat4::from_translation(Vec3::new(0., lower + upper, 0.))
                    * Mat4::from_scale(Vec3::new(1., -1., 1.))
            }
        }
    }

    /// Swap only output endpoints, after a direct shader's logarithmic mapping.
    pub fn ndc_endpoints(self) -> (f32, f32) {
        (self.orient(-1., -1., 1.), self.orient(1., -1., 1.))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn direction_survives_figure_metadata_serialization_and_old_payload_defaults() {
        let mut figure = super::super::Figure::new();
        figure.set_axes_y_direction(0, YDirection::Reverse);
        let metadata = figure.axes_metadata(0).unwrap().clone();
        let serialized = crate::event::SerializedAxesMetadata::from(metadata);
        let mut value = serde_json::to_value(serialized).unwrap();
        // The event transport's established serde contract is camelCase.
        assert_eq!(value["yDirection"], "reverse");
        assert!(value.get("y_direction").is_none());
        let restored: crate::event::SerializedAxesMetadata =
            serde_json::from_value(value.clone()).unwrap();
        let restored: super::super::AxesMetadata = restored.into();
        assert_eq!(restored.y_direction, YDirection::Reverse);
        value.as_object_mut().unwrap().remove("yDirection");
        let old: crate::event::SerializedAxesMetadata = serde_json::from_value(value).unwrap();
        assert_eq!(old.y_direction, YDirection::Normal);
    }

    #[test]
    fn camera_projection_reflects_data_y_not_screen_axes() {
        let mut camera = crate::core::Camera::new();
        let projection = camera.view_proj_matrix();
        let direction = YDirection::Reverse.model_matrix(2., 8.);
        let lower = Vec3::new(4., 2., 7.).extend(1.);
        let upper = Vec3::new(4., 8., 7.).extend(1.);
        assert!((projection * direction * lower - projection * upper).length() < 1e-5);
        assert!((projection * direction * upper - projection * lower).length() < 1e-5);
    }
    #[test]
    fn reflection_keeps_limits_ordered_and_is_involutive() {
        let limits = (2., 8.);
        let matrix = YDirection::Reverse.model_matrix(limits.0, limits.1);
        assert_eq!(
            matrix.transform_point3(Vec3::new(4., 2., 7.)),
            Vec3::new(4., 8., 7.)
        );
        assert_eq!(
            matrix.transform_point3(Vec3::new(4., 8., 7.)),
            Vec3::new(4., 2., 7.)
        );
        assert_eq!(matrix * matrix, Mat4::IDENTITY);
        assert_eq!(limits, (2., 8.));
        assert_eq!(YDirection::Reverse.ndc_endpoints(), (1., -1.));
        assert_eq!(YDirection::Reverse.orient(20., 10., 90.), 80.);
    }
}
