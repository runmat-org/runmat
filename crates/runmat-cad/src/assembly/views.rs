//! Saved model views of an assembly, each a named view with its camera:
//! `view hero = { from = [-0.62, 0.7, -0.36], up = [0, 0, 1], default = true }`. `from` points
//! from the model toward the eye (any length); the canvas frames the model along it with `up`
//! kept upright, and opens on the `default` view.

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ModelView {
    pub name: String,
    /// From the model toward the eye.
    pub from: [f64; 3],
    /// The direction kept upright on screen (+z unless set).
    #[serde(default = "z_up")]
    pub up: [f64; 3],
    /// The canvas opens on this view.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub default: bool,
}

fn z_up() -> [f64; 3] {
    [0.0, 0.0, 1.0]
}

impl ModelView {
    /// The view to open on: the one marked `default`, else none.
    pub fn default_of(views: &[ModelView]) -> Option<&ModelView> {
        views.iter().find(|v| v.default)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_view_reads_with_upright_z_unless_set() {
        let v: ModelView =
            serde_json::from_str(r#"{ "name": "hero", "from": [-1, 1, -0.5], "default": true }"#)
                .unwrap();
        assert_eq!(v.up, [0.0, 0.0, 1.0]);
        assert_eq!(
            ModelView::default_of(std::slice::from_ref(&v)).map(|v| v.name.as_str()),
            Some("hero")
        );
        let plain: ModelView =
            serde_json::from_str(r#"{ "name": "side", "from": [0, 1, 0] }"#).unwrap();
        assert!(ModelView::default_of(&[plain]).is_none());
    }
}
