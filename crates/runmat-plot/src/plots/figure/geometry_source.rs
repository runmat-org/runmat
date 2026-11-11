//! The geometry scene a figure was imported from.
//!
//! A saved field result or model view is a figure scene written from a geometry scene: it
//! carries a legend and color bar (the overlay), outline lines and the physical unit context
//! next to its meshes. A figure restored from such a scene keeps that geometry scene, and the
//! renderer draws it the way it draws a live result. Adding, removing or clearing plots drops
//! it, because the figure then no longer shows the saved scene.

use std::sync::Arc;

use super::Figure;
use crate::geometry_scene::GeometryScene;

impl Figure {
    /// Attach the geometry scene this figure was imported from.
    pub fn with_geometry_source(mut self, scene: GeometryScene) -> Self {
        self.geometry_source = Some(Arc::new(scene));
        self
    }

    /// The geometry scene this figure was imported from, while its plots are unchanged.
    pub fn geometry_source(&self) -> Option<&GeometryScene> {
        self.geometry_source.as_deref()
    }
}
