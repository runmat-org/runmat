//! Reset axes appearance without replacing their structural identity.
use super::*;

impl Figure {
    pub fn reset_axes_properties(&mut self, axes_index: usize) {
        let Some(previous) = self.axes_metadata.get(axes_index) else {
            return;
        };
        let mut reset = self.inherited_axes_metadata();
        reset.position = previous.position;
        reset.position_explicit = previous.position_explicit;
        reset.units = previous.units.clone();
        reset.axes_kind = previous.axes_kind;
        reset.overlay_parent = previous.overlay_parent;
        if reset.overlay_parent.is_some() {
            reset.y_axis_location = previous.y_axis_location.clone();
        }
        // A linked colorbar is a figure child with an independent lifetime.
        reset.colorbar_enabled = previous.colorbar_enabled;
        reset.view_revision = previous.view_revision.wrapping_add(1);
        self.axes_metadata[axes_index] = reset;
        if axes_index == self.active_axes_index {
            self.sync_legacy_fields_from_active_axes();
        }
        self.dirty = true;
    }
}
