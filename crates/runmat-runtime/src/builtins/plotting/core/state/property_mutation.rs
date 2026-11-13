//! Ephemeral authority for one property operation and its own scene successors.

use super::*;
use std::cell::RefCell;
#[path = "property_mutation/notifications.rs"]
mod notifications;
use notifications::PreparedFigureUpdates;
#[path = "property_mutation/axes_reads.rs"]
mod axes_reads;
pub(crate) use axes_reads::{
    axis_color_limits_snapshot_for_axes_for_property,
    axis_displayed_color_limits_for_axes_for_property,
    axis_displayed_limits_snapshot_for_axes_for_property, axis_limit_method_for_axes_for_property,
    axis_tick_angles_snapshot_for_axes_for_property,
    axis_tick_formats_snapshot_for_axes_for_property,
    axis_tick_labels_snapshot_for_axes_for_property, axis_ticks_snapshot_for_axes_for_property,
    axis_z_limits_snapshot_for_axes_for_property,
};
#[path = "property_mutation/current_axes.rs"]
mod current_axes;
#[path = "property_mutation/current_limits.rs"]
#[cfg(test)]
mod current_limits;
pub(crate) use current_axes::axes_state_snapshot_for_property;
#[cfg(test)]
pub(crate) use current_axes::current_axes_handle_for_figure_for_property;
pub(crate) use current_axes::current_axes_state_for_property;
#[cfg(test)]
pub(crate) use current_limits::{
    set_current_x_limits_for_property, set_current_y_limits_for_property,
};
#[path = "property_mutation/layout.rs"]
mod layout;
pub(crate) use layout::{color_order_for_axes_for_property, color_order_for_figure_for_property};
pub(crate) use layout::{
    configure_subplot_for_property, configure_subplot_state_for_property,
    set_color_order_for_axes_for_property, set_color_order_for_figure_for_property,
};
#[path = "property_mutation/axes_properties.rs"]
mod axes_properties;
pub(crate) use axes_properties::{
    set_axes_position_for_axes_for_property, set_axes_style_for_axes_for_property,
    set_axes_units_for_axes_for_property, set_box_enabled_for_axes_for_property,
    set_color_limits_for_axes_for_property, set_grid_enabled_for_axes_for_property,
    set_legend_for_axes_for_property, set_minor_grid_enabled_for_axes_for_property,
};
#[path = "property_mutation/figure.rs"]
mod figure;
pub(crate) use figure::{
    set_figure_background_color_for_property, set_figure_name_for_property,
    set_figure_number_title_for_property, set_figure_position_for_property,
    set_figure_tag_for_property, set_figure_visible_for_property,
    set_sg_title_properties_for_figure_for_property,
};
#[path = "property_mutation/annotations.rs"]
mod annotations;
pub(crate) use annotations::{
    set_text_annotation_properties_for_axes_for_property,
    set_text_properties_for_axes_for_property, update_text_annotation_position_source_for_property,
};
#[path = "property_mutation/axis_limits.rs"]
mod axis_limits;
pub(crate) use axis_limits::{
    set_axis_color_limits_preserving_selection_for_property, set_axis_equal_for_axes_for_property,
    set_axis_manual_limits_for_axes_for_property, set_axis_tick_angles_for_axes_for_property,
    set_axis_tick_formats_for_axes_for_property, set_axis_tick_labels_for_axes_for_property,
    set_axis_ticks_for_axes_for_property, set_axis_z_limits_preserving_selection_for_property,
    set_data_aspect_ratio_for_axes_for_property, set_log_modes_for_axes_for_property,
    set_view_for_axes_for_property, set_y_axis_location_for_axes_for_property,
};
#[path = "property_mutation/selection.rs"]
mod selection;
pub(crate) use selection::{
    axes_metadata_snapshot_for_property, axis_display_bounds_snapshot_for_axes_for_property,
    data_aspect_ratio_snapshot_for_axes_for_property, plot_child_handle_snapshot_for_property,
    select_axes_for_figure_for_property, select_axes_receiver_for_property,
    select_current_figure_if_exists_for_property,
};
#[path = "property_mutation/chart_metadata.rs"]
mod chart_metadata;
pub(crate) use chart_metadata::{
    set_animated_line_maximum_num_points_for_property, set_heatmap_color_limits_for_property,
    set_heatmap_display_labels_for_property, set_histogram_handle_display_name_for_property,
    update_binscatter_handle_for_plot_for_property, update_histogram2_handle_for_plot_for_property,
    update_histogram_handle_for_plot_for_property,
    update_histogram_handle_metadata_for_plot_for_property,
    update_image_handle_for_plot_for_property,
};
#[path = "property_mutation/aggregate.rs"]
mod aggregate;
pub(crate) use aggregate::{
    update_stackedplot_figure_for_property, update_stackedplot_handle_state_for_property,
    update_textscatter_figure_for_property, update_textscatter_handle_state_for_property,
    update_wordcloud_figure_for_property, update_wordcloud_handle_state_for_property,
};
#[path = "property_mutation/plot_data.rs"]
mod plot_data;
pub(crate) use plot_data::{
    try_update_plot_element_for_property, update_area_plot_for_property,
    update_errorbar_plot_for_property, update_histogram_plot_data_for_property,
    update_image_plot_for_property, update_plot_element_for_property,
    update_quiver_plot_for_property, update_stem_plot_for_property,
};
#[path = "property_mutation/transactions.rs"]
mod transactions;
pub(super) use transactions::{
    with_annotation_target_mut_for_property, with_axes_target_mut_for_property,
    with_figure_mut_for_property,
};

pub(crate) struct PropertyMutationTarget {
    selected: Option<RefCell<Option<GraphicsPropertySelection>>>,
    callback: Option<crate::context::GraphicsCallbackMutationScope>,
}

#[derive(Clone, Copy)]
pub(super) enum PropertyLocation {
    CurrentFigure,
    Figure(FigureHandle),
    Axes(FigureHandle, usize),
    Plot(FigureHandle, usize),
    Child(u64),
    Annotation(FigureHandle, usize, usize),
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum PropertyMutationError {
    #[error(transparent)]
    Admission(#[from] FigureError),
    #[error("{0}")]
    LegacyMessage(String),
}

impl From<String> for PropertyMutationError {
    fn from(message: String) -> Self {
        Self::LegacyMessage(message)
    }
}

impl PropertyMutationTarget {
    pub(crate) const fn alias() -> Self {
        Self {
            selected: None,
            callback: None,
        }
    }

    pub(crate) fn is_selected(&self) -> bool {
        self.selected.is_some()
    }

    pub(super) fn child_key(&self, alias: f64) -> Result<u64, FigureError> {
        match self.selection()? {
            Some(selection) => match selection.receiver().kind() {
                GraphicsReceiverKind::PlotChild { id } => Ok(id),
                _ => Err(FigureError::InvalidPlotObjectHandle),
            },
            None if alias.is_finite() && alias > 0.0 => Ok(alias.round() as u64),
            None => Err(FigureError::InvalidPlotObjectHandle),
        }
    }

    pub(crate) fn selected(selection: GraphicsPropertySelection) -> Self {
        Self {
            selected: Some(RefCell::new(Some(selection))),
            callback: None,
        }
    }

    pub(crate) fn with_runtime(mut self, runtime: Option<&crate::context::RuntimeContext>) -> Self {
        self.callback =
            runtime.and_then(crate::context::RuntimeContext::graphics_callback_mutation);
        self
    }

    pub(super) fn selecting(&self, selection: GraphicsPropertySelection) -> Self {
        Self {
            selected: Some(RefCell::new(Some(selection))),
            callback: self.callback.clone(),
        }
    }

    pub(crate) fn selection(&self) -> Result<Option<GraphicsPropertySelection>, FigureError> {
        self.selected
            .as_ref()
            .map(|slot| slot.borrow().clone().ok_or(FigureError::StalePlotSnapshot))
            .transpose()
    }

    pub(super) fn registry(
        &self,
        location: PropertyLocation,
    ) -> Result<PropertyMutationGuard<'_>, FigureError> {
        let registry = registry();
        let location = match location {
            PropertyLocation::CurrentFigure => PropertyLocation::Figure(registry.current),
            location => location,
        };
        if let Some(selection) = self.selection()? {
            selection.validate(&registry)?;
            selection.validate_location(location)?;
        }
        let figure = match location {
            PropertyLocation::CurrentFigure => None,
            PropertyLocation::Figure(figure)
            | PropertyLocation::Axes(figure, _)
            | PropertyLocation::Plot(figure, _)
            | PropertyLocation::Annotation(figure, _, _) => Some(figure),
            PropertyLocation::Child(id) => registry
                .plot_children
                .get(&id)
                .map(|child| child.figure_axes().0),
        };
        let callback = self
            .callback
            .as_ref()
            .filter(|scope| figure.is_some_and(|figure| figure.as_u32() == scope.figure()))
            .map(|scope| {
                let state = registry
                    .figures
                    .get(&FigureHandle::from(scope.figure()))
                    .ok_or(FigureError::StalePlotSnapshot)?;
                let receiver = state
                    .figure_receiver
                    .as_ref()
                    .ok_or(FigureError::StalePlotSnapshot)?;
                if !scope.admits(receiver, &state.scene_lease) {
                    return Err(FigureError::StalePlotSnapshot);
                }
                Ok((scope.clone(), receiver.clone(), state.scene_lease.clone()))
            })
            .transpose()?;
        Ok(PropertyMutationGuard {
            registry,
            target: self,
            retired: None,
            callback,
        })
    }

    pub(crate) fn ensure_current(&self) -> Result<(), FigureError> {
        if let Some(selection) = self.selection()? {
            selection.ensure_current()?;
        }
        Ok(())
    }

    pub(crate) fn figure_snapshot(
        &self,
        handle: FigureHandle,
    ) -> Result<super::shared_figure::FigureOwner, FigureError> {
        let registry = self.registry(PropertyLocation::Figure(handle))?;
        registry
            .figures
            .get(&handle)
            .map(|state| state.figure.snapshot())
            .ok_or(FigureError::InvalidHandle(handle.as_u32()))
    }

    pub(crate) fn plot_snapshot(
        &self,
        figure: FigureHandle,
        index: usize,
    ) -> Result<PlotElement, FigureError> {
        let registry = self.registry(PropertyLocation::Plot(figure, index))?;
        registry
            .figures
            .get(&figure)
            .and_then(|state| state.figure.plots().nth(index))
            .cloned()
            .ok_or(FigureError::InvalidPlotObjectHandle)
    }

    pub(crate) fn colormap_length(
        &self,
        figure: FigureHandle,
        index: usize,
    ) -> Result<usize, FigureError> {
        let registry = self.registry(PropertyLocation::Axes(figure, index))?;
        registry
            .figures
            .get(&figure)
            .and_then(|state| state.figure.axes_metadata(index))
            .map(|meta| meta.colormap_length)
            .ok_or(FigureError::InvalidAxesHandle)
    }

    pub(super) fn notify(
        &self,
        handle: FigureHandle,
        figure: &Figure,
        kind: FigureEventKind,
    ) -> Result<(), FigureError> {
        if self.selected.is_none() && self.callback.is_none() {
            notify_with_figure(handle, figure, kind);
            return Ok(());
        }
        let (revision, receiver, scene) = {
            let mut registry = self.registry(PropertyLocation::Figure(handle))?;
            super::colorbar::reconcile(&mut registry, handle, figure);
            let state = registry
                .figures
                .get(&handle)
                .ok_or(FigureError::StalePlotSnapshot)?;
            (
                Some(state.revision),
                state
                    .figure_receiver
                    .clone()
                    .ok_or(FigureError::StalePlotSnapshot)?,
                state.scene_lease.clone(),
            )
        };
        notify_event(FigureEventView {
            handle,
            kind,
            revision,
            figure: Some(figure),
            receiver: Some(&receiver),
            scene: Some(&scene),
        });
        Ok(())
    }
}

/// A helper cannot authorize a later helper by rereading an arbitrary scene.
/// Only its already-admitted guard may carry the exact resulting owner forward.
pub(super) struct PropertyMutationGuard<'a> {
    registry: PlotRegistryGuard<'static>,
    target: &'a PropertyMutationTarget,
    // The original backing may own callable roots: retire it after the lock.
    retired: Option<GraphicsPropertySelection>,
    callback: Option<(
        crate::context::GraphicsCallbackMutationScope,
        GraphicsReceiverRef,
        GraphicsSceneLease,
    )>,
}

impl PropertyMutationGuard<'_> {
    fn selected_child_id(&self) -> Option<u64> {
        self.target.selected.as_ref().and_then(|slot| {
            slot.borrow()
                .as_ref()
                .and_then(|selection| match selection.receiver().kind() {
                    GraphicsReceiverKind::PlotChild { id } => Some(id),
                    _ => None,
                })
        })
    }

    pub(super) fn for_each_child_mut_where(
        &mut self,
        mut predicate: impl FnMut(&PlotChildHandleState) -> bool,
        mut update: impl FnMut(&mut PlotChildHandleState),
    ) {
        match self.selected_child_id() {
            Some(id) => {
                if let Some(entry) = self
                    .registry
                    .plot_children
                    .entries
                    .get_mut(&id)
                    .filter(|entry| predicate(entry.state.as_ref()))
                {
                    update(Arc::make_mut(&mut entry.state));
                }
            }
            None => self
                .registry
                .plot_children
                .for_each_mut_where(predicate, update),
        }
    }

    pub(super) fn find_child_mut(
        &mut self,
        mut predicate: impl FnMut(&PlotChildHandleState) -> bool,
    ) -> Option<&mut PlotChildHandleState> {
        let selected = self.selected_child_id();
        match selected {
            Some(id) => self
                .registry
                .plot_children
                .entries
                .get_mut(&id)
                .filter(|entry| predicate(entry.state.as_ref()))
                .map(|entry| Arc::make_mut(&mut entry.state)),
            None => self.registry.plot_children.find_mut(predicate),
        }
    }
}

impl Deref for PropertyMutationGuard<'_> {
    type Target = PlotRegistry;
    fn deref(&self) -> &Self::Target {
        &self.registry
    }
}

impl DerefMut for PropertyMutationGuard<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.registry
    }
}

impl Drop for PropertyMutationGuard<'_> {
    fn drop(&mut self) {
        if let Some((scope, receiver, predecessor)) = &self.callback {
            if let Some(state) = self
                .registry
                .figures
                .get(&FigureHandle::from(scope.figure()))
            {
                if state
                    .figure_receiver
                    .as_ref()
                    .is_some_and(|current| current.shares_lease(receiver))
                {
                    scope.record_successor(receiver, predecessor, &state.scene_lease);
                }
            }
        }
        if let Some(slot) = &self.target.selected {
            let previous = slot.borrow_mut().take();
            let successor = previous
                .as_ref()
                .and_then(|selection| selection.successor(&self.registry).ok());
            *slot.borrow_mut() = successor;
            self.retired = previous;
        }
    }
}
