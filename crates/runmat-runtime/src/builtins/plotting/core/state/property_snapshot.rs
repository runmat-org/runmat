//! Coherent metadata snapshots selected under the actual registry owner.

use super::*;

pub(crate) struct FigurePropertySnapshot {
    pub(crate) selection: Option<GraphicsPropertySelection>,
    pub(crate) handle: FigureHandle,
    pub(crate) figure: super::shared_figure::FigureOwner,
    pub(crate) axes: Vec<f64>,
    pub(crate) current_axes: f64,
    pub(crate) has_sg_title: bool,
    pub(crate) tag: String,
    pub(crate) waitbar: Option<WaitbarState>,
}

pub(crate) struct AxesPropertySnapshot {
    pub(crate) selection: Option<GraphicsPropertySelection>,
    pub(crate) handle: FigureHandle,
    pub(crate) index: usize,
    pub(crate) meta: runmat_plot::plots::AxesMetadata,
    pub(crate) axes: FigureAxesState,
    pub(crate) display_bounds: AxisDisplayBoundsSnapshot,
    pub(crate) x_limits: Option<(f64, f64)>,
    pub(crate) y_limits: Option<(f64, f64)>,
    pub(crate) z_limits: Option<(f64, f64)>,
    pub(crate) color_limits: Option<(f64, f64)>,
}

pub(crate) fn axes_property_snapshot_alias(
    handle: FigureHandle,
    index: usize,
) -> Result<AxesPropertySnapshot, FigureError> {
    axes_snapshot_from_registry(&mut registry(), handle, index, None)
}

pub(crate) fn figure_property_snapshot_alias(
    handle: FigureHandle,
) -> Result<FigurePropertySnapshot, FigureError> {
    figure_snapshot_from_registry(&registry(), handle, None)
}

impl GraphicsPropertySelection {
    pub(crate) fn axes_snapshot(&self) -> Result<AxesPropertySnapshot, FigureError> {
        let GraphicsReceiverKind::Axes { index, .. } = self.receiver().kind() else {
            return Err(FigureError::InvalidAxesHandle);
        };
        let mut registry = registry();
        self.validate(&registry)?;
        axes_snapshot_from_registry(
            &mut registry,
            self.figure_handle(),
            index,
            Some(self.clone()),
        )
    }
    pub(crate) fn figure_snapshot(&self) -> Result<FigurePropertySnapshot, FigureError> {
        if !matches!(self.receiver().kind(), GraphicsReceiverKind::Figure { .. }) {
            return Err(FigureError::InvalidPlotObjectHandle);
        }
        let registry = registry();
        self.validate(&registry)?;
        figure_snapshot_from_registry(&registry, self.figure_handle(), Some(self.clone()))
    }
}

fn axes_snapshot_from_registry(
    registry: &mut PlotRegistry,
    handle: FigureHandle,
    index: usize,
    selection: Option<GraphicsPropertySelection>,
) -> Result<AxesPropertySnapshot, FigureError> {
    let state = registry
        .figures
        .get_mut(&handle)
        .ok_or(FigureError::InvalidAxesHandle)?;
    validate_axes_index(state, index)?;
    let meta = state
        .figure
        .axes_metadata(index)
        .cloned()
        .ok_or(FigureError::InvalidAxesHandle)?;
    let display_bounds = display_bounds_for_state_axes(state, index);
    let x_limits = displayed_xy_limit(display_bounds, LinkAxesAxis::X).or(meta.x_limits);
    let y_limits = displayed_xy_limit(display_bounds, LinkAxesAxis::Y).or(meta.y_limits);
    let z_limits = super::axes_modes::z_limits_for_state_axes(state, index);
    let color_limits = super::axes_modes::displayed_color_limits(
        super::axes_modes::color_limits_for_state_axes(state, index),
    );
    Ok(AxesPropertySnapshot {
        selection,
        handle,
        index,
        meta,
        axes: FigureAxesState {
            handle,
            rows: state.figure.axes_rows.max(1),
            cols: state.figure.axes_cols.max(1),
            active_index: state.active_axes,
        },
        display_bounds,
        x_limits,
        y_limits,
        z_limits,
        color_limits,
    })
}

fn figure_snapshot_from_registry(
    registry: &PlotRegistry,
    handle: FigureHandle,
    selection: Option<GraphicsPropertySelection>,
) -> Result<FigurePropertySnapshot, FigureError> {
    let state = registry
        .figures
        .get(&handle)
        .ok_or(FigureError::InvalidHandle(handle.as_u32()))?;
    let mut axes = axes_handles_from_state(state, handle);
    axes.extend(super::colorbar::handles_in_registry(registry, handle));
    Ok(FigurePropertySnapshot {
        selection,
        handle,
        figure: state.figure.snapshot(),
        axes,
        current_axes: encode_axes_handle(handle, state.active_axes),
        has_sg_title: state.figure.sg_title.is_some(),
        tag: state.tag.clone(),
        waitbar: state.waitbar.clone(),
    })
}
