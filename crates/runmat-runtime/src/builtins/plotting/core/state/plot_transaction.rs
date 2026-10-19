//! Plot publication shares one guarded mutation owner with callback properties.
use super::*;

/// A plot prepares data without retaining a registry guard and commits only to
/// the original live axes and its admitted scene successors.
pub(crate) struct PreparedPlotTarget {
    property: PropertyMutationTarget,
    handle: FigureHandle,
    axes: usize,
}

impl PreparedPlotTarget {
    pub(crate) fn prepare(
        resolved: ResolvedAxesTarget,
        explicit: bool,
    ) -> Result<Self, FigureError> {
        let (handle, axes) = resolved.coordinates();
        let property = resolved.into_authority();
        if explicit {
            select_axes_for_figure_for_property(&property, handle, axes)?;
        }
        Ok(Self {
            property,
            handle,
            axes,
        })
    }

    pub(crate) fn hold_enabled(&self) -> Result<bool, FigureError> {
        let reg = self
            .property
            .registry(PropertyLocation::Axes(self.handle, self.axes))?;
        let state = reg
            .figures
            .get(&self.handle)
            .ok_or(FigureError::StalePlotSnapshot)?;
        Ok(*state.hold_per_axes.get(&self.axes).unwrap_or(&false))
    }

    pub(crate) fn set_style_order(&self, order: &[LineStyle]) -> Result<(), FigureError> {
        let mut reg = self
            .property
            .registry(PropertyLocation::Axes(self.handle, self.axes))?;
        get_state_for_future_plot_style(&mut reg, self.handle)
            .cycle_for_axes_mut(self.axes)
            .set_order(order);
        Ok(())
    }

    pub(crate) fn series_style(
        &self,
        series: usize,
        hold: bool,
        needs_line: bool,
        needs_color: bool,
    ) -> Result<(Option<LineStyle>, Option<Vec4>), FigureError> {
        let mut reg = self
            .property
            .registry(PropertyLocation::Axes(self.handle, self.axes))?;
        let state = get_state_for_future_plot_style(&mut reg, self.handle);
        let line = needs_line.then(|| {
            if hold {
                state.cycle_for_axes_mut(self.axes).next()
            } else {
                LineStyle::Solid
            }
        });
        let color = needs_color.then(|| {
            if hold {
                state.color_cycle_for_axes_mut(self.axes).next()
            } else {
                state.color_cycle_for_axes_mut(self.axes).color_at(series)
            }
        });
        Ok((line, color))
    }

    pub(crate) fn register_line(&self, axes: usize, plot: usize) -> Result<f64, FigureError> {
        if axes != self.axes {
            return Err(FigureError::InvalidAxesHandle);
        }
        let mut reg = self
            .property
            .registry(PropertyLocation::Axes(self.handle, axes))?;
        let state = reg
            .figures
            .get(&self.handle)
            .ok_or(FigureError::StalePlotSnapshot)?;
        if !matches!(state.figure.plots().nth(plot), Some(PlotElement::Line(_))) {
            return Err(FigureError::InvalidPlotObjectHandle);
        }
        Ok(register_simple_plot_handle_in_registry(
            &mut reg,
            self.handle,
            axes,
            plot,
            PlotChildHandleState::Line,
        ))
    }

    pub(crate) fn render<F>(
        &self,
        runtime: Option<&crate::context::RuntimeContext>,
        builtin: &'static str,
        options: PlotRenderOptions<'_>,
        apply: F,
    ) -> BuiltinResult<String>
    where
        F: FnOnce(&mut Figure, usize) -> BuiltinResult<()>,
    {
        render_for_property(
            &self.property,
            Some((self.handle, self.axes)),
            runtime,
            false,
            builtin,
            options,
            apply,
        )
    }
}

pub(super) fn render_for_property<F>(
    property: &PropertyMutationTarget,
    coordinates: Option<(FigureHandle, usize)>,
    runtime: Option<&crate::context::RuntimeContext>,
    append: bool,
    builtin: &'static str,
    opts: PlotRenderOptions<'_>,
    apply: F,
) -> BuiltinResult<String>
where
    F: FnOnce(&mut Figure, usize) -> BuiltinResult<()>,
{
    let rendering_disabled = interactive_rendering_disabled();
    let host_managed_rendering = host_managed_rendering_enabled();
    let (handle, figure_clone, receiver) = {
        let location = coordinates.map_or(PropertyLocation::CurrentFigure, |(handle, axes)| {
            PropertyLocation::Axes(handle, axes)
        });
        let mut reg = property.registry(location).map_err(|source| {
            super::plotting_error_with_source(builtin, "plot target changed before commit", source)
        })?;
        let handle = coordinates.map_or(reg.current, |(handle, _)| handle);
        let axes_index = coordinates.map_or_else(
            || get_state_mut(&mut reg, handle).active_axes,
            |(_, axes)| axes,
        );
        let should_clear = !append && !get_state_mut(&mut reg, handle).hold();
        if should_clear {
            clear_axes::clear_registered_axes(&mut reg, handle, &[axes_index]);
        }
        {
            let state = get_state_mut(&mut reg, handle);
            state.figure.set_active_axes_index(axes_index);
            if should_clear {
                state.figure.reset_axes_text(axes_index);
                state.figure.set_axes_kind(axes_index, AxesKind::Cartesian);
                state.figure.set_axes_limits(axes_index, None, None);
                state.figure.set_axes_z_limits(axes_index, None);
                state.reset_cycle(axes_index);
            }
        }
        {
            let state = get_state_mut(&mut reg, handle);
            if !opts.title.is_empty() {
                state.figure.set_axes_title(axes_index, opts.title);
            }
            if !opts.x_label.is_empty() || !opts.y_label.is_empty() {
                state
                    .figure
                    .set_axes_labels(axes_index, opts.x_label, opts.y_label);
            }
            state.figure.set_grid(opts.grid);
            state.figure.set_axis_equal(opts.axis_equal);

            let first_new_plot = state.figure.plots().count();
            let _axes_context = AxesContextGuard::install(state, axes_index);
            apply(&mut state.figure, axes_index)
                .map_err(|flow| map_control_flow_with_builtin(flow, builtin))?;
            // New colormapped graphics use their axes' colormap, as in MATLAB.
            state.figure.apply_axes_colormaps_from(first_new_plot);

            // Increment revision after a successful mutation so surfaces can avoid
            // re-rendering unchanged figures when "presenting" an already-loaded handle.
            state.revision = state.revision.wrapping_add(1);
        }
        let figure_clone = reg
            .figures
            .get(&handle)
            .expect("figure exists")
            .figure
            .owned_clone();
        let receiver = reg
            .figures
            .get(&handle)
            .and_then(|state| state.figure_receiver.clone());
        (handle, figure_clone, receiver)
    };
    property
        .notify(handle, &figure_clone, FigureEventKind::Updated)
        .map_err(|source| {
            super::plotting_error_with_source(
                builtin,
                "plot target changed before notification",
                source,
            )
        })?;
    if let (Some(runtime), Some(receiver)) = (runtime, receiver) {
        if let Some(service) = runtime.service_ports().graphics_presentation() {
            if service.stage_scene(receiver)?
                == crate::context::GraphicsSceneStaging::DeferredToFlush
            {
                return Ok(format!("Figure {} updated", handle.as_u32()));
            }
        }
    }

    present_figure_update_with_staging(
        builtin,
        handle,
        figure_clone,
        rendering_disabled,
        host_managed_rendering,
        runtime.is_none(),
    )
}
