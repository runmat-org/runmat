//! Async plot-source materialization at the graphics property boundary.
//! Snapshotting releases the registry guard before a provider readback awaits.
use super::super::state::{plot_child_element_snapshot, PlotChildHandleState};
use super::*;
use runmat_plot::plots::figure::PlotElement;

async fn line_xy(
    line: &runmat_plot::plots::LinePlot,
    builtin: &'static str,
) -> BuiltinResult<PlotXY> {
    line.export_numeric_xy_data_typed()
        .await
        .map_err(|err| map_figure_error(builtin, err))?
        .ok_or_else(|| plotting_error(builtin, format!("{builtin}: line has no source data")))
}

async fn scatter_xy(
    scatter: &runmat_plot::plots::ScatterPlot,
    builtin: &'static str,
) -> BuiltinResult<PlotXY> {
    scatter
        .export_numeric_xy_data_typed()
        .await
        .map_err(|err| map_figure_error(builtin, err))?
        .ok_or_else(|| plotting_error(builtin, format!("{builtin}: scatter has no source data")))
}

async fn line3_xyz(
    line: &runmat_plot::plots::Line3Plot,
    builtin: &'static str,
) -> BuiltinResult<PlotXYZ> {
    line.export_numeric_xyz_data_typed()
        .await
        .map_err(|err| map_figure_error(builtin, err))?
        .ok_or_else(|| plotting_error(builtin, format!("{builtin}: plot3 has no source data")))
}

pub async fn get_properties_async(
    handle: PlotHandle,
    property: Option<&str>,
    builtin: &'static str,
) -> BuiltinResult<Value> {
    let PlotHandle::PlotChild(id, state) = &handle else {
        return get_properties(handle, property, builtin);
    };
    if !matches!(
        state.as_ref(),
        PlotChildHandleState::Line(_)
            | PlotChildHandleState::Scatter(_)
            | PlotChildHandleState::Line3(_)
    ) {
        return get_properties(handle, property, builtin);
    }
    let (child, plot, _) =
        plot_child_element_snapshot(*id).map_err(|err| map_figure_error(builtin, err))?;
    match (child, plot) {
        (PlotChildHandleState::Line(state), PlotElement::Line(line)) => {
            if let SourcePropertyRead::Ready(value) =
                line_property_without_data(&state, property, builtin, &line)?
            {
                return Ok(value);
            }
            let (x, y) = line_xy(&line, builtin).await?;
            line_property_with_data(&state, property, builtin, line, x, y)
        }
        (PlotChildHandleState::Scatter(state), PlotElement::Scatter(scatter)) => {
            if let SourcePropertyRead::Ready(value) =
                scatter_property_without_data(&state, property, builtin, &scatter)?
            {
                return Ok(value);
            }
            let (x, y) = scatter_xy(&scatter, builtin).await?;
            scatter_property_with_data(&state, property, builtin, scatter, x, y)
        }
        (PlotChildHandleState::Line3(state), PlotElement::Line3(line))
            if line3_property_needs_data(property) =>
        {
            let data = line3_xyz(&line, builtin).await?;
            line3_property_with_data(&state, property, builtin, line, Some(data))
        }
        (PlotChildHandleState::Line3(state), PlotElement::Line3(line)) => {
            line3_property_with_data(&state, property, builtin, line, None)
        }
        (
            PlotChildHandleState::Line(_)
            | PlotChildHandleState::Scatter(_)
            | PlotChildHandleState::Line3(_),
            _,
        ) => Err(plotting_error(
            builtin,
            format!("{builtin}: invalid plot handle"),
        )),
        _ => get_properties(handle, property, builtin),
    }
}

pub async fn set_properties_async(
    handle: PlotHandle,
    args: &[Value],
    builtin: &'static str,
) -> BuiltinResult<()> {
    let PlotHandle::PlotChild(id, state) = &handle else {
        return set_properties(handle, args, builtin);
    };
    if !matches!(
        state.as_ref(),
        PlotChildHandleState::Line(_)
            | PlotChildHandleState::Scatter(_)
            | PlotChildHandleState::Line3(_)
    ) {
        return set_properties(handle, args, builtin);
    }
    if args.is_empty() || !args.len().is_multiple_of(2) {
        return Err(plotting_error(
            builtin,
            format!("{builtin}: property/value arguments must come in pairs"),
        ));
    }
    let mut pairs = Vec::with_capacity(args.len() / 2);
    for pair in args.chunks_exact(2) {
        pairs.push((property_name(&pair[0], builtin)?, &pair[1]));
    }
    let (child, plot, revision) =
        plot_child_element_snapshot(*id).map_err(|err| map_figure_error(builtin, err))?;
    let guard = PlotUpdateGuard {
        child_handle: *id,
        revision,
    };
    match (child, plot) {
        (PlotChildHandleState::Line(state), PlotElement::Line(line)) => {
            let prepared = prepare_line_data_updates(&pairs, builtin)?;
            if prepared.is_empty() {
                return apply_line_style_only(&state, &pairs, builtin, guard);
            }
            let data = line_xy(&line, builtin).await?;
            apply_line_properties_with_data(&state, &pairs, builtin, data, prepared, Some(guard))
        }
        (PlotChildHandleState::Scatter(state), PlotElement::Scatter(scatter)) => {
            let prepared = prepare_scatter_data_updates(&pairs, builtin)?;
            if prepared.is_empty() {
                return apply_scatter_style_only(&state, &pairs, builtin, guard);
            }
            let data = scatter_xy(&scatter, builtin).await?;
            apply_scatter_properties_with_data(
                &state,
                &pairs,
                builtin,
                scatter,
                data,
                prepared,
                Some(guard),
            )
        }
        (PlotChildHandleState::Line3(state), PlotElement::Line3(line)) => {
            let prepared = prepare_line3_data_updates(&pairs, builtin)?;
            if prepared.is_empty() {
                return apply_line3_style_only(&state, &pairs, builtin, guard);
            }
            let data = line3_xyz(&line, builtin).await?;
            apply_line3_properties_with_data(&state, &pairs, builtin, data, prepared, Some(guard))
        }
        (
            PlotChildHandleState::Line(_)
            | PlotChildHandleState::Scatter(_)
            | PlotChildHandleState::Line3(_),
            _,
        ) => Err(plotting_error(
            builtin,
            format!("{builtin}: invalid plot handle"),
        )),
        _ => set_properties(handle, args, builtin),
    }
}

#[cfg(test)]
#[path = "async_readback/tests/mod.rs"]
mod tests;
