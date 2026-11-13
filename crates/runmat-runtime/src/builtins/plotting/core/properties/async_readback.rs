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
            | PlotChildHandleState::Scatter3(_)
            | PlotChildHandleState::Line3(_)
    ) {
        return get_properties(handle, property, builtin);
    }
    let (child, plot, _) =
        plot_child_element_snapshot(*id).map_err(|err| map_figure_error(builtin, err))?;
    get_child_properties_from_source(
        &child,
        Some(&plot),
        property,
        builtin,
        PropertyReadSource::CompatibilityAlias,
    )
    .await
}

pub(super) async fn get_selected_child_properties(
    selection: &super::super::state::GraphicsPropertySelection,
    property: Option<&str>,
    builtin: &'static str,
) -> BuiltinResult<Value> {
    let snapshot = selection
        .child_snapshot()
        .map_err(|error| map_figure_error(builtin, error))?;
    let plot = snapshot
        .child
        .plot_index()
        .and_then(|index| snapshot.figure.plots().nth(index));
    selection
        .ensure_current()
        .map_err(|error| map_figure_error(builtin, error))?;
    let result = get_child_properties_from_source(
        &snapshot.child,
        plot,
        property,
        builtin,
        PropertyReadSource::Selected(&snapshot),
    )
    .await?;
    selection
        .ensure_current()
        .map_err(|error| map_figure_error(builtin, error))?;
    Ok(result)
}

async fn get_child_properties_from_source(
    child: &PlotChildHandleState,
    plot: Option<&PlotElement>,
    property: Option<&str>,
    builtin: &'static str,
    source: PropertyReadSource<'_>,
) -> BuiltinResult<Value> {
    match (child, plot) {
        (PlotChildHandleState::Line(state), Some(PlotElement::Line(line))) => {
            if let SourcePropertyRead::Ready(value) =
                line_property_without_data(state, property, builtin, line)?
            {
                return Ok(value);
            }
            let (x, y) = line_xy(line, builtin).await?;
            line_property_with_data(state, property, builtin, line, x, y)
        }
        (PlotChildHandleState::Scatter(state), Some(PlotElement::Scatter(scatter))) => {
            if let SourcePropertyRead::Ready(value) =
                scatter_property_without_data(state, property, builtin, scatter)?
            {
                return Ok(value);
            }
            let (x, y) = scatter_xy(scatter, builtin).await?;
            scatter_property_with_data(state, property, builtin, scatter, x, y)
        }
        (PlotChildHandleState::Scatter3(state), Some(PlotElement::Scatter3(scatter))) => {
            scatter3_data::get_async(state, scatter, property, builtin).await
        }
        (PlotChildHandleState::Line3(state), Some(PlotElement::Line3(line)))
            if line3_property_needs_data(property) =>
        {
            let data = line3_xyz(line, builtin).await?;
            line3_property_with_data(state, property, builtin, line, Some(data))
        }
        (PlotChildHandleState::Line3(state), Some(PlotElement::Line3(line))) => {
            line3_property_with_data(state, property, builtin, line, None)
        }
        (
            PlotChildHandleState::Line(_)
            | PlotChildHandleState::Scatter(_)
            | PlotChildHandleState::Scatter3(_)
            | PlotChildHandleState::Line3(_),
            _,
        ) => Err(plotting_error(
            builtin,
            format!("{builtin}: invalid plot handle"),
        )),
        _ => get_plot_child_property(child, property, builtin, source),
    }
}

#[cfg(test)]
pub async fn set_properties_async(
    handle: PlotHandle,
    args: &[Value],
    builtin: &'static str,
) -> BuiltinResult<()> {
    set_properties_async_with_runtime(
        None,
        handle,
        crate::call::FunctionArgumentView::from_slice(args),
        builtin,
    )
    .await
}

pub(crate) async fn set_properties_async_with_runtime(
    runtime: Option<&crate::context::RuntimeContext>,
    handle: PlotHandle,
    args: crate::call::FunctionArgumentView<'_>,
    builtin: &'static str,
) -> BuiltinResult<()> {
    let PlotHandle::PlotChild(id, state) = &handle else {
        return set_properties_with_runtime(runtime, handle, args, builtin);
    };
    if !matches!(
        state.as_ref(),
        PlotChildHandleState::Line(_)
            | PlotChildHandleState::Scatter(_)
            | PlotChildHandleState::Scatter3(_)
            | PlotChildHandleState::Line3(_)
    ) {
        return set_properties_with_runtime(runtime, handle, args, builtin);
    }
    let (child, plot, revision) =
        plot_child_element_snapshot(*id).map_err(|err| map_figure_error(builtin, err))?;
    let authority = super::super::state::PropertyMutationTarget::alias().with_runtime(runtime);
    let guard = Some(PlotUpdateGuard {
        child_handle: *id,
        revision,
    });
    set_child_properties_from_source(&authority, *id, &child, Some(&plot), args, builtin, guard)
        .await
}

pub(super) async fn set_selected_child_properties(
    authority: &super::super::state::PropertyMutationTarget,
    selection: &super::super::state::GraphicsPropertySelection,
    id: u64,
    args: crate::call::FunctionArgumentView<'_>,
    builtin: &'static str,
) -> BuiltinResult<()> {
    let snapshot = selection
        .child_snapshot()
        .map_err(|err| map_figure_error(builtin, err))?;
    let plot = snapshot
        .child
        .plot_index()
        .and_then(|index| snapshot.figure.plots().nth(index));
    authority
        .ensure_current()
        .map_err(|err| map_figure_error(builtin, err))?;
    set_child_properties_from_source(
        authority,
        id as f64,
        &snapshot.child,
        plot,
        args,
        builtin,
        None,
    )
    .await?;
    authority
        .ensure_current()
        .map_err(|err| map_figure_error(builtin, err))
}

async fn set_child_properties_from_source(
    authority: &super::super::state::PropertyMutationTarget,
    id: f64,
    child: &PlotChildHandleState,
    plot: Option<&PlotElement>,
    args: crate::call::FunctionArgumentView<'_>,
    builtin: &'static str,
    guard: Option<PlotUpdateGuard>,
) -> BuiltinResult<()> {
    if args.is_empty() || !args.len().is_multiple_of(2) {
        return Err(plotting_error(
            builtin,
            format!("{builtin}: property/value arguments must come in pairs"),
        ));
    }
    let mut pairs = Vec::with_capacity(args.len() / 2);
    for pair in args.chunks_exact(2) {
        pairs.push((
            property_name(&pair[0], builtin)?,
            pair.get(1).expect("paired property value"),
        ));
    }
    match (child, plot) {
        (PlotChildHandleState::Line(state), Some(PlotElement::Line(line))) => {
            let prepared = prepare_line_data_updates(&pairs, builtin)?;
            if prepared.is_empty() {
                return apply_line_style_only(authority, state, &pairs, builtin, guard);
            }
            let data = line_xy(line, builtin).await?;
            apply_line_properties_with_data(
                authority, state, &pairs, builtin, data, prepared, guard,
            )
        }
        (PlotChildHandleState::Scatter(state), Some(PlotElement::Scatter(scatter))) => {
            let prepared = prepare_scatter_data_updates(&pairs, builtin)?;
            if prepared.is_empty() {
                return apply_scatter_style_only(authority, state, &pairs, builtin, guard);
            }
            let data = scatter_xy(scatter, builtin).await?;
            apply_scatter_properties_with_data(
                authority, state, &pairs, builtin, scatter, data, prepared, guard,
            )
        }
        (PlotChildHandleState::Scatter3(state), Some(PlotElement::Scatter3(scatter))) => {
            scatter3_data::apply_async(authority, state, scatter, &pairs, builtin, guard).await
        }
        (PlotChildHandleState::Line3(state), Some(PlotElement::Line3(line))) => {
            let prepared = prepare_line3_data_updates(&pairs, builtin)?;
            if prepared.is_empty() {
                return apply_line3_style_only(authority, state, &pairs, builtin, guard);
            }
            let data = line3_xyz(line, builtin).await?;
            apply_line3_properties_with_data(
                authority, state, &pairs, builtin, data, prepared, guard,
            )
        }
        (
            PlotChildHandleState::Line(_)
            | PlotChildHandleState::Scatter(_)
            | PlotChildHandleState::Scatter3(_)
            | PlotChildHandleState::Line3(_),
            _,
        ) => Err(plotting_error(
            builtin,
            format!("{builtin}: invalid plot handle"),
        )),
        _ => set_properties_for_target(
            authority,
            PropertySetHandle::PlotChild(id, child),
            args,
            builtin,
        )?
        .complete_compatibility(builtin),
    }
}

#[cfg(test)]
#[path = "async_readback/tests/mod.rs"]
mod tests;
