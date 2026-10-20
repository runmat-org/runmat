//! Axes property rendering from one authenticated metadata snapshot.

use super::super::state::{AxesPropertySnapshot, GraphicsPropertySelection};
use super::*;

pub(super) fn get_compatibility(
    handle: FigureHandle,
    index: usize,
    property: Option<&str>,
    builtin: &'static str,
) -> BuiltinResult<Value> {
    let snapshot = super::super::state::axes_property_snapshot_alias(handle, index)
        .map_err(|error| map_figure_error(builtin, error))?;
    render(snapshot, property, builtin)
}

pub(super) fn get_selected(
    selection: &GraphicsPropertySelection,
    property: Option<&str>,
    builtin: &'static str,
) -> BuiltinResult<Value> {
    let snapshot = selection
        .axes_snapshot()
        .map_err(|error| map_figure_error(builtin, error))?;
    render(snapshot, property, builtin)
}

fn render(
    snapshot: AxesPropertySnapshot,
    property: Option<&str>,
    builtin: &'static str,
) -> BuiltinResult<Value> {
    let AxesPropertySnapshot {
        selection,
        handle,
        index: axes_index,
        meta,
        axes,
        display_bounds,
        x_limits,
        y_limits,
        z_limits,
        color_limits,
    } = snapshot;
    let displayed_axis = |axis| -> BuiltinResult<Value> {
        Ok(limit_value(match axis {
            AxisLimitAxis::X => x_limits,
            AxisLimitAxis::Y => y_limits,
            AxisLimitAxis::Z => z_limits,
        }))
    };
    let displayed_color = || -> BuiltinResult<Value> { Ok(limit_value(color_limits)) };
    let displayed_z = || -> BuiltinResult<Value> { Ok(limit_value(z_limits.or(Some((0.0, 1.0))))) };
    let result = match property.map(canonical_property_name).as_deref() {
        None => {
            let mut st = runmat_value::UnboundedStructValueBuilder::new();
            st.insert(
                "Handle",
                Value::Num(super::super::state::encode_axes_handle(handle, axes_index)),
            );
            st.insert("Figure", Value::Num(handle.as_u32() as f64));
            st.insert("Rows", Value::Num(axes.rows as f64));
            st.insert("Cols", Value::Num(axes.cols as f64));
            st.insert("Index", Value::Num((axes_index + 1) as f64));
            st.insert(
                "Title",
                Value::Num(super::super::state::encode_plot_object_handle(
                    handle,
                    axes_index,
                    PlotObjectKind::Title,
                )),
            );
            st.insert(
                "Subtitle",
                Value::Num(super::super::state::encode_plot_object_handle(
                    handle,
                    axes_index,
                    PlotObjectKind::Subtitle,
                )),
            );
            st.insert(
                "XLabel",
                Value::Num(super::super::state::encode_plot_object_handle(
                    handle,
                    axes_index,
                    PlotObjectKind::XLabel,
                )),
            );
            st.insert(
                "YLabel",
                Value::Num(super::super::state::encode_plot_object_handle(
                    handle,
                    axes_index,
                    PlotObjectKind::YLabel,
                )),
            );
            st.insert(
                "ZLabel",
                Value::Num(super::super::state::encode_plot_object_handle(
                    handle,
                    axes_index,
                    PlotObjectKind::ZLabel,
                )),
            );
            st.insert(
                "Legend",
                Value::Num(super::super::state::encode_plot_object_handle(
                    handle,
                    axes_index,
                    PlotObjectKind::Legend,
                )),
            );
            st.insert(
                "XAxis",
                Value::Num(super::super::state::encode_plot_object_handle(
                    handle,
                    axes_index,
                    PlotObjectKind::XAxis,
                )),
            );
            st.insert(
                "YAxis",
                Value::Num(super::super::state::encode_plot_object_handle(
                    handle,
                    axes_index,
                    PlotObjectKind::YAxis,
                )),
            );
            st.insert("LegendVisible", Value::Bool(meta.legend_enabled));
            st.insert("Type", Value::String("axes".into()));
            st.insert("Parent", Value::Num(handle.as_u32() as f64));
            st.insert("Position", figure_position_value(meta.position));
            st.insert("Units", Value::string(meta.units.clone()));
            st.insert(
                "Children",
                handles_value(vec![
                    super::super::state::encode_plot_object_handle(
                        handle,
                        axes_index,
                        PlotObjectKind::Title,
                    ),
                    super::super::state::encode_plot_object_handle(
                        handle,
                        axes_index,
                        PlotObjectKind::Subtitle,
                    ),
                    super::super::state::encode_plot_object_handle(
                        handle,
                        axes_index,
                        PlotObjectKind::XLabel,
                    ),
                    super::super::state::encode_plot_object_handle(
                        handle,
                        axes_index,
                        PlotObjectKind::YLabel,
                    ),
                    super::super::state::encode_plot_object_handle(
                        handle,
                        axes_index,
                        PlotObjectKind::ZLabel,
                    ),
                    super::super::state::encode_plot_object_handle(
                        handle,
                        axes_index,
                        PlotObjectKind::Legend,
                    ),
                ]),
            );
            st.insert("Grid", Value::Bool(meta.grid_enabled));
            st.insert("MinorGrid", Value::Bool(meta.minor_grid_enabled));
            st.insert(
                "HiddenLineRemoval",
                Value::String(on_off(meta.hidden_line_removal).into()),
            );
            st.insert("Box", Value::Bool(meta.box_enabled));
            st.insert("AxisEqual", Value::Bool(meta.axis_equal));
            st.insert(
                "DataAspectRatio",
                tensor_from_vec(meta.data_aspect_ratio.to_vec()),
            );
            st.insert(
                "DataAspectRatioMode",
                Value::string(meta.data_aspect_ratio_mode.clone()),
            );
            st.insert("Colorbar", Value::Bool(meta.colorbar_enabled));
            st.insert(
                "Colormap",
                Value::string(format!("{:?}", meta.colormap).to_ascii_lowercase()),
            );
            st.insert("ColorLimits", displayed_color()?);
            st.insert("XLim", displayed_axis(AxisLimitAxis::X)?);
            st.insert("YLim", displayed_axis(AxisLimitAxis::Y)?);
            st.insert("ZLim", displayed_z()?);
            st.insert("CLim", displayed_color()?);
            st.insert(
                "XTick",
                tick_value(ticks_or_auto(
                    meta.x_ticks.as_deref(),
                    x_bounds(display_bounds),
                )),
            );
            st.insert(
                "YTick",
                tick_value(ticks_or_auto(
                    meta.y_ticks.as_deref(),
                    y_bounds(display_bounds),
                )),
            );
            st.insert("XTickMode", tick_mode_value(meta.x_ticks.as_ref()));
            st.insert("YTickMode", tick_mode_value(meta.y_ticks.as_ref()));
            st.insert(
                "XTickLabel",
                tick_label_value(tick_labels_or_auto(
                    meta.x_tick_labels.as_deref(),
                    meta.x_ticks.as_deref(),
                    x_bounds(display_bounds),
                    meta.x_tick_format.as_deref(),
                )),
            );
            st.insert(
                "YTickLabel",
                tick_label_value(tick_labels_or_auto(
                    meta.y_tick_labels.as_deref(),
                    meta.y_ticks.as_deref(),
                    y_bounds(display_bounds),
                    meta.y_tick_format.as_deref(),
                )),
            );
            st.insert(
                "XTickLabelMode",
                tick_mode_value(meta.x_tick_labels.as_ref()),
            );
            st.insert(
                "YTickLabelMode",
                tick_mode_value(meta.y_tick_labels.as_ref()),
            );
            st.insert(
                "XTickLabelFormat",
                Value::string(tick_format_or_default(meta.x_tick_format.as_deref())),
            );
            st.insert(
                "YTickLabelFormat",
                Value::string(tick_format_or_default(meta.y_tick_format.as_deref())),
            );
            st.insert(
                "XTickLabelRotation",
                Value::Num(meta.x_tick_label_rotation.unwrap_or(0.0)),
            );
            st.insert(
                "YTickLabelRotation",
                Value::Num(meta.y_tick_label_rotation.unwrap_or(0.0)),
            );
            st.insert(
                "FontSize",
                Value::Num(meta.axes_style.font_size.unwrap_or(10.0) as f64),
            );
            st.insert("Visible", Value::Bool(meta.axes_style.visible));
            st.insert(
                "XScale",
                Value::String(if meta.x_log { "log" } else { "linear" }.into()),
            );
            st.insert(
                "YScale",
                Value::String(if meta.y_log { "log" } else { "linear" }.into()),
            );
            st.insert("YAxisLocation", Value::string(meta.y_axis_location.clone()));
            st.insert("YDir", Value::String(meta.y_direction.as_str().into()));
            Ok(Value::Struct(st.finish()))
        }
        Some("title") => Ok(Value::Num(super::super::state::encode_plot_object_handle(
            handle,
            axes_index,
            PlotObjectKind::Title,
        ))),
        Some("subtitle") => Ok(Value::Num(super::super::state::encode_plot_object_handle(
            handle,
            axes_index,
            PlotObjectKind::Subtitle,
        ))),
        Some("xlabel") => Ok(Value::Num(super::super::state::encode_plot_object_handle(
            handle,
            axes_index,
            PlotObjectKind::XLabel,
        ))),
        Some("ylabel") => Ok(Value::Num(super::super::state::encode_plot_object_handle(
            handle,
            axes_index,
            PlotObjectKind::YLabel,
        ))),
        Some("zlabel") => Ok(Value::Num(super::super::state::encode_plot_object_handle(
            handle,
            axes_index,
            PlotObjectKind::ZLabel,
        ))),
        Some("legend") => Ok(Value::Num(super::super::state::encode_plot_object_handle(
            handle,
            axes_index,
            PlotObjectKind::Legend,
        ))),
        Some("xaxis") => Ok(Value::Num(super::super::state::encode_plot_object_handle(
            handle,
            axes_index,
            PlotObjectKind::XAxis,
        ))),
        Some("yaxis") => Ok(Value::Num(super::super::state::encode_plot_object_handle(
            handle,
            axes_index,
            PlotObjectKind::YAxis,
        ))),
        Some("view") => {
            let az = meta.view_azimuth_deg.unwrap_or(-37.5) as f64;
            let el = meta.view_elevation_deg.unwrap_or(30.0) as f64;
            Ok(Value::Tensor(
                Tensor::new(vec![az, el], vec![1, 2]).expect("view vector"),
            ))
        }
        Some("grid") => Ok(Value::Bool(meta.grid_enabled)),
        Some("minorgrid") => Ok(Value::Bool(meta.minor_grid_enabled)),
        Some("hiddenlineremoval") => Ok(Value::String(on_off(meta.hidden_line_removal).into())),
        Some("box") => Ok(Value::Bool(meta.box_enabled)),
        Some("axisequal") => Ok(Value::Bool(meta.axis_equal)),
        Some("dataaspectratio") => Ok(tensor_from_vec(meta.data_aspect_ratio.to_vec())),
        Some("dataaspectratiomode") => Ok(Value::string(meta.data_aspect_ratio_mode.clone())),
        Some("colorbar") => Ok(Value::Bool(meta.colorbar_enabled)),
        Some("colormap") => Ok(Value::string(
            format!("{:?}", meta.colormap).to_ascii_lowercase(),
        )),
        Some("colorlimits") => displayed_color(),
        Some("xlim") => displayed_axis(AxisLimitAxis::X),
        Some("ylim") => displayed_axis(AxisLimitAxis::Y),
        Some("zlim") => displayed_z(),
        Some("clim") => displayed_color(),
        Some("xtick") => Ok(tick_value(ticks_or_auto(
            meta.x_ticks.as_deref(),
            x_bounds(display_bounds),
        ))),
        Some("ytick") => Ok(tick_value(ticks_or_auto(
            meta.y_ticks.as_deref(),
            y_bounds(display_bounds),
        ))),
        Some("xtickmode") => Ok(tick_mode_value(meta.x_ticks.as_ref())),
        Some("ytickmode") => Ok(tick_mode_value(meta.y_ticks.as_ref())),
        Some("xticklabel") => Ok(tick_label_value(tick_labels_or_auto(
            meta.x_tick_labels.as_deref(),
            meta.x_ticks.as_deref(),
            x_bounds(display_bounds),
            meta.x_tick_format.as_deref(),
        ))),
        Some("yticklabel") => Ok(tick_label_value(tick_labels_or_auto(
            meta.y_tick_labels.as_deref(),
            meta.y_ticks.as_deref(),
            y_bounds(display_bounds),
            meta.y_tick_format.as_deref(),
        ))),
        Some("xticklabelmode") => Ok(tick_mode_value(meta.x_tick_labels.as_ref())),
        Some("yticklabelmode") => Ok(tick_mode_value(meta.y_tick_labels.as_ref())),
        Some("xticklabelformat") => Ok(Value::string(tick_format_or_default(
            meta.x_tick_format.as_deref(),
        ))),
        Some("yticklabelformat") => Ok(Value::string(tick_format_or_default(
            meta.y_tick_format.as_deref(),
        ))),
        Some("xticklabelrotation") => Ok(Value::Num(meta.x_tick_label_rotation.unwrap_or(0.0))),
        Some("yticklabelrotation") => Ok(Value::Num(meta.y_tick_label_rotation.unwrap_or(0.0))),
        Some("fontsize") => Ok(Value::Num(meta.axes_style.font_size.unwrap_or(10.0) as f64)),
        Some("visible") => Ok(Value::Bool(meta.axes_style.visible)),
        Some("xscale") => Ok(Value::String(
            if meta.x_log { "log" } else { "linear" }.into(),
        )),
        Some("yscale") => Ok(Value::String(
            if meta.y_log { "log" } else { "linear" }.into(),
        )),
        Some("yaxislocation") => Ok(Value::string(meta.y_axis_location)),
        Some("ydir") => Ok(Value::String(meta.y_direction.as_str().into())),
        Some("type") => Ok(Value::String("axes".into())),
        Some("parent") => Ok(Value::Num(handle.as_u32() as f64)),
        Some("position") => Ok(figure_position_value(meta.position)),
        Some("units") => Ok(Value::string(meta.units)),
        Some("legendvisible") => Ok(Value::Bool(meta.legend_enabled)),
        Some("children") => Ok(handles_value(vec![
            super::super::state::encode_plot_object_handle(
                handle,
                axes_index,
                PlotObjectKind::Title,
            ),
            super::super::state::encode_plot_object_handle(
                handle,
                axes_index,
                PlotObjectKind::Subtitle,
            ),
            super::super::state::encode_plot_object_handle(
                handle,
                axes_index,
                PlotObjectKind::XLabel,
            ),
            super::super::state::encode_plot_object_handle(
                handle,
                axes_index,
                PlotObjectKind::YLabel,
            ),
            super::super::state::encode_plot_object_handle(
                handle,
                axes_index,
                PlotObjectKind::ZLabel,
            ),
            super::super::state::encode_plot_object_handle(
                handle,
                axes_index,
                PlotObjectKind::Legend,
            ),
        ])),
        Some(other) => Err(plotting_error(
            builtin,
            format!("{builtin}: unsupported axes property `{other}`"),
        )),
    }?;
    if let Some(selection) = selection {
        selection
            .ensure_current()
            .map_err(|error| map_figure_error(builtin, error))?;
    }
    Ok(result)
}
