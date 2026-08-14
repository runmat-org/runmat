use runmat_builtins::{
    CellArray, CharArray, NumericScalar, StringArray, StructValue, Tensor, Value,
};
use runmat_plot::plots::{
    ColorMap, LegendStyle, NumericPlotData, PatchData, PolarHistogramDisplayStyle, ShadingMode,
    TextStyle,
};
use std::borrow::Cow;

use super::point::{marker_area_points2_to_diameter_px, marker_diameter_px_to_area_points2};
use super::state::{
    axes_handle_exists, axes_handles_for_figure, axes_metadata_snapshot, axes_state_snapshot,
    axis_display_bounds_snapshot_for_axes, current_axes_handle_for_figure,
    current_figure_handle_if_exists, decode_axes_handle, decode_plot_object_handle,
    figure_handle_exists, figure_has_sg_title, figure_tag, legend_entries_snapshot,
    present_figure_update, root_default_properties, root_default_property, root_figure_handles,
    root_show_hidden_handles, root_units, select_axes_for_figure, select_current_figure_if_exists,
    set_axes_position_for_axes, set_axes_style_for_axes, set_axes_units_for_axes,
    set_axis_tick_angles_for_axes, set_axis_tick_formats_for_axes, set_axis_tick_labels_for_axes,
    set_axis_ticks_for_axes, set_figure_background_color, set_figure_name, set_figure_number_title,
    set_figure_position, set_figure_tag, set_figure_visible, set_legend_for_axes,
    set_root_default_property, set_root_show_hidden_handles, set_root_units,
    set_sg_title_properties_for_figure, set_text_annotation_properties_for_axes,
    set_text_properties_for_axes, FigureHandle, PlotObjectKind, RootPropertyValue,
};
use super::style::{
    color_from_name_or_token, parse_color_value, parse_handle_visibility, value_as_bool,
    value_as_f64, value_as_string, LineStyleParseOptions,
};
use super::{plotting_error, plotting_error_with_source};
use crate::builtins::common::tensor;
use crate::builtins::plotting::axis_scale::scale_mode_from_value;
use crate::builtins::plotting::histogram2;
use crate::builtins::plotting::op_common::limits::limit_value;
use crate::builtins::plotting::op_common::value_as_text_string;
use crate::builtins::stats::summary::binscatter;
use crate::BuiltinResult;

const MAX_AXES_FONT_SIZE_POINTS: f64 = 512.0;

#[derive(Clone, Debug)]
pub enum PlotHandle {
    Root,
    Figure(FigureHandle),
    Axes(FigureHandle, usize),
    Ruler(FigureHandle, usize, PlotObjectKind),
    Text(FigureHandle, usize, PlotObjectKind),
    Legend(FigureHandle, usize),
    PlotChild(f64, Box<super::state::PlotChildHandleState>),
}

pub fn resolve_plot_handle(value: &Value, builtin: &'static str) -> BuiltinResult<PlotHandle> {
    let scalar = handle_scalar(value, builtin)?;
    if scalar == 0.0 {
        return Ok(PlotHandle::Root);
    }
    if !scalar.is_finite() || scalar < 0.0 {
        return Err(plotting_error(
            builtin,
            format!("{builtin}: unsupported or invalid plotting handle"),
        ));
    }
    if let Ok(state) = super::state::plot_child_handle_snapshot(scalar) {
        return Ok(PlotHandle::PlotChild(scalar.round(), Box::new(state)));
    }
    if let Ok((handle, axes_index, kind)) = decode_plot_object_handle(scalar) {
        if axes_handle_exists(handle, axes_index) {
            return Ok(match kind {
                PlotObjectKind::Legend => PlotHandle::Legend(handle, axes_index),
                PlotObjectKind::XAxis | PlotObjectKind::YAxis => {
                    PlotHandle::Ruler(handle, axes_index, kind)
                }
                _ => PlotHandle::Text(handle, axes_index, kind),
            });
        }
    }
    if let Ok((handle, axes_index)) = decode_axes_handle(scalar) {
        if axes_handle_exists(handle, axes_index) {
            return Ok(PlotHandle::Axes(handle, axes_index));
        }
        return Err(plotting_error(
            builtin,
            format!("{builtin}: invalid axes handle"),
        ));
    }
    let figure = FigureHandle::from(scalar.round() as u32);
    if figure_handle_exists(figure) {
        return Ok(PlotHandle::Figure(figure));
    }
    Err(plotting_error(
        builtin,
        format!("{builtin}: unsupported or invalid plotting handle"),
    ))
}

pub fn get_properties(
    handle: PlotHandle,
    property: Option<&str>,
    builtin: &'static str,
) -> BuiltinResult<Value> {
    match handle {
        PlotHandle::Root => get_root_property(property, builtin),
        PlotHandle::Axes(handle, axes_index) => {
            get_axes_property(handle, axes_index, property, builtin)
        }
        PlotHandle::Ruler(handle, axes_index, kind) => {
            get_ruler_property(handle, axes_index, kind, property, builtin)
        }
        PlotHandle::Text(handle, axes_index, kind) => {
            get_text_property(handle, axes_index, kind, property, builtin)
        }
        PlotHandle::Legend(handle, axes_index) => {
            get_legend_property(handle, axes_index, property, builtin)
        }
        PlotHandle::Figure(handle) => get_figure_property(handle, property, builtin),
        PlotHandle::PlotChild(_, state) => get_plot_child_property(&state, property, builtin),
    }
}

pub fn set_properties(
    handle: PlotHandle,
    args: &[Value],
    builtin: &'static str,
) -> BuiltinResult<()> {
    if args.is_empty() || !args.len().is_multiple_of(2) {
        return Err(plotting_error(
            builtin,
            format!("{builtin}: property/value arguments must come in pairs"),
        ));
    }
    match handle {
        PlotHandle::Root => {
            for pair in args.chunks_exact(2) {
                let raw_key = property_name_text(&pair[0], builtin)?;
                let key = canonical_property_name(raw_key.trim()).into_owned();
                apply_root_property(&key, raw_key.trim(), &pair[1], builtin)?;
            }
            Ok(())
        }
        PlotHandle::Figure(handle) => {
            let mut needs_present = false;
            for pair in args.chunks_exact(2) {
                validate_figure_property_value(&pair[0], &pair[1], Some(handle), builtin)?;
            }
            for pair in args.chunks_exact(2) {
                let key = property_name(&pair[0], builtin)?;
                needs_present |= apply_figure_property(handle, &key, &pair[1], builtin)?;
            }
            if needs_present {
                let figure = super::state::clone_figure(handle)
                    .ok_or_else(|| plotting_error(builtin, format!("{builtin}: invalid figure")))?;
                let _ = present_figure_update(builtin, handle, figure)?;
            }
            Ok(())
        }
        PlotHandle::Axes(handle, axes_index) => {
            for pair in args.chunks_exact(2) {
                let key = property_name(&pair[0], builtin)?;
                apply_axes_property(handle, axes_index, &key, &pair[1], builtin)?;
            }
            Ok(())
        }
        PlotHandle::Ruler(handle, axes_index, kind) => {
            for pair in args.chunks_exact(2) {
                let key = property_name(&pair[0], builtin)?;
                apply_ruler_property(handle, axes_index, kind, &key, &pair[1], builtin)?;
            }
            Ok(())
        }
        PlotHandle::Text(handle, axes_index, kind) => {
            let mut text: Option<String> = None;
            let mut style = if matches!(kind, PlotObjectKind::SuperTitle) {
                super::state::clone_figure(handle)
                    .ok_or_else(|| plotting_error(builtin, format!("{builtin}: invalid figure")))?
                    .sg_title_style
            } else {
                axes_metadata_snapshot(handle, axes_index)
                    .map_err(|err| map_figure_error(builtin, err))?
                    .text_style_for(kind)
            };
            for pair in args.chunks_exact(2) {
                let key = property_name(&pair[0], builtin)?;
                apply_text_property(&mut text, &mut style, &key, &pair[1], builtin)?;
            }
            if matches!(kind, PlotObjectKind::SuperTitle) {
                set_sg_title_properties_for_figure(handle, text, Some(style))
                    .map_err(|err| map_figure_error(builtin, err))?;
            } else {
                set_text_properties_for_axes(handle, axes_index, kind, text, Some(style))
                    .map_err(|err| map_figure_error(builtin, err))?;
            }
            Ok(())
        }
        PlotHandle::Legend(handle, axes_index) => {
            let snapshot = axes_metadata_snapshot(handle, axes_index)
                .map_err(|err| map_figure_error(builtin, err))?;
            let mut style = snapshot.legend_style;
            let mut enabled = snapshot.legend_enabled;
            let mut labels: Option<Vec<String>> = None;
            for pair in args.chunks_exact(2) {
                let key = property_name(&pair[0], builtin)?;
                apply_legend_property(
                    &mut style,
                    &mut enabled,
                    &mut labels,
                    &key,
                    &pair[1],
                    builtin,
                )?;
            }
            set_legend_for_axes(handle, axes_index, enabled, labels.as_deref(), Some(style))
                .map_err(|err| map_figure_error(builtin, err))?;
            Ok(())
        }
        PlotHandle::PlotChild(handle, state) => {
            apply_plot_child_properties(handle, &state, args, builtin)
        }
    }
}

pub fn parse_text_style_pairs(builtin: &'static str, args: &[Value]) -> BuiltinResult<TextStyle> {
    if args.is_empty() {
        return Ok(TextStyle::default());
    }
    if !args.len().is_multiple_of(2) {
        return Err(plotting_error(
            builtin,
            format!("{builtin}: property/value arguments must come in pairs"),
        ));
    }
    let mut style = TextStyle::default();
    let mut text = None;
    for pair in args.chunks_exact(2) {
        let key = property_name(&pair[0], builtin)?;
        apply_text_property(&mut text, &mut style, &key, &pair[1], builtin)?;
    }
    Ok(style)
}

pub fn validate_heatmap_property_pairs(
    args: &[Value],
    x_label_len: usize,
    y_label_len: usize,
    builtin: &'static str,
) -> BuiltinResult<()> {
    if args.is_empty() {
        return Ok(());
    }
    if !args.len().is_multiple_of(2) {
        return Err(plotting_error(
            builtin,
            format!("{builtin}: property/value arguments must come in pairs"),
        ));
    }
    for pair in args.chunks_exact(2) {
        let key = property_name(&pair[0], builtin)?;
        match key.as_str() {
            "title" => validate_axes_text_alias(PlotObjectKind::Title, &pair[1], builtin)?,
            "subtitle" => validate_axes_text_alias(PlotObjectKind::Subtitle, &pair[1], builtin)?,
            "xlabel" => validate_axes_text_alias(PlotObjectKind::XLabel, &pair[1], builtin)?,
            "ylabel" => validate_axes_text_alias(PlotObjectKind::YLabel, &pair[1], builtin)?,
            "colorbar" | "colorbarvisible" => {
                heatmap_on_off_value(&pair[1]).ok_or_else(|| {
                    plotting_error(
                        builtin,
                        format!("{builtin}: ColorbarVisible must be 'on', 'off', 0, or 1"),
                    )
                })?;
            }
            "gridvisible" => {
                heatmap_on_off_value(&pair[1]).ok_or_else(|| {
                    plotting_error(
                        builtin,
                        format!("{builtin}: GridVisible must be 'on', 'off', 0, or 1"),
                    )
                })?;
            }
            "fontsize" => {
                let font_size = heatmap_numeric_scalar_f64(&pair[1]).ok_or_else(|| {
                    plotting_error(
                        builtin,
                        format!("{builtin}: FontSize must be a numeric scalar"),
                    )
                })?;
                if !font_size.is_finite()
                    || font_size <= 0.0
                    || font_size > MAX_AXES_FONT_SIZE_POINTS
                {
                    return Err(plotting_error(
                        builtin,
                        format!(
                            "{builtin}: FontSize must be a positive finite value no larger than {MAX_AXES_FONT_SIZE_POINTS}"
                        ),
                    ));
                }
            }
            "colorlimits" => {
                crate::builtins::plotting::op_common::limits::limits_from_value(&pair[1], builtin)?;
            }
            "colormap" => {
                let name = value_as_string(&pair[1]).ok_or_else(|| {
                    plotting_error(builtin, format!("{builtin}: Colormap must be a string"))
                })?;
                parse_colormap_name(&name, builtin)?;
            }
            "xdisplaylabels" => {
                let labels = label_strings_from_value(&pair[1], builtin, "labels")?;
                if labels.len() != x_label_len {
                    return Err(plotting_error(
                        builtin,
                        format!("{builtin}: XDisplayLabels length must match heatmap columns"),
                    ));
                }
            }
            "ydisplaylabels" => {
                let labels = label_strings_from_value(&pair[1], builtin, "labels")?;
                if labels.len() != y_label_len {
                    return Err(plotting_error(
                        builtin,
                        format!("{builtin}: YDisplayLabels length must match heatmap rows"),
                    ));
                }
            }
            other => {
                return Err(plotting_error(
                    builtin,
                    format!("{builtin}: unsupported heatmap property `{other}`"),
                ));
            }
        }
    }
    Ok(())
}

pub fn split_legend_style_pairs<'a>(
    builtin: &'static str,
    args: &'a [Value],
) -> BuiltinResult<(&'a [Value], LegendStyle)> {
    let mut style = LegendStyle::default();
    let mut enabled = true;
    let mut labels = None;
    let mut split = args.len();
    while split >= 2 {
        let key_idx = split - 2;
        let Ok(key) = property_name(&args[key_idx], builtin) else {
            break;
        };
        if !matches!(
            key.as_str(),
            "location"
                | "fontsize"
                | "fontweight"
                | "fontangle"
                | "interpreter"
                | "textcolor"
                | "color"
                | "visible"
                | "string"
                | "box"
                | "orientation"
        ) {
            break;
        }
        apply_legend_property(
            &mut style,
            &mut enabled,
            &mut labels,
            &key,
            &args[key_idx + 1],
            builtin,
        )?;
        split -= 2;
    }
    Ok((&args[..split], style))
}

pub fn map_figure_error(
    builtin: &'static str,
    err: impl std::error::Error + Send + Sync + 'static,
) -> crate::RuntimeError {
    let message = format!("{builtin}: {err}");
    plotting_error_with_source(builtin, message, err)
}

fn get_figure_property(
    handle: FigureHandle,
    property: Option<&str>,
    builtin: &'static str,
) -> BuiltinResult<Value> {
    let figure = super::state::clone_figure(handle)
        .ok_or_else(|| plotting_error(builtin, format!("{builtin}: invalid figure")))?;
    let axes = axes_handles_for_figure(handle).map_err(|err| map_figure_error(builtin, err))?;
    let current_axes =
        current_axes_handle_for_figure(handle).map_err(|err| map_figure_error(builtin, err))?;
    let sg_title_handle =
        super::state::encode_plot_object_handle(handle, 0, PlotObjectKind::SuperTitle);
    let has_sg_title = figure_has_sg_title(handle);
    match property.map(canonical_property_name).as_deref() {
        None => {
            let mut st = StructValue::new();
            st.insert("Handle", Value::Num(handle.as_u32() as f64));
            st.insert("Number", Value::Num(handle.as_u32() as f64));
            st.insert("Type", Value::String("figure".into()));
            st.insert("CurrentAxes", Value::Num(current_axes));
            let mut children = axes;
            if has_sg_title {
                children.push(sg_title_handle);
            }
            st.insert("Children", handles_value(children));
            st.insert("Parent", Value::Num(f64::NAN));
            st.insert(
                "Name",
                Value::String(figure.name.clone().unwrap_or_default()),
            );
            st.insert("NumberTitle", Value::Bool(figure.number_title));
            st.insert("Visible", Value::Bool(figure.visible));
            st.insert("Position", figure_position_value(figure.position));
            st.insert(
                "Color",
                Value::String(color_to_short_name(figure.background_color)),
            );
            st.insert("Tag", Value::String(figure_tag(handle).unwrap_or_default()));
            if let Some(waitbar) = super::state::waitbar_state_snapshot(handle)
                .map_err(|err| map_figure_error(builtin, err))?
            {
                st.insert("WaitbarProgress", Value::Num(waitbar.progress));
                st.insert("WaitbarMessage", Value::String(waitbar.message));
            }
            st.insert("SGTitle", Value::Num(sg_title_handle));
            Ok(Value::Struct(st))
        }
        Some("number") => Ok(Value::Num(handle.as_u32() as f64)),
        Some("type") => Ok(Value::String("figure".into())),
        Some("currentaxes") => Ok(Value::Num(current_axes)),
        Some("children") => Ok(handles_value({
            let mut children = axes;
            if has_sg_title {
                children.push(sg_title_handle);
            }
            children
        })),
        Some("parent") => Ok(Value::Num(f64::NAN)),
        Some("name") => Ok(Value::String(figure.name.unwrap_or_default())),
        Some("numbertitle") => Ok(Value::Bool(figure.number_title)),
        Some("visible") => Ok(Value::Bool(figure.visible)),
        Some("position") => Ok(figure_position_value(figure.position)),
        Some("color") => Ok(Value::String(color_to_short_name(figure.background_color))),
        Some("waitbarprogress") => Ok(Value::Num(
            super::state::waitbar_state_snapshot(handle)
                .map_err(|err| map_figure_error(builtin, err))?
                .map(|state| state.progress)
                .unwrap_or(f64::NAN),
        )),
        Some("waitbarmessage") => Ok(Value::String(
            super::state::waitbar_state_snapshot(handle)
                .map_err(|err| map_figure_error(builtin, err))?
                .map(|state| state.message)
                .unwrap_or_default(),
        )),
        Some("tag") => Ok(Value::String(figure_tag(handle).unwrap_or_default())),
        Some("sgtitle") => Ok(Value::Num(sg_title_handle)),
        Some(other) => Err(plotting_error(
            builtin,
            format!("{builtin}: unsupported figure property `{other}`"),
        )),
    }
}

fn get_root_property(property: Option<&str>, builtin: &'static str) -> BuiltinResult<Value> {
    match property.map(canonical_property_name).as_deref() {
        None => {
            let mut st = StructValue::new();
            st.insert("Handle", Value::Num(0.0));
            st.insert("Type", Value::String("root".into()));
            st.insert("CurrentFigure", current_root_figure_value());
            st.insert("Children", root_children_value());
            st.insert("Parent", empty_handle_value());
            st.insert("ScreenSize", root_screen_size_value());
            st.insert("MonitorPositions", root_screen_size_value());
            st.insert("Units", Value::String(root_units()));
            st.insert(
                "ShowHiddenHandles",
                Value::String(on_off(root_show_hidden_handles()).into()),
            );
            for (name, value) in root_default_properties() {
                st.insert(name, root_property_value_to_value(value));
            }
            Ok(Value::Struct(st))
        }
        Some("handle") => Ok(Value::Num(0.0)),
        Some("type") => Ok(Value::String("root".into())),
        Some("currentfigure") => Ok(current_root_figure_value()),
        Some("children") => Ok(root_children_value()),
        Some("parent") => Ok(empty_handle_value()),
        Some("screensize") => Ok(root_screen_size_value()),
        Some("monitorpositions") => Ok(root_screen_size_value()),
        Some("units") => Ok(Value::String(root_units())),
        Some("showhiddenhandles") => Ok(Value::String(on_off(root_show_hidden_handles()).into())),
        Some(default) if is_root_default_property(default) => Ok(root_default_property(default)
            .map(root_property_value_to_value)
            .unwrap_or_else(|| Value::String(String::new()))),
        Some(other) => Err(plotting_error(
            builtin,
            format!("{builtin}: unsupported root property `{other}`"),
        )),
    }
}

fn apply_root_property(
    key: &str,
    display_key: &str,
    value: &Value,
    builtin: &'static str,
) -> BuiltinResult<()> {
    match key {
        "currentfigure" => {
            let resolved = resolve_plot_handle(value, builtin)?;
            let PlotHandle::Figure(handle) = resolved else {
                return Err(plotting_error(
                    builtin,
                    format!("{builtin}: CurrentFigure must be a figure handle"),
                ));
            };
            select_current_figure_if_exists(handle)
                .map_err(|err| map_figure_error(builtin, err))?;
            Ok(())
        }
        "units" => {
            let units = value_as_text_string(value)
                .ok_or_else(|| plotting_error(builtin, format!("{builtin}: Units must be text")))?
                .trim()
                .to_ascii_lowercase();
            if !matches!(
                units.as_str(),
                "pixels" | "normalized" | "inches" | "centimeters" | "points" | "characters"
            ) {
                return Err(plotting_error(
                    builtin,
                    format!("{builtin}: unsupported root Units `{units}`"),
                ));
            }
            set_root_units(units);
            Ok(())
        }
        "showhiddenhandles" => {
            let enabled = value_as_bool(value).ok_or_else(|| {
                plotting_error(
                    builtin,
                    format!("{builtin}: ShowHiddenHandles must be 'on' or 'off'"),
                )
            })?;
            set_root_show_hidden_handles(enabled);
            Ok(())
        }
        "handle" | "type" | "children" | "parent" | "screensize" | "monitorpositions" => {
            Err(plotting_error(
                builtin,
                format!("{builtin}: root property `{key}` is read-only"),
            ))
        }
        default if is_root_default_property(default) => {
            let stored = root_property_value_from_value(value, builtin)?;
            set_root_default_property(default.to_string(), display_key.to_string(), stored);
            Ok(())
        }
        other => Err(plotting_error(
            builtin,
            format!("{builtin}: unsupported root property `{other}`"),
        )),
    }
}

fn current_root_figure_value() -> Value {
    current_figure_handle_if_exists()
        .map(|handle| Value::Num(handle.as_u32() as f64))
        .unwrap_or_else(empty_handle_value)
}

fn root_children_value() -> Value {
    handles_value(
        root_figure_handles()
            .into_iter()
            .map(|handle| handle.as_u32() as f64)
            .collect(),
    )
}

fn root_screen_size_value() -> Value {
    tensor_from_vec(vec![1.0, 1.0, 1920.0, 1080.0])
}

fn empty_handle_value() -> Value {
    Value::Tensor(Tensor::new(Vec::new(), vec![0, 0]).expect("empty tensor shape is valid"))
}

fn on_off(value: bool) -> &'static str {
    if value {
        "on"
    } else {
        "off"
    }
}

fn is_root_default_property(name: &str) -> bool {
    name.starts_with("default") && name.len() > "default".len()
}

fn root_property_value_from_value(
    value: &Value,
    builtin: &'static str,
) -> BuiltinResult<RootPropertyValue> {
    match value {
        Value::Bool(value) => Ok(RootPropertyValue::Bool(*value)),
        Value::Num(value) => Ok(RootPropertyValue::Num(*value)),
        Value::Int(value) => Ok(RootPropertyValue::Num(value.to_f64())),
        Value::String(value) => Ok(RootPropertyValue::String(value.clone())),
        Value::CharArray(value) => Ok(RootPropertyValue::String(value.data.iter().collect())),
        Value::Tensor(value) => Ok(RootPropertyValue::Tensor(value.clone())),
        Value::StringArray(value) => Ok(RootPropertyValue::StringArray {
            rows: value.rows,
            cols: value.cols,
            shape: value.shape.clone(),
            data: value.data.clone(),
        }),
        _ => Err(plotting_error(
            builtin,
            format!("{builtin}: unsupported root default property value"),
        )),
    }
}

fn root_property_value_to_value(value: RootPropertyValue) -> Value {
    match value {
        RootPropertyValue::Bool(value) => Value::Bool(value),
        RootPropertyValue::Num(value) => Value::Num(value),
        RootPropertyValue::String(value) => Value::String(value),
        RootPropertyValue::Tensor(value) => Value::Tensor(value),
        RootPropertyValue::StringArray {
            rows,
            cols,
            shape,
            data,
        } => Value::StringArray(StringArray {
            rows,
            cols,
            shape,
            data,
        }),
    }
}

fn get_axes_property(
    handle: FigureHandle,
    axes_index: usize,
    property: Option<&str>,
    builtin: &'static str,
) -> BuiltinResult<Value> {
    let meta =
        axes_metadata_snapshot(handle, axes_index).map_err(|err| map_figure_error(builtin, err))?;
    let axes =
        axes_state_snapshot(handle, axes_index).map_err(|err| map_figure_error(builtin, err))?;
    let display_bounds = axis_display_bounds_snapshot_for_axes(handle, axes_index)
        .map_err(|err| map_figure_error(builtin, err))?;
    match property.map(canonical_property_name).as_deref() {
        None => {
            let mut st = StructValue::new();
            st.insert(
                "Handle",
                Value::Num(super::state::encode_axes_handle(handle, axes_index)),
            );
            st.insert("Figure", Value::Num(handle.as_u32() as f64));
            st.insert("Rows", Value::Num(axes.rows as f64));
            st.insert("Cols", Value::Num(axes.cols as f64));
            st.insert("Index", Value::Num((axes_index + 1) as f64));
            st.insert(
                "Title",
                Value::Num(super::state::encode_plot_object_handle(
                    handle,
                    axes_index,
                    PlotObjectKind::Title,
                )),
            );
            st.insert(
                "Subtitle",
                Value::Num(super::state::encode_plot_object_handle(
                    handle,
                    axes_index,
                    PlotObjectKind::Subtitle,
                )),
            );
            st.insert(
                "XLabel",
                Value::Num(super::state::encode_plot_object_handle(
                    handle,
                    axes_index,
                    PlotObjectKind::XLabel,
                )),
            );
            st.insert(
                "YLabel",
                Value::Num(super::state::encode_plot_object_handle(
                    handle,
                    axes_index,
                    PlotObjectKind::YLabel,
                )),
            );
            st.insert(
                "ZLabel",
                Value::Num(super::state::encode_plot_object_handle(
                    handle,
                    axes_index,
                    PlotObjectKind::ZLabel,
                )),
            );
            st.insert(
                "Legend",
                Value::Num(super::state::encode_plot_object_handle(
                    handle,
                    axes_index,
                    PlotObjectKind::Legend,
                )),
            );
            st.insert(
                "XAxis",
                Value::Num(super::state::encode_plot_object_handle(
                    handle,
                    axes_index,
                    PlotObjectKind::XAxis,
                )),
            );
            st.insert(
                "YAxis",
                Value::Num(super::state::encode_plot_object_handle(
                    handle,
                    axes_index,
                    PlotObjectKind::YAxis,
                )),
            );
            st.insert("LegendVisible", Value::Bool(meta.legend_enabled));
            st.insert("Type", Value::String("axes".into()));
            st.insert("Parent", Value::Num(handle.as_u32() as f64));
            st.insert("Position", figure_position_value(meta.position));
            st.insert("Units", Value::String(meta.units.clone()));
            st.insert(
                "Children",
                handles_value(vec![
                    super::state::encode_plot_object_handle(
                        handle,
                        axes_index,
                        PlotObjectKind::Title,
                    ),
                    super::state::encode_plot_object_handle(
                        handle,
                        axes_index,
                        PlotObjectKind::Subtitle,
                    ),
                    super::state::encode_plot_object_handle(
                        handle,
                        axes_index,
                        PlotObjectKind::XLabel,
                    ),
                    super::state::encode_plot_object_handle(
                        handle,
                        axes_index,
                        PlotObjectKind::YLabel,
                    ),
                    super::state::encode_plot_object_handle(
                        handle,
                        axes_index,
                        PlotObjectKind::ZLabel,
                    ),
                    super::state::encode_plot_object_handle(
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
                Value::String(meta.data_aspect_ratio_mode.clone()),
            );
            st.insert("Colorbar", Value::Bool(meta.colorbar_enabled));
            st.insert(
                "Colormap",
                Value::String(format!("{:?}", meta.colormap).to_ascii_lowercase()),
            );
            st.insert(
                "ColorLimits",
                crate::builtins::plotting::op_common::limits::limit_value(meta.color_limits),
            );
            st.insert("XLim", limit_value(meta.x_limits));
            st.insert("YLim", limit_value(meta.y_limits));
            st.insert("ZLim", limit_value(meta.z_limits));
            st.insert("CLim", limit_value(meta.color_limits));
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
                Value::String(tick_format_or_default(meta.x_tick_format.as_deref())),
            );
            st.insert(
                "YTickLabelFormat",
                Value::String(tick_format_or_default(meta.y_tick_format.as_deref())),
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
            st.insert("YAxisLocation", Value::String(meta.y_axis_location.clone()));
            Ok(Value::Struct(st))
        }
        Some("title") => Ok(Value::Num(super::state::encode_plot_object_handle(
            handle,
            axes_index,
            PlotObjectKind::Title,
        ))),
        Some("subtitle") => Ok(Value::Num(super::state::encode_plot_object_handle(
            handle,
            axes_index,
            PlotObjectKind::Subtitle,
        ))),
        Some("xlabel") => Ok(Value::Num(super::state::encode_plot_object_handle(
            handle,
            axes_index,
            PlotObjectKind::XLabel,
        ))),
        Some("ylabel") => Ok(Value::Num(super::state::encode_plot_object_handle(
            handle,
            axes_index,
            PlotObjectKind::YLabel,
        ))),
        Some("zlabel") => Ok(Value::Num(super::state::encode_plot_object_handle(
            handle,
            axes_index,
            PlotObjectKind::ZLabel,
        ))),
        Some("legend") => Ok(Value::Num(super::state::encode_plot_object_handle(
            handle,
            axes_index,
            PlotObjectKind::Legend,
        ))),
        Some("xaxis") => Ok(Value::Num(super::state::encode_plot_object_handle(
            handle,
            axes_index,
            PlotObjectKind::XAxis,
        ))),
        Some("yaxis") => Ok(Value::Num(super::state::encode_plot_object_handle(
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
        Some("dataaspectratiomode") => Ok(Value::String(meta.data_aspect_ratio_mode.clone())),
        Some("colorbar") => Ok(Value::Bool(meta.colorbar_enabled)),
        Some("colormap") => Ok(Value::String(
            format!("{:?}", meta.colormap).to_ascii_lowercase(),
        )),
        Some("colorlimits") => Ok(crate::builtins::plotting::op_common::limits::limit_value(
            meta.color_limits,
        )),
        Some("xlim") => Ok(limit_value(meta.x_limits)),
        Some("ylim") => Ok(limit_value(meta.y_limits)),
        Some("zlim") => Ok(limit_value(meta.z_limits)),
        Some("clim") => Ok(limit_value(meta.color_limits)),
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
        Some("xticklabelformat") => Ok(Value::String(tick_format_or_default(
            meta.x_tick_format.as_deref(),
        ))),
        Some("yticklabelformat") => Ok(Value::String(tick_format_or_default(
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
        Some("yaxislocation") => Ok(Value::String(meta.y_axis_location)),
        Some("type") => Ok(Value::String("axes".into())),
        Some("parent") => Ok(Value::Num(handle.as_u32() as f64)),
        Some("position") => Ok(figure_position_value(meta.position)),
        Some("units") => Ok(Value::String(meta.units)),
        Some("legendvisible") => Ok(Value::Bool(meta.legend_enabled)),
        Some("children") => Ok(handles_value(vec![
            super::state::encode_plot_object_handle(handle, axes_index, PlotObjectKind::Title),
            super::state::encode_plot_object_handle(handle, axes_index, PlotObjectKind::Subtitle),
            super::state::encode_plot_object_handle(handle, axes_index, PlotObjectKind::XLabel),
            super::state::encode_plot_object_handle(handle, axes_index, PlotObjectKind::YLabel),
            super::state::encode_plot_object_handle(handle, axes_index, PlotObjectKind::ZLabel),
            super::state::encode_plot_object_handle(handle, axes_index, PlotObjectKind::Legend),
        ])),
        Some(other) => Err(plotting_error(
            builtin,
            format!("{builtin}: unsupported axes property `{other}`"),
        )),
    }
}

fn get_text_property(
    handle: FigureHandle,
    axes_index: usize,
    kind: PlotObjectKind,
    property: Option<&str>,
    builtin: &'static str,
) -> BuiltinResult<Value> {
    let (text, style) = match kind {
        PlotObjectKind::SuperTitle => {
            let figure = super::state::clone_figure(handle)
                .ok_or_else(|| plotting_error(builtin, format!("{builtin}: invalid figure")))?;
            (figure.sg_title, figure.sg_title_style)
        }
        PlotObjectKind::Title
        | PlotObjectKind::Subtitle
        | PlotObjectKind::XLabel
        | PlotObjectKind::YLabel
        | PlotObjectKind::ZLabel => {
            let meta = axes_metadata_snapshot(handle, axes_index)
                .map_err(|err| map_figure_error(builtin, err))?;
            match kind {
                PlotObjectKind::Title => (meta.title, meta.title_style),
                PlotObjectKind::Subtitle => (meta.subtitle, meta.subtitle_style),
                PlotObjectKind::XLabel => (meta.x_label, meta.x_label_style),
                PlotObjectKind::YLabel => (meta.y_label, meta.y_label_style),
                PlotObjectKind::ZLabel => (meta.z_label, meta.z_label_style),
                PlotObjectKind::Legend
                | PlotObjectKind::SuperTitle
                | PlotObjectKind::XAxis
                | PlotObjectKind::YAxis => unreachable!(),
            }
        }
        PlotObjectKind::Legend | PlotObjectKind::XAxis | PlotObjectKind::YAxis => unreachable!(),
    };
    let parent = if matches!(kind, PlotObjectKind::SuperTitle) {
        Value::Num(handle.as_u32() as f64)
    } else {
        Value::Num(super::state::encode_axes_handle(handle, axes_index))
    };
    match property.map(canonical_property_name).as_deref() {
        None => {
            let mut st = StructValue::new();
            st.insert("Type", Value::String("text".into()));
            st.insert("Parent", parent.clone());
            st.insert("Children", handles_value(Vec::new()));
            st.insert("String", text_value(text));
            st.insert("Visible", Value::Bool(style.visible));
            if let Some(size) = style.font_size {
                st.insert("FontSize", Value::Num(size as f64));
            }
            if let Some(weight) = style.font_weight {
                st.insert("FontWeight", Value::String(weight));
            }
            if let Some(angle) = style.font_angle {
                st.insert("FontAngle", Value::String(angle));
            }
            if let Some(interpreter) = style.interpreter {
                st.insert("Interpreter", Value::String(interpreter));
            }
            if let Some(color) = style.color {
                st.insert("Color", Value::String(color_to_short_name(color)));
            }
            Ok(Value::Struct(st))
        }
        Some("type") => Ok(Value::String("text".into())),
        Some("parent") => Ok(parent),
        Some("children") => Ok(handles_value(Vec::new())),
        Some("string") => Ok(text_value(text)),
        Some("visible") => Ok(Value::Bool(style.visible)),
        Some("fontsize") => Ok(style
            .font_size
            .map(|v| Value::Num(v as f64))
            .unwrap_or(Value::Num(f64::NAN))),
        Some("fontweight") => Ok(style
            .font_weight
            .map(Value::String)
            .unwrap_or_else(|| Value::String(String::new()))),
        Some("fontangle") => Ok(style
            .font_angle
            .map(Value::String)
            .unwrap_or_else(|| Value::String(String::new()))),
        Some("interpreter") => Ok(style
            .interpreter
            .map(Value::String)
            .unwrap_or_else(|| Value::String(String::new()))),
        Some("color") => Ok(style
            .color
            .map(|c| Value::String(color_to_short_name(c)))
            .unwrap_or_else(|| Value::String(String::new()))),
        Some(other) => Err(plotting_error(
            builtin,
            format!("{builtin}: unsupported text property `{other}`"),
        )),
    }
}

fn get_ruler_property(
    handle: FigureHandle,
    axes_index: usize,
    kind: PlotObjectKind,
    property: Option<&str>,
    builtin: &'static str,
) -> BuiltinResult<Value> {
    let meta =
        axes_metadata_snapshot(handle, axes_index).map_err(|err| map_figure_error(builtin, err))?;
    let (axis_name, format, rotation) = match kind {
        PlotObjectKind::XAxis => (
            "x",
            meta.x_tick_format.as_deref(),
            meta.x_tick_label_rotation,
        ),
        PlotObjectKind::YAxis => (
            "y",
            meta.y_tick_format.as_deref(),
            meta.y_tick_label_rotation,
        ),
        _ => {
            return Err(plotting_error(
                builtin,
                format!("{builtin}: invalid ruler handle"),
            ))
        }
    };
    let parent = Value::Num(super::state::encode_axes_handle(handle, axes_index));
    match property.map(canonical_property_name).as_deref() {
        None => {
            let mut st = StructValue::new();
            st.insert("Type", Value::String("numericruler".into()));
            st.insert("Parent", parent);
            st.insert("Children", handles_value(Vec::new()));
            st.insert("Axis", Value::String(axis_name.into()));
            st.insert(
                "TickLabelFormat",
                Value::String(tick_format_or_default(format)),
            );
            st.insert("TickLabelRotation", Value::Num(rotation.unwrap_or(0.0)));
            Ok(Value::Struct(st))
        }
        Some("type") => Ok(Value::String("numericruler".into())),
        Some("parent") => Ok(parent),
        Some("children") => Ok(handles_value(Vec::new())),
        Some("axis") => Ok(Value::String(axis_name.into())),
        Some("ticklabelformat") => Ok(Value::String(tick_format_or_default(format))),
        Some("ticklabelrotation") => Ok(Value::Num(rotation.unwrap_or(0.0))),
        Some(other) => Err(plotting_error(
            builtin,
            format!("{builtin}: unsupported ruler property `{other}`"),
        )),
    }
}

fn get_legend_property(
    handle: FigureHandle,
    axes_index: usize,
    property: Option<&str>,
    builtin: &'static str,
) -> BuiltinResult<Value> {
    let meta =
        axes_metadata_snapshot(handle, axes_index).map_err(|err| map_figure_error(builtin, err))?;
    let entries = legend_entries_snapshot(handle, axes_index)
        .map_err(|err| map_figure_error(builtin, err))?;
    match property.map(canonical_property_name).as_deref() {
        None => {
            let mut st = StructValue::new();
            st.insert("Type", Value::String("legend".into()));
            st.insert(
                "Parent",
                Value::Num(super::state::encode_axes_handle(handle, axes_index)),
            );
            st.insert("Children", handles_value(Vec::new()));
            st.insert(
                "Visible",
                Value::Bool(meta.legend_enabled && meta.legend_style.visible),
            );
            st.insert(
                "String",
                legend_labels_value(entries.iter().map(|e| e.label.clone()).collect()),
            );
            if let Some(location) = meta.legend_style.location {
                st.insert("Location", Value::String(location));
            }
            if let Some(size) = meta.legend_style.font_size {
                st.insert("FontSize", Value::Num(size as f64));
            }
            if let Some(weight) = meta.legend_style.font_weight {
                st.insert("FontWeight", Value::String(weight));
            }
            if let Some(angle) = meta.legend_style.font_angle {
                st.insert("FontAngle", Value::String(angle));
            }
            if let Some(interpreter) = meta.legend_style.interpreter {
                st.insert("Interpreter", Value::String(interpreter));
            }
            if let Some(box_visible) = meta.legend_style.box_visible {
                st.insert("Box", Value::Bool(box_visible));
            }
            if let Some(orientation) = meta.legend_style.orientation {
                st.insert("Orientation", Value::String(orientation));
            }
            if let Some(color) = meta.legend_style.text_color {
                st.insert("TextColor", Value::String(color_to_short_name(color)));
            }
            Ok(Value::Struct(st))
        }
        Some("visible") => Ok(Value::Bool(
            meta.legend_enabled && meta.legend_style.visible,
        )),
        Some("type") => Ok(Value::String("legend".into())),
        Some("parent") => Ok(Value::Num(super::state::encode_axes_handle(
            handle, axes_index,
        ))),
        Some("children") => Ok(handles_value(Vec::new())),
        Some("string") => Ok(legend_labels_value(
            entries.into_iter().map(|e| e.label).collect(),
        )),
        Some("location") => Ok(meta
            .legend_style
            .location
            .map(Value::String)
            .unwrap_or_else(|| Value::String(String::new()))),
        Some("fontsize") => Ok(meta
            .legend_style
            .font_size
            .map(|v| Value::Num(v as f64))
            .unwrap_or(Value::Num(f64::NAN))),
        Some("fontweight") => Ok(meta
            .legend_style
            .font_weight
            .map(Value::String)
            .unwrap_or_else(|| Value::String(String::new()))),
        Some("fontangle") => Ok(meta
            .legend_style
            .font_angle
            .map(Value::String)
            .unwrap_or_else(|| Value::String(String::new()))),
        Some("interpreter") => Ok(meta
            .legend_style
            .interpreter
            .map(Value::String)
            .unwrap_or_else(|| Value::String(String::new()))),
        Some("box") => Ok(meta
            .legend_style
            .box_visible
            .map(Value::Bool)
            .unwrap_or(Value::Bool(true))),
        Some("orientation") => Ok(meta
            .legend_style
            .orientation
            .map(Value::String)
            .unwrap_or_else(|| Value::String(String::new()))),
        Some("textcolor") | Some("color") => Ok(meta
            .legend_style
            .text_color
            .map(|c| Value::String(color_to_short_name(c)))
            .unwrap_or_else(|| Value::String(String::new()))),
        Some(other) => Err(plotting_error(
            builtin,
            format!("{builtin}: unsupported legend property `{other}`"),
        )),
    }
}

fn property_name(value: &Value, builtin: &'static str) -> BuiltinResult<String> {
    property_name_text(value, builtin).map(|s| canonical_property_name(s.trim()).into_owned())
}

pub(crate) fn is_heatmap_property_name(value: &Value) -> bool {
    let Some(name) = value_as_string(value) else {
        return false;
    };
    matches!(
        canonical_property_name(name.trim()).as_ref(),
        "title"
            | "subtitle"
            | "xlabel"
            | "ylabel"
            | "colorbar"
            | "colorbarvisible"
            | "gridvisible"
            | "fontsize"
            | "colorlimits"
            | "colormap"
            | "xdisplaylabels"
            | "ydisplaylabels"
    )
}

fn property_name_text(value: &Value, builtin: &'static str) -> BuiltinResult<String> {
    value_as_string(value)
        .map(|s| s.trim().to_string())
        .ok_or_else(|| {
            plotting_error(
                builtin,
                format!("{builtin}: property names must be strings"),
            )
        })
}

pub(crate) fn data_aspect_ratio_from_value(
    value: &Value,
    builtin: &'static str,
) -> BuiltinResult<[f64; 3]> {
    let tensor =
        Tensor::try_from(value).map_err(|e| plotting_error(builtin, format!("{builtin}: {e}")))?;
    let data = tensor::tensor_values_f64(&tensor);
    if data.len() != 3 || data.iter().any(|value| !value.is_finite() || *value <= 0.0) {
        return Err(plotting_error(
            builtin,
            format!(
                "{builtin}: DataAspectRatio must be a 3-element positive finite numeric vector"
            ),
        ));
    }
    Ok([data[0], data[1], data[2]])
}

pub(crate) fn data_aspect_ratio_mode_from_value(
    value: &Value,
    builtin: &'static str,
) -> BuiltinResult<&'static str> {
    let mode = value_as_string(value).ok_or_else(|| {
        plotting_error(
            builtin,
            format!("{builtin}: DataAspectRatioMode must be text"),
        )
    })?;
    match mode.trim().to_ascii_lowercase().as_str() {
        "auto" => Ok("auto"),
        "manual" => Ok("manual"),
        other => Err(plotting_error(
            builtin,
            format!("{builtin}: unsupported DataAspectRatioMode `{other}`"),
        )),
    }
}

fn canonical_property_name(name: &str) -> Cow<'_, str> {
    match name.to_ascii_lowercase().as_str() {
        "textcolor" => Cow::Borrowed("textcolor"),
        "color" | "backgroundcolor" => Cow::Borrowed("color"),
        "fontsize" => Cow::Borrowed("fontsize"),
        "fontweight" => Cow::Borrowed("fontweight"),
        "fontangle" => Cow::Borrowed("fontangle"),
        "interpreter" => Cow::Borrowed("interpreter"),
        "visible" => Cow::Borrowed("visible"),
        "location" => Cow::Borrowed("location"),
        "box" => Cow::Borrowed("box"),
        "orientation" => Cow::Borrowed("orientation"),
        "string" => Cow::Borrowed("string"),
        "title" => Cow::Borrowed("title"),
        "xlabel" => Cow::Borrowed("xlabel"),
        "ylabel" => Cow::Borrowed("ylabel"),
        "zlabel" => Cow::Borrowed("zlabel"),
        "xaxis" => Cow::Borrowed("xaxis"),
        "yaxis" => Cow::Borrowed("yaxis"),
        "axis" => Cow::Borrowed("axis"),
        "view" => Cow::Borrowed("view"),
        "grid" | "xgrid" | "ygrid" | "zgrid" => Cow::Borrowed("grid"),
        "minorgrid" | "xminorgrid" | "yminorgrid" | "zminorgrid" => Cow::Borrowed("minorgrid"),
        "axisequal" => Cow::Borrowed("axisequal"),
        "colorbar" => Cow::Borrowed("colorbar"),
        "colorbarvisible" => Cow::Borrowed("colorbarvisible"),
        "gridvisible" => Cow::Borrowed("gridvisible"),
        "colormap" => Cow::Borrowed("colormap"),
        "xdisplaylabels" => Cow::Borrowed("xdisplaylabels"),
        "ydisplaylabels" => Cow::Borrowed("ydisplaylabels"),
        "colordata" | "cdata" => Cow::Borrowed("colordata"),
        "colorlimits" => Cow::Borrowed("colorlimits"),
        "xlim" => Cow::Borrowed("xlim"),
        "ylim" => Cow::Borrowed("ylim"),
        "zlim" => Cow::Borrowed("zlim"),
        "clim" | "caxis" => Cow::Borrowed("clim"),
        "xtick" => Cow::Borrowed("xtick"),
        "ytick" => Cow::Borrowed("ytick"),
        "xtickmode" => Cow::Borrowed("xtickmode"),
        "ytickmode" => Cow::Borrowed("ytickmode"),
        "xticklabel" => Cow::Borrowed("xticklabel"),
        "yticklabel" => Cow::Borrowed("yticklabel"),
        "xticklabelmode" => Cow::Borrowed("xticklabelmode"),
        "yticklabelmode" => Cow::Borrowed("yticklabelmode"),
        "xticklabelformat" => Cow::Borrowed("xticklabelformat"),
        "yticklabelformat" => Cow::Borrowed("yticklabelformat"),
        "ticklabelformat" => Cow::Borrowed("ticklabelformat"),
        "xticklabelrotation" => Cow::Borrowed("xticklabelrotation"),
        "yticklabelrotation" => Cow::Borrowed("yticklabelrotation"),
        "ticklabelrotation" => Cow::Borrowed("ticklabelrotation"),
        "hiddenlineremoval" => Cow::Borrowed("hiddenlineremoval"),
        "xscale" => Cow::Borrowed("xscale"),
        "yscale" => Cow::Borrowed("yscale"),
        "yaxislocation" => Cow::Borrowed("yaxislocation"),
        "currentaxes" => Cow::Borrowed("currentaxes"),
        "currentfigure" => Cow::Borrowed("currentfigure"),
        "sgtitle" | "supertitle" => Cow::Borrowed("sgtitle"),
        "children" => Cow::Borrowed("children"),
        "handle" => Cow::Borrowed("handle"),
        "parent" => Cow::Borrowed("parent"),
        "type" => Cow::Borrowed("type"),
        "screensize" => Cow::Borrowed("screensize"),
        "monitorpositions" => Cow::Borrowed("monitorpositions"),
        "units" => Cow::Borrowed("units"),
        "showhiddenhandles" => Cow::Borrowed("showhiddenhandles"),
        "number" => Cow::Borrowed("number"),
        "name" => Cow::Borrowed("name"),
        "numbertitle" => Cow::Borrowed("numbertitle"),
        "legend" => Cow::Borrowed("legend"),
        "legendvisible" => Cow::Borrowed("legendvisible"),
        "autoscalefactor" => Cow::Borrowed("autoscalefactor"),
        "barwidth" => Cow::Borrowed("barwidth"),
        "baseline" => Cow::Borrowed("baseline"),
        "basevalue" => Cow::Borrowed("basevalue"),
        "bincounts" => Cow::Borrowed("bincounts"),
        "binedges" => Cow::Borrowed("binedges"),
        "capsize" => Cow::Borrowed("capsize"),
        "cdatamapping" => Cow::Borrowed("cdatamapping"),
        "displayname" => Cow::Borrowed("displayname"),
        "edgealpha" => Cow::Borrowed("edgealpha"),
        "edgecolor" => Cow::Borrowed("edgecolor"),
        "facealpha" => Cow::Borrowed("facealpha"),
        "facecolor" => Cow::Borrowed("facecolor"),
        "faces" => Cow::Borrowed("faces"),
        "filled" => Cow::Borrowed("filled"),
        "label" => Cow::Borrowed("label"),
        "labelorientation" => Cow::Borrowed("labelorientation"),
        "linestyle" => Cow::Borrowed("linestyle"),
        "linewidth" => Cow::Borrowed("linewidth"),
        "marker" => Cow::Borrowed("marker"),
        "markeredgecolor" => Cow::Borrowed("markeredgecolor"),
        "markerfacecolor" => Cow::Borrowed("markerfacecolor"),
        "markersize" => Cow::Borrowed("markersize"),
        "maxheadsize" => Cow::Borrowed("maxheadsize"),
        "normalization" => Cow::Borrowed("normalization"),
        "numbins" => Cow::Borrowed("numbins"),
        "position" => Cow::Borrowed("position"),
        "sizedata" => Cow::Borrowed("sizedata"),
        "value" => Cow::Borrowed("value"),
        "vertices" => Cow::Borrowed("vertices"),
        "xdata" => Cow::Borrowed("xdata"),
        "ydata" => Cow::Borrowed("ydata"),
        "zdata" => Cow::Borrowed("zdata"),
        other => Cow::Owned(other.to_string()),
    }
}

fn apply_text_property(
    text: &mut Option<String>,
    style: &mut TextStyle,
    key: &str,
    value: &Value,
    builtin: &'static str,
) -> BuiltinResult<()> {
    let opts = LineStyleParseOptions::generic(builtin);
    match key {
        "string" => {
            *text = Some(value_as_text_string(value).ok_or_else(|| {
                plotting_error(builtin, format!("{builtin}: String must be text"))
            })?);
        }
        "color" => style.color = Some(parse_color_value(&opts, value)?),
        "fontsize" => {
            style.font_size = Some(value_as_f64(value).ok_or_else(|| {
                plotting_error(builtin, format!("{builtin}: FontSize must be numeric"))
            })? as f32)
        }
        "fontweight" => {
            style.font_weight = Some(value_as_string(value).ok_or_else(|| {
                plotting_error(builtin, format!("{builtin}: FontWeight must be a string"))
            })?)
        }
        "fontangle" => {
            style.font_angle = Some(value_as_string(value).ok_or_else(|| {
                plotting_error(builtin, format!("{builtin}: FontAngle must be a string"))
            })?)
        }
        "interpreter" => {
            style.interpreter = Some(value_as_string(value).ok_or_else(|| {
                plotting_error(builtin, format!("{builtin}: Interpreter must be a string"))
            })?)
        }
        "visible" => {
            style.visible = value_as_bool(value).ok_or_else(|| {
                plotting_error(builtin, format!("{builtin}: Visible must be logical"))
            })?
        }
        other => {
            return Err(plotting_error(
                builtin,
                format!("{builtin}: unsupported property `{other}`"),
            ))
        }
    }
    Ok(())
}

fn apply_legend_property(
    style: &mut LegendStyle,
    enabled: &mut bool,
    labels: &mut Option<Vec<String>>,
    key: &str,
    value: &Value,
    builtin: &'static str,
) -> BuiltinResult<()> {
    let opts = LineStyleParseOptions::generic(builtin);
    match key {
        "string" => *labels = Some(collect_label_strings(builtin, std::slice::from_ref(value))?),
        "location" => {
            style.location = Some(
                value_as_string(value)
                    .ok_or_else(|| plotting_error(builtin, "legend: Location must be a string"))?,
            )
        }
        "fontsize" => {
            style.font_size = Some(
                value_as_f64(value)
                    .ok_or_else(|| plotting_error(builtin, "legend: FontSize must be numeric"))?
                    as f32,
            )
        }
        "fontweight" => {
            style.font_weight =
                Some(value_as_string(value).ok_or_else(|| {
                    plotting_error(builtin, "legend: FontWeight must be a string")
                })?)
        }
        "fontangle" => {
            style.font_angle = Some(
                value_as_string(value)
                    .ok_or_else(|| plotting_error(builtin, "legend: FontAngle must be a string"))?,
            )
        }
        "interpreter" => {
            style.interpreter =
                Some(value_as_string(value).ok_or_else(|| {
                    plotting_error(builtin, "legend: Interpreter must be a string")
                })?)
        }
        "textcolor" | "color" => style.text_color = Some(parse_color_value(&opts, value)?),
        "visible" => {
            let visible = value_as_bool(value)
                .ok_or_else(|| plotting_error(builtin, "legend: Visible must be logical"))?;
            style.visible = visible;
            *enabled = visible;
        }
        "box" => {
            style.box_visible = Some(
                value_as_bool(value)
                    .ok_or_else(|| plotting_error(builtin, "legend: Box must be logical"))?,
            )
        }
        "orientation" => {
            style.orientation =
                Some(value_as_string(value).ok_or_else(|| {
                    plotting_error(builtin, "legend: Orientation must be a string")
                })?)
        }
        other => {
            return Err(plotting_error(
                builtin,
                format!("{builtin}: unsupported property `{other}`"),
            ))
        }
    }
    Ok(())
}

fn apply_axes_property(
    handle: FigureHandle,
    axes_index: usize,
    key: &str,
    value: &Value,
    builtin: &'static str,
) -> BuiltinResult<()> {
    match key {
        "legendvisible" => {
            let visible = value_as_bool(value).ok_or_else(|| {
                plotting_error(builtin, format!("{builtin}: LegendVisible must be logical"))
            })?;
            set_legend_for_axes(handle, axes_index, visible, None, None)
                .map_err(|err| map_figure_error(builtin, err))?;
            Ok(())
        }
        "title" => apply_axes_text_alias(handle, axes_index, PlotObjectKind::Title, value, builtin),
        "subtitle" => {
            apply_axes_text_alias(handle, axes_index, PlotObjectKind::Subtitle, value, builtin)
        }
        "xlabel" => {
            apply_axes_text_alias(handle, axes_index, PlotObjectKind::XLabel, value, builtin)
        }
        "ylabel" => {
            apply_axes_text_alias(handle, axes_index, PlotObjectKind::YLabel, value, builtin)
        }
        "zlabel" => {
            apply_axes_text_alias(handle, axes_index, PlotObjectKind::ZLabel, value, builtin)
        }
        "view" => {
            let tensor = runmat_builtins::Tensor::try_from(value)
                .map_err(|e| plotting_error(builtin, format!("{builtin}: {e}")))?;
            let data = tensor::tensor_values_f64(&tensor);
            if data.len() != 2 || !data[0].is_finite() || !data[1].is_finite() {
                return Err(plotting_error(
                    builtin,
                    format!("{builtin}: View must be a 2-element finite numeric vector"),
                ));
            }
            crate::builtins::plotting::state::set_view_for_axes(
                handle,
                axes_index,
                data[0] as f32,
                data[1] as f32,
            )
            .map_err(|err| map_figure_error(builtin, err))?;
            Ok(())
        }
        "grid" => {
            let enabled = value_as_bool(value).ok_or_else(|| {
                plotting_error(builtin, format!("{builtin}: Grid must be logical"))
            })?;
            crate::builtins::plotting::state::set_grid_enabled_for_axes(
                handle, axes_index, enabled,
            )
            .map_err(|err| map_figure_error(builtin, err))?;
            Ok(())
        }
        "minorgrid" => {
            let enabled = value_as_bool(value).ok_or_else(|| {
                plotting_error(builtin, format!("{builtin}: MinorGrid must be logical"))
            })?;
            crate::builtins::plotting::state::set_minor_grid_enabled_for_axes(
                handle, axes_index, enabled,
            )
            .map_err(|err| map_figure_error(builtin, err))?;
            Ok(())
        }
        "box" => {
            let enabled = value_as_bool(value).ok_or_else(|| {
                plotting_error(builtin, format!("{builtin}: Box must be logical"))
            })?;
            crate::builtins::plotting::state::set_box_enabled_for_axes(handle, axes_index, enabled)
                .map_err(|err| map_figure_error(builtin, err))?;
            Ok(())
        }
        "hiddenlineremoval" => {
            let enabled = value_as_bool(value).ok_or_else(|| {
                plotting_error(
                    builtin,
                    format!("{builtin}: HiddenLineRemoval must be 'on' or 'off'"),
                )
            })?;
            crate::builtins::plotting::state::set_hidden_line_removal_for_axes(
                handle, axes_index, enabled,
            )
            .map_err(|err| map_figure_error(builtin, err))?;
            Ok(())
        }
        "axisequal" => {
            let enabled = value_as_bool(value).ok_or_else(|| {
                plotting_error(builtin, format!("{builtin}: AxisEqual must be logical"))
            })?;
            crate::builtins::plotting::state::set_axis_equal_for_axes(handle, axes_index, enabled)
                .map_err(|err| map_figure_error(builtin, err))?;
            Ok(())
        }
        "dataaspectratio" => {
            let ratio = data_aspect_ratio_from_value(value, builtin)?;
            crate::builtins::plotting::state::set_data_aspect_ratio_for_axes(
                handle, axes_index, ratio, "manual",
            )
            .map_err(|err| map_figure_error(builtin, err))?;
            Ok(())
        }
        "dataaspectratiomode" => {
            let mode = data_aspect_ratio_mode_from_value(value, builtin)?;
            let (ratio, _) = crate::builtins::plotting::state::data_aspect_ratio_snapshot_for_axes(
                handle, axes_index,
            )
            .map_err(|err| map_figure_error(builtin, err))?;
            crate::builtins::plotting::state::set_data_aspect_ratio_for_axes(
                handle, axes_index, ratio, mode,
            )
            .map_err(|err| map_figure_error(builtin, err))?;
            Ok(())
        }
        "colorbar" => {
            let enabled = value_as_bool(value).ok_or_else(|| {
                plotting_error(builtin, format!("{builtin}: Colorbar must be logical"))
            })?;
            crate::builtins::plotting::state::set_colorbar_enabled_for_axes(
                handle, axes_index, enabled,
            )
            .map_err(|err| map_figure_error(builtin, err))?;
            Ok(())
        }
        "colormap" => {
            let name = value_as_string(value).ok_or_else(|| {
                plotting_error(builtin, format!("{builtin}: Colormap must be a string"))
            })?;
            let cmap = parse_colormap_name(&name, builtin)?;
            crate::builtins::plotting::state::set_colormap_for_axes(handle, axes_index, cmap)
                .map_err(|err| map_figure_error(builtin, err))?;
            Ok(())
        }
        "fontsize" => {
            let font_size = value_as_f64(value).ok_or_else(|| {
                plotting_error(builtin, format!("{builtin}: FontSize must be numeric"))
            })?;
            if !font_size.is_finite() || font_size <= 0.0 || font_size > MAX_AXES_FONT_SIZE_POINTS {
                return Err(plotting_error(
                    builtin,
                    format!(
                        "{builtin}: FontSize must be a positive finite value no larger than {MAX_AXES_FONT_SIZE_POINTS}"
                    ),
                ));
            }
            let meta = axes_metadata_snapshot(handle, axes_index)
                .map_err(|err| map_figure_error(builtin, err))?;
            let mut style = meta.axes_style;
            style.font_size = Some(font_size as f32);
            set_axes_style_for_axes(handle, axes_index, style)
                .map_err(|err| map_figure_error(builtin, err))?;
            Ok(())
        }
        "visible" => {
            let visible = value_as_bool(value).ok_or_else(|| {
                plotting_error(builtin, format!("{builtin}: Visible must be logical"))
            })?;
            let meta = axes_metadata_snapshot(handle, axes_index)
                .map_err(|err| map_figure_error(builtin, err))?;
            let mut style = meta.axes_style;
            style.visible = visible;
            set_axes_style_for_axes(handle, axes_index, style)
                .map_err(|err| map_figure_error(builtin, err))?;
            Ok(())
        }
        "xlim" => {
            let limits = limits_from_optional_value(value, builtin)?;
            let meta = axes_metadata_snapshot(handle, axes_index)
                .map_err(|err| map_figure_error(builtin, err))?;
            crate::builtins::plotting::state::set_axis_limits_for_axes(
                handle,
                axes_index,
                limits,
                meta.y_limits,
            )
            .map_err(|err| map_figure_error(builtin, err))?;
            Ok(())
        }
        "ylim" => {
            let limits = limits_from_optional_value(value, builtin)?;
            let meta = axes_metadata_snapshot(handle, axes_index)
                .map_err(|err| map_figure_error(builtin, err))?;
            crate::builtins::plotting::state::set_axis_limits_for_axes(
                handle,
                axes_index,
                meta.x_limits,
                limits,
            )
            .map_err(|err| map_figure_error(builtin, err))?;
            Ok(())
        }
        "zlim" => {
            let limits = limits_from_optional_value(value, builtin)?;
            crate::builtins::plotting::state::set_z_limits_for_axes(handle, axes_index, limits)
                .map_err(|err| map_figure_error(builtin, err))?;
            Ok(())
        }
        "clim" => {
            let limits = limits_from_optional_value(value, builtin)?;
            crate::builtins::plotting::state::set_color_limits_for_axes(handle, axes_index, limits)
                .map_err(|err| map_figure_error(builtin, err))?;
            Ok(())
        }
        "xtick" => {
            let ticks = ticks_from_value(value, builtin)?;
            let meta = axes_metadata_snapshot(handle, axes_index)
                .map_err(|err| map_figure_error(builtin, err))?;
            set_axis_ticks_for_axes(handle, axes_index, Some(ticks), meta.y_ticks)
                .map_err(|err| map_figure_error(builtin, err))?;
            Ok(())
        }
        "ytick" => {
            let ticks = ticks_from_value(value, builtin)?;
            let meta = axes_metadata_snapshot(handle, axes_index)
                .map_err(|err| map_figure_error(builtin, err))?;
            set_axis_ticks_for_axes(handle, axes_index, meta.x_ticks, Some(ticks))
                .map_err(|err| map_figure_error(builtin, err))?;
            Ok(())
        }
        "xtickmode" => {
            let mode = tick_mode_from_value(value, builtin, "XTickMode")?;
            let meta = axes_metadata_snapshot(handle, axes_index)
                .map_err(|err| map_figure_error(builtin, err))?;
            let display_bounds = axis_display_bounds_snapshot_for_axes(handle, axes_index)
                .map_err(|err| map_figure_error(builtin, err))?;
            let x = match mode {
                TickMode::Auto => None,
                TickMode::Manual => Some(ticks_or_auto(
                    meta.x_ticks.as_deref(),
                    x_bounds(display_bounds),
                )),
            };
            set_axis_ticks_for_axes(handle, axes_index, x, meta.y_ticks)
                .map_err(|err| map_figure_error(builtin, err))?;
            Ok(())
        }
        "ytickmode" => {
            let mode = tick_mode_from_value(value, builtin, "YTickMode")?;
            let meta = axes_metadata_snapshot(handle, axes_index)
                .map_err(|err| map_figure_error(builtin, err))?;
            let display_bounds = axis_display_bounds_snapshot_for_axes(handle, axes_index)
                .map_err(|err| map_figure_error(builtin, err))?;
            let y = match mode {
                TickMode::Auto => None,
                TickMode::Manual => Some(ticks_or_auto(
                    meta.y_ticks.as_deref(),
                    y_bounds(display_bounds),
                )),
            };
            set_axis_ticks_for_axes(handle, axes_index, meta.x_ticks, y)
                .map_err(|err| map_figure_error(builtin, err))?;
            Ok(())
        }
        "xticklabel" => {
            let labels = tick_labels_from_value(value, builtin)?;
            let meta = axes_metadata_snapshot(handle, axes_index)
                .map_err(|err| map_figure_error(builtin, err))?;
            let display_bounds = axis_display_bounds_snapshot_for_axes(handle, axes_index)
                .map_err(|err| map_figure_error(builtin, err))?;
            let ticks = ticks_or_auto(meta.x_ticks.as_deref(), x_bounds(display_bounds));
            set_axis_ticks_for_axes(handle, axes_index, Some(ticks.clone()), meta.y_ticks)
                .map_err(|err| map_figure_error(builtin, err))?;
            set_axis_tick_labels_for_axes(
                handle,
                axes_index,
                Some(labels_padded_to_ticks(labels, ticks.len())),
                meta.y_tick_labels,
            )
            .map_err(|err| map_figure_error(builtin, err))?;
            Ok(())
        }
        "yticklabel" => {
            let labels = tick_labels_from_value(value, builtin)?;
            let meta = axes_metadata_snapshot(handle, axes_index)
                .map_err(|err| map_figure_error(builtin, err))?;
            let display_bounds = axis_display_bounds_snapshot_for_axes(handle, axes_index)
                .map_err(|err| map_figure_error(builtin, err))?;
            let ticks = ticks_or_auto(meta.y_ticks.as_deref(), y_bounds(display_bounds));
            set_axis_ticks_for_axes(handle, axes_index, meta.x_ticks, Some(ticks.clone()))
                .map_err(|err| map_figure_error(builtin, err))?;
            set_axis_tick_labels_for_axes(
                handle,
                axes_index,
                meta.x_tick_labels,
                Some(labels_padded_to_ticks(labels, ticks.len())),
            )
            .map_err(|err| map_figure_error(builtin, err))?;
            Ok(())
        }
        "xticklabelmode" => {
            let mode = tick_mode_from_value(value, builtin, "XTickLabelMode")?;
            let meta = axes_metadata_snapshot(handle, axes_index)
                .map_err(|err| map_figure_error(builtin, err))?;
            let display_bounds = axis_display_bounds_snapshot_for_axes(handle, axes_index)
                .map_err(|err| map_figure_error(builtin, err))?;
            let x_labels = match mode {
                TickMode::Auto => None,
                TickMode::Manual => Some(tick_labels_or_auto(
                    meta.x_tick_labels.as_deref(),
                    meta.x_ticks.as_deref(),
                    x_bounds(display_bounds),
                    meta.x_tick_format.as_deref(),
                )),
            };
            if matches!(mode, TickMode::Manual) {
                let ticks = ticks_or_auto(meta.x_ticks.as_deref(), x_bounds(display_bounds));
                set_axis_ticks_for_axes(handle, axes_index, Some(ticks), meta.y_ticks)
                    .map_err(|err| map_figure_error(builtin, err))?;
            }
            set_axis_tick_labels_for_axes(handle, axes_index, x_labels, meta.y_tick_labels)
                .map_err(|err| map_figure_error(builtin, err))?;
            Ok(())
        }
        "yticklabelmode" => {
            let mode = tick_mode_from_value(value, builtin, "YTickLabelMode")?;
            let meta = axes_metadata_snapshot(handle, axes_index)
                .map_err(|err| map_figure_error(builtin, err))?;
            let display_bounds = axis_display_bounds_snapshot_for_axes(handle, axes_index)
                .map_err(|err| map_figure_error(builtin, err))?;
            let y_labels = match mode {
                TickMode::Auto => None,
                TickMode::Manual => Some(tick_labels_or_auto(
                    meta.y_tick_labels.as_deref(),
                    meta.y_ticks.as_deref(),
                    y_bounds(display_bounds),
                    meta.y_tick_format.as_deref(),
                )),
            };
            if matches!(mode, TickMode::Manual) {
                let ticks = ticks_or_auto(meta.y_ticks.as_deref(), y_bounds(display_bounds));
                set_axis_ticks_for_axes(handle, axes_index, meta.x_ticks, Some(ticks))
                    .map_err(|err| map_figure_error(builtin, err))?;
            }
            set_axis_tick_labels_for_axes(handle, axes_index, meta.x_tick_labels, y_labels)
                .map_err(|err| map_figure_error(builtin, err))?;
            Ok(())
        }
        "xticklabelformat" => {
            let format = tick_format_from_value(value, builtin, "XTickLabelFormat")?;
            let meta = axes_metadata_snapshot(handle, axes_index)
                .map_err(|err| map_figure_error(builtin, err))?;
            set_axis_tick_formats_for_axes(handle, axes_index, Some(format), meta.y_tick_format)
                .map_err(|err| map_figure_error(builtin, err))?;
            Ok(())
        }
        "yticklabelformat" => {
            let format = tick_format_from_value(value, builtin, "YTickLabelFormat")?;
            let meta = axes_metadata_snapshot(handle, axes_index)
                .map_err(|err| map_figure_error(builtin, err))?;
            set_axis_tick_formats_for_axes(handle, axes_index, meta.x_tick_format, Some(format))
                .map_err(|err| map_figure_error(builtin, err))?;
            Ok(())
        }
        "xticklabelrotation" => {
            let angle = tick_angle_from_value(value, builtin, "XTickLabelRotation")?;
            let meta = axes_metadata_snapshot(handle, axes_index)
                .map_err(|err| map_figure_error(builtin, err))?;
            set_axis_tick_angles_for_axes(
                handle,
                axes_index,
                Some(angle),
                meta.y_tick_label_rotation,
            )
            .map_err(|err| map_figure_error(builtin, err))?;
            Ok(())
        }
        "yticklabelrotation" => {
            let angle = tick_angle_from_value(value, builtin, "YTickLabelRotation")?;
            let meta = axes_metadata_snapshot(handle, axes_index)
                .map_err(|err| map_figure_error(builtin, err))?;
            set_axis_tick_angles_for_axes(
                handle,
                axes_index,
                meta.x_tick_label_rotation,
                Some(angle),
            )
            .map_err(|err| map_figure_error(builtin, err))?;
            Ok(())
        }
        "xscale" => {
            let mode = scale_mode_from_value(value, builtin)?;
            let meta = axes_metadata_snapshot(handle, axes_index)
                .map_err(|err| map_figure_error(builtin, err))?;
            crate::builtins::plotting::state::set_log_modes_for_axes(
                handle,
                axes_index,
                mode.is_log(),
                meta.y_log,
            )
            .map_err(|err| map_figure_error(builtin, err))?;
            Ok(())
        }
        "yscale" => {
            let mode = scale_mode_from_value(value, builtin)?;
            let meta = axes_metadata_snapshot(handle, axes_index)
                .map_err(|err| map_figure_error(builtin, err))?;
            crate::builtins::plotting::state::set_log_modes_for_axes(
                handle,
                axes_index,
                meta.x_log,
                mode.is_log(),
            )
            .map_err(|err| map_figure_error(builtin, err))?;
            Ok(())
        }
        "yaxislocation" => {
            let location = value_as_string(value)
                .ok_or_else(|| {
                    plotting_error(builtin, format!("{builtin}: YAxisLocation must be text"))
                })?
                .trim()
                .to_ascii_lowercase();
            if !matches!(location.as_str(), "left" | "right") {
                return Err(plotting_error(
                    builtin,
                    format!("{builtin}: YAxisLocation must be 'left' or 'right'"),
                ));
            }
            crate::builtins::plotting::state::set_y_axis_location_for_axes(
                handle, axes_index, location,
            )
            .map_err(|err| map_figure_error(builtin, err))?;
            Ok(())
        }
        "position" => {
            let position = parse_figure_position(value, builtin)?;
            set_axes_position_for_axes(handle, axes_index, position)
                .map_err(|err| map_figure_error(builtin, err))?;
            Ok(())
        }
        "units" => {
            let units = value_as_string(value)
                .ok_or_else(|| plotting_error(builtin, format!("{builtin}: Units must be text")))?
                .trim()
                .to_ascii_lowercase();
            if !matches!(
                units.as_str(),
                "normalized" | "pixels" | "inches" | "centimeters" | "points" | "characters"
            ) {
                return Err(plotting_error(
                    builtin,
                    format!("{builtin}: unsupported axes Units `{units}`"),
                ));
            }
            set_axes_units_for_axes(handle, axes_index, units)
                .map_err(|err| map_figure_error(builtin, err))?;
            Ok(())
        }
        other => Err(plotting_error(
            builtin,
            format!("{builtin}: unsupported axes property `{other}`"),
        )),
    }
}

fn apply_ruler_property(
    handle: FigureHandle,
    axes_index: usize,
    kind: PlotObjectKind,
    key: &str,
    value: &Value,
    builtin: &'static str,
) -> BuiltinResult<()> {
    match key {
        "ticklabelformat" => {
            let format = tick_format_from_value(value, builtin, "TickLabelFormat")?;
            let meta = axes_metadata_snapshot(handle, axes_index)
                .map_err(|err| map_figure_error(builtin, err))?;
            let (x_format, y_format) = match kind {
                PlotObjectKind::XAxis => (Some(format), meta.y_tick_format),
                PlotObjectKind::YAxis => (meta.x_tick_format, Some(format)),
                _ => {
                    return Err(plotting_error(
                        builtin,
                        format!("{builtin}: invalid ruler handle"),
                    ))
                }
            };
            set_axis_tick_formats_for_axes(handle, axes_index, x_format, y_format)
                .map_err(|err| map_figure_error(builtin, err))?;
            Ok(())
        }
        "ticklabelrotation" => {
            let angle = tick_angle_from_value(value, builtin, "TickLabelRotation")?;
            let meta = axes_metadata_snapshot(handle, axes_index)
                .map_err(|err| map_figure_error(builtin, err))?;
            let (x_angle, y_angle) = match kind {
                PlotObjectKind::XAxis => (Some(angle), meta.y_tick_label_rotation),
                PlotObjectKind::YAxis => (meta.x_tick_label_rotation, Some(angle)),
                _ => {
                    return Err(plotting_error(
                        builtin,
                        format!("{builtin}: invalid ruler handle"),
                    ))
                }
            };
            set_axis_tick_angles_for_axes(handle, axes_index, x_angle, y_angle)
                .map_err(|err| map_figure_error(builtin, err))?;
            Ok(())
        }
        other => Err(plotting_error(
            builtin,
            format!("{builtin}: unsupported ruler property `{other}`"),
        )),
    }
}

fn apply_figure_property(
    figure_handle: FigureHandle,
    key: &str,
    value: &Value,
    builtin: &'static str,
) -> BuiltinResult<bool> {
    let opts = LineStyleParseOptions::generic(builtin);
    match key {
        "name" => {
            let name = value_as_text_string(value)
                .ok_or_else(|| plotting_error(builtin, format!("{builtin}: Name must be text")))?;
            set_figure_name(figure_handle, name).map_err(|err| map_figure_error(builtin, err))?;
            Ok(true)
        }
        "numbertitle" => {
            let enabled = value_as_bool(value).ok_or_else(|| {
                plotting_error(builtin, format!("{builtin}: NumberTitle must be logical"))
            })?;
            set_figure_number_title(figure_handle, enabled)
                .map_err(|err| map_figure_error(builtin, err))?;
            Ok(true)
        }
        "visible" => {
            let visible = value_as_bool(value).ok_or_else(|| {
                plotting_error(builtin, format!("{builtin}: Visible must be logical"))
            })?;
            let _ = set_figure_visible(figure_handle, visible)
                .map_err(|err| map_figure_error(builtin, err))?;
            Ok(true)
        }
        "position" => {
            let position = parse_figure_position(value, builtin)?;
            set_figure_position(figure_handle, position)
                .map_err(|err| map_figure_error(builtin, err))?;
            Ok(true)
        }
        "color" => {
            let color = parse_color_value(&opts, value)?;
            set_figure_background_color(figure_handle, color)
                .map_err(|err| map_figure_error(builtin, err))?;
            Ok(true)
        }
        "tag" => {
            let tag = value_as_text_string(value)
                .ok_or_else(|| plotting_error(builtin, format!("{builtin}: Tag must be text")))?;
            set_figure_tag(figure_handle, tag).map_err(|err| map_figure_error(builtin, err))?;
            Ok(true)
        }
        "currentaxes" => {
            let resolved = resolve_plot_handle(value, builtin)?;
            let PlotHandle::Axes(fig, axes_index) = resolved else {
                return Err(plotting_error(
                    builtin,
                    format!("{builtin}: CurrentAxes must be an axes handle"),
                ));
            };
            if fig != figure_handle {
                return Err(plotting_error(
                    builtin,
                    format!("{builtin}: CurrentAxes must belong to the target figure"),
                ));
            }
            select_axes_for_figure(figure_handle, axes_index)
                .map_err(|err| map_figure_error(builtin, err))?;
            Ok(true)
        }
        "sgtitle" => {
            apply_figure_text_alias(figure_handle, PlotObjectKind::SuperTitle, value, builtin)?;
            Ok(true)
        }
        other => Err(plotting_error(
            builtin,
            format!("{builtin}: unsupported figure property `{other}`"),
        )),
    }
}

pub(crate) fn validate_figure_property_value(
    key_value: &Value,
    property_value: &Value,
    target_figure: Option<FigureHandle>,
    builtin: &'static str,
) -> BuiltinResult<()> {
    let key = property_name(key_value, builtin)?;
    let opts = LineStyleParseOptions::generic(builtin);
    match key.as_str() {
        "name" => {
            value_as_text_string(property_value)
                .ok_or_else(|| plotting_error(builtin, format!("{builtin}: Name must be text")))?;
        }
        "numbertitle" => {
            value_as_bool(property_value).ok_or_else(|| {
                plotting_error(builtin, format!("{builtin}: NumberTitle must be logical"))
            })?;
        }
        "visible" => {
            value_as_bool(property_value).ok_or_else(|| {
                plotting_error(builtin, format!("{builtin}: Visible must be logical"))
            })?;
        }
        "position" => {
            let _ = parse_figure_position(property_value, builtin)?;
        }
        "color" => {
            let _ = parse_color_value(&opts, property_value)?;
        }
        "tag" => {
            value_as_text_string(property_value)
                .ok_or_else(|| plotting_error(builtin, format!("{builtin}: Tag must be text")))?;
        }
        "currentaxes" => {
            let resolved = resolve_plot_handle(property_value, builtin)?;
            let PlotHandle::Axes(fig, _) = resolved else {
                return Err(plotting_error(
                    builtin,
                    format!("{builtin}: CurrentAxes must be an axes handle"),
                ));
            };
            let Some(target) = target_figure else {
                return Err(plotting_error(
                    builtin,
                    format!("{builtin}: CurrentAxes requires an existing target figure"),
                ));
            };
            if fig != target {
                return Err(plotting_error(
                    builtin,
                    format!("{builtin}: CurrentAxes must belong to the target figure"),
                ));
            }
        }
        "sgtitle" => {
            validate_figure_text_alias(PlotObjectKind::SuperTitle, property_value, builtin)?
        }
        other => {
            return Err(plotting_error(
                builtin,
                format!("{builtin}: unsupported figure property `{other}`"),
            ));
        }
    }
    Ok(())
}

fn get_histogram_property(
    hist: &super::state::HistogramHandleState,
    property: Option<&str>,
    builtin: &'static str,
) -> BuiltinResult<Value> {
    match property.map(canonical_property_name).as_deref() {
        None => {
            let mut st = StructValue::new();
            st.insert("Type", Value::String("histogram".into()));
            st.insert(
                "Parent",
                Value::Num(super::state::encode_axes_handle(
                    hist.figure,
                    hist.axes_index,
                )),
            );
            st.insert("Children", handles_value(Vec::new()));
            st.insert("BinEdges", tensor_from_vec(hist.bin_edges.clone()));
            st.insert("BinCounts", tensor_from_vec(hist.raw_counts.clone()));
            st.insert(
                "Values",
                tensor_from_vec(apply_histogram_normalization(
                    &hist.raw_counts,
                    &hist.bin_edges,
                    &hist.normalization,
                    hist.normalization_denominator,
                )),
            );
            if let Some(data) = &hist.metadata.data {
                st.insert("Data", Value::Tensor(data.clone()));
            }
            st.insert("Normalization", Value::String(hist.normalization.clone()));
            st.insert("NumBins", Value::Num(hist.raw_counts.len() as f64));
            st.insert("BinWidth", Value::Num(hist.metadata.bin_width));
            st.insert(
                "BinLimits",
                tensor_from_vec(vec![hist.metadata.bin_limits.0, hist.metadata.bin_limits.1]),
            );
            st.insert("FaceColor", Value::String(hist.metadata.face_color.clone()));
            st.insert("FaceAlpha", Value::Num(hist.metadata.face_alpha));
            st.insert("EdgeColor", Value::String(hist.metadata.edge_color.clone()));
            st.insert(
                "DisplayStyle",
                Value::String(hist.metadata.display_style.clone()),
            );
            st.insert(
                "DisplayName",
                Value::String(hist.display_name.clone().unwrap_or_default()),
            );
            Ok(Value::Struct(st))
        }
        Some("type") => Ok(Value::String("histogram".into())),
        Some("parent") => Ok(Value::Num(super::state::encode_axes_handle(
            hist.figure,
            hist.axes_index,
        ))),
        Some("children") => Ok(handles_value(Vec::new())),
        Some("binedges") => Ok(tensor_from_vec(hist.bin_edges.clone())),
        Some("bincounts") => Ok(tensor_from_vec(hist.raw_counts.clone())),
        Some("values") => Ok(tensor_from_vec(apply_histogram_normalization(
            &hist.raw_counts,
            &hist.bin_edges,
            &hist.normalization,
            hist.normalization_denominator,
        ))),
        Some("data") => Ok(Value::Tensor(hist.metadata.data.clone().unwrap_or_else(
            || Tensor::new(Vec::new(), vec![0, 0]).expect("valid empty tensor"),
        ))),
        Some("normalization") => Ok(Value::String(hist.normalization.clone())),
        Some("numbins") => Ok(Value::Num(hist.raw_counts.len() as f64)),
        Some("binwidth") => Ok(Value::Num(hist.metadata.bin_width)),
        Some("binlimits") => Ok(tensor_from_vec(vec![
            hist.metadata.bin_limits.0,
            hist.metadata.bin_limits.1,
        ])),
        Some("facecolor") => Ok(Value::String(hist.metadata.face_color.clone())),
        Some("facealpha") => Ok(Value::Num(hist.metadata.face_alpha)),
        Some("edgecolor") => Ok(Value::String(hist.metadata.edge_color.clone())),
        Some("displaystyle") => Ok(Value::String(hist.metadata.display_style.clone())),
        Some("displayname") => Ok(Value::String(hist.display_name.clone().unwrap_or_default())),
        Some(other) => Err(plotting_error(
            builtin,
            format!("{builtin}: unsupported histogram property `{other}`"),
        )),
    }
}

fn get_histogram2_property(
    hist: &super::state::Histogram2HandleState,
    property: Option<&str>,
    builtin: &'static str,
) -> BuiltinResult<Value> {
    match property.map(canonical_property_name).as_deref() {
        None => {
            let mut st = child_base_struct("histogram2", hist.figure, hist.axes_index);
            st.insert("Values", Value::Tensor(hist.values.clone()));
            st.insert("BinCounts", Value::Tensor(hist.raw_counts.clone()));
            if let Some(data) = &hist.data {
                st.insert("Data", Value::Tensor(data.clone()));
            }
            st.insert("XBinEdges", tensor_from_vec(hist.x_bin_edges.clone()));
            st.insert("YBinEdges", tensor_from_vec(hist.y_bin_edges.clone()));
            st.insert("NumBins", tensor_from_vec(histogram2_num_bins(hist)));
            st.insert("Normalization", Value::String(hist.normalization.clone()));
            st.insert(
                "DisplayStyle",
                Value::String(hist.display_style.as_str().into()),
            );
            st.insert("ShowEmptyBins", Value::Bool(hist.show_empty_bins));
            st.insert("FaceAlpha", Value::Num(hist.face_alpha));
            st.insert(
                "DisplayName",
                Value::String(hist.display_name.clone().unwrap_or_default()),
            );
            Ok(Value::Struct(st))
        }
        Some("type") => Ok(Value::String("histogram2".into())),
        Some("parent") => Ok(child_parent_handle(hist.figure, hist.axes_index)),
        Some("children") => Ok(handles_value(Vec::new())),
        Some("values") => Ok(Value::Tensor(hist.values.clone())),
        Some("bincounts") => Ok(Value::Tensor(hist.raw_counts.clone())),
        Some("bindata") => Ok(Value::Tensor(hist.values.clone())),
        Some("data") => Ok(Value::Tensor(hist.data.clone().unwrap_or_else(|| {
            Tensor::new(Vec::new(), vec![0, 0]).expect("valid empty tensor")
        }))),
        Some("xbinedges") => Ok(tensor_from_vec(hist.x_bin_edges.clone())),
        Some("ybinedges") => Ok(tensor_from_vec(hist.y_bin_edges.clone())),
        Some("numbins") => Ok(tensor_from_vec(histogram2_num_bins(hist))),
        Some("normalization") => Ok(Value::String(hist.normalization.clone())),
        Some("displaystyle") => Ok(Value::String(hist.display_style.as_str().into())),
        Some("showemptybins") => Ok(Value::Bool(hist.show_empty_bins)),
        Some("facealpha") => Ok(Value::Num(hist.face_alpha)),
        Some("displayname") => Ok(Value::String(hist.display_name.clone().unwrap_or_default())),
        Some(other) => Err(plotting_error(
            builtin,
            format!("{builtin}: unsupported histogram2 property `{other}`"),
        )),
    }
}

fn histogram2_num_bins(hist: &super::state::Histogram2HandleState) -> Vec<f64> {
    vec![
        hist.x_bin_edges.len().saturating_sub(1) as f64,
        hist.y_bin_edges.len().saturating_sub(1) as f64,
    ]
}

fn get_binscatter_property(
    binscatter: &super::state::BinscatterHandleState,
    property: Option<&str>,
    builtin: &'static str,
) -> BuiltinResult<Value> {
    match property.map(canonical_property_name).as_deref() {
        None => {
            let mut st = child_base_struct("binscatter", binscatter.figure, binscatter.axes_index);
            st.insert("Values", Value::Tensor(binscatter.values.clone()));
            st.insert("XData", Value::Tensor(binscatter.x_data.clone()));
            st.insert("YData", Value::Tensor(binscatter.y_data.clone()));
            st.insert("XBinEdges", tensor_from_vec(binscatter.x_bin_edges.clone()));
            st.insert("YBinEdges", tensor_from_vec(binscatter.y_bin_edges.clone()));
            st.insert("NumBins", tensor_from_vec(num_bins_value(binscatter)));
            st.insert("XLimits", binscatter_x_limits(binscatter));
            st.insert("YLimits", binscatter_y_limits(binscatter));
            st.insert("ShowEmptyBins", Value::Bool(binscatter.show_empty_bins));
            st.insert("FaceAlpha", Value::Num(binscatter.face_alpha));
            st.insert(
                "DisplayName",
                Value::String(binscatter.display_name.clone().unwrap_or_default()),
            );
            Ok(Value::Struct(st))
        }
        Some("type") => Ok(Value::String("binscatter".into())),
        Some("parent") => Ok(child_parent_handle(
            binscatter.figure,
            binscatter.axes_index,
        )),
        Some("children") => Ok(handles_value(Vec::new())),
        Some("values") | Some("bindata") => Ok(Value::Tensor(binscatter.values.clone())),
        Some("xdata") => Ok(Value::Tensor(binscatter.x_data.clone())),
        Some("ydata") => Ok(Value::Tensor(binscatter.y_data.clone())),
        Some("xbinedges") => Ok(tensor_from_vec(binscatter.x_bin_edges.clone())),
        Some("ybinedges") => Ok(tensor_from_vec(binscatter.y_bin_edges.clone())),
        Some("numbins") => Ok(tensor_from_vec(num_bins_value(binscatter))),
        Some("xlimits") => Ok(binscatter_x_limits(binscatter)),
        Some("ylimits") => Ok(binscatter_y_limits(binscatter)),
        Some("showemptybins") => Ok(Value::Bool(binscatter.show_empty_bins)),
        Some("facealpha") => Ok(Value::Num(binscatter.face_alpha)),
        Some("displayname") => Ok(Value::String(
            binscatter.display_name.clone().unwrap_or_default(),
        )),
        Some(other) => Err(plotting_error(
            builtin,
            format!("{builtin}: unsupported binscatter property `{other}`"),
        )),
    }
}

fn num_bins_value(binscatter: &super::state::BinscatterHandleState) -> Vec<f64> {
    vec![binscatter.num_bins[0] as f64, binscatter.num_bins[1] as f64]
}

fn binscatter_x_limits(binscatter: &super::state::BinscatterHandleState) -> Value {
    binscatter
        .x_limits_option
        .clone()
        .map(Value::Tensor)
        .unwrap_or_else(|| tensor_from_vec(vec![binscatter.x_limits.0, binscatter.x_limits.1]))
}

fn binscatter_y_limits(binscatter: &super::state::BinscatterHandleState) -> Value {
    binscatter
        .y_limits_option
        .clone()
        .map(Value::Tensor)
        .unwrap_or_else(|| tensor_from_vec(vec![binscatter.y_limits.0, binscatter.y_limits.1]))
}

fn get_plot_child_property(
    state: &super::state::PlotChildHandleState,
    property: Option<&str>,
    builtin: &'static str,
) -> BuiltinResult<Value> {
    match state {
        super::state::PlotChildHandleState::Histogram(hist) => {
            get_histogram_property(hist, property, builtin)
        }
        super::state::PlotChildHandleState::Histogram2(hist) => {
            get_histogram2_property(hist, property, builtin)
        }
        super::state::PlotChildHandleState::Line(plot) => {
            get_line_property(plot, property, builtin)
        }
        super::state::PlotChildHandleState::AnimatedLine(plot) => {
            get_animated_line_property(plot, property, builtin)
        }
        super::state::PlotChildHandleState::Scatter(plot) => {
            get_scatter_property(plot, property, builtin)
        }
        super::state::PlotChildHandleState::Bar(plot) => get_bar_property(plot, property, builtin),
        super::state::PlotChildHandleState::Stem(stem) => {
            get_stem_property(stem, property, builtin)
        }
        super::state::PlotChildHandleState::ErrorBar(errorbar) => {
            get_errorbar_property(errorbar, property, builtin)
        }
        super::state::PlotChildHandleState::Stairs(plot) => {
            get_stairs_property(plot, property, builtin)
        }
        super::state::PlotChildHandleState::Quiver(quiver) => {
            get_quiver_property(quiver, property, builtin)
        }
        super::state::PlotChildHandleState::Image(image) => {
            get_image_property(image, property, builtin)
        }
        super::state::PlotChildHandleState::Heatmap(heatmap) => {
            get_heatmap_property(heatmap, property, builtin)
        }
        super::state::PlotChildHandleState::Binscatter(binscatter) => {
            get_binscatter_property(binscatter, property, builtin)
        }
        super::state::PlotChildHandleState::FunctionSurface(function_surface) => {
            get_function_surface_property(function_surface, property, builtin)
        }
        super::state::PlotChildHandleState::FunctionContour(function_contour) => {
            get_function_contour_property(function_contour, property, builtin)
        }
        super::state::PlotChildHandleState::Area(area) => {
            get_area_property(area, property, builtin)
        }
        super::state::PlotChildHandleState::Surface(plot) => {
            get_surface_property(plot, property, builtin)
        }
        super::state::PlotChildHandleState::Patch(plot) => {
            get_patch_property(plot, property, builtin)
        }
        super::state::PlotChildHandleState::Line3(plot) => {
            get_line3_property(plot, property, builtin)
        }
        super::state::PlotChildHandleState::Scatter3(plot) => {
            get_scatter3_property(plot, property, builtin)
        }
        super::state::PlotChildHandleState::Contour(plot) => {
            get_contour_property(plot, property, builtin)
        }
        super::state::PlotChildHandleState::ContourFill(plot) => {
            get_contour_fill_property(plot, property, builtin)
        }
        super::state::PlotChildHandleState::ReferenceLine(plot) => {
            get_reference_line_property(plot, property, builtin)
        }
        super::state::PlotChildHandleState::Pie(plot) => get_pie_property(plot, property, builtin),
        super::state::PlotChildHandleState::Text(text) => {
            get_world_text_property(text, property, builtin)
        }
        super::state::PlotChildHandleState::TextScatter(textscatter) => {
            crate::builtins::plotting::textscatter::get_textscatter_property(
                textscatter,
                property,
                builtin,
            )
        }
        super::state::PlotChildHandleState::WordCloud(wordcloud) => {
            crate::builtins::plotting::wordcloud::get_wordcloud_property(
                wordcloud, property, builtin,
            )
        }
        super::state::PlotChildHandleState::StackedPlot(stacked) => {
            crate::builtins::plotting::stackedplot::get_stackedplot_property(
                stacked, property, builtin,
            )
        }
    }
}

fn apply_plot_child_property(
    handle: f64,
    state: &super::state::PlotChildHandleState,
    key: &str,
    value: &Value,
    builtin: &'static str,
) -> BuiltinResult<()> {
    match state {
        super::state::PlotChildHandleState::Histogram(hist) => {
            apply_histogram_property(hist, key, value, builtin)
        }
        super::state::PlotChildHandleState::Histogram2(hist) => {
            apply_histogram2_property(hist, key, value, builtin)
        }
        super::state::PlotChildHandleState::Line(plot) => {
            apply_line_property(plot, key, value, builtin)
        }
        super::state::PlotChildHandleState::AnimatedLine(plot) => {
            apply_animated_line_property(plot, key, value, builtin)
        }
        super::state::PlotChildHandleState::Scatter(plot) => {
            apply_scatter_property(plot, key, value, builtin)
        }
        super::state::PlotChildHandleState::Bar(plot) => {
            apply_bar_property(plot, key, value, builtin)
        }
        super::state::PlotChildHandleState::Stem(stem) => {
            apply_stem_property(stem, key, value, builtin)
        }
        super::state::PlotChildHandleState::ErrorBar(errorbar) => {
            apply_errorbar_property(errorbar, key, value, builtin)
        }
        super::state::PlotChildHandleState::Stairs(plot) => {
            apply_stairs_property(plot, key, value, builtin)
        }
        super::state::PlotChildHandleState::Quiver(quiver) => {
            apply_quiver_property(quiver, key, value, builtin)
        }
        super::state::PlotChildHandleState::Image(image) => {
            apply_image_property(image, key, value, builtin)
        }
        super::state::PlotChildHandleState::Heatmap(heatmap) => {
            apply_heatmap_property(handle, heatmap, key, value, builtin)
        }
        super::state::PlotChildHandleState::Binscatter(binscatter) => {
            apply_binscatter_property(binscatter, key, value, builtin)
        }
        super::state::PlotChildHandleState::FunctionSurface(function_surface) => {
            apply_function_surface_property(function_surface, key, value, builtin)
        }
        super::state::PlotChildHandleState::FunctionContour(function_contour) => {
            apply_function_contour_property(function_contour, key, value, builtin)
        }
        super::state::PlotChildHandleState::Area(area) => {
            apply_area_property(area, key, value, builtin)
        }
        super::state::PlotChildHandleState::Surface(plot) => {
            apply_surface_property(plot, key, value, builtin)
        }
        super::state::PlotChildHandleState::Patch(plot) => {
            apply_patch_property(plot, key, value, builtin)
        }
        super::state::PlotChildHandleState::Line3(plot) => {
            apply_line3_property(plot, key, value, builtin)
        }
        super::state::PlotChildHandleState::Scatter3(plot) => {
            apply_scatter3_property(plot, key, value, builtin)
        }
        super::state::PlotChildHandleState::Contour(plot) => {
            apply_contour_property(plot, key, value, builtin)
        }
        super::state::PlotChildHandleState::ContourFill(plot) => {
            apply_contour_fill_property(plot, key, value, builtin)
        }
        super::state::PlotChildHandleState::ReferenceLine(plot) => {
            apply_reference_line_property(plot, key, value, builtin)
        }
        super::state::PlotChildHandleState::Pie(plot) => {
            apply_pie_property(plot, key, value, builtin)
        }
        super::state::PlotChildHandleState::Text(text) => {
            apply_world_text_property(text, key, value, builtin)
        }
        super::state::PlotChildHandleState::TextScatter(textscatter) => {
            crate::builtins::plotting::textscatter::apply_textscatter_property(
                handle,
                textscatter,
                key,
                value,
                builtin,
            )
        }
        super::state::PlotChildHandleState::WordCloud(wordcloud) => {
            crate::builtins::plotting::wordcloud::apply_wordcloud_property(
                handle, wordcloud, key, value, builtin,
            )
        }
        super::state::PlotChildHandleState::StackedPlot(stacked) => {
            crate::builtins::plotting::stackedplot::apply_stackedplot_property(
                handle, stacked, key, value, builtin,
            )
        }
    }
}

fn apply_plot_child_properties(
    handle: f64,
    state: &super::state::PlotChildHandleState,
    args: &[Value],
    builtin: &'static str,
) -> BuiltinResult<()> {
    let mut pairs = Vec::with_capacity(args.len() / 2);
    for pair in args.chunks_exact(2) {
        pairs.push((property_name(&pair[0], builtin)?, &pair[1]));
    }
    match state {
        super::state::PlotChildHandleState::Line(plot) => {
            apply_line_properties(plot, &pairs, builtin)
        }
        super::state::PlotChildHandleState::AnimatedLine(plot) => {
            apply_animated_line_properties(plot, &pairs, builtin)
        }
        super::state::PlotChildHandleState::Line3(plot) => {
            apply_line3_properties(plot, &pairs, builtin)
        }
        super::state::PlotChildHandleState::Histogram2(hist) => {
            apply_histogram2_properties(hist, &pairs, builtin)
        }
        super::state::PlotChildHandleState::Scatter(plot) => {
            apply_scatter_properties(plot, &pairs, builtin)
        }
        _ => {
            for (key, value) in pairs {
                apply_plot_child_property(handle, state, &key, value, builtin)?;
            }
            Ok(())
        }
    }
}

fn child_parent_handle(figure: FigureHandle, axes_index: usize) -> Value {
    Value::Num(super::state::encode_axes_handle(figure, axes_index))
}

fn child_base_struct(kind: &str, figure: FigureHandle, axes_index: usize) -> StructValue {
    let mut st = StructValue::new();
    st.insert("Type", Value::String(kind.into()));
    st.insert("Parent", child_parent_handle(figure, axes_index));
    st.insert("Children", handles_value(Vec::new()));
    st
}

fn figure_position_value(position: [f64; 4]) -> Value {
    Value::Tensor(Tensor::new(position.to_vec(), vec![1, 4]).expect("figure position vector"))
}

fn parse_figure_position(value: &Value, builtin: &'static str) -> BuiltinResult<[f64; 4]> {
    let values = match value {
        Value::Tensor(t)
            if tensor::tensor_element_len(t) == 4 && is_figure_position_vector_shape(t) =>
        {
            tensor::tensor_values_f64(t)
        }
        _ => {
            return Err(plotting_error(
                builtin,
                format!("{builtin}: Position must be a 4-element numeric vector"),
            ))
        }
    };
    let mut position = [0.0; 4];
    position.copy_from_slice(&values);
    if !position.iter().all(|value| value.is_finite()) {
        return Err(plotting_error(
            builtin,
            format!("{builtin}: Position values must be finite"),
        ));
    }
    if position[2] <= 0.0 || position[3] <= 0.0 {
        return Err(plotting_error(
            builtin,
            format!("{builtin}: Position width and height must be positive"),
        ));
    }
    Ok(position)
}

fn is_figure_position_vector_shape(tensor: &Tensor) -> bool {
    tensor.shape.len() <= 1 || (tensor.shape.len() == 2 && (tensor.rows == 1 || tensor.cols == 1))
}

fn text_position_value(position: glam::Vec3) -> Value {
    Value::Tensor(
        Tensor::new(
            vec![position.x as f64, position.y as f64, position.z as f64],
            vec![1, 3],
        )
        .expect("text position vector"),
    )
}

fn parse_text_position(value: &Value, builtin: &'static str) -> BuiltinResult<glam::Vec3> {
    match value {
        Value::Tensor(t) if matches!(tensor::tensor_element_len(t), 2 | 3) => {
            let data = tensor::tensor_values_f64(t);
            Ok(glam::Vec3::new(
                data[0] as f32,
                data[1] as f32,
                data.get(2).copied().unwrap_or(0.0) as f32,
            ))
        }
        _ => Err(plotting_error(
            builtin,
            format!("{builtin}: Position must be a 2-element or 3-element vector"),
        )),
    }
}

fn get_world_text_property(
    handle: &super::state::TextAnnotationHandleState,
    property: Option<&str>,
    builtin: &'static str,
) -> BuiltinResult<Value> {
    let figure = super::state::clone_figure(handle.figure)
        .ok_or_else(|| plotting_error(builtin, format!("{builtin}: invalid text figure")))?;
    let annotation = figure
        .axes_text_annotation(handle.axes_index, handle.annotation_index)
        .cloned()
        .ok_or_else(|| plotting_error(builtin, format!("{builtin}: invalid text handle")))?;
    match property.map(canonical_property_name).as_deref() {
        None => {
            let mut st = child_base_struct("text", handle.figure, handle.axes_index);
            st.insert("String", Value::String(annotation.text.clone()));
            st.insert("Position", text_position_value(annotation.position));
            if let Some(weight) = annotation.style.font_weight.clone() {
                st.insert("FontWeight", Value::String(weight));
            }
            if let Some(angle) = annotation.style.font_angle.clone() {
                st.insert("FontAngle", Value::String(angle));
            }
            if let Some(interpreter) = annotation.style.interpreter.clone() {
                st.insert("Interpreter", Value::String(interpreter));
            }
            if let Some(color) = annotation.style.color {
                st.insert("Color", Value::String(color_to_short_name(color)));
            }
            if let Some(font_size) = annotation.style.font_size {
                st.insert("FontSize", Value::Num(font_size as f64));
            }
            st.insert("Visible", Value::Bool(annotation.style.visible));
            Ok(Value::Struct(st))
        }
        Some("type") => Ok(Value::String("text".into())),
        Some("parent") => Ok(child_parent_handle(handle.figure, handle.axes_index)),
        Some("children") => Ok(handles_value(Vec::new())),
        Some("string") => Ok(Value::String(annotation.text)),
        Some("position") => Ok(text_position_value(annotation.position)),
        Some("fontweight") => Ok(annotation
            .style
            .font_weight
            .map(Value::String)
            .unwrap_or_else(|| Value::String(String::new()))),
        Some("fontangle") => Ok(annotation
            .style
            .font_angle
            .map(Value::String)
            .unwrap_or_else(|| Value::String(String::new()))),
        Some("interpreter") => Ok(annotation
            .style
            .interpreter
            .map(Value::String)
            .unwrap_or_else(|| Value::String(String::new()))),
        Some("color") => Ok(annotation
            .style
            .color
            .map(|c| Value::String(color_to_short_name(c)))
            .unwrap_or_else(|| Value::String(String::new()))),
        Some("fontsize") => Ok(Value::Num(
            annotation.style.font_size.unwrap_or_default() as f64
        )),
        Some("visible") => Ok(Value::Bool(annotation.style.visible)),
        Some(other) => Err(plotting_error(
            builtin,
            format!("{builtin}: unsupported text property `{other}`"),
        )),
    }
}

fn apply_world_text_property(
    handle: &super::state::TextAnnotationHandleState,
    key: &str,
    value: &Value,
    builtin: &'static str,
) -> BuiltinResult<()> {
    let figure = super::state::clone_figure(handle.figure)
        .ok_or_else(|| plotting_error(builtin, format!("{builtin}: invalid text figure")))?;
    let annotation = figure
        .axes_text_annotation(handle.axes_index, handle.annotation_index)
        .cloned()
        .ok_or_else(|| plotting_error(builtin, format!("{builtin}: invalid text handle")))?;
    let mut text = None;
    let mut position = None;
    let mut style = annotation.style;
    match canonical_property_name(key).as_ref() {
        "string" => {
            text = Some(value_as_text_string(value).ok_or_else(|| {
                plotting_error(builtin, format!("{builtin}: String must be text"))
            })?);
        }
        "position" => position = Some(parse_text_position(value, builtin)?),
        other => apply_text_property(&mut text, &mut style, other, value, builtin)?,
    }
    set_text_annotation_properties_for_axes(
        handle.figure,
        handle.axes_index,
        handle.annotation_index,
        text,
        position,
        Some(style),
    )
    .map_err(|err| map_figure_error(builtin, err))?;
    Ok(())
}

fn get_simple_plot(
    plot: &super::state::SimplePlotHandleState,
    builtin: &'static str,
) -> BuiltinResult<runmat_plot::plots::figure::PlotElement> {
    let figure = super::state::clone_figure(plot.figure)
        .ok_or_else(|| plotting_error(builtin, format!("{builtin}: invalid plot figure")))?;
    let resolved = figure
        .plots()
        .nth(plot.plot_index)
        .cloned()
        .ok_or_else(|| plotting_error(builtin, format!("{builtin}: invalid plot handle")))?;
    Ok(resolved)
}

fn get_line_property(
    line_handle: &super::state::SimplePlotHandleState,
    property: Option<&str>,
    builtin: &'static str,
) -> BuiltinResult<Value> {
    let plot = get_simple_plot(line_handle, builtin)?;
    let runmat_plot::plots::figure::PlotElement::Line(line) = plot else {
        return Err(plotting_error(
            builtin,
            format!("{builtin}: invalid line handle"),
        ));
    };
    let (x_data, y_data) = line_xy_data_for_properties(&line, builtin)?;
    match property.map(canonical_property_name).as_deref() {
        None => {
            let mut st = child_base_struct("line", line_handle.figure, line_handle.axes_index);
            st.insert("XData", numeric_plot_data_value(&x_data));
            st.insert("YData", numeric_plot_data_value(&y_data));
            st.insert("Color", Value::String(color_to_short_name(line.color)));
            st.insert("LineWidth", Value::Num(line.line_width as f64));
            st.insert(
                "LineStyle",
                Value::String(line_style_name(line.line_style).into()),
            );
            st.insert(
                "DisplayName",
                Value::String(line.label.clone().unwrap_or_default()),
            );
            st.insert(
                "HandleVisibility",
                Value::String(line.handle_visibility.clone()),
            );
            st.insert(
                "Visible",
                Value::String(if line.visible { "on" } else { "off" }.into()),
            );
            insert_line_marker_struct_props(&mut st, line.marker.as_ref());
            Ok(Value::Struct(st))
        }
        Some("type") => Ok(Value::String("line".into())),
        Some("parent") => Ok(child_parent_handle(
            line_handle.figure,
            line_handle.axes_index,
        )),
        Some("children") => Ok(handles_value(Vec::new())),
        Some("xdata") => Ok(numeric_plot_data_value(&x_data)),
        Some("ydata") => Ok(numeric_plot_data_value(&y_data)),
        Some("color") => Ok(Value::String(color_to_short_name(line.color))),
        Some("linewidth") => Ok(Value::Num(line.line_width as f64)),
        Some("linestyle") => Ok(Value::String(line_style_name(line.line_style).into())),
        Some("displayname") => Ok(Value::String(line.label.unwrap_or_default())),
        Some("handlevisibility") => Ok(Value::String(line.handle_visibility)),
        Some("visible") => Ok(Value::String(
            if line.visible { "on" } else { "off" }.into(),
        )),
        Some(name) => line_marker_property_value(&line.marker, name, builtin),
    }
}

fn get_animated_line_property(
    line_handle: &super::state::AnimatedLineHandleState,
    property: Option<&str>,
    builtin: &'static str,
) -> BuiltinResult<Value> {
    let simple = animated_line_simple_state(line_handle);
    let plot = get_simple_plot(&simple, builtin)?;
    let property = property.map(canonical_property_name);
    match plot {
        runmat_plot::plots::figure::PlotElement::Line(line) => {
            let (x_data, y_data) = line_xy_data_for_properties(&line, builtin)?;
            match property.as_deref() {
                None => {
                    let mut st = child_base_struct(
                        "animatedline",
                        line_handle.figure,
                        line_handle.axes_index,
                    );
                    st.insert("XData", numeric_plot_data_value(&x_data));
                    st.insert("YData", numeric_plot_data_value(&y_data));
                    st.insert("Color", Value::String(color_to_short_name(line.color)));
                    st.insert("LineWidth", Value::Num(line.line_width as f64));
                    st.insert(
                        "LineStyle",
                        Value::String(line_style_name(line.line_style).into()),
                    );
                    st.insert(
                        "DisplayName",
                        Value::String(line.label.clone().unwrap_or_default()),
                    );
                    st.insert(
                        "Visible",
                        Value::String(if line.visible { "on" } else { "off" }.into()),
                    );
                    st.insert(
                        "MaximumNumPoints",
                        animated_line_maximum_value(line_handle.maximum_num_points),
                    );
                    insert_line_marker_struct_props(&mut st, line.marker.as_ref());
                    Ok(Value::Struct(st))
                }
                Some("type") => Ok(Value::String("animatedline".into())),
                Some("parent") => Ok(child_parent_handle(
                    line_handle.figure,
                    line_handle.axes_index,
                )),
                Some("children") => Ok(handles_value(Vec::new())),
                Some("xdata") => Ok(numeric_plot_data_value(&x_data)),
                Some("ydata") => Ok(numeric_plot_data_value(&y_data)),
                Some("zdata") => Ok(tensor_from_vec(Vec::new())),
                Some("color") => Ok(Value::String(color_to_short_name(line.color))),
                Some("linewidth") => Ok(Value::Num(line.line_width as f64)),
                Some("linestyle") => Ok(Value::String(line_style_name(line.line_style).into())),
                Some("displayname") => Ok(Value::String(line.label.unwrap_or_default())),
                Some("visible") => Ok(Value::String(
                    if line.visible { "on" } else { "off" }.into(),
                )),
                Some("maximumnumpoints") => {
                    Ok(animated_line_maximum_value(line_handle.maximum_num_points))
                }
                Some(name) => line_marker_property_value(&line.marker, name, builtin),
            }
        }
        runmat_plot::plots::figure::PlotElement::Line3(line) => {
            let (x_data, y_data, z_data) = line_xyz_data_for_properties(&line, builtin)?;
            match property.as_deref() {
                None => {
                    let mut st = child_base_struct(
                        "animatedline",
                        line_handle.figure,
                        line_handle.axes_index,
                    );
                    st.insert("XData", numeric_plot_data_value(&x_data));
                    st.insert("YData", numeric_plot_data_value(&y_data));
                    st.insert("ZData", numeric_plot_data_value(&z_data));
                    st.insert("Color", Value::String(color_to_short_name(line.color)));
                    st.insert("LineWidth", Value::Num(line.line_width as f64));
                    st.insert(
                        "LineStyle",
                        Value::String(line_style_name(line.line_style).into()),
                    );
                    st.insert(
                        "DisplayName",
                        Value::String(line.label.clone().unwrap_or_default()),
                    );
                    st.insert(
                        "Visible",
                        Value::String(if line.visible { "on" } else { "off" }.into()),
                    );
                    st.insert(
                        "MaximumNumPoints",
                        animated_line_maximum_value(line_handle.maximum_num_points),
                    );
                    Ok(Value::Struct(st))
                }
                Some("type") => Ok(Value::String("animatedline".into())),
                Some("parent") => Ok(child_parent_handle(
                    line_handle.figure,
                    line_handle.axes_index,
                )),
                Some("children") => Ok(handles_value(Vec::new())),
                Some("xdata") => Ok(numeric_plot_data_value(&x_data)),
                Some("ydata") => Ok(numeric_plot_data_value(&y_data)),
                Some("zdata") => Ok(numeric_plot_data_value(&z_data)),
                Some("color") => Ok(Value::String(color_to_short_name(line.color))),
                Some("linewidth") => Ok(Value::Num(line.line_width as f64)),
                Some("linestyle") => Ok(Value::String(line_style_name(line.line_style).into())),
                Some("displayname") => Ok(Value::String(line.label.unwrap_or_default())),
                Some("visible") => Ok(Value::String(
                    if line.visible { "on" } else { "off" }.into(),
                )),
                Some("maximumnumpoints") => {
                    Ok(animated_line_maximum_value(line_handle.maximum_num_points))
                }
                Some(other) => Err(plotting_error(
                    builtin,
                    format!("{builtin}: unsupported animatedline property `{other}`"),
                )),
            }
        }
        _ => Err(plotting_error(
            builtin,
            format!("{builtin}: invalid animatedline handle"),
        )),
    }
}

fn animated_line_maximum_value(maximum: Option<usize>) -> Value {
    match maximum {
        Some(value) => Value::Num(value as f64),
        None => Value::Num(f64::INFINITY),
    }
}

fn animated_line_simple_state(
    line_handle: &super::state::AnimatedLineHandleState,
) -> super::state::SimplePlotHandleState {
    super::state::SimplePlotHandleState {
        figure: line_handle.figure,
        axes_index: line_handle.axes_index,
        plot_index: line_handle.plot_index,
    }
}

fn get_reference_line_property(
    line_handle: &super::state::SimplePlotHandleState,
    property: Option<&str>,
    builtin: &'static str,
) -> BuiltinResult<Value> {
    let plot = get_simple_plot(line_handle, builtin)?;
    let runmat_plot::plots::figure::PlotElement::ReferenceLine(line) = plot else {
        return Err(plotting_error(
            builtin,
            format!("{builtin}: invalid reference line handle"),
        ));
    };
    let orientation = match line.orientation {
        runmat_plot::plots::ReferenceLineOrientation::Vertical => "vertical",
        runmat_plot::plots::ReferenceLineOrientation::Horizontal => "horizontal",
    };
    match property.map(canonical_property_name).as_deref() {
        None => {
            let mut st =
                child_base_struct("constantline", line_handle.figure, line_handle.axes_index);
            st.insert("Value", Value::Num(line.value));
            st.insert("Orientation", Value::String(orientation.into()));
            st.insert("Color", Value::String(color_to_short_name(line.color)));
            st.insert("LineWidth", Value::Num(line.line_width as f64));
            st.insert(
                "LineStyle",
                Value::String(line_style_name(line.line_style).into()),
            );
            st.insert(
                "Label",
                Value::String(line.label.clone().unwrap_or_default()),
            );
            st.insert(
                "LabelOrientation",
                Value::String(line.label_orientation.clone()),
            );
            st.insert(
                "DisplayName",
                Value::String(line.display_name.clone().unwrap_or_default()),
            );
            st.insert(
                "Visible",
                Value::String(if line.visible { "on" } else { "off" }.into()),
            );
            Ok(Value::Struct(st))
        }
        Some("type") => Ok(Value::String("constantline".into())),
        Some("parent") => Ok(child_parent_handle(
            line_handle.figure,
            line_handle.axes_index,
        )),
        Some("children") => Ok(handles_value(Vec::new())),
        Some("value") => Ok(Value::Num(line.value)),
        Some("orientation") => Ok(Value::String(orientation.into())),
        Some("color") => Ok(Value::String(color_to_short_name(line.color))),
        Some("linewidth") => Ok(Value::Num(line.line_width as f64)),
        Some("linestyle") => Ok(Value::String(line_style_name(line.line_style).into())),
        Some("label") => Ok(Value::String(line.label.unwrap_or_default())),
        Some("labelorientation") => Ok(Value::String(line.label_orientation)),
        Some("displayname") => Ok(Value::String(line.display_name.unwrap_or_default())),
        Some("visible") => Ok(Value::String(
            if line.visible { "on" } else { "off" }.into(),
        )),
        Some(other) => Err(plotting_error(
            builtin,
            format!("{builtin}: unsupported reference line property `{other}`"),
        )),
    }
}

fn get_stairs_property(
    stairs_handle: &super::state::SimplePlotHandleState,
    property: Option<&str>,
    builtin: &'static str,
) -> BuiltinResult<Value> {
    let plot = get_simple_plot(stairs_handle, builtin)?;
    let runmat_plot::plots::figure::PlotElement::Stairs(stairs) = plot else {
        return Err(plotting_error(
            builtin,
            format!("{builtin}: invalid stairs handle"),
        ));
    };
    match property.map(canonical_property_name).as_deref() {
        None => {
            let mut st =
                child_base_struct("stairs", stairs_handle.figure, stairs_handle.axes_index);
            st.insert("XData", tensor_from_vec(stairs.x.clone()));
            st.insert("YData", tensor_from_vec(stairs.y.clone()));
            st.insert("Color", Value::String(color_to_short_name(stairs.color)));
            st.insert("LineWidth", Value::Num(stairs.line_width as f64));
            if let Some(label) = stairs.label.clone() {
                st.insert("DisplayName", Value::String(label));
            }
            Ok(Value::Struct(st))
        }
        Some("type") => Ok(Value::String("stairs".into())),
        Some("parent") => Ok(child_parent_handle(
            stairs_handle.figure,
            stairs_handle.axes_index,
        )),
        Some("children") => Ok(handles_value(Vec::new())),
        Some("xdata") => Ok(tensor_from_vec(stairs.x.clone())),
        Some("ydata") => Ok(tensor_from_vec(stairs.y.clone())),
        Some("color") => Ok(Value::String(color_to_short_name(stairs.color))),
        Some("linewidth") => Ok(Value::Num(stairs.line_width as f64)),
        Some("displayname") => Ok(Value::String(stairs.label.unwrap_or_default())),
        Some(other) => Err(plotting_error(
            builtin,
            format!("{builtin}: unsupported stairs property `{other}`"),
        )),
    }
}

fn get_scatter_property(
    scatter_handle: &super::state::SimplePlotHandleState,
    property: Option<&str>,
    builtin: &'static str,
) -> BuiltinResult<Value> {
    let plot = get_simple_plot(scatter_handle, builtin)?;
    let runmat_plot::plots::figure::PlotElement::Scatter(scatter) = plot else {
        return Err(plotting_error(
            builtin,
            format!("{builtin}: invalid scatter handle"),
        ));
    };
    let (x_data, y_data) = scatter_xy_data_for_properties(&scatter, builtin)?;
    let x_render = x_data.materialize_f64();
    let y_render = y_data.materialize_f64();
    match property.map(canonical_property_name).as_deref() {
        None => {
            let mut st =
                child_base_struct("scatter", scatter_handle.figure, scatter_handle.axes_index);
            st.insert("XData", numeric_plot_data_value(&x_data));
            st.insert("YData", numeric_plot_data_value(&y_data));
            if let Some(theta) = scatter.theta_data.clone() {
                st.insert("ThetaData", tensor_from_vec(theta));
            }
            if let Some(r) = scatter.r_data.clone() {
                st.insert("RData", tensor_from_vec(r));
            }
            st.insert(
                "Marker",
                Value::String(marker_style_name(scatter.marker_style).into()),
            );
            st.insert("SizeData", scatter_size_data_value(&scatter));
            st.insert(
                "MarkerFaceColor",
                Value::String(color_to_short_name(scatter.color)),
            );
            st.insert(
                "MarkerEdgeColor",
                Value::String(color_to_short_name(scatter.edge_color)),
            );
            st.insert("LineWidth", Value::Num(scatter.edge_thickness as f64));
            if let Some(label) = scatter.label.clone() {
                st.insert("DisplayName", Value::String(label));
            }
            Ok(Value::Struct(st))
        }
        Some("type") => Ok(Value::String("scatter".into())),
        Some("parent") => Ok(child_parent_handle(
            scatter_handle.figure,
            scatter_handle.axes_index,
        )),
        Some("children") => Ok(handles_value(Vec::new())),
        Some("xdata") => Ok(numeric_plot_data_value(&x_data)),
        Some("ydata") => Ok(numeric_plot_data_value(&y_data)),
        Some("thetadata") => Ok(tensor_from_vec(
            scatter
                .theta_data
                .clone()
                .unwrap_or_else(|| cartesian_theta(&x_render, &y_render)),
        )),
        Some("rdata") => Ok(tensor_from_vec(
            scatter
                .r_data
                .clone()
                .unwrap_or_else(|| cartesian_radius(&x_render, &y_render)),
        )),
        Some("marker") => Ok(Value::String(
            marker_style_name(scatter.marker_style).into(),
        )),
        Some("sizedata") => Ok(scatter_size_data_value(&scatter)),
        Some("markerfacecolor") => Ok(Value::String(color_to_short_name(scatter.color))),
        Some("markeredgecolor") => Ok(Value::String(color_to_short_name(scatter.edge_color))),
        Some("linewidth") => Ok(Value::Num(scatter.edge_thickness as f64)),
        Some("displayname") => Ok(Value::String(scatter.label.unwrap_or_default())),
        Some(other) => Err(plotting_error(
            builtin,
            format!("{builtin}: unsupported scatter property `{other}`"),
        )),
    }
}

fn get_bar_property(
    bar_handle: &super::state::SimplePlotHandleState,
    property: Option<&str>,
    builtin: &'static str,
) -> BuiltinResult<Value> {
    let plot = get_simple_plot(bar_handle, builtin)?;
    let runmat_plot::plots::figure::PlotElement::Bar(bar) = plot else {
        return Err(plotting_error(
            builtin,
            format!("{builtin}: invalid bar handle"),
        ));
    };
    match property.map(canonical_property_name).as_deref() {
        None => {
            let mut st = child_base_struct("bar", bar_handle.figure, bar_handle.axes_index);
            st.insert("FaceColor", Value::String(color_to_short_name(bar.color)));
            st.insert("BarWidth", Value::Num(bar.bar_width as f64));
            if let Some(label) = bar.label.clone() {
                st.insert("DisplayName", Value::String(label));
            }
            Ok(Value::Struct(st))
        }
        Some("type") => Ok(Value::String("bar".into())),
        Some("parent") => Ok(child_parent_handle(
            bar_handle.figure,
            bar_handle.axes_index,
        )),
        Some("children") => Ok(handles_value(Vec::new())),
        Some("facecolor") | Some("color") => Ok(Value::String(color_to_short_name(bar.color))),
        Some("barwidth") => Ok(Value::Num(bar.bar_width as f64)),
        Some("displayname") => Ok(Value::String(bar.label.unwrap_or_default())),
        Some(other) => Err(plotting_error(
            builtin,
            format!("{builtin}: unsupported bar property `{other}`"),
        )),
    }
}

fn get_surface_property(
    surface_handle: &super::state::SimplePlotHandleState,
    property: Option<&str>,
    builtin: &'static str,
) -> BuiltinResult<Value> {
    let plot = get_simple_plot(surface_handle, builtin)?;
    let runmat_plot::plots::figure::PlotElement::Surface(surface) = plot else {
        return Err(plotting_error(
            builtin,
            format!("{builtin}: invalid surface handle"),
        ));
    };
    match property.map(canonical_property_name).as_deref() {
        None => {
            let mut st =
                child_base_struct("surface", surface_handle.figure, surface_handle.axes_index);
            st.insert("XData", surface_x_data_value(&surface));
            st.insert("YData", surface_y_data_value(&surface));
            st.insert("ZData", surface_z_data_value(&surface));
            st.insert("FaceAlpha", Value::Num(surface.alpha as f64));
            st.insert("FaceColor", surface_face_color_value(&surface));
            st.insert("EdgeColor", surface_edge_color_value(&surface));
            st.insert(
                "Shading",
                Value::String(surface_shading_name(surface.shading_mode).into()),
            );
            st.insert(
                "Lighting",
                Value::String(surface_lighting_name(surface.lighting_enabled).into()),
            );
            st.insert("Visible", Value::Bool(surface.visible));
            st.insert(
                "DisplayName",
                Value::String(surface.label.clone().unwrap_or_default()),
            );
            Ok(Value::Struct(st))
        }
        Some("type") => Ok(Value::String("surface".into())),
        Some("parent") => Ok(child_parent_handle(
            surface_handle.figure,
            surface_handle.axes_index,
        )),
        Some("children") => Ok(handles_value(Vec::new())),
        Some("xdata") => Ok(surface_x_data_value(&surface)),
        Some("ydata") => Ok(surface_y_data_value(&surface)),
        Some("zdata") => Ok(surface_z_data_value(&surface)),
        Some("facealpha") => Ok(Value::Num(surface.alpha as f64)),
        Some("facecolor") | Some("color") => Ok(surface_face_color_value(&surface)),
        Some("edgecolor") => Ok(surface_edge_color_value(&surface)),
        Some("shading") => Ok(Value::String(
            surface_shading_name(surface.shading_mode).into(),
        )),
        Some("lighting") => Ok(Value::String(
            surface_lighting_name(surface.lighting_enabled).into(),
        )),
        Some("visible") => Ok(Value::Bool(surface.visible)),
        Some("displayname") => Ok(Value::String(surface.label.unwrap_or_default())),
        Some(other) => Err(plotting_error(
            builtin,
            format!("{builtin}: unsupported surface property `{other}`"),
        )),
    }
}

fn get_function_surface_property(
    function_surface: &super::state::FunctionSurfaceHandleState,
    property: Option<&str>,
    builtin: &'static str,
) -> BuiltinResult<Value> {
    let surface_handle = super::state::SimplePlotHandleState {
        figure: function_surface.figure,
        axes_index: function_surface.axes_index,
        plot_index: function_surface.plot_index,
    };
    let plot = get_simple_plot(&surface_handle, builtin)?;
    let runmat_plot::plots::figure::PlotElement::Surface(surface) = plot else {
        return Err(plotting_error(
            builtin,
            format!("{builtin}: invalid function surface handle"),
        ));
    };
    match property.map(canonical_property_name).as_deref() {
        None => {
            let mut st = child_base_struct(
                "functionsurface",
                function_surface.figure,
                function_surface.axes_index,
            );
            insert_function_surface_metadata(&mut st, function_surface);
            st.insert("XData", surface_x_data_value(&surface));
            st.insert("YData", surface_y_data_value(&surface));
            st.insert("ZData", surface_z_data_value(&surface));
            st.insert("FaceAlpha", Value::Num(surface.alpha as f64));
            st.insert("FaceColor", surface_face_color_value(&surface));
            st.insert("EdgeColor", surface_edge_color_value(&surface));
            st.insert(
                "Shading",
                Value::String(surface_shading_name(surface.shading_mode).into()),
            );
            st.insert(
                "Lighting",
                Value::String(surface_lighting_name(surface.lighting_enabled).into()),
            );
            st.insert("Visible", Value::Bool(surface.visible));
            st.insert(
                "DisplayName",
                Value::String(surface.label.clone().unwrap_or_default()),
            );
            Ok(Value::Struct(st))
        }
        Some("type") => Ok(Value::String("functionsurface".into())),
        Some("parent") => Ok(child_parent_handle(
            function_surface.figure,
            function_surface.axes_index,
        )),
        Some("children") => Ok(handles_value(Vec::new())),
        Some("function") => match &function_surface.function {
            super::state::FunctionSurfaceFunctionState::Explicit(function) => {
                Ok(function_surface_function_value(function))
            }
            super::state::FunctionSurfaceFunctionState::Parametric { .. } => Err(plotting_error(
                builtin,
                format!(
                    "{builtin}: parametric function surfaces use XFunction/YFunction/ZFunction"
                ),
            )),
        },
        Some("xfunction") => match &function_surface.function {
            super::state::FunctionSurfaceFunctionState::Parametric { x, .. } => {
                Ok(function_surface_function_value(x))
            }
            super::state::FunctionSurfaceFunctionState::Explicit(_) => Err(plotting_error(
                builtin,
                format!("{builtin}: non-parametric function surfaces use Function"),
            )),
        },
        Some("yfunction") => match &function_surface.function {
            super::state::FunctionSurfaceFunctionState::Parametric { y, .. } => {
                Ok(function_surface_function_value(y))
            }
            super::state::FunctionSurfaceFunctionState::Explicit(_) => Err(plotting_error(
                builtin,
                format!("{builtin}: non-parametric function surfaces use Function"),
            )),
        },
        Some("zfunction") => match &function_surface.function {
            super::state::FunctionSurfaceFunctionState::Parametric { z, .. } => {
                Ok(function_surface_function_value(z))
            }
            super::state::FunctionSurfaceFunctionState::Explicit(_) => Err(plotting_error(
                builtin,
                format!("{builtin}: non-parametric function surfaces use Function"),
            )),
        },
        Some("meshdensity") => Ok(Value::Num(function_surface.mesh_density as f64)),
        Some("xrange") => Ok(tensor_from_vec(vec![
            function_surface.x_range.0,
            function_surface.x_range.1,
        ])),
        Some("yrange") => Ok(tensor_from_vec(vec![
            function_surface.y_range.0,
            function_surface.y_range.1,
        ])),
        Some("xdata") => Ok(tensor_from_vec(surface.x_data.clone())),
        Some("ydata") => Ok(tensor_from_vec(surface.y_data.clone())),
        Some("zdata") => Ok(surface
            .z_data
            .clone()
            .map(tensor_from_matrix)
            .unwrap_or_else(|| tensor_from_vec(Vec::new()))),
        Some("facealpha") => Ok(Value::Num(surface.alpha as f64)),
        Some("displayname") => Ok(Value::String(surface.label.unwrap_or_default())),
        Some(other) => Err(plotting_error(
            builtin,
            format!("{builtin}: unsupported functionsurface property `{other}`"),
        )),
    }
}

fn insert_function_surface_metadata(
    st: &mut StructValue,
    function_surface: &super::state::FunctionSurfaceHandleState,
) {
    st.insert(
        "MeshDensity",
        Value::Num(function_surface.mesh_density as f64),
    );
    st.insert(
        "XRange",
        tensor_from_vec(vec![function_surface.x_range.0, function_surface.x_range.1]),
    );
    st.insert(
        "YRange",
        tensor_from_vec(vec![function_surface.y_range.0, function_surface.y_range.1]),
    );
    match &function_surface.function {
        super::state::FunctionSurfaceFunctionState::Explicit(function) => {
            st.insert("Function", function_surface_function_value(function));
        }
        super::state::FunctionSurfaceFunctionState::Parametric { x, y, z } => {
            st.insert("XFunction", function_surface_function_value(x));
            st.insert("YFunction", function_surface_function_value(y));
            st.insert("ZFunction", function_surface_function_value(z));
        }
    }
}

fn function_surface_function_value(function: &super::state::FunctionSurfaceFunctionRef) -> Value {
    match function {
        super::state::FunctionSurfaceFunctionRef::FunctionHandle(name) => {
            Value::FunctionHandle(name.clone())
        }
        super::state::FunctionSurfaceFunctionRef::ExternalFunctionHandle(name) => {
            Value::ExternalFunctionHandle(name.clone())
        }
        super::state::FunctionSurfaceFunctionRef::MethodFunctionHandle(name) => {
            Value::MethodFunctionHandle(name.clone())
        }
        super::state::FunctionSurfaceFunctionRef::BoundFunctionHandle { name, function } => {
            Value::BoundFunctionHandle {
                name: name.clone(),
                function: *function,
            }
        }
        super::state::FunctionSurfaceFunctionRef::ClosureSummary {
            function_name,
            bound_function,
        } => match bound_function {
            Some(function) => Value::BoundFunctionHandle {
                name: function_name.clone(),
                function: *function,
            },
            None => Value::String(function_name.clone()),
        },
    }
}

fn get_function_contour_property(
    function_contour: &super::state::FunctionContourHandleState,
    property: Option<&str>,
    builtin: &'static str,
) -> BuiltinResult<Value> {
    let contour_handle = super::state::SimplePlotHandleState {
        figure: function_contour.figure,
        axes_index: function_contour.axes_index,
        plot_index: function_contour.plot_index,
    };
    let plot = get_simple_plot(&contour_handle, builtin)?;
    let runmat_plot::plots::figure::PlotElement::Contour(contour) = plot else {
        return Err(plotting_error(
            builtin,
            format!("{builtin}: invalid function contour handle"),
        ));
    };
    match property.map(canonical_property_name).as_deref() {
        None => {
            let mut st = child_base_struct(
                "functioncontour",
                function_contour.figure,
                function_contour.axes_index,
            );
            st.insert(
                "Function",
                function_surface_function_value(&function_contour.function),
            );
            st.insert(
                "MeshDensity",
                Value::Num(function_contour.mesh_density as f64),
            );
            st.insert(
                "XRange",
                tensor_from_vec(vec![function_contour.x_range.0, function_contour.x_range.1]),
            );
            st.insert(
                "YRange",
                tensor_from_vec(vec![function_contour.y_range.0, function_contour.y_range.1]),
            );
            st.insert("Fill", Value::Bool(function_contour.fill));
            st.insert("LineWidth", Value::Num(contour.line_width as f64));
            st.insert(
                "DisplayName",
                Value::String(contour.label.clone().unwrap_or_default()),
            );
            Ok(Value::Struct(st))
        }
        Some("type") => Ok(Value::String("functioncontour".into())),
        Some("parent") => Ok(child_parent_handle(
            function_contour.figure,
            function_contour.axes_index,
        )),
        Some("children") => Ok(handles_value(Vec::new())),
        Some("function") => Ok(function_surface_function_value(&function_contour.function)),
        Some("meshdensity") => Ok(Value::Num(function_contour.mesh_density as f64)),
        Some("xrange") => Ok(tensor_from_vec(vec![
            function_contour.x_range.0,
            function_contour.x_range.1,
        ])),
        Some("yrange") => Ok(tensor_from_vec(vec![
            function_contour.y_range.0,
            function_contour.y_range.1,
        ])),
        Some("fill") => Ok(Value::Bool(function_contour.fill)),
        Some("linewidth") => Ok(Value::Num(contour.line_width as f64)),
        Some("displayname") => Ok(Value::String(contour.label.unwrap_or_default())),
        Some("zdata") => Ok(Value::Num(contour.base_z as f64)),
        Some(other) => Err(plotting_error(
            builtin,
            format!("{builtin}: unsupported functioncontour property `{other}`"),
        )),
    }
}

fn get_patch_property(
    patch_handle: &super::state::SimplePlotHandleState,
    property: Option<&str>,
    builtin: &'static str,
) -> BuiltinResult<Value> {
    let plot = get_simple_plot(patch_handle, builtin)?;
    let runmat_plot::plots::figure::PlotElement::Patch(patch) = plot else {
        return Err(plotting_error(
            builtin,
            format!("{builtin}: invalid patch handle"),
        ));
    };
    let (source_x, source_y, source_z, source_c) = patch.source_data();
    match property.map(canonical_property_name).as_deref() {
        None => {
            let mut st = child_base_struct("patch", patch_handle.figure, patch_handle.axes_index);
            st.insert("Faces", faces_tensor(patch.faces()));
            st.insert("Vertices", vertices_tensor(patch.vertices()));
            st.insert("XData", patch_source_or_vertices(source_x, &patch, 0));
            st.insert("YData", patch_source_or_vertices(source_y, &patch, 1));
            st.insert("ZData", patch_source_or_vertices(source_z, &patch, 2));
            if let Some(c_data) = source_c {
                st.insert("CData", patch_data_value(c_data));
            }
            st.insert(
                "FaceColor",
                patch_color_property(patch.face_color_mode(), patch.face_color()),
            );
            st.insert(
                "EdgeColor",
                patch_edge_color_property(patch.edge_color_mode(), patch.edge_color()),
            );
            st.insert("FaceAlpha", Value::Num(patch.face_alpha() as f64));
            st.insert("EdgeAlpha", Value::Num(patch.edge_alpha() as f64));
            st.insert("LineWidth", Value::Num(patch.line_width() as f64));
            st.insert("Visible", Value::Bool(patch.is_visible()));
            if let Some(label) = patch.label() {
                st.insert("DisplayName", Value::String(label.to_string()));
            }
            Ok(Value::Struct(st))
        }
        Some("type") => Ok(Value::String("patch".into())),
        Some("parent") => Ok(child_parent_handle(
            patch_handle.figure,
            patch_handle.axes_index,
        )),
        Some("children") => Ok(handles_value(Vec::new())),
        Some("faces") => Ok(faces_tensor(patch.faces())),
        Some("vertices") => Ok(vertices_tensor(patch.vertices())),
        Some("xdata") => Ok(patch_source_or_vertices(source_x, &patch, 0)),
        Some("ydata") => Ok(patch_source_or_vertices(source_y, &patch, 1)),
        Some("zdata") => Ok(patch_source_or_vertices(source_z, &patch, 2)),
        Some("cdata") | Some("colordata") => source_c
            .map(patch_data_value)
            .ok_or_else(|| plotting_error(builtin, format!("{builtin}: patch has no CData"))),
        Some("facecolor") | Some("color") => Ok(patch_color_property(
            patch.face_color_mode(),
            patch.face_color(),
        )),
        Some("edgecolor") => Ok(patch_edge_color_property(
            patch.edge_color_mode(),
            patch.edge_color(),
        )),
        Some("facealpha") => Ok(Value::Num(patch.face_alpha() as f64)),
        Some("edgealpha") => Ok(Value::Num(patch.edge_alpha() as f64)),
        Some("linewidth") => Ok(Value::Num(patch.line_width() as f64)),
        Some("displayname") => Ok(Value::String(patch.label().unwrap_or_default().to_string())),
        Some("visible") => Ok(Value::Bool(patch.is_visible())),
        Some(other) => Err(plotting_error(
            builtin,
            format!("{builtin}: unsupported patch property `{other}`"),
        )),
    }
}

fn patch_source_or_vertices(
    source: Option<&PatchData>,
    patch: &runmat_plot::plots::PatchPlot,
    component: usize,
) -> Value {
    source.map(patch_data_value).unwrap_or_else(|| {
        tensor_from_vec(
            patch
                .vertices()
                .iter()
                .map(|point| match component {
                    0 => point.x as f64,
                    1 => point.y as f64,
                    _ => point.z as f64,
                })
                .collect(),
        )
    })
}

fn patch_data_value(data: &PatchData) -> Value {
    Value::Tensor(
        Tensor::from_numeric_storage(data.storage.clone(), data.shape.clone())
            .expect("patch source shape matches storage"),
    )
}

fn get_line3_property(
    line_handle: &super::state::SimplePlotHandleState,
    property: Option<&str>,
    builtin: &'static str,
) -> BuiltinResult<Value> {
    let plot = get_simple_plot(line_handle, builtin)?;
    let runmat_plot::plots::figure::PlotElement::Line3(line) = plot else {
        return Err(plotting_error(
            builtin,
            format!("{builtin}: invalid plot3 handle"),
        ));
    };
    match property.map(canonical_property_name).as_deref() {
        None => {
            let mut st = child_base_struct("line", line_handle.figure, line_handle.axes_index);
            let (x_data, y_data, z_data) = line_xyz_data_for_properties(&line, builtin)?;
            st.insert("XData", numeric_plot_data_value(&x_data));
            st.insert("YData", numeric_plot_data_value(&y_data));
            st.insert("ZData", numeric_plot_data_value(&z_data));
            st.insert("Color", Value::String(color_to_short_name(line.color)));
            st.insert("LineWidth", Value::Num(line.line_width as f64));
            st.insert(
                "LineStyle",
                Value::String(line_style_name(line.line_style).into()),
            );
            st.insert(
                "DisplayName",
                Value::String(line.label.clone().unwrap_or_default()),
            );
            st.insert(
                "Visible",
                Value::String(if line.visible { "on" } else { "off" }.into()),
            );
            Ok(Value::Struct(st))
        }
        Some("type") => Ok(Value::String("line".into())),
        Some("parent") => Ok(child_parent_handle(
            line_handle.figure,
            line_handle.axes_index,
        )),
        Some("children") => Ok(handles_value(Vec::new())),
        Some("xdata") => {
            let (x, _, _) = line_xyz_data_for_properties(&line, builtin)?;
            Ok(numeric_plot_data_value(&x))
        }
        Some("ydata") => {
            let (_, y, _) = line_xyz_data_for_properties(&line, builtin)?;
            Ok(numeric_plot_data_value(&y))
        }
        Some("zdata") => {
            let (_, _, z) = line_xyz_data_for_properties(&line, builtin)?;
            Ok(numeric_plot_data_value(&z))
        }
        Some("color") => Ok(Value::String(color_to_short_name(line.color))),
        Some("linewidth") => Ok(Value::Num(line.line_width as f64)),
        Some("linestyle") => Ok(Value::String(line_style_name(line.line_style).into())),
        Some("displayname") => Ok(Value::String(line.label.unwrap_or_default())),
        Some("visible") => Ok(Value::String(
            if line.visible { "on" } else { "off" }.into(),
        )),
        Some(other) => Err(plotting_error(
            builtin,
            format!("{builtin}: unsupported plot3 property `{other}`"),
        )),
    }
}

fn get_scatter3_property(
    scatter_handle: &super::state::SimplePlotHandleState,
    property: Option<&str>,
    builtin: &'static str,
) -> BuiltinResult<Value> {
    let plot = get_simple_plot(scatter_handle, builtin)?;
    let runmat_plot::plots::figure::PlotElement::Scatter3(scatter) = plot else {
        return Err(plotting_error(
            builtin,
            format!("{builtin}: invalid scatter3 handle"),
        ));
    };
    let (x, y, z): (Vec<f64>, Vec<f64>, Vec<f64>) = scatter
        .points
        .iter()
        .map(|p| (p.x as f64, p.y as f64, p.z as f64))
        .unzip_n_vec();
    match property.map(canonical_property_name).as_deref() {
        None => {
            let mut st =
                child_base_struct("scatter", scatter_handle.figure, scatter_handle.axes_index);
            st.insert("XData", tensor_from_vec(x));
            st.insert("YData", tensor_from_vec(y));
            st.insert("ZData", tensor_from_vec(z));
            st.insert(
                "Marker",
                Value::String(marker_style_name(scatter.marker_style).into()),
            );
            st.insert(
                "SizeData",
                Value::Num(marker_diameter_px_to_area_points2(scatter.point_size)),
            );
            st.insert(
                "MarkerFaceColor",
                Value::String(color_to_short_name(
                    scatter.colors.first().copied().unwrap_or_default(),
                )),
            );
            st.insert(
                "MarkerEdgeColor",
                Value::String(color_to_short_name(scatter.edge_color)),
            );
            st.insert("LineWidth", Value::Num(scatter.edge_thickness as f64));
            if let Some(label) = scatter.label.clone() {
                st.insert("DisplayName", Value::String(label));
            }
            Ok(Value::Struct(st))
        }
        Some("type") => Ok(Value::String("scatter".into())),
        Some("parent") => Ok(child_parent_handle(
            scatter_handle.figure,
            scatter_handle.axes_index,
        )),
        Some("children") => Ok(handles_value(Vec::new())),
        Some("marker") => Ok(Value::String(
            marker_style_name(scatter.marker_style).into(),
        )),
        Some("sizedata") => Ok(Value::Num(marker_diameter_px_to_area_points2(
            scatter.point_size,
        ))),
        Some("markerfacecolor") => Ok(Value::String(color_to_short_name(
            scatter.colors.first().copied().unwrap_or_default(),
        ))),
        Some("markeredgecolor") => Ok(Value::String(color_to_short_name(scatter.edge_color))),
        Some("linewidth") => Ok(Value::Num(scatter.edge_thickness as f64)),
        Some("displayname") => Ok(Value::String(scatter.label.unwrap_or_default())),
        Some(other) => Err(plotting_error(
            builtin,
            format!("{builtin}: unsupported scatter3 property `{other}`"),
        )),
    }
}

fn get_pie_property(
    pie_handle: &super::state::SimplePlotHandleState,
    property: Option<&str>,
    builtin: &'static str,
) -> BuiltinResult<Value> {
    let plot = get_simple_plot(pie_handle, builtin)?;
    let runmat_plot::plots::figure::PlotElement::Pie(pie) = plot else {
        return Err(plotting_error(
            builtin,
            format!("{builtin}: invalid pie handle"),
        ));
    };
    match property.map(canonical_property_name).as_deref() {
        None => {
            let mut st = child_base_struct("pie", pie_handle.figure, pie_handle.axes_index);
            if let Some(label) = pie.label.clone() {
                st.insert("DisplayName", Value::String(label));
            }
            Ok(Value::Struct(st))
        }
        Some("type") => Ok(Value::String("pie".into())),
        Some("parent") => Ok(child_parent_handle(
            pie_handle.figure,
            pie_handle.axes_index,
        )),
        Some("children") => Ok(handles_value(Vec::new())),
        Some("displayname") => Ok(Value::String(pie.label.unwrap_or_default())),
        Some(other) => Err(plotting_error(
            builtin,
            format!("{builtin}: unsupported pie property `{other}`"),
        )),
    }
}

fn get_contour_property(
    contour_handle: &super::state::SimplePlotHandleState,
    property: Option<&str>,
    builtin: &'static str,
) -> BuiltinResult<Value> {
    let plot = get_simple_plot(contour_handle, builtin)?;
    let runmat_plot::plots::figure::PlotElement::Contour(contour) = plot else {
        return Err(plotting_error(
            builtin,
            format!("{builtin}: invalid contour handle"),
        ));
    };
    match property.map(canonical_property_name).as_deref() {
        None => {
            let mut st =
                child_base_struct("contour", contour_handle.figure, contour_handle.axes_index);
            st.insert("ZData", Value::Num(contour.base_z as f64));
            if let Some(label) = contour.label.clone() {
                st.insert("DisplayName", Value::String(label));
            }
            Ok(Value::Struct(st))
        }
        Some("type") => Ok(Value::String("contour".into())),
        Some("parent") => Ok(child_parent_handle(
            contour_handle.figure,
            contour_handle.axes_index,
        )),
        Some("children") => Ok(handles_value(Vec::new())),
        Some("zdata") => Ok(Value::Num(contour.base_z as f64)),
        Some("displayname") => Ok(Value::String(contour.label.unwrap_or_default())),
        Some(other) => Err(plotting_error(
            builtin,
            format!("{builtin}: unsupported contour property `{other}`"),
        )),
    }
}

fn get_contour_fill_property(
    fill_handle: &super::state::SimplePlotHandleState,
    property: Option<&str>,
    builtin: &'static str,
) -> BuiltinResult<Value> {
    let plot = get_simple_plot(fill_handle, builtin)?;
    let runmat_plot::plots::figure::PlotElement::ContourFill(fill) = plot else {
        return Err(plotting_error(
            builtin,
            format!("{builtin}: invalid contourf handle"),
        ));
    };
    match property.map(canonical_property_name).as_deref() {
        None => {
            let mut st = child_base_struct("contour", fill_handle.figure, fill_handle.axes_index);
            if let Some(label) = fill.label.clone() {
                st.insert("DisplayName", Value::String(label));
            }
            Ok(Value::Struct(st))
        }
        Some("type") => Ok(Value::String("contour".into())),
        Some("parent") => Ok(child_parent_handle(
            fill_handle.figure,
            fill_handle.axes_index,
        )),
        Some("children") => Ok(handles_value(Vec::new())),
        Some("displayname") => Ok(Value::String(fill.label.unwrap_or_default())),
        Some(other) => Err(plotting_error(
            builtin,
            format!("{builtin}: unsupported contourf property `{other}`"),
        )),
    }
}

fn get_stem_property(
    stem_handle: &super::state::StemHandleState,
    property: Option<&str>,
    builtin: &'static str,
) -> BuiltinResult<Value> {
    let figure = super::state::clone_figure(stem_handle.figure)
        .ok_or_else(|| plotting_error(builtin, format!("{builtin}: invalid stem figure")))?;
    let plot = figure
        .plots()
        .nth(stem_handle.plot_index)
        .ok_or_else(|| plotting_error(builtin, format!("{builtin}: invalid stem handle")))?;
    let runmat_plot::plots::figure::PlotElement::Stem(stem) = plot else {
        return Err(plotting_error(
            builtin,
            format!("{builtin}: invalid stem handle"),
        ));
    };
    match property.map(canonical_property_name).as_deref() {
        None => {
            let mut st = StructValue::new();
            st.insert("Type", Value::String("stem".into()));
            st.insert(
                "Parent",
                Value::Num(super::state::encode_axes_handle(
                    stem_handle.figure,
                    stem_handle.axes_index,
                )),
            );
            st.insert("Children", handles_value(Vec::new()));
            st.insert("BaseValue", Value::Num(stem.baseline));
            st.insert("BaseLine", Value::Bool(stem.baseline_visible));
            st.insert("LineWidth", Value::Num(stem.line_width as f64));
            st.insert(
                "LineStyle",
                Value::String(line_style_name(stem.line_style).into()),
            );
            st.insert("Color", Value::String(color_to_short_name(stem.color)));
            if let Some(marker) = &stem.marker {
                st.insert(
                    "Marker",
                    Value::String(marker_style_name(marker.kind).into()),
                );
                st.insert("MarkerSize", Value::Num(marker.size as f64));
                st.insert(
                    "MarkerFaceColor",
                    Value::String(color_to_short_name(marker.face_color)),
                );
                st.insert(
                    "MarkerEdgeColor",
                    Value::String(color_to_short_name(marker.edge_color)),
                );
                st.insert("Filled", Value::Bool(marker.filled));
            }
            Ok(Value::Struct(st))
        }
        Some("type") => Ok(Value::String("stem".into())),
        Some("parent") => Ok(Value::Num(super::state::encode_axes_handle(
            stem_handle.figure,
            stem_handle.axes_index,
        ))),
        Some("children") => Ok(handles_value(Vec::new())),
        Some("basevalue") => Ok(Value::Num(stem.baseline)),
        Some("baseline") => Ok(Value::Bool(stem.baseline_visible)),
        Some("linewidth") => Ok(Value::Num(stem.line_width as f64)),
        Some("linestyle") => Ok(Value::String(line_style_name(stem.line_style).into())),
        Some("color") => Ok(Value::String(color_to_short_name(stem.color))),
        Some("marker") => Ok(Value::String(
            stem.marker
                .as_ref()
                .map(|m| marker_style_name(m.kind).to_string())
                .unwrap_or("none".into()),
        )),
        Some("markersize") => Ok(Value::Num(
            stem.marker.as_ref().map(|m| m.size as f64).unwrap_or(0.0),
        )),
        Some("markerfacecolor") => Ok(Value::String(
            stem.marker
                .as_ref()
                .map(|m| color_to_short_name(m.face_color))
                .unwrap_or("none".into()),
        )),
        Some("markeredgecolor") => Ok(Value::String(
            stem.marker
                .as_ref()
                .map(|m| color_to_short_name(m.edge_color))
                .unwrap_or("none".into()),
        )),
        Some("filled") => Ok(Value::Bool(
            stem.marker.as_ref().map(|m| m.filled).unwrap_or(false),
        )),
        Some(other) => Err(plotting_error(
            builtin,
            format!("{builtin}: unsupported stem property `{other}`"),
        )),
    }
}

fn get_errorbar_property(
    error_handle: &super::state::ErrorBarHandleState,
    property: Option<&str>,
    builtin: &'static str,
) -> BuiltinResult<Value> {
    let figure = super::state::clone_figure(error_handle.figure)
        .ok_or_else(|| plotting_error(builtin, format!("{builtin}: invalid errorbar figure")))?;
    let plot = figure
        .plots()
        .nth(error_handle.plot_index)
        .ok_or_else(|| plotting_error(builtin, format!("{builtin}: invalid errorbar handle")))?;
    let runmat_plot::plots::figure::PlotElement::ErrorBar(errorbar) = plot else {
        return Err(plotting_error(
            builtin,
            format!("{builtin}: invalid errorbar handle"),
        ));
    };
    match property.map(canonical_property_name).as_deref() {
        None => {
            let mut st = StructValue::new();
            st.insert("Type", Value::String("errorbar".into()));
            st.insert(
                "Parent",
                Value::Num(super::state::encode_axes_handle(
                    error_handle.figure,
                    error_handle.axes_index,
                )),
            );
            st.insert("Children", handles_value(Vec::new()));
            st.insert("LineWidth", Value::Num(errorbar.line_width as f64));
            st.insert(
                "LineStyle",
                Value::String(line_style_name(errorbar.line_style).into()),
            );
            st.insert("Color", Value::String(color_to_short_name(errorbar.color)));
            st.insert("CapSize", Value::Num(errorbar.cap_size as f64));
            if let Some(marker) = &errorbar.marker {
                st.insert(
                    "Marker",
                    Value::String(marker_style_name(marker.kind).into()),
                );
                st.insert("MarkerSize", Value::Num(marker.size as f64));
            }
            Ok(Value::Struct(st))
        }
        Some("type") => Ok(Value::String("errorbar".into())),
        Some("parent") => Ok(Value::Num(super::state::encode_axes_handle(
            error_handle.figure,
            error_handle.axes_index,
        ))),
        Some("children") => Ok(handles_value(Vec::new())),
        Some("linewidth") => Ok(Value::Num(errorbar.line_width as f64)),
        Some("linestyle") => Ok(Value::String(line_style_name(errorbar.line_style).into())),
        Some("color") => Ok(Value::String(color_to_short_name(errorbar.color))),
        Some("capsize") => Ok(Value::Num(errorbar.cap_size as f64)),
        Some("marker") => Ok(Value::String(
            errorbar
                .marker
                .as_ref()
                .map(|m| marker_style_name(m.kind).to_string())
                .unwrap_or("none".into()),
        )),
        Some("markersize") => Ok(Value::Num(
            errorbar
                .marker
                .as_ref()
                .map(|m| m.size as f64)
                .unwrap_or(0.0),
        )),
        Some(other) => Err(plotting_error(
            builtin,
            format!("{builtin}: unsupported errorbar property `{other}`"),
        )),
    }
}

fn get_quiver_property(
    quiver_handle: &super::state::QuiverHandleState,
    property: Option<&str>,
    builtin: &'static str,
) -> BuiltinResult<Value> {
    let figure = super::state::clone_figure(quiver_handle.figure)
        .ok_or_else(|| plotting_error(builtin, format!("{builtin}: invalid quiver figure")))?;
    let plot = figure
        .plots()
        .nth(quiver_handle.plot_index)
        .ok_or_else(|| plotting_error(builtin, format!("{builtin}: invalid quiver handle")))?;
    let runmat_plot::plots::figure::PlotElement::Quiver(quiver) = plot else {
        return Err(plotting_error(
            builtin,
            format!("{builtin}: invalid quiver handle"),
        ));
    };
    let has_cpu_data = quiver.has_cpu_vector_data();
    match property.map(canonical_property_name).as_deref() {
        None => {
            let mut st = StructValue::new();
            st.insert("Type", Value::String("quiver".into()));
            st.insert(
                "Parent",
                Value::Num(super::state::encode_axes_handle(
                    quiver_handle.figure,
                    quiver_handle.axes_index,
                )),
            );
            st.insert("Children", handles_value(Vec::new()));
            st.insert("Color", Value::String(color_to_short_name(quiver.color)));
            st.insert("LineWidth", Value::Num(quiver.line_width as f64));
            st.insert("AutoScaleFactor", Value::Num(quiver.scale as f64));
            st.insert("MaxHeadSize", Value::Num(quiver.head_size as f64));
            if has_cpu_data {
                st.insert("XData", tensor_from_vec(quiver.x.clone()));
                st.insert("YData", tensor_from_vec(quiver.y.clone()));
                if quiver_handle.is_3d {
                    st.insert(
                        "ZData",
                        tensor_from_vec(quiver.z.clone().unwrap_or_default()),
                    );
                    st.insert(
                        "WData",
                        tensor_from_vec(quiver.w.clone().unwrap_or_default()),
                    );
                }
                st.insert("UData", tensor_from_vec(quiver.u.clone()));
                st.insert("VData", tensor_from_vec(quiver.v.clone()));
            }
            Ok(Value::Struct(st))
        }
        Some("type") => Ok(Value::String("quiver".into())),
        Some("parent") => Ok(Value::Num(super::state::encode_axes_handle(
            quiver_handle.figure,
            quiver_handle.axes_index,
        ))),
        Some("children") => Ok(handles_value(Vec::new())),
        Some("color") => Ok(Value::String(color_to_short_name(quiver.color))),
        Some("linewidth") => Ok(Value::Num(quiver.line_width as f64)),
        Some("autoscalefactor") => Ok(Value::Num(quiver.scale as f64)),
        Some("maxheadsize") => Ok(Value::Num(quiver.head_size as f64)),
        Some("xdata") if has_cpu_data => Ok(tensor_from_vec(quiver.x.clone())),
        Some("ydata") if has_cpu_data => Ok(tensor_from_vec(quiver.y.clone())),
        Some("zdata") if quiver_handle.is_3d => {
            if has_cpu_data {
                Ok(tensor_from_vec(quiver.z.clone().unwrap_or_default()))
            } else {
                Err(quiver_data_unavailable_error(builtin))
            }
        }
        Some("udata") if has_cpu_data => Ok(tensor_from_vec(quiver.u.clone())),
        Some("vdata") if has_cpu_data => Ok(tensor_from_vec(quiver.v.clone())),
        Some("wdata") if quiver_handle.is_3d => {
            if has_cpu_data {
                Ok(tensor_from_vec(quiver.w.clone().unwrap_or_default()))
            } else {
                Err(quiver_data_unavailable_error(builtin))
            }
        }
        Some("xdata") | Some("ydata") | Some("udata") | Some("vdata") => {
            Err(quiver_data_unavailable_error(builtin))
        }
        Some(other) => Err(plotting_error(
            builtin,
            format!("{builtin}: unsupported quiver property `{other}`"),
        )),
    }
}

fn quiver_data_unavailable_error(builtin: &'static str) -> crate::RuntimeError {
    plotting_error(
        builtin,
        format!("{builtin}: quiver data properties are unavailable for GPU-backed quiver handles"),
    )
}

fn get_image_property(
    image_handle: &super::state::ImageHandleState,
    property: Option<&str>,
    builtin: &'static str,
) -> BuiltinResult<Value> {
    let figure = super::state::clone_figure(image_handle.figure)
        .ok_or_else(|| plotting_error(builtin, format!("{builtin}: invalid image figure")))?;
    let plot = figure
        .plots()
        .nth(image_handle.plot_index)
        .ok_or_else(|| plotting_error(builtin, format!("{builtin}: invalid image handle")))?;
    let runmat_plot::plots::figure::PlotElement::Surface(surface) = plot else {
        return Err(plotting_error(
            builtin,
            format!("{builtin}: invalid image handle"),
        ));
    };
    if !surface.image_mode {
        return Err(plotting_error(
            builtin,
            format!("{builtin}: handle does not reference an image plot"),
        ));
    }
    match property.map(canonical_property_name).as_deref() {
        None => {
            let mut st = StructValue::new();
            st.insert("Type", Value::String("image".into()));
            st.insert(
                "Parent",
                Value::Num(super::state::encode_axes_handle(
                    image_handle.figure,
                    image_handle.axes_index,
                )),
            );
            st.insert("Children", handles_value(Vec::new()));
            st.insert("XData", tensor_from_vec(surface.x_data.clone()));
            st.insert("YData", tensor_from_vec(surface.y_data.clone()));
            if let Some(c_data) = &image_handle.c_data {
                st.insert("CData", Value::Tensor(c_data.clone()));
            }
            st.insert(
                "CDataMapping",
                Value::String(image_handle.c_data_mapping.clone()),
            );
            Ok(Value::Struct(st))
        }
        Some("type") => Ok(Value::String("image".into())),
        Some("parent") => Ok(Value::Num(super::state::encode_axes_handle(
            image_handle.figure,
            image_handle.axes_index,
        ))),
        Some("children") => Ok(handles_value(Vec::new())),
        Some("xdata") => Ok(tensor_from_vec(surface.x_data.clone())),
        Some("ydata") => Ok(tensor_from_vec(surface.y_data.clone())),
        Some("colordata") => Ok(Value::Tensor(image_handle.c_data.clone().unwrap_or_else(
            || Tensor::new(Vec::new(), vec![0, 0]).expect("valid empty tensor"),
        ))),
        Some("cdatamapping") => Ok(Value::String(image_handle.c_data_mapping.clone())),
        Some(other) => Err(plotting_error(
            builtin,
            format!("{builtin}: unsupported image property `{other}`"),
        )),
    }
}

fn get_heatmap_property(
    heatmap_handle: &super::state::HeatmapHandleState,
    property: Option<&str>,
    builtin: &'static str,
) -> BuiltinResult<Value> {
    let figure = super::state::clone_figure(heatmap_handle.figure)
        .ok_or_else(|| plotting_error(builtin, format!("{builtin}: invalid heatmap figure")))?;
    let plot = figure
        .plots()
        .nth(heatmap_handle.plot_index)
        .ok_or_else(|| plotting_error(builtin, format!("{builtin}: invalid heatmap handle")))?;
    let runmat_plot::plots::figure::PlotElement::Surface(surface) = plot else {
        return Err(plotting_error(
            builtin,
            format!("{builtin}: invalid heatmap handle"),
        ));
    };
    if !surface.image_mode {
        return Err(plotting_error(
            builtin,
            format!("{builtin}: handle does not reference a heatmap plot"),
        ));
    }
    let meta = figure
        .axes_metadata(heatmap_handle.axes_index)
        .ok_or_else(|| plotting_error(builtin, format!("{builtin}: invalid heatmap axes")))?;
    match property.map(canonical_property_name).as_deref() {
        None => {
            let mut st = StructValue::new();
            st.insert("Type", Value::String("heatmap".into()));
            st.insert(
                "Parent",
                Value::Num(super::state::encode_axes_handle(
                    heatmap_handle.figure,
                    heatmap_handle.axes_index,
                )),
            );
            st.insert("Children", handles_value(Vec::new()));
            st.insert(
                "Title",
                Value::String(meta.title.clone().unwrap_or_default()),
            );
            st.insert(
                "XLabel",
                Value::String(meta.x_label.clone().unwrap_or_default()),
            );
            st.insert(
                "YLabel",
                Value::String(meta.y_label.clone().unwrap_or_default()),
            );
            st.insert(
                "XDisplayLabels",
                string_array_from_vec(heatmap_handle.x_labels.clone())?,
            );
            st.insert(
                "YDisplayLabels",
                string_array_from_vec(heatmap_handle.y_labels.clone())?,
            );
            st.insert(
                "ColorData",
                Value::Tensor(heatmap_handle.color_data.clone()),
            );
            st.insert("ColorbarVisible", Value::Bool(meta.colorbar_enabled));
            st.insert("GridVisible", Value::Bool(meta.grid_enabled));
            st.insert(
                "FontSize",
                Value::Num(meta.axes_style.font_size.unwrap_or(10.0) as f64),
            );
            st.insert(
                "Colormap",
                Value::String(format!("{:?}", meta.colormap).to_ascii_lowercase()),
            );
            st.insert(
                "ColorLimits",
                heatmap_handle
                    .color_limits
                    .clone()
                    .map(Value::Tensor)
                    .unwrap_or_else(|| {
                        crate::builtins::plotting::op_common::limits::limit_value(meta.color_limits)
                    }),
            );
            Ok(Value::Struct(st))
        }
        Some("type") => Ok(Value::String("heatmap".into())),
        Some("parent") => Ok(Value::Num(super::state::encode_axes_handle(
            heatmap_handle.figure,
            heatmap_handle.axes_index,
        ))),
        Some("children") => Ok(handles_value(Vec::new())),
        Some("title") => Ok(Value::String(meta.title.clone().unwrap_or_default())),
        Some("xlabel") => Ok(Value::String(meta.x_label.clone().unwrap_or_default())),
        Some("ylabel") => Ok(Value::String(meta.y_label.clone().unwrap_or_default())),
        Some("xdisplaylabels") => string_array_from_vec(heatmap_handle.x_labels.clone()),
        Some("ydisplaylabels") => string_array_from_vec(heatmap_handle.y_labels.clone()),
        Some("colordata") => Ok(Value::Tensor(heatmap_handle.color_data.clone())),
        Some("colorbarvisible") | Some("colorbar") => Ok(Value::Bool(meta.colorbar_enabled)),
        Some("gridvisible") => Ok(Value::Bool(meta.grid_enabled)),
        Some("fontsize") => Ok(Value::Num(meta.axes_style.font_size.unwrap_or(10.0) as f64)),
        Some("colormap") => Ok(Value::String(
            format!("{:?}", meta.colormap).to_ascii_lowercase(),
        )),
        Some("colorlimits") => Ok(heatmap_handle
            .color_limits
            .clone()
            .map(Value::Tensor)
            .unwrap_or_else(|| {
                crate::builtins::plotting::op_common::limits::limit_value(meta.color_limits)
            })),
        Some(other) => Err(plotting_error(
            builtin,
            format!("{builtin}: unsupported heatmap property `{other}`"),
        )),
    }
}

fn get_area_property(
    area_handle: &super::state::AreaHandleState,
    property: Option<&str>,
    builtin: &'static str,
) -> BuiltinResult<Value> {
    let figure = super::state::clone_figure(area_handle.figure)
        .ok_or_else(|| plotting_error(builtin, format!("{builtin}: invalid area figure")))?;
    let plot = figure
        .plots()
        .nth(area_handle.plot_index)
        .ok_or_else(|| plotting_error(builtin, format!("{builtin}: invalid area handle")))?;
    let runmat_plot::plots::figure::PlotElement::Area(area) = plot else {
        return Err(plotting_error(
            builtin,
            format!("{builtin}: invalid area handle"),
        ));
    };
    match property.map(canonical_property_name).as_deref() {
        None => {
            let mut st = StructValue::new();
            st.insert("Type", Value::String("area".into()));
            st.insert(
                "Parent",
                Value::Num(super::state::encode_axes_handle(
                    area_handle.figure,
                    area_handle.axes_index,
                )),
            );
            st.insert("Children", handles_value(Vec::new()));
            st.insert("XData", tensor_from_vec(area.x.clone()));
            st.insert("YData", tensor_from_vec(area.y.clone()));
            st.insert("BaseValue", Value::Num(area.baseline));
            st.insert("Color", Value::String(color_to_short_name(area.color)));
            Ok(Value::Struct(st))
        }
        Some("type") => Ok(Value::String("area".into())),
        Some("parent") => Ok(Value::Num(super::state::encode_axes_handle(
            area_handle.figure,
            area_handle.axes_index,
        ))),
        Some("children") => Ok(handles_value(Vec::new())),
        Some("xdata") => Ok(tensor_from_vec(area.x.clone())),
        Some("ydata") => Ok(tensor_from_vec(area.y.clone())),
        Some("basevalue") => Ok(Value::Num(area.baseline)),
        Some("color") => Ok(Value::String(color_to_short_name(area.color))),
        Some(other) => Err(plotting_error(
            builtin,
            format!("{builtin}: unsupported area property `{other}`"),
        )),
    }
}

fn apply_histogram_property(
    hist: &super::state::HistogramHandleState,
    key: &str,
    value: &Value,
    builtin: &'static str,
) -> BuiltinResult<()> {
    match key {
        "normalization" => {
            let norm = value_as_string(value)
                .ok_or_else(|| {
                    plotting_error(
                        builtin,
                        format!("{builtin}: Normalization must be a string"),
                    )
                })?
                .trim()
                .to_ascii_lowercase();
            validate_histogram_normalization(&norm, builtin)?;
            let normalized = apply_histogram_normalization(
                &hist.raw_counts,
                &hist.bin_edges,
                &norm,
                hist.normalization_denominator,
            );
            let labels = histogram_labels_from_edges(&hist.bin_edges);
            super::state::update_histogram_plot_data(
                hist.figure,
                hist.plot_index,
                labels,
                normalized,
            )
            .map_err(|err| map_figure_error(builtin, err))?;
            super::state::update_histogram_handle_for_plot(
                hist.figure,
                hist.axes_index,
                hist.plot_index,
                norm,
                hist.raw_counts.clone(),
            )
            .map_err(|err| map_figure_error(builtin, err))?;
            Ok(())
        }
        "displayname" => {
            let display_name = value_as_string(value).map(|s| s.to_string());
            super::state::update_plot_element(hist.figure, hist.plot_index, |plot| {
                if let runmat_plot::plots::figure::PlotElement::Bar(bar) = plot {
                    bar.label = display_name.clone();
                }
            })
            .map_err(|err| map_figure_error(builtin, err))?;
            super::state::set_histogram_handle_display_name(
                hist.figure,
                hist.axes_index,
                hist.plot_index,
                display_name,
            )
            .map_err(|err| map_figure_error(builtin, err))?;
            Ok(())
        }
        "displaystyle" => {
            let display_style = value_as_string(value)
                .ok_or_else(|| {
                    plotting_error(builtin, format!("{builtin}: DisplayStyle must be a string"))
                })?
                .trim()
                .to_ascii_lowercase();
            let style = match display_style.as_str() {
                "bar" => PolarHistogramDisplayStyle::Bar,
                "stairs" => PolarHistogramDisplayStyle::Stairs,
                other => {
                    return Err(plotting_error(
                        builtin,
                        format!("{builtin}: unsupported histogram DisplayStyle `{other}`"),
                    ));
                }
            };
            if !hist.metadata.is_polar && display_style == "stairs" {
                return Err(plotting_error(
                    builtin,
                    format!(
                        "{builtin}: DisplayStyle 'stairs' is only supported for polar histograms"
                    ),
                ));
            }
            super::state::update_plot_element(hist.figure, hist.plot_index, |plot| {
                if let runmat_plot::plots::figure::PlotElement::Bar(bar) = plot {
                    if bar.is_polar_histogram() {
                        bar.set_polar_histogram_display_style(style);
                    }
                }
            })
            .map_err(|err| map_figure_error(builtin, err))?;
            super::state::update_histogram_handle_metadata_for_plot(
                hist.figure,
                hist.axes_index,
                hist.plot_index,
                |metadata| metadata.display_style = display_style.clone(),
            )
            .map_err(|err| map_figure_error(builtin, err))?;
            Ok(())
        }
        "facealpha" => {
            let alpha = patch_scalar_f64(value).ok_or_else(|| {
                plotting_error(builtin, format!("{builtin}: FaceAlpha must be numeric"))
            })?;
            let alpha = alpha.clamp(0.0, 1.0);
            super::state::update_plot_element(hist.figure, hist.plot_index, |plot| {
                if let runmat_plot::plots::figure::PlotElement::Bar(bar) = plot {
                    let mut color = bar.color;
                    color.w = alpha as f32;
                    bar.apply_face_style(color, bar.bar_width);
                }
            })
            .map_err(|err| map_figure_error(builtin, err))?;
            super::state::update_histogram_handle_metadata_for_plot(
                hist.figure,
                hist.axes_index,
                hist.plot_index,
                |metadata| metadata.face_alpha = alpha,
            )
            .map_err(|err| map_figure_error(builtin, err))?;
            Ok(())
        }
        "facecolor" | "edgecolor" => {
            let Some(color_text) = value_as_string(value) else {
                return Err(plotting_error(
                    builtin,
                    format!("{builtin}: color property must be a string"),
                ));
            };
            let lower = color_text.trim().to_ascii_lowercase();
            let color = match lower.as_str() {
                "auto" => None,
                "none" if key == "edgecolor" => None,
                _ => Some(color_from_name_or_token(&lower).ok_or_else(|| {
                    plotting_error(builtin, format!("{builtin}: unsupported color `{lower}`"))
                })?),
            };
            super::state::update_plot_element(hist.figure, hist.plot_index, |plot| {
                if let runmat_plot::plots::figure::PlotElement::Bar(bar) = plot {
                    if key == "facecolor" {
                        if let Some(mut color) = color {
                            color.w = hist.metadata.face_alpha as f32;
                            bar.apply_face_style(color, bar.bar_width);
                        }
                    } else {
                        bar.apply_outline_style(color, bar.outline_width);
                    }
                }
            })
            .map_err(|err| map_figure_error(builtin, err))?;
            super::state::update_histogram_handle_metadata_for_plot(
                hist.figure,
                hist.axes_index,
                hist.plot_index,
                |metadata| {
                    if key == "facecolor" {
                        metadata.face_color = lower.clone();
                    } else {
                        metadata.edge_color = lower.clone();
                    }
                },
            )
            .map_err(|err| map_figure_error(builtin, err))?;
            Ok(())
        }
        other => Err(plotting_error(
            builtin,
            format!("{builtin}: unsupported histogram property `{other}`"),
        )),
    }
}

fn apply_histogram2_property(
    hist: &super::state::Histogram2HandleState,
    key: &str,
    value: &Value,
    builtin: &'static str,
) -> BuiltinResult<()> {
    let mut next = hist.clone();
    apply_histogram2_property_to_state(&mut next, key, value, builtin)?;
    refresh_histogram2_chart(next, builtin)
}

fn apply_histogram2_properties(
    hist: &super::state::Histogram2HandleState,
    pairs: &[(String, &Value)],
    builtin: &'static str,
) -> BuiltinResult<()> {
    let mut next = hist.clone();
    for (key, value) in pairs {
        apply_histogram2_property_to_state(&mut next, key, value, builtin)?;
    }
    refresh_histogram2_chart(next, builtin)
}

fn apply_histogram2_property_to_state(
    next: &mut super::state::Histogram2HandleState,
    key: &str,
    value: &Value,
    builtin: &'static str,
) -> BuiltinResult<()> {
    match key {
        "normalization" => {
            next.normalization = value_as_string(value)
                .ok_or_else(|| {
                    plotting_error(
                        builtin,
                        format!("{builtin}: Normalization must be a string"),
                    )
                })?
                .trim()
                .to_ascii_lowercase();
            next.values = histogram2::apply_normalization_2d(
                &next.raw_counts,
                &next.x_bin_edges,
                &next.y_bin_edges,
                &next.normalization,
                next.normalization_denominator,
            )?;
        }
        "displaystyle" => {
            next.display_style = histogram2::parse_display_style(value)?;
        }
        "showemptybins" => {
            next.show_empty_bins = histogram2::option_bool(value, "ShowEmptyBins")?;
        }
        "facealpha" => {
            let alpha = histogram2::option_scalar(value, "FaceAlpha")?;
            histogram2::validate_face_alpha(alpha)?;
            next.face_alpha = alpha;
        }
        "displayname" => {
            next.display_name = value_as_string(value).map(|s| s.to_string());
        }
        other => Err(plotting_error(
            builtin,
            format!("{builtin}: unsupported histogram2 property `{other}`"),
        ))?,
    }
    Ok(())
}

fn refresh_histogram2_chart(
    next: super::state::Histogram2HandleState,
    builtin: &'static str,
) -> BuiltinResult<()> {
    let color_limits = current_histogram2_color_limits(&next, builtin)?;
    let surface = histogram2::build_surface_from_values(
        &next.values,
        &next.x_bin_edges,
        &next.y_bin_edges,
        next.display_style,
        next.show_empty_bins,
        next.face_alpha,
        next.display_name.as_deref(),
        color_limits,
    )?;
    super::state::update_plot_element(next.figure, next.plot_index, |plot| {
        if let runmat_plot::plots::figure::PlotElement::Surface(existing) = plot {
            *existing = surface;
        }
    })
    .map_err(|err| map_figure_error(builtin, err))?;
    super::state::update_histogram2_handle_for_plot(
        next.figure,
        next.axes_index,
        next.plot_index,
        |state| {
            state.values = next.values;
            state.raw_counts = next.raw_counts;
            state.x_bin_edges = next.x_bin_edges;
            state.y_bin_edges = next.y_bin_edges;
            state.normalization = next.normalization;
            state.display_style = next.display_style;
            state.show_empty_bins = next.show_empty_bins;
            state.face_alpha = next.face_alpha;
            state.display_name = next.display_name;
            state.data = next.data;
        },
    )
    .map_err(|err| map_figure_error(builtin, err))?;
    Ok(())
}

fn current_histogram2_color_limits(
    hist: &super::state::Histogram2HandleState,
    builtin: &'static str,
) -> BuiltinResult<Option<(f64, f64)>> {
    let figure = super::state::clone_figure(hist.figure)
        .ok_or_else(|| plotting_error(builtin, format!("{builtin}: invalid histogram2 figure")))?;
    let plot = figure
        .plots()
        .nth(hist.plot_index)
        .ok_or_else(|| plotting_error(builtin, format!("{builtin}: invalid histogram2 handle")))?;
    let runmat_plot::plots::figure::PlotElement::Surface(surface) = plot else {
        return Err(plotting_error(
            builtin,
            format!("{builtin}: invalid histogram2 handle"),
        ));
    };
    Ok(surface.color_limits)
}

fn apply_stem_property(
    stem_handle: &super::state::StemHandleState,
    key: &str,
    value: &Value,
    builtin: &'static str,
) -> BuiltinResult<()> {
    super::state::update_stem_plot(
        stem_handle.figure,
        stem_handle.plot_index,
        |stem| match key {
            "basevalue" => {
                if let Some(v) = value_as_f64(value) {
                    stem.baseline = v;
                }
            }
            "baseline" => {
                if let Some(v) = value_as_bool(value) {
                    stem.baseline_visible = v;
                }
            }
            "linewidth" => {
                if let Some(v) = value_as_f64(value) {
                    stem.line_width = v as f32;
                }
            }
            "linestyle" => {
                if let Some(s) = value_as_string(value) {
                    stem.line_style = parse_line_style_name_for_props(&s);
                }
            }
            "color" => {
                if let Ok(c) = parse_color_value(&LineStyleParseOptions::generic(builtin), value) {
                    stem.color = c;
                }
            }
            "marker" => {
                if let Some(s) = value_as_string(value) {
                    stem.marker = marker_from_name(&s, stem.marker.clone());
                }
            }
            "markersize" => {
                if let Some(v) = value_as_f64(value) {
                    if let Some(marker) = &mut stem.marker {
                        marker.size = v as f32;
                    }
                }
            }
            "markerfacecolor" => {
                if let Ok(c) = parse_color_value(&LineStyleParseOptions::generic(builtin), value) {
                    if let Some(marker) = &mut stem.marker {
                        marker.face_color = c;
                    }
                }
            }
            "markeredgecolor" => {
                if let Ok(c) = parse_color_value(&LineStyleParseOptions::generic(builtin), value) {
                    if let Some(marker) = &mut stem.marker {
                        marker.edge_color = c;
                    }
                }
            }
            "filled" => {
                if let Some(v) = value_as_bool(value) {
                    if let Some(marker) = &mut stem.marker {
                        marker.filled = v;
                    }
                }
            }
            _ => {}
        },
    )
    .map_err(|err| map_figure_error(builtin, err))?;
    Ok(())
}

fn apply_errorbar_property(
    error_handle: &super::state::ErrorBarHandleState,
    key: &str,
    value: &Value,
    builtin: &'static str,
) -> BuiltinResult<()> {
    super::state::update_errorbar_plot(error_handle.figure, error_handle.plot_index, |errorbar| {
        match key {
            "linewidth" => {
                if let Some(v) = value_as_f64(value) {
                    errorbar.line_width = v as f32;
                }
            }
            "linestyle" => {
                if let Some(s) = value_as_string(value) {
                    errorbar.line_style = parse_line_style_name_for_props(&s);
                }
            }
            "color" => {
                if let Ok(c) = parse_color_value(&LineStyleParseOptions::generic(builtin), value) {
                    errorbar.color = c;
                }
            }
            "capsize" => {
                if let Some(v) = value_as_f64(value) {
                    errorbar.cap_size = v as f32;
                }
            }
            "marker" => {
                if let Some(s) = value_as_string(value) {
                    errorbar.marker = marker_from_name(&s, errorbar.marker.clone());
                }
            }
            "markersize" => {
                if let Some(v) = value_as_f64(value) {
                    if let Some(marker) = &mut errorbar.marker {
                        marker.size = v as f32;
                    }
                }
            }
            _ => {}
        }
    })
    .map_err(|err| map_figure_error(builtin, err))?;
    Ok(())
}

fn apply_quiver_property(
    quiver_handle: &super::state::QuiverHandleState,
    key: &str,
    value: &Value,
    builtin: &'static str,
) -> BuiltinResult<()> {
    if matches!(
        key,
        "xdata" | "ydata" | "zdata" | "udata" | "vdata" | "wdata"
    ) {
        return apply_quiver_data_property(quiver_handle, key, value, builtin);
    }
    super::state::update_quiver_plot(quiver_handle.figure, quiver_handle.plot_index, |quiver| {
        match key {
            "color" => {
                if let Ok(c) = parse_color_value(&LineStyleParseOptions::generic(builtin), value) {
                    quiver.color = c;
                    quiver.mark_dirty();
                }
            }
            "linewidth" => {
                if let Some(v) = value_as_f64(value) {
                    quiver.line_width = v as f32;
                }
            }
            "autoscalefactor" => {
                if let Some(v) = value_as_f64(value) {
                    quiver.scale = v as f32;
                    quiver.mark_dirty();
                }
            }
            "maxheadsize" => {
                if let Some(v) = value_as_f64(value) {
                    quiver.head_size = v as f32;
                    quiver.mark_dirty();
                }
            }
            _ => {}
        }
    })
    .map_err(|err| map_figure_error(builtin, err))?;
    Ok(())
}

fn apply_quiver_data_property(
    quiver_handle: &super::state::QuiverHandleState,
    key: &str,
    value: &Value,
    builtin: &'static str,
) -> BuiltinResult<()> {
    if matches!(key, "zdata" | "wdata") && !quiver_handle.is_3d {
        return Err(plotting_error(
            builtin,
            format!("{builtin}: unsupported quiver property `{key}`"),
        ));
    }
    let figure = super::state::clone_figure(quiver_handle.figure)
        .ok_or_else(|| plotting_error(builtin, format!("{builtin}: invalid quiver figure")))?;
    let plot = figure
        .plots()
        .nth(quiver_handle.plot_index)
        .ok_or_else(|| plotting_error(builtin, format!("{builtin}: invalid quiver handle")))?;
    let runmat_plot::plots::figure::PlotElement::Quiver(quiver) = plot else {
        return Err(plotting_error(
            builtin,
            format!("{builtin}: invalid quiver handle"),
        ));
    };
    let expected_len = quiver
        .cpu_vector_data_len()
        .ok_or_else(|| quiver_data_unavailable_error(builtin))?;
    let tensor = Tensor::try_from(value).map_err(|err| {
        plotting_error(builtin, format!("{builtin}: {key} must be numeric: {err}"))
    })?;
    if tensor::tensor_element_len(&tensor) != expected_len {
        return Err(plotting_error(
            builtin,
            format!("{builtin}: {key} length must match existing quiver data length"),
        ));
    }
    let data = tensor::tensor_values_f64(&tensor);
    super::state::update_quiver_plot(quiver_handle.figure, quiver_handle.plot_index, |quiver| {
        match key {
            "xdata" => quiver.x = data,
            "ydata" => quiver.y = data,
            "zdata" => quiver.z = Some(data),
            "udata" => quiver.u = data,
            "vdata" => quiver.v = data,
            "wdata" => quiver.w = Some(data),
            _ => {}
        }
        quiver.mark_dirty();
    })
    .map_err(|err| map_figure_error(builtin, err))?;
    Ok(())
}

fn apply_image_property(
    image_handle: &super::state::ImageHandleState,
    key: &str,
    value: &Value,
    builtin: &'static str,
) -> BuiltinResult<()> {
    if key == "colordata" {
        return Err(plotting_error(
            builtin,
            format!("{builtin}: setting image CData is not implemented"),
        ));
    }
    if key == "cdatamapping" {
        let mapping = value_as_string(value)
            .map(|text| text.trim().to_ascii_lowercase())
            .ok_or_else(|| {
                plotting_error(builtin, format!("{builtin}: CDataMapping must be text"))
            })?;
        if !matches!(mapping.as_str(), "direct" | "scaled") {
            return Err(plotting_error(
                builtin,
                format!("{builtin}: CDataMapping must be 'direct' or 'scaled'"),
            ));
        }
        let renderer_limits = if mapping == "direct" {
            let integer = image_handle
                .c_data
                .as_ref()
                .is_some_and(|tensor| tensor.integer_storage().is_some());
            Some(crate::builtins::plotting::image::direct_colormap_limits(
                integer,
                super::state::colormap_length_for_axes(
                    image_handle.figure,
                    image_handle.axes_index,
                ),
            ))
        } else {
            super::state::clone_figure(image_handle.figure).and_then(|figure| {
                figure
                    .axes_metadata(image_handle.axes_index)
                    .and_then(|metadata| metadata.color_limits)
            })
        };
        super::state::update_image_plot(image_handle.figure, image_handle.plot_index, |surface| {
            surface.set_color_limits(renderer_limits);
        })
        .map_err(|err| map_figure_error(builtin, err))?;
        super::state::update_image_handle_for_plot(
            image_handle.figure,
            image_handle.axes_index,
            image_handle.plot_index,
            |state| state.c_data_mapping = mapping,
        )
        .map_err(|err| map_figure_error(builtin, err))?;
        return Ok(());
    }
    let mut retained_mapping = None;
    super::state::update_image_plot(image_handle.figure, image_handle.plot_index, |surface| {
        match key {
            "xdata" => {
                if let Ok(tensor) = Tensor::try_from(value) {
                    surface.x_data = tensor.materialize_f64();
                }
            }
            "ydata" => {
                if let Ok(tensor) = Tensor::try_from(value) {
                    surface.y_data = tensor.materialize_f64();
                }
            }
            "cdatamapping" => {
                if let Some(text) = value_as_string(value) {
                    let normalized = text.trim().to_ascii_lowercase();
                    if normalized == "direct" {
                        surface.image_mode = true;
                    }
                    if matches!(normalized.as_str(), "direct" | "scaled") {
                        retained_mapping = Some(normalized);
                    }
                }
            }
            _ => {}
        }
    })
    .map_err(|err| map_figure_error(builtin, err))?;
    if retained_mapping.is_some() {
        super::state::update_image_handle_for_plot(
            image_handle.figure,
            image_handle.axes_index,
            image_handle.plot_index,
            |state| {
                if let Some(mapping) = retained_mapping {
                    state.c_data_mapping = mapping;
                }
            },
        )
        .map_err(|err| map_figure_error(builtin, err))?;
    }
    Ok(())
}

fn apply_heatmap_property(
    handle: f64,
    heatmap_handle: &super::state::HeatmapHandleState,
    key: &str,
    value: &Value,
    builtin: &'static str,
) -> BuiltinResult<()> {
    match key {
        "title" => apply_axes_property(
            heatmap_handle.figure,
            heatmap_handle.axes_index,
            "title",
            value,
            builtin,
        ),
        "xlabel" => apply_axes_property(
            heatmap_handle.figure,
            heatmap_handle.axes_index,
            "xlabel",
            value,
            builtin,
        ),
        "ylabel" => apply_axes_property(
            heatmap_handle.figure,
            heatmap_handle.axes_index,
            "ylabel",
            value,
            builtin,
        ),
        "colorbar" | "colorbarvisible" => {
            let enabled = heatmap_on_off_value(value).ok_or_else(|| {
                plotting_error(
                    builtin,
                    format!("{builtin}: ColorbarVisible must be 'on', 'off', 0, or 1"),
                )
            })?;
            crate::builtins::plotting::state::set_colorbar_enabled_for_axes(
                heatmap_handle.figure,
                heatmap_handle.axes_index,
                enabled,
            )
            .map_err(|err| map_figure_error(builtin, err))
        }
        "gridvisible" => {
            let enabled = heatmap_on_off_value(value).ok_or_else(|| {
                plotting_error(
                    builtin,
                    format!("{builtin}: GridVisible must be 'on', 'off', 0, or 1"),
                )
            })?;
            crate::builtins::plotting::state::set_grid_enabled_for_axes(
                heatmap_handle.figure,
                heatmap_handle.axes_index,
                enabled,
            )
            .map_err(|err| map_figure_error(builtin, err))
        }
        "fontsize" => apply_axes_property(
            heatmap_handle.figure,
            heatmap_handle.axes_index,
            "fontsize",
            value,
            builtin,
        ),
        "colormap" => apply_axes_property(
            heatmap_handle.figure,
            heatmap_handle.axes_index,
            "colormap",
            value,
            builtin,
        ),
        "colorlimits" => {
            let public_value = Tensor::try_from(value)
                .map_err(|error| plotting_error(builtin, format!("{builtin}: {error}")))?;
            let public_limits =
                crate::builtins::plotting::heatmap::public_color_limits_for_value(value, builtin)?;
            let renderer_limits =
                crate::builtins::plotting::heatmap::renderer_color_limits_for_value(
                    &heatmap_handle.color_data,
                    value,
                    builtin,
                )?;
            crate::builtins::plotting::state::set_color_limits_for_axes(
                heatmap_handle.figure,
                heatmap_handle.axes_index,
                Some(public_limits),
            )
            .map_err(|err| map_figure_error(builtin, err))?;
            super::state::set_heatmap_color_limits(handle, public_value)
                .map_err(|err| map_figure_error(builtin, err))?;
            super::state::update_plot_element(
                heatmap_handle.figure,
                heatmap_handle.plot_index,
                |plot| {
                    if let runmat_plot::plots::figure::PlotElement::Surface(surface) = plot {
                        surface.set_color_limits(Some(renderer_limits));
                    }
                },
            )
            .map_err(|err| map_figure_error(builtin, err))
        }
        "xdisplaylabels" => {
            let labels = label_strings_from_value(value, builtin, "labels")?;
            if labels.len() != heatmap_handle.x_labels.len() {
                return Err(plotting_error(
                    builtin,
                    format!("{builtin}: XDisplayLabels length must match heatmap columns"),
                ));
            }
            super::state::set_heatmap_display_labels(
                heatmap_handle.figure,
                heatmap_handle.axes_index,
                heatmap_handle.plot_index,
                Some(labels),
                None,
            )
            .map_err(|err| map_figure_error(builtin, err))
        }
        "ydisplaylabels" => {
            let labels = label_strings_from_value(value, builtin, "labels")?;
            if labels.len() != heatmap_handle.y_labels.len() {
                return Err(plotting_error(
                    builtin,
                    format!("{builtin}: YDisplayLabels length must match heatmap rows"),
                ));
            }
            super::state::set_heatmap_display_labels(
                heatmap_handle.figure,
                heatmap_handle.axes_index,
                heatmap_handle.plot_index,
                None,
                Some(labels),
            )
            .map_err(|err| map_figure_error(builtin, err))
        }
        other => Err(plotting_error(
            builtin,
            format!("{builtin}: unsupported heatmap property `{other}`"),
        )),
    }
}

fn apply_area_property(
    area_handle: &super::state::AreaHandleState,
    key: &str,
    value: &Value,
    builtin: &'static str,
) -> BuiltinResult<()> {
    super::state::update_area_plot(
        area_handle.figure,
        area_handle.plot_index,
        |area| match key {
            "color" => {
                if let Ok(c) = parse_color_value(&LineStyleParseOptions::generic(builtin), value) {
                    area.color = c;
                }
            }
            "basevalue" => {
                if let Some(v) = value_as_f64(value) {
                    area.baseline = v;
                    area.lower_y = None;
                }
            }
            _ => {}
        },
    )
    .map_err(|err| map_figure_error(builtin, err))?;
    Ok(())
}

fn apply_line_property(
    line_handle: &super::state::SimplePlotHandleState,
    key: &str,
    value: &Value,
    builtin: &'static str,
) -> BuiltinResult<()> {
    apply_line_properties(line_handle, &[(key.to_string(), value)], builtin)
}

fn apply_line_properties(
    line_handle: &super::state::SimplePlotHandleState,
    pairs: &[(String, &Value)],
    builtin: &'static str,
) -> BuiltinResult<()> {
    let current = get_simple_plot(line_handle, builtin)?;
    let runmat_plot::plots::figure::PlotElement::Line(current_line) = current else {
        return Err(plotting_error(
            builtin,
            format!("{builtin}: invalid line handle"),
        ));
    };

    let (current_x, current_y) = line_xy_data_for_properties(&current_line, builtin)?;
    let mut x_update = None;
    let mut y_update = None;
    let mut style_pairs = Vec::new();
    for (key, value) in pairs {
        match key.as_str() {
            "xdata" => x_update = Some(line_numeric_data(value, "XData", builtin)?),
            "ydata" => y_update = Some(line_numeric_data(value, "YData", builtin)?),
            _ => style_pairs.push((key.as_str(), *value)),
        }
    }

    let next_x = x_update.unwrap_or_else(|| current_x.clone());
    let next_y = y_update.unwrap_or_else(|| current_y.clone());
    let update_data = next_x != current_x || next_y != current_y;
    if update_data && next_x.len() != next_y.len() {
        return Err(plotting_error(
            builtin,
            format!("{builtin}: XData and YData must contain the same number of elements"),
        ));
    }
    for (key, value) in &style_pairs {
        if *key == "handlevisibility" {
            parse_handle_visibility(&LineStyleParseOptions::generic(builtin), value)?;
        }
    }

    super::state::update_plot_element(line_handle.figure, line_handle.plot_index, |plot| {
        if let runmat_plot::plots::figure::PlotElement::Line(line) = plot {
            if update_data {
                let _ = line.update_numeric_data(next_x.clone(), next_y.clone());
            }
            for (key, value) in &style_pairs {
                apply_line_plot_properties(line, key, value, builtin);
            }
        }
    })
    .map_err(|err| map_figure_error(builtin, err))?;
    Ok(())
}

fn apply_animated_line_property(
    line_handle: &super::state::AnimatedLineHandleState,
    key: &str,
    value: &Value,
    builtin: &'static str,
) -> BuiltinResult<()> {
    apply_animated_line_properties(line_handle, &[(key.to_string(), value)], builtin)
}

fn apply_animated_line_properties(
    line_handle: &super::state::AnimatedLineHandleState,
    pairs: &[(String, &Value)],
    builtin: &'static str,
) -> BuiltinResult<()> {
    let mut maximum = None;
    let mut plot_pairs = Vec::new();
    for (key, value) in pairs {
        match key.as_str() {
            "maximumnumpoints" => {
                maximum = Some(animated_line_maximum_from_value(value, builtin)?);
            }
            _ => plot_pairs.push((key.clone(), *value)),
        }
    }

    if !plot_pairs.is_empty() {
        let simple = animated_line_simple_state(line_handle);
        if line_handle.is_3d {
            apply_line3_properties(&simple, &plot_pairs, builtin)?;
        } else {
            if plot_pairs.iter().any(|(key, _)| key == "zdata") {
                return Err(plotting_error(
                    builtin,
                    format!("{builtin}: ZData requires a 3-D animated line"),
                ));
            }
            apply_line_properties(&simple, &plot_pairs, builtin)?;
        }
    }

    if let Some(maximum) = maximum {
        super::state::set_animated_line_maximum_num_points(line_handle, maximum)
            .map_err(|err| plotting_error(builtin, format!("{builtin}: {err}")))?;
    }
    Ok(())
}

pub(crate) fn animated_line_maximum_from_value(
    value: &Value,
    builtin: &'static str,
) -> BuiltinResult<Option<usize>> {
    if matches!(value, Value::Bool(_) | Value::LogicalArray(_)) {
        return Err(plotting_error(
            builtin,
            format!("{builtin}: MaximumNumPoints must be numeric"),
        ));
    }
    let integer = match value {
        Value::Int(integer) => Some(integer.clone()),
        Value::Tensor(tensor) if tensor.len() == 1 => tensor
            .integer_storage()
            .and_then(|storage| storage.value_at(0)),
        _ => None,
    };
    if let Some(integer) = integer {
        let Some(maximum) = integer.try_to_usize() else {
            return Err(plotting_error(
                builtin,
                format!("{builtin}: MaximumNumPoints must be positive or Inf"),
            ));
        };
        if maximum == 0 {
            return Err(plotting_error(
                builtin,
                format!("{builtin}: MaximumNumPoints must be positive or Inf"),
            ));
        }
        return Ok(Some(maximum));
    }
    let Some(maximum) = value_as_f64(value) else {
        return Err(plotting_error(
            builtin,
            format!("{builtin}: MaximumNumPoints must be numeric"),
        ));
    };
    if maximum.is_infinite() && maximum.is_sign_positive() {
        return Ok(None);
    }
    if !maximum.is_finite() || maximum <= 0.0 {
        return Err(plotting_error(
            builtin,
            format!("{builtin}: MaximumNumPoints must be positive or Inf"),
        ));
    }
    let rounded = maximum.round();
    if (rounded - maximum).abs() > f64::EPSILON {
        return Err(plotting_error(
            builtin,
            format!("{builtin}: MaximumNumPoints must be a positive integer or Inf"),
        ));
    }
    if rounded > usize::MAX as f64 || (usize::BITS == 64 && rounded == usize::MAX as f64) {
        return Err(plotting_error(
            builtin,
            format!("{builtin}: MaximumNumPoints is too large"),
        ));
    }
    Ok(Some(rounded as usize))
}

fn apply_reference_line_property(
    line_handle: &super::state::SimplePlotHandleState,
    key: &str,
    value: &Value,
    builtin: &'static str,
) -> BuiltinResult<()> {
    super::state::update_plot_element(line_handle.figure, line_handle.plot_index, |plot| {
        if let runmat_plot::plots::figure::PlotElement::ReferenceLine(line) = plot {
            match key {
                "value" => {
                    if let Some(v) = value_as_f64(value) {
                        if v.is_finite() {
                            line.value = v;
                        }
                    }
                }
                "color" => {
                    if let Ok(c) =
                        parse_color_value(&LineStyleParseOptions::generic(builtin), value)
                    {
                        line.color = c;
                    }
                }
                "linewidth" => {
                    if let Some(v) = value_as_f64(value) {
                        if v > 0.0 {
                            line.line_width = v as f32;
                        }
                    }
                }
                "linestyle" => {
                    if let Some(s) = value_as_string(value) {
                        line.line_style = parse_line_style_name_for_props(&s);
                    }
                }
                "label" => {
                    line.label = value_as_string(value).map(|s| s.to_string());
                }
                "labelorientation" => {
                    if let Some(s) = value_as_string(value) {
                        line.label_orientation = s.to_ascii_lowercase();
                    }
                }
                "displayname" => {
                    line.display_name = value_as_string(value).map(|s| s.to_string());
                }
                "visible" => {
                    if let Some(v) = value_as_bool(value) {
                        line.visible = v;
                    } else if let Some(s) = value_as_string(value) {
                        line.visible =
                            !matches!(s.trim().to_ascii_lowercase().as_str(), "off" | "false");
                    }
                }
                _ => {}
            }
        }
    })
    .map_err(|err| map_figure_error(builtin, err))?;
    Ok(())
}

fn apply_stairs_property(
    stairs_handle: &super::state::SimplePlotHandleState,
    key: &str,
    value: &Value,
    builtin: &'static str,
) -> BuiltinResult<()> {
    super::state::update_plot_element(stairs_handle.figure, stairs_handle.plot_index, |plot| {
        if let runmat_plot::plots::figure::PlotElement::Stairs(stairs) = plot {
            match key {
                "color" => {
                    if let Ok(c) =
                        parse_color_value(&LineStyleParseOptions::generic(builtin), value)
                    {
                        stairs.color = c;
                    }
                }
                "linewidth" => {
                    if let Some(v) = value_as_f64(value) {
                        stairs.line_width = v as f32;
                    }
                }
                "displayname" => {
                    stairs.label = value_as_string(value).map(|s| s.to_string());
                }
                _ => {}
            }
        }
    })
    .map_err(|err| map_figure_error(builtin, err))?;
    Ok(())
}

fn apply_scatter_property(
    scatter_handle: &super::state::SimplePlotHandleState,
    key: &str,
    value: &Value,
    builtin: &'static str,
) -> BuiltinResult<()> {
    apply_scatter_properties(scatter_handle, &[(key.to_string(), value)], builtin)
}

fn apply_scatter_properties(
    scatter_handle: &super::state::SimplePlotHandleState,
    pairs: &[(String, &Value)],
    builtin: &'static str,
) -> BuiltinResult<()> {
    let current = get_simple_plot(scatter_handle, builtin)?;
    let runmat_plot::plots::figure::PlotElement::Scatter(current_scatter) = current else {
        return Err(plotting_error(
            builtin,
            format!("{builtin}: invalid scatter handle"),
        ));
    };

    let (current_x, current_y) = scatter_xy_data_for_properties(&current_scatter, builtin)?;
    let current_x_render = current_x.materialize_f64();
    let current_y_render = current_y.materialize_f64();

    let current_theta = current_scatter
        .theta_data
        .clone()
        .unwrap_or_else(|| cartesian_theta(&current_x_render, &current_y_render));
    let current_r = current_scatter
        .r_data
        .clone()
        .unwrap_or_else(|| cartesian_radius(&current_x_render, &current_y_render));
    let mut x_update = None;
    let mut y_update = None;
    let mut theta_update = None;
    let mut r_update = None;
    let mut style_pairs = Vec::new();
    for (key, value) in pairs {
        match key.as_str() {
            "xdata" => x_update = Some(line_numeric_data(value, "XData", builtin)?),
            "ydata" => y_update = Some(line_numeric_data(value, "YData", builtin)?),
            "thetadata" => {
                theta_update =
                    Some(line_numeric_data(value, "ThetaData", builtin)?.materialize_f64())
            }
            "rdata" => {
                r_update = Some(line_numeric_data(value, "RData", builtin)?.materialize_f64())
            }
            _ => style_pairs.push((key.as_str(), *value)),
        }
    }

    let next_theta = theta_update.unwrap_or_else(|| current_theta.clone());
    let next_r = r_update.unwrap_or_else(|| current_r.clone());
    let polar_update =
        (next_theta != current_theta || next_r != current_r).then_some((next_theta, next_r));
    if let Some((theta, r)) = polar_update.as_ref() {
        validate_polar_scatter_data(theta, r, builtin)?;
    }
    let next_x = x_update.unwrap_or_else(|| current_x.clone());
    let next_y = y_update.unwrap_or_else(|| current_y.clone());
    let cartesian_update = (next_x != current_x || next_y != current_y).then_some((next_x, next_y));
    if let Some((x, y)) = cartesian_update.as_ref() {
        if x.len() != y.len() || x.is_empty() {
            return Err(plotting_error(
                builtin,
                format!(
                    "{builtin}: XData and YData must contain the same nonzero number of elements"
                ),
            ));
        }
    }
    if cartesian_update.is_some() && polar_update.is_some() {
        return Err(plotting_error(
            builtin,
            format!("{builtin}: cannot update Cartesian and polar coordinate properties together"),
        ));
    }

    super::state::update_plot_element(scatter_handle.figure, scatter_handle.plot_index, |plot| {
        if let runmat_plot::plots::figure::PlotElement::Scatter(scatter) = plot {
            if let Some((x, y)) = cartesian_update.as_ref() {
                let _ = scatter.update_numeric_data(x.clone(), y.clone());
            }
            if let Some((theta, r)) = polar_update.as_ref() {
                replace_polar_scatter_data_unchecked(scatter, theta.clone(), r.clone());
            }
            for (key, value) in &style_pairs {
                match *key {
                    "marker" => {
                        if let Some(s) = value_as_string(value) {
                            scatter.marker_style =
                                scatter_marker_from_name(&s, scatter.marker_style);
                        }
                    }
                    "sizedata" => {
                        if let Some(v) = value_as_f64(value) {
                            scatter.marker_size = marker_area_points2_to_diameter_px(v);
                        }
                    }
                    "markerfacecolor" => {
                        if let Ok(c) =
                            parse_color_value(&LineStyleParseOptions::generic(builtin), value)
                        {
                            scatter.set_face_color(c);
                        }
                    }
                    "markeredgecolor" => {
                        if let Ok(c) =
                            parse_color_value(&LineStyleParseOptions::generic(builtin), value)
                        {
                            scatter.set_edge_color(c);
                        }
                    }
                    "linewidth" => {
                        if let Some(v) = value_as_f64(value) {
                            scatter.set_edge_thickness(v as f32);
                        }
                    }
                    "displayname" => {
                        scatter.label = value_as_string(value).map(|s| s.to_string());
                    }
                    _ => {}
                }
            }
        }
    })
    .map_err(|err| map_figure_error(builtin, err))?;
    Ok(())
}

fn validate_polar_scatter_data(
    theta: &[f64],
    r: &[f64],
    builtin: &'static str,
) -> BuiltinResult<()> {
    if theta.len() != r.len() {
        return Err(plotting_error(
            builtin,
            format!("{builtin}: ThetaData and RData must contain the same number of elements"),
        ));
    }
    if theta.is_empty() {
        return Err(plotting_error(
            builtin,
            format!("{builtin}: ThetaData and RData cannot be empty"),
        ));
    }
    Ok(())
}

fn replace_polar_scatter_data_unchecked(
    scatter: &mut runmat_plot::plots::ScatterPlot,
    theta: Vec<f64>,
    r: Vec<f64>,
) {
    let x_data = theta
        .iter()
        .zip(r.iter())
        .map(|(theta, r)| r * theta.cos())
        .collect();
    let y_data = theta
        .iter()
        .zip(r.iter())
        .map(|(theta, r)| r * theta.sin())
        .collect();
    let point_count = theta.len();
    scatter
        .update_data(x_data, y_data)
        .expect("validated polar scatter data can update cartesian scatter data");
    if scatter
        .per_point_sizes
        .as_ref()
        .is_some_and(|sizes| sizes.len() != point_count)
    {
        scatter.per_point_sizes = None;
    }
    if scatter
        .per_point_colors
        .as_ref()
        .is_some_and(|colors| colors.len() != point_count)
    {
        scatter.per_point_colors = None;
    }
    if scatter
        .color_values
        .as_ref()
        .is_some_and(|values| values.len() != point_count)
    {
        scatter.color_values = None;
        scatter.color_limits = None;
    }
    scatter.theta_data = Some(theta);
    scatter.r_data = Some(r);
}

fn scatter_size_data_value(scatter: &runmat_plot::plots::ScatterPlot) -> Value {
    match scatter.per_point_sizes.as_ref() {
        Some(sizes) => tensor_from_vec(
            sizes
                .iter()
                .map(|size| marker_diameter_px_to_area_points2(*size))
                .collect(),
        ),
        None => Value::Num(marker_diameter_px_to_area_points2(scatter.marker_size)),
    }
}

fn cartesian_theta(x: &[f64], y: &[f64]) -> Vec<f64> {
    x.iter().zip(y.iter()).map(|(x, y)| y.atan2(*x)).collect()
}

fn cartesian_radius(x: &[f64], y: &[f64]) -> Vec<f64> {
    x.iter().zip(y.iter()).map(|(x, y)| x.hypot(*y)).collect()
}

fn apply_bar_property(
    bar_handle: &super::state::SimplePlotHandleState,
    key: &str,
    value: &Value,
    builtin: &'static str,
) -> BuiltinResult<()> {
    super::state::update_plot_element(bar_handle.figure, bar_handle.plot_index, |plot| {
        if let runmat_plot::plots::figure::PlotElement::Bar(bar) = plot {
            match key {
                "facecolor" | "color" => {
                    if let Ok(c) =
                        parse_color_value(&LineStyleParseOptions::generic(builtin), value)
                    {
                        bar.color = c;
                    }
                }
                "barwidth" => {
                    if let Some(v) = value_as_f64(value) {
                        bar.bar_width = v as f32;
                    }
                }
                "displayname" => {
                    bar.label = value_as_string(value).map(|s| s.to_string());
                }
                _ => {}
            }
        }
    })
    .map_err(|err| map_figure_error(builtin, err))?;
    Ok(())
}

#[derive(Clone, Debug)]
enum SurfacePropertyUpdate {
    FaceAlpha(f32),
    DisplayName(Option<String>),
    Visible(bool),
    Colormap(ColorMap),
    Shading(ShadingMode),
    Wireframe(bool),
    FlattenZ(bool),
    Lighting(bool),
    FaceColor(glam::Vec4),
    FaceNone,
}

fn apply_surface_property(
    surface_handle: &super::state::SimplePlotHandleState,
    key: &str,
    value: &Value,
    builtin: &'static str,
) -> BuiltinResult<()> {
    if matches!(key, "xdata" | "ydata" | "zdata") {
        return apply_surface_data_property(surface_handle, key, value, builtin);
    }

    let update = surface_property_update(key, value, builtin)?;
    super::state::update_plot_element(surface_handle.figure, surface_handle.plot_index, |plot| {
        if let runmat_plot::plots::figure::PlotElement::Surface(surface) = plot {
            match &update {
                SurfacePropertyUpdate::FaceAlpha(alpha) => surface.alpha = *alpha,
                SurfacePropertyUpdate::DisplayName(label) => surface.label = label.clone(),
                SurfacePropertyUpdate::Visible(visible) => surface.visible = *visible,
                SurfacePropertyUpdate::Colormap(colormap) => surface.colormap = colormap.clone(),
                SurfacePropertyUpdate::Shading(shading) => surface.shading_mode = *shading,
                SurfacePropertyUpdate::Wireframe(wireframe) => surface.wireframe = *wireframe,
                SurfacePropertyUpdate::FlattenZ(flatten_z) => surface.flatten_z = *flatten_z,
                SurfacePropertyUpdate::Lighting(enabled) => surface.lighting_enabled = *enabled,
                SurfacePropertyUpdate::FaceColor(color) => {
                    surface.colormap = ColorMap::Custom(*color, *color);
                    surface.shading_mode = ShadingMode::None;
                    surface.lighting_enabled = false;
                }
                SurfacePropertyUpdate::FaceNone => surface.alpha = 0.0,
            }
        }
    })
    .map_err(|err| map_figure_error(builtin, err))?;
    Ok(())
}

fn apply_surface_data_property(
    surface_handle: &super::state::SimplePlotHandleState,
    key: &str,
    value: &Value,
    builtin: &'static str,
) -> BuiltinResult<()> {
    let current = get_simple_plot(surface_handle, builtin)?;
    let runmat_plot::plots::figure::PlotElement::Surface(current_surface) = current else {
        return Err(plotting_error(
            builtin,
            format!("{builtin}: invalid surface handle"),
        ));
    };
    let z_grid = current_surface.z_data.clone().ok_or_else(|| {
        plotting_error(
            builtin,
            format!("{builtin}: changing {key} for GPU-only surface data is not supported yet"),
        )
    })?;
    let (x_len, y_len) = surface_grid_size(&z_grid)?;

    let next = match key {
        "xdata" => {
            let data = surface_coordinate_data_from_value(value, "XData", builtin)?;
            let (x_grid, y_grid, z_grid) = match data {
                SurfaceCoordinateData::Vector(axis) => {
                    if axis.len() != x_len {
                        return Err(plotting_error(
                            builtin,
                            format!("{builtin}: XData vector length must match surface column count {x_len}"),
                        ));
                    }
                    let y_grid = current_surface
                        .y_grid
                        .clone()
                        .unwrap_or_else(|| axis_to_y_grid(&current_surface.y_data, x_len));
                    (axis_to_x_grid(&axis, y_len), y_grid, z_grid)
                }
                SurfaceCoordinateData::Grid(grid) => {
                    validate_surface_grid_shape(&grid, x_len, y_len, "XData", builtin)?;
                    let y_grid = current_surface
                        .y_grid
                        .clone()
                        .unwrap_or_else(|| axis_to_y_grid(&current_surface.y_data, x_len));
                    (grid, y_grid, z_grid)
                }
            };
            SurfaceDataUpdate::CoordinateGrids {
                x_grid,
                y_grid,
                z_grid,
            }
        }
        "ydata" => {
            let data = surface_coordinate_data_from_value(value, "YData", builtin)?;
            let (x_grid, y_grid, z_grid) = match data {
                SurfaceCoordinateData::Vector(axis) => {
                    if axis.len() != y_len {
                        return Err(plotting_error(
                            builtin,
                            format!("{builtin}: YData vector length must match surface row count {y_len}"),
                        ));
                    }
                    let x_grid = current_surface
                        .x_grid
                        .clone()
                        .unwrap_or_else(|| axis_to_x_grid(&current_surface.x_data, y_len));
                    (x_grid, axis_to_y_grid(&axis, x_len), z_grid)
                }
                SurfaceCoordinateData::Grid(grid) => {
                    validate_surface_grid_shape(&grid, x_len, y_len, "YData", builtin)?;
                    let x_grid = current_surface
                        .x_grid
                        .clone()
                        .unwrap_or_else(|| axis_to_x_grid(&current_surface.x_data, y_len));
                    (x_grid, grid, z_grid)
                }
            };
            SurfaceDataUpdate::CoordinateGrids {
                x_grid,
                y_grid,
                z_grid,
            }
        }
        "zdata" => {
            let z_grid = surface_z_grid_from_value(value, y_len, x_len, builtin)?;
            if let (Some(x_grid), Some(y_grid)) = (
                current_surface.x_grid.clone(),
                current_surface.y_grid.clone(),
            ) {
                SurfaceDataUpdate::CoordinateGrids {
                    x_grid,
                    y_grid,
                    z_grid,
                }
            } else {
                SurfaceDataUpdate::AxisData {
                    x_data: current_surface.x_data.clone(),
                    y_data: current_surface.y_data.clone(),
                    z_grid,
                }
            }
        }
        other => {
            return Err(plotting_error(
                builtin,
                format!("{builtin}: unsupported surface property `{other}`"),
            ))
        }
    };

    super::state::update_plot_element(surface_handle.figure, surface_handle.plot_index, |plot| {
        if let runmat_plot::plots::figure::PlotElement::Surface(surface) = plot {
            match &next {
                SurfaceDataUpdate::AxisData {
                    x_data,
                    y_data,
                    z_grid,
                } => {
                    let _ =
                        surface.update_axis_data(x_data.clone(), y_data.clone(), z_grid.clone());
                }
                SurfaceDataUpdate::CoordinateGrids {
                    x_grid,
                    y_grid,
                    z_grid,
                } => {
                    let _ = surface.update_coordinate_grids(
                        x_grid.clone(),
                        y_grid.clone(),
                        z_grid.clone(),
                    );
                }
            }
        }
    })
    .map_err(|err| map_figure_error(builtin, err))?;
    Ok(())
}

enum SurfaceDataUpdate {
    AxisData {
        x_data: Vec<f64>,
        y_data: Vec<f64>,
        z_grid: Vec<Vec<f64>>,
    },
    CoordinateGrids {
        x_grid: Vec<Vec<f64>>,
        y_grid: Vec<Vec<f64>>,
        z_grid: Vec<Vec<f64>>,
    },
}

enum SurfaceCoordinateData {
    Vector(Vec<f64>),
    Grid(Vec<Vec<f64>>),
}

fn surface_property_update(
    key: &str,
    value: &Value,
    builtin: &'static str,
) -> BuiltinResult<SurfacePropertyUpdate> {
    match key {
        "facealpha" | "alpha" => {
            let alpha = value_as_f64(value).ok_or_else(|| {
                plotting_error(builtin, format!("{builtin}: FaceAlpha must be numeric"))
            })?;
            Ok(SurfacePropertyUpdate::FaceAlpha(
                alpha.clamp(0.0, 1.0) as f32
            ))
        }
        "displayname" | "label" => Ok(SurfacePropertyUpdate::DisplayName(
            value_as_string(value).map(|s| s.to_string()),
        )),
        "visible" => {
            let visible = value_as_bool(value).ok_or_else(|| {
                plotting_error(builtin, format!("{builtin}: Visible must be logical"))
            })?;
            Ok(SurfacePropertyUpdate::Visible(visible))
        }
        "colormap" => {
            if let Some(name) = value_as_string(value) {
                return Ok(SurfacePropertyUpdate::Colormap(parse_colormap_name(
                    &name, builtin,
                )?));
            }
            let color = parse_color_value(&LineStyleParseOptions::generic(builtin), value)?;
            Ok(SurfacePropertyUpdate::Colormap(ColorMap::Custom(
                color, color,
            )))
        }
        "shading" => Ok(SurfacePropertyUpdate::Shading(surface_shading_from_value(
            value, builtin,
        )?)),
        "edgecolor" => {
            let text = value_as_string(value).ok_or_else(|| {
                plotting_error(
                    builtin,
                    format!("{builtin}: EdgeColor must be 'auto', 'flat', 'interp', or 'none'"),
                )
            })?;
            match text.trim().to_ascii_lowercase().as_str() {
                "none" => Ok(SurfacePropertyUpdate::Wireframe(false)),
                "auto" | "flat" | "interp" => Ok(SurfacePropertyUpdate::Wireframe(true)),
                other => Err(plotting_error(
                    builtin,
                    format!("{builtin}: unsupported EdgeColor `{other}`"),
                )),
            }
        }
        "facecolor" | "color" => surface_face_color_update(value, builtin),
        "flattenz" => {
            let flatten_z = value_as_bool(value).ok_or_else(|| {
                plotting_error(builtin, format!("{builtin}: FlattenZ must be logical"))
            })?;
            Ok(SurfacePropertyUpdate::FlattenZ(flatten_z))
        }
        "lighting" => Ok(SurfacePropertyUpdate::Lighting(
            surface_lighting_from_value(value, builtin)?,
        )),
        other => Err(plotting_error(
            builtin,
            format!("{builtin}: unsupported surface property `{other}`"),
        )),
    }
}

fn apply_function_surface_property(
    function_surface: &super::state::FunctionSurfaceHandleState,
    key: &str,
    value: &Value,
    builtin: &'static str,
) -> BuiltinResult<()> {
    match key {
        "facealpha" | "displayname" => {
            let surface_handle = super::state::SimplePlotHandleState {
                figure: function_surface.figure,
                axes_index: function_surface.axes_index,
                plot_index: function_surface.plot_index,
            };
            apply_surface_property(&surface_handle, key, value, builtin)
        }
        "meshdensity" | "xrange" | "yrange" | "function" | "xfunction" | "yfunction"
        | "zfunction" => Err(plotting_error(
            builtin,
            format!("{builtin}: changing {key} after fsurf sampling is not supported yet"),
        )),
        _ => Ok(()),
    }
}

fn apply_function_contour_property(
    function_contour: &super::state::FunctionContourHandleState,
    key: &str,
    value: &Value,
    builtin: &'static str,
) -> BuiltinResult<()> {
    match key {
        "linewidth" | "displayname" => {
            let contour_handle = super::state::SimplePlotHandleState {
                figure: function_contour.figure,
                axes_index: function_contour.axes_index,
                plot_index: function_contour.plot_index,
            };
            apply_contour_property(&contour_handle, key, value, builtin)
        }
        "meshdensity" | "xrange" | "yrange" | "function" | "fill" => Err(plotting_error(
            builtin,
            format!("{builtin}: changing {key} after fcontour sampling is not supported yet"),
        )),
        _ => Ok(()),
    }
}

fn apply_binscatter_property(
    binscatter: &super::state::BinscatterHandleState,
    key: &str,
    value: &Value,
    builtin: &'static str,
) -> BuiltinResult<()> {
    let mut next = binscatter.clone();
    match key {
        "numbins" => {
            next.num_bins = binscatter::parse_num_bins(value)?;
            next.auto_bins = false;
        }
        "showemptybins" => {
            next.show_empty_bins = binscatter::option_bool(value, "ShowEmptyBins")?;
        }
        "xlimits" => {
            next.x_limits_option = Some(binscatter::parse_limits(value, "XLimits")?);
        }
        "ylimits" => {
            next.y_limits_option = Some(binscatter::parse_limits(value, "YLimits")?);
        }
        "xdata" => {
            next.x_data = binscatter_numeric_data(value, "XData", builtin)?;
        }
        "ydata" => {
            next.y_data = binscatter_numeric_data(value, "YData", builtin)?;
        }
        "facealpha" => {
            let alpha = value_as_f64(value).ok_or_else(|| {
                plotting_error(builtin, format!("{builtin}: FaceAlpha must be numeric"))
            })?;
            binscatter::validate_face_alpha(alpha)?;
            next.face_alpha = alpha;
        }
        "displayname" => {
            next.display_name = value_as_string(value).map(|s| s.to_string());
        }
        other => Err(plotting_error(
            builtin,
            format!("{builtin}: unsupported binscatter property `{other}`"),
        ))?,
    }
    recompute_binscatter_chart(next, builtin)
}

fn binscatter_numeric_data(
    value: &Value,
    name: &str,
    builtin: &'static str,
) -> BuiltinResult<Tensor> {
    let tensor = tensor::value_to_tensor(value)
        .map_err(|_| plotting_error(builtin, format!("{builtin}: {name} must be numeric")))?;
    if tensor.shape.iter().filter(|&&dim| dim > 1).count() > 1 {
        return Err(plotting_error(
            builtin,
            format!("{builtin}: {name} must be a real numeric vector"),
        ));
    }
    Ok(tensor)
}

fn recompute_binscatter_chart(
    mut next: super::state::BinscatterHandleState,
    builtin: &'static str,
) -> BuiltinResult<()> {
    if next.x_data.len() != next.y_data.len() {
        return Err(plotting_error(
            builtin,
            format!("{builtin}: XData and YData must contain the same number of elements"),
        ));
    }
    if next.auto_bins {
        next.num_bins = [
            binscatter::auto_bin_count(&next.x_data, next.x_limits_option.as_ref(), 100)?,
            binscatter::auto_bin_count(&next.y_data, next.y_limits_option.as_ref(), 100)?,
        ];
    }
    let color_limits = current_binscatter_color_limits(&next, builtin)?;
    let chart = binscatter::build_binscatter_chart(
        &next.x_data,
        &next.y_data,
        next.num_bins,
        next.x_limits_option.as_ref(),
        next.y_limits_option.as_ref(),
        next.show_empty_bins,
        next.face_alpha,
        next.display_name.as_deref(),
        color_limits,
    )?;
    let x_limits = (
        *chart.x_bin_edges.first().unwrap_or(&0.0),
        *chart.x_bin_edges.last().unwrap_or(&1.0),
    );
    let y_limits = (
        *chart.y_bin_edges.first().unwrap_or(&0.0),
        *chart.y_bin_edges.last().unwrap_or(&1.0),
    );
    let surface = chart.surface.clone();
    super::state::update_image_plot(next.figure, next.plot_index, |existing| {
        *existing = surface;
    })
    .map_err(|err| map_figure_error(builtin, err))?;
    super::state::update_binscatter_handle_for_plot(next.figure, next.plot_index, |state| {
        state.values = chart.values;
        state.x_bin_edges = chart.x_bin_edges;
        state.y_bin_edges = chart.y_bin_edges;
        state.x_data = next.x_data;
        state.y_data = next.y_data;
        state.num_bins = next.num_bins;
        state.auto_bins = next.auto_bins;
        state.x_limits_option = next.x_limits_option;
        state.y_limits_option = next.y_limits_option;
        state.x_limits = x_limits;
        state.y_limits = y_limits;
        state.show_empty_bins = next.show_empty_bins;
        state.face_alpha = next.face_alpha;
        state.display_name = next.display_name;
    })
    .map_err(|err| map_figure_error(builtin, err))?;
    Ok(())
}

fn current_binscatter_color_limits(
    binscatter: &super::state::BinscatterHandleState,
    builtin: &'static str,
) -> BuiltinResult<Option<(f64, f64)>> {
    let figure = super::state::clone_figure(binscatter.figure)
        .ok_or_else(|| plotting_error(builtin, format!("{builtin}: invalid binscatter figure")))?;
    let plot = figure
        .plots()
        .nth(binscatter.plot_index)
        .ok_or_else(|| plotting_error(builtin, format!("{builtin}: invalid binscatter handle")))?;
    let runmat_plot::plots::figure::PlotElement::Surface(surface) = plot else {
        return Err(plotting_error(
            builtin,
            format!("{builtin}: invalid binscatter handle"),
        ));
    };
    Ok(surface.color_limits)
}

fn apply_patch_property(
    patch_handle: &super::state::SimplePlotHandleState,
    key: &str,
    value: &Value,
    builtin: &'static str,
) -> BuiltinResult<()> {
    if !matches!(
        key,
        "facecolor"
            | "color"
            | "edgecolor"
            | "facealpha"
            | "edgealpha"
            | "linewidth"
            | "displayname"
            | "visible"
    ) {
        return Err(plotting_error(
            builtin,
            format!("{builtin}: unsupported patch property `{key}`"),
        ));
    }
    match key {
        "facealpha" | "edgealpha" => {
            let alpha = value_as_f64(value).ok_or_else(|| {
                plotting_error(builtin, format!("{builtin}: {key} must be numeric"))
            })?;
            if !alpha.is_finite() || !(0.0..=1.0).contains(&alpha) {
                return Err(plotting_error(
                    builtin,
                    format!("{builtin}: {key} must be in the range [0, 1]"),
                ));
            }
        }
        "linewidth" => {
            let width = patch_scalar_f64(value).ok_or_else(|| {
                plotting_error(builtin, format!("{builtin}: LineWidth must be numeric"))
            })?;
            if !width.is_finite() || width <= 0.0 {
                return Err(plotting_error(
                    builtin,
                    format!("{builtin}: LineWidth must be a positive finite value"),
                ));
            }
        }
        "visible" => {
            validate_patch_visible(value, builtin)?;
        }
        _ => {}
    }
    super::state::update_plot_element(patch_handle.figure, patch_handle.plot_index, |plot| {
        if let runmat_plot::plots::figure::PlotElement::Patch(patch) = plot {
            match key {
                "facecolor" | "color" => {
                    if let Some(text) = value_as_string(value) {
                        match text.trim().to_ascii_lowercase().as_str() {
                            "none" => patch
                                .set_face_color_mode(runmat_plot::plots::PatchFaceColorMode::None),
                            "flat" => patch
                                .set_face_color_mode(runmat_plot::plots::PatchFaceColorMode::Flat),
                            _ => {
                                if let Ok(c) = parse_color_value(
                                    &LineStyleParseOptions::generic(builtin),
                                    value,
                                ) {
                                    patch.set_face_color(c);
                                    patch.set_face_color_mode(
                                        runmat_plot::plots::PatchFaceColorMode::Color,
                                    );
                                }
                            }
                        }
                    } else if let Ok(c) =
                        parse_color_value(&LineStyleParseOptions::generic(builtin), value)
                    {
                        patch.set_face_color(c);
                        patch.set_face_color_mode(runmat_plot::plots::PatchFaceColorMode::Color);
                    }
                }
                "edgecolor" => {
                    if let Some(text) = value_as_string(value) {
                        if text.trim().eq_ignore_ascii_case("none") {
                            patch.set_edge_color_mode(runmat_plot::plots::PatchEdgeColorMode::None);
                        } else if let Ok(c) =
                            parse_color_value(&LineStyleParseOptions::generic(builtin), value)
                        {
                            patch.set_edge_color(c);
                            patch
                                .set_edge_color_mode(runmat_plot::plots::PatchEdgeColorMode::Color);
                        }
                    } else if let Ok(c) =
                        parse_color_value(&LineStyleParseOptions::generic(builtin), value)
                    {
                        patch.set_edge_color(c);
                        patch.set_edge_color_mode(runmat_plot::plots::PatchEdgeColorMode::Color);
                    }
                }
                "facealpha" => {
                    if let Some(v) = value_as_f64(value) {
                        patch.set_face_alpha(v as f32);
                    }
                }
                "edgealpha" => {
                    if let Some(v) = value_as_f64(value) {
                        patch.set_edge_alpha(v as f32);
                    }
                }
                "linewidth" => {
                    if let Some(v) = value_as_f64(value) {
                        patch.set_line_width(v as f32);
                    }
                }
                "displayname" => {
                    patch.set_label(value_as_string(value).map(|s| s.to_string()));
                }
                "visible" => {
                    if let Some(v) = value_as_bool(value) {
                        patch.set_visible(v);
                    }
                }
                _ => {}
            }
        }
    })
    .map_err(|err| map_figure_error(builtin, err))?;
    Ok(())
}

fn validate_patch_visible(value: &Value, builtin: &'static str) -> BuiltinResult<()> {
    if let Some(text) = value_as_string(value) {
        if matches!(text.trim().to_ascii_lowercase().as_str(), "on" | "off") {
            return Ok(());
        }
    } else if matches!(value, Value::Bool(_)) {
        return Ok(());
    } else if let Some(value) = patch_scalar_f64(value) {
        if value == 0.0 || value == 1.0 {
            return Ok(());
        }
    }
    Err(plotting_error(
        builtin,
        format!("{builtin}: Visible must be on/off or numeric/logical 0 or 1"),
    ))
}

fn patch_scalar_f64(value: &Value) -> Option<f64> {
    match value {
        Value::Tensor(tensor) if tensor.len() != 1 => None,
        _ => value_as_f64(value),
    }
}

fn apply_line3_property(
    line_handle: &super::state::SimplePlotHandleState,
    key: &str,
    value: &Value,
    builtin: &'static str,
) -> BuiltinResult<()> {
    apply_line3_properties(line_handle, &[(key.to_string(), value)], builtin)
}

fn apply_line3_properties(
    line_handle: &super::state::SimplePlotHandleState,
    pairs: &[(String, &Value)],
    builtin: &'static str,
) -> BuiltinResult<()> {
    let current = get_simple_plot(line_handle, builtin)?;
    let runmat_plot::plots::figure::PlotElement::Line3(current_line) = current else {
        return Err(plotting_error(
            builtin,
            format!("{builtin}: invalid line handle"),
        ));
    };

    let (current_x, current_y, current_z) = line_xyz_data_for_properties(&current_line, builtin)?;
    let mut x_update = None;
    let mut y_update = None;
    let mut z_update = None;
    let mut style_pairs = Vec::new();
    for (key, value) in pairs {
        match key.as_str() {
            "xdata" => x_update = Some(line_numeric_data(value, "XData", builtin)?),
            "ydata" => y_update = Some(line_numeric_data(value, "YData", builtin)?),
            "zdata" => z_update = Some(line_numeric_data(value, "ZData", builtin)?),
            _ => style_pairs.push((key.as_str(), *value)),
        }
    }

    let next_x = x_update.unwrap_or_else(|| current_x.clone());
    let next_y = y_update.unwrap_or_else(|| current_y.clone());
    let next_z = z_update.unwrap_or_else(|| current_z.clone());
    let update_data = next_x != current_x || next_y != current_y || next_z != current_z;
    if update_data
        && (next_x.is_empty() || next_x.len() != next_y.len() || next_x.len() != next_z.len())
    {
        return Err(plotting_error(
            builtin,
            format!("{builtin}: XData, YData, and ZData must contain the same nonzero number of elements"),
        ));
    }

    super::state::update_plot_element(line_handle.figure, line_handle.plot_index, |plot| {
        if let runmat_plot::plots::figure::PlotElement::Line3(line) = plot {
            if update_data {
                let _ = line.update_numeric_data(next_x.clone(), next_y.clone(), next_z.clone());
            }
            for (key, value) in &style_pairs {
                match *key {
                    "color" => {
                        if let Ok(c) =
                            parse_color_value(&LineStyleParseOptions::generic(builtin), value)
                        {
                            line.color = c;
                        }
                    }
                    "linewidth" => {
                        if let Some(v) = value_as_f64(value) {
                            line.line_width = v as f32;
                        }
                    }
                    "linestyle" => {
                        if let Some(s) = value_as_string(value) {
                            line.line_style = parse_line_style_name_for_props(&s);
                        }
                    }
                    "displayname" => {
                        line.label = value_as_string(value).map(|s| s.to_string());
                    }
                    "visible" => {
                        if let Some(v) = value_as_bool(value) {
                            line.set_visible(v);
                        } else if let Some(s) = value_as_string(value) {
                            line.set_visible(!matches!(
                                s.trim().to_ascii_lowercase().as_str(),
                                "off" | "false"
                            ));
                        }
                    }
                    _ => {}
                }
            }
        }
    })
    .map_err(|err| map_figure_error(builtin, err))?;
    Ok(())
}

fn apply_scatter3_property(
    scatter_handle: &super::state::SimplePlotHandleState,
    key: &str,
    value: &Value,
    builtin: &'static str,
) -> BuiltinResult<()> {
    super::state::update_plot_element(scatter_handle.figure, scatter_handle.plot_index, |plot| {
        if let runmat_plot::plots::figure::PlotElement::Scatter3(scatter) = plot {
            match key {
                "sizedata" => {
                    if let Some(v) = value_as_f64(value) {
                        scatter.point_size = marker_area_points2_to_diameter_px(v);
                    }
                }
                "marker" => {
                    if let Some(s) = value_as_string(value) {
                        scatter.marker_style = scatter_marker_from_name(&s, scatter.marker_style);
                    }
                }
                "markerfacecolor" => {
                    if let Ok(c) =
                        parse_color_value(&LineStyleParseOptions::generic(builtin), value)
                    {
                        let count = scatter.points.len().max(1);
                        scatter.colors = vec![c; count];
                    }
                }
                "markeredgecolor" => {
                    if let Ok(c) =
                        parse_color_value(&LineStyleParseOptions::generic(builtin), value)
                    {
                        scatter.edge_color = c;
                    }
                }
                "linewidth" => {
                    if let Some(v) = value_as_f64(value) {
                        scatter.edge_thickness = v as f32;
                    }
                }
                "displayname" => {
                    scatter.label = value_as_string(value).map(|s| s.to_string());
                }
                _ => {}
            }
        }
    })
    .map_err(|err| map_figure_error(builtin, err))?;
    Ok(())
}

fn apply_pie_property(
    pie_handle: &super::state::SimplePlotHandleState,
    key: &str,
    value: &Value,
    builtin: &'static str,
) -> BuiltinResult<()> {
    super::state::update_plot_element(pie_handle.figure, pie_handle.plot_index, |plot| {
        if let runmat_plot::plots::figure::PlotElement::Pie(pie) = plot {
            if key == "displayname" {
                pie.label = value_as_string(value).map(|s| s.to_string());
            }
        }
    })
    .map_err(|err| map_figure_error(builtin, err))?;
    Ok(())
}

fn apply_contour_property(
    contour_handle: &super::state::SimplePlotHandleState,
    key: &str,
    value: &Value,
    builtin: &'static str,
) -> BuiltinResult<()> {
    super::state::update_plot_element(contour_handle.figure, contour_handle.plot_index, |plot| {
        if let runmat_plot::plots::figure::PlotElement::Contour(contour) = plot {
            if key == "displayname" {
                contour.label = value_as_string(value).map(|s| s.to_string());
            } else if key == "linewidth" {
                if let Some(width) = value_as_f64(value) {
                    contour.line_width = (width as f32).max(0.5);
                }
            }
        }
    })
    .map_err(|err| map_figure_error(builtin, err))?;
    Ok(())
}

fn apply_contour_fill_property(
    fill_handle: &super::state::SimplePlotHandleState,
    key: &str,
    value: &Value,
    builtin: &'static str,
) -> BuiltinResult<()> {
    super::state::update_plot_element(fill_handle.figure, fill_handle.plot_index, |plot| {
        if let runmat_plot::plots::figure::PlotElement::ContourFill(fill) = plot {
            if key == "displayname" {
                fill.label = value_as_string(value).map(|s| s.to_string());
            }
        }
    })
    .map_err(|err| map_figure_error(builtin, err))?;
    Ok(())
}

fn apply_line_plot_properties(
    line: &mut runmat_plot::plots::LinePlot,
    key: &str,
    value: &Value,
    builtin: &'static str,
) {
    match key {
        "color" => {
            if let Ok(c) = parse_color_value(&LineStyleParseOptions::generic(builtin), value) {
                line.color = c;
            }
        }
        "linewidth" => {
            if let Some(v) = value_as_f64(value) {
                line.line_width = v as f32;
            }
        }
        "linestyle" => {
            if let Some(s) = value_as_string(value) {
                line.line_style = parse_line_style_name_for_props(&s);
            }
        }
        "displayname" => {
            line.label = value_as_string(value).map(|s| s.to_string());
        }
        "handlevisibility" => {
            if let Ok(visibility) =
                parse_handle_visibility(&LineStyleParseOptions::generic(builtin), value)
            {
                line.handle_visibility = visibility;
            }
        }
        "marker" => {
            if let Some(s) = value_as_string(value) {
                line.marker = marker_from_name(&s, line.marker.clone());
            }
        }
        "markersize" => {
            if let Some(v) = value_as_f64(value) {
                if let Some(marker) = &mut line.marker {
                    marker.size = v as f32;
                }
            }
        }
        "markerfacecolor" => {
            if let Ok(c) = parse_color_value(&LineStyleParseOptions::generic(builtin), value) {
                if let Some(marker) = &mut line.marker {
                    marker.face_color = c;
                }
            }
        }
        "markeredgecolor" => {
            if let Ok(c) = parse_color_value(&LineStyleParseOptions::generic(builtin), value) {
                if let Some(marker) = &mut line.marker {
                    marker.edge_color = c;
                }
            }
        }
        "filled" => {
            if let Some(v) = value_as_bool(value) {
                if let Some(marker) = &mut line.marker {
                    marker.filled = v;
                }
            }
        }
        "visible" => {
            if let Some(v) = value_as_bool(value) {
                line.set_visible(v);
            } else if let Some(s) = value_as_string(value) {
                line.set_visible(!matches!(
                    s.trim().to_ascii_lowercase().as_str(),
                    "off" | "false"
                ));
            }
        }
        _ => {}
    }
}

fn line_numeric_data(
    value: &Value,
    name: &str,
    builtin: &'static str,
) -> BuiltinResult<NumericPlotData> {
    let tensor = tensor::value_to_tensor(value)
        .map_err(|_| plotting_error(builtin, format!("{builtin}: {name} must be numeric")))?;
    let shape = tensor.shape.clone();
    let storage = tensor
        .into_numeric_storage()
        .map_err(|err| plotting_error(builtin, format!("{builtin}: {name}: {err}")))?;
    NumericPlotData::new(storage, shape)
        .map_err(|err| plotting_error(builtin, format!("{builtin}: {name}: {err}")))
}

fn line_xy_data_for_properties(
    line: &runmat_plot::plots::LinePlot,
    builtin: &'static str,
) -> BuiltinResult<(NumericPlotData, NumericPlotData)> {
    if let (Some(x), Some(y)) = line.source_data() {
        return Ok((x.clone(), y.clone()));
    }
    futures::executor::block_on(line.export_numeric_xy_data())
        .map_err(|err| {
            plotting_error(
                builtin,
                format!("{builtin}: unable to read line source data: {err}"),
            )
        })?
        .ok_or_else(|| plotting_error(builtin, format!("{builtin}: line has no source data")))
}

fn line_xyz_data_for_properties(
    line: &runmat_plot::plots::Line3Plot,
    builtin: &'static str,
) -> BuiltinResult<(NumericPlotData, NumericPlotData, NumericPlotData)> {
    if let (Some(x), Some(y), Some(z)) = line.source_data() {
        return Ok((x.clone(), y.clone(), z.clone()));
    }
    futures::executor::block_on(line.export_numeric_xyz_data())
        .map_err(|err| {
            plotting_error(
                builtin,
                format!("{builtin}: unable to read plot3 source data: {err}"),
            )
        })?
        .ok_or_else(|| plotting_error(builtin, format!("{builtin}: plot3 has no source data")))
}

fn scatter_xy_data_for_properties(
    scatter: &runmat_plot::plots::ScatterPlot,
    builtin: &'static str,
) -> BuiltinResult<(NumericPlotData, NumericPlotData)> {
    if let (Some(x), Some(y)) = scatter.source_data() {
        return Ok((x.clone(), y.clone()));
    }
    futures::executor::block_on(scatter.export_numeric_xy_data())
        .map_err(|err| {
            plotting_error(
                builtin,
                format!("{builtin}: unable to read scatter source data: {err}"),
            )
        })?
        .ok_or_else(|| plotting_error(builtin, format!("{builtin}: scatter has no source data")))
}

fn numeric_plot_data_value(data: &NumericPlotData) -> Value {
    Value::Tensor(
        Tensor::from_numeric_storage(data.storage().clone(), data.shape().to_vec())
            .expect("validated graphics source data"),
    )
}

fn limits_from_optional_value(
    value: &Value,
    builtin: &'static str,
) -> BuiltinResult<Option<(f64, f64)>> {
    if let Some(text) = value_as_string(value) {
        let norm = text.trim().to_ascii_lowercase();
        if matches!(norm.as_str(), "auto" | "tight") {
            return Ok(None);
        }
    }
    Ok(Some(
        crate::builtins::plotting::op_common::limits::limits_from_value(value, builtin)?,
    ))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TickMode {
    Auto,
    Manual,
}

fn ticks_from_value(value: &Value, builtin: &'static str) -> BuiltinResult<Vec<f64>> {
    let tensor = runmat_builtins::Tensor::try_from(value)
        .map_err(|e| plotting_error(builtin, format!("{builtin}: {e}")))?;
    let ticks = tensor::tensor_values_f64(&tensor);
    if ticks.iter().any(|value| !value.is_finite()) {
        return Err(plotting_error(
            builtin,
            format!("{builtin}: tick values must be finite"),
        ));
    }
    if ticks.windows(2).any(|pair| pair[1] <= pair[0]) {
        return Err(plotting_error(
            builtin,
            format!("{builtin}: tick values must be strictly increasing"),
        ));
    }
    Ok(ticks)
}

fn tick_mode_from_value(
    value: &Value,
    builtin: &'static str,
    property: &str,
) -> BuiltinResult<TickMode> {
    let mode = value_as_string(value).ok_or_else(|| {
        plotting_error(builtin, format!("{builtin}: {property} must be a string"))
    })?;
    match mode.trim().to_ascii_lowercase().as_str() {
        "auto" => Ok(TickMode::Auto),
        "manual" => Ok(TickMode::Manual),
        _ => Err(plotting_error(
            builtin,
            format!("{builtin}: {property} must be 'auto' or 'manual'"),
        )),
    }
}

fn tick_format_from_value(
    value: &Value,
    builtin: &'static str,
    property: &'static str,
) -> BuiltinResult<String> {
    let format = value_as_string(value)
        .ok_or_else(|| plotting_error(builtin, format!("{builtin}: {property} must be text")))?;
    Ok(runmat_plot::core::plot_renderer::plot_utils::canonical_tick_label_format(&format))
}

fn tick_angle_from_value(
    value: &Value,
    builtin: &'static str,
    property: &'static str,
) -> BuiltinResult<f64> {
    let angle = scalar_numeric_value(value)
        .ok_or_else(|| plotting_error(builtin, format!("{builtin}: {property} must be numeric")))?;
    if angle.is_finite() {
        Ok(angle)
    } else {
        Err(plotting_error(
            builtin,
            format!("{builtin}: {property} must be finite"),
        ))
    }
}

fn scalar_numeric_value(value: &Value) -> Option<f64> {
    match value {
        Value::Num(value) => Some(*value),
        Value::Int(value) => Some(value.to_f64()),
        Value::Tensor(tensor) if tensor::is_scalar_tensor(tensor) => {
            Some(tensor::tensor_values_f64(tensor)[0])
        }
        _ => None,
    }
}

fn tick_format_or_default(format: Option<&str>) -> String {
    runmat_plot::core::plot_renderer::plot_utils::canonical_tick_label_format(
        format.unwrap_or("%g"),
    )
}

fn x_bounds(bounds: Option<(f64, f64, f64, f64)>) -> Option<(f64, f64)> {
    bounds.map(|(x_min, x_max, _, _)| (x_min, x_max))
}

fn y_bounds(bounds: Option<(f64, f64, f64, f64)>) -> Option<(f64, f64)> {
    bounds.map(|(_, _, y_min, y_max)| (y_min, y_max))
}

fn ticks_or_auto(explicit: Option<&[f64]>, bounds: Option<(f64, f64)>) -> Vec<f64> {
    if let Some(ticks) = explicit {
        return ticks.to_vec();
    }
    let (lo, hi) = bounds.unwrap_or((-1.0, 1.0));
    runmat_plot::core::plot_utils::generate_major_ticks(lo, hi)
}

fn tick_value(data: Vec<f64>) -> Value {
    tensor_from_vec(data)
}

fn tick_mode_value<T>(ticks: Option<&Vec<T>>) -> Value {
    Value::String(if ticks.is_some() { "manual" } else { "auto" }.into())
}

fn tick_labels_from_value(value: &Value, builtin: &'static str) -> BuiltinResult<Vec<String>> {
    match value {
        Value::String(s) => Ok(vec![s.clone()]),
        Value::StringArray(strings) => Ok(strings.data.clone()),
        Value::CharArray(chars) => Ok(char_array_rows(chars)),
        Value::Cell(cell) => {
            let mut labels = Vec::with_capacity(cell.data.len());
            for entry in &cell.data {
                labels.extend(tick_labels_from_value(entry, builtin)?);
            }
            Ok(labels)
        }
        Value::Tensor(tensor) if tensor::tensor_element_len(tensor) == 0 => Ok(Vec::new()),
        other => Err(plotting_error(
            builtin,
            format!("{builtin}: tick labels must be a string array or cell array of text, got {other:?}"),
        )),
    }
}

fn tick_labels_or_auto(
    explicit: Option<&[String]>,
    ticks: Option<&[f64]>,
    bounds: Option<(f64, f64)>,
    format: Option<&str>,
) -> Vec<String> {
    if let Some(labels) = explicit {
        return labels.to_vec();
    }
    let formatter = runmat_plot::core::plot_utils::TickLabelFormatter::new(format);
    ticks_or_auto(ticks, bounds)
        .into_iter()
        .map(|tick| formatter.format(tick))
        .collect()
}

fn labels_padded_to_ticks(mut labels: Vec<String>, tick_count: usize) -> Vec<String> {
    if labels.len() < tick_count {
        labels.resize(tick_count, String::new());
    }
    labels
}

fn tick_label_value(labels: Vec<String>) -> Value {
    let values = labels
        .iter()
        .map(|label| Value::CharArray(CharArray::new_row(label)))
        .collect::<Vec<_>>();
    Value::Cell(CellArray::new(values, 1, labels.len()).expect("valid tick-label cell array"))
}

fn char_array_rows(chars: &CharArray) -> Vec<String> {
    if chars.rows == 0 || chars.cols == 0 {
        return Vec::new();
    }
    (0..chars.rows)
        .map(|row| {
            let start = row * chars.cols;
            let end = start + chars.cols;
            chars.data[start..end]
                .iter()
                .collect::<String>()
                .trim_end()
                .to_string()
        })
        .collect()
}

fn parse_colormap_name(
    name: &str,
    builtin: &'static str,
) -> BuiltinResult<runmat_plot::plots::surface::ColorMap> {
    runmat_plot::plots::surface::ColorMap::from_name(name).ok_or_else(|| {
        plotting_error(
            builtin,
            format!("{builtin}: unknown colormap '{}'", name.trim()),
        )
    })
}

fn apply_axes_text_alias(
    handle: FigureHandle,
    axes_index: usize,
    kind: PlotObjectKind,
    value: &Value,
    builtin: &'static str,
) -> BuiltinResult<()> {
    if let Some(text) = value_as_string(value) {
        set_text_properties_for_axes(handle, axes_index, kind, Some(text), None)
            .map_err(|err| map_figure_error(builtin, err))?;
        return Ok(());
    }

    let scalar = handle_scalar(value, builtin)?;
    let (src_handle, src_axes, src_kind) =
        decode_plot_object_handle(scalar).map_err(|err| map_figure_error(builtin, err))?;
    if src_kind != kind {
        return Err(plotting_error(
            builtin,
            format!(
                "{builtin}: expected a matching text handle for `{}`",
                key_name(kind)
            ),
        ));
    }
    let meta = axes_metadata_snapshot(src_handle, src_axes)
        .map_err(|err| map_figure_error(builtin, err))?;
    let (text, style) = match kind {
        PlotObjectKind::Title => (meta.title, meta.title_style),
        PlotObjectKind::Subtitle => (meta.subtitle, meta.subtitle_style),
        PlotObjectKind::XLabel => (meta.x_label, meta.x_label_style),
        PlotObjectKind::YLabel => (meta.y_label, meta.y_label_style),
        PlotObjectKind::ZLabel => (meta.z_label, meta.z_label_style),
        PlotObjectKind::Legend
        | PlotObjectKind::SuperTitle
        | PlotObjectKind::XAxis
        | PlotObjectKind::YAxis => unreachable!(),
    };
    set_text_properties_for_axes(handle, axes_index, kind, text, Some(style))
        .map_err(|err| map_figure_error(builtin, err))?;
    Ok(())
}

fn validate_axes_text_alias(
    kind: PlotObjectKind,
    value: &Value,
    builtin: &'static str,
) -> BuiltinResult<()> {
    if value_as_string(value).is_some() {
        return Ok(());
    }

    let scalar = handle_scalar(value, builtin)?;
    let (src_handle, src_axes, src_kind) =
        decode_plot_object_handle(scalar).map_err(|err| map_figure_error(builtin, err))?;
    if src_kind != kind {
        return Err(plotting_error(
            builtin,
            format!(
                "{builtin}: expected a matching text handle for `{}`",
                key_name(kind)
            ),
        ));
    }
    axes_metadata_snapshot(src_handle, src_axes).map_err(|err| map_figure_error(builtin, err))?;
    Ok(())
}

fn apply_figure_text_alias(
    handle: FigureHandle,
    kind: PlotObjectKind,
    value: &Value,
    builtin: &'static str,
) -> BuiltinResult<()> {
    if let Some(text) = value_as_text_string(value) {
        match kind {
            PlotObjectKind::SuperTitle => {
                set_sg_title_properties_for_figure(handle, Some(text), None)
                    .map_err(|err| map_figure_error(builtin, err))?;
            }
            _ => unreachable!(),
        }
        return Ok(());
    }

    let scalar = handle_scalar(value, builtin)?;
    let (src_handle, _src_axes, src_kind) =
        decode_plot_object_handle(scalar).map_err(|err| map_figure_error(builtin, err))?;
    if src_kind != kind {
        return Err(plotting_error(
            builtin,
            format!(
                "{builtin}: expected a matching text handle for `{}`",
                key_name(kind)
            ),
        ));
    }

    let figure = super::state::clone_figure(src_handle)
        .ok_or_else(|| plotting_error(builtin, format!("{builtin}: invalid figure handle")))?;
    let (text, style) = match kind {
        PlotObjectKind::SuperTitle => (figure.sg_title, figure.sg_title_style),
        _ => unreachable!(),
    };
    set_sg_title_properties_for_figure(handle, text, Some(style))
        .map_err(|err| map_figure_error(builtin, err))?;
    Ok(())
}

fn validate_figure_text_alias(
    kind: PlotObjectKind,
    value: &Value,
    builtin: &'static str,
) -> BuiltinResult<()> {
    if value_as_text_string(value).is_some() {
        return Ok(());
    }

    let scalar = handle_scalar(value, builtin)?;
    let (src_handle, _src_axes, src_kind) =
        decode_plot_object_handle(scalar).map_err(|err| map_figure_error(builtin, err))?;
    if src_kind != kind {
        return Err(plotting_error(
            builtin,
            format!(
                "{builtin}: expected a matching text handle for `{}`",
                key_name(kind)
            ),
        ));
    }

    super::state::clone_figure(src_handle)
        .ok_or_else(|| plotting_error(builtin, format!("{builtin}: invalid figure handle")))?;
    Ok(())
}

fn collect_label_strings(builtin: &'static str, args: &[Value]) -> BuiltinResult<Vec<String>> {
    let mut labels = Vec::new();
    for arg in args {
        match arg {
            Value::StringArray(arr) => labels.extend(arr.data.iter().cloned()),
            Value::Cell(cell) => {
                for row in 0..cell.rows {
                    for col in 0..cell.cols {
                        let value = cell.get(row, col).map_err(|err| {
                            plotting_error(builtin, format!("legend: invalid label cell: {err}"))
                        })?;
                        labels.push(value_as_string(&value).ok_or_else(|| {
                            plotting_error(builtin, "legend: labels must be strings or char arrays")
                        })?);
                    }
                }
            }
            _ => labels.push(value_as_string(arg).ok_or_else(|| {
                plotting_error(builtin, "legend: labels must be strings or char arrays")
            })?),
        }
    }
    Ok(labels)
}

fn handle_scalar(value: &Value, builtin: &'static str) -> BuiltinResult<f64> {
    match value {
        Value::Num(v) => Ok(*v),
        Value::Int(i) => Ok(i.to_f64()),
        Value::Tensor(t) if tensor::is_scalar_tensor(t) => Ok(tensor::tensor_values_f64(t)[0]),
        _ => Err(plotting_error(
            builtin,
            format!("{builtin}: expected plotting handle"),
        )),
    }
}

fn legend_labels_value(labels: Vec<String>) -> Value {
    Value::StringArray(StringArray {
        rows: 1,
        cols: labels.len().max(1),
        shape: vec![1, labels.len().max(1)],
        data: labels,
    })
}

fn text_value(text: Option<String>) -> Value {
    match text {
        Some(text) if text.contains('\n') => {
            let lines: Vec<String> = text.split('\n').map(|s| s.to_string()).collect();
            Value::StringArray(StringArray {
                rows: 1,
                cols: lines.len().max(1),
                shape: vec![1, lines.len().max(1)],
                data: lines,
            })
        }
        Some(text) => Value::String(text),
        None => Value::String(String::new()),
    }
}

fn handles_value(handles: Vec<f64>) -> Value {
    let len = handles.len();
    Value::Tensor(Tensor::new(handles, vec![1, len]).expect("handle row vector"))
}

fn tensor_from_vec(data: Vec<f64>) -> Value {
    let len = data.len();
    Value::Tensor(Tensor::new(data, vec![1, len]).expect("property row vector"))
}

fn string_array_from_vec(data: Vec<String>) -> BuiltinResult<Value> {
    let cols = data.len();
    let array = StringArray::new(data, vec![1, cols])
        .map_err(|e| plotting_error("get", format!("get: {e}")))?;
    Ok(Value::StringArray(array))
}

pub(crate) fn label_strings_from_value(
    value: &Value,
    builtin: &'static str,
    label_context: &str,
) -> BuiltinResult<Vec<String>> {
    match value {
        Value::StringArray(array) => Ok(array.data.clone()),
        Value::Cell(cell) => cell
            .data
            .iter()
            .map(|item| {
                value_as_text_string(item).ok_or_else(|| {
                    plotting_error(
                        builtin,
                        format!("{builtin}: {label_context} must contain text values"),
                    )
                })
            })
            .collect(),
        Value::CharArray(chars) if chars.rows == 1 => Ok(vec![chars.data.iter().collect()]),
        Value::String(text) => Ok(vec![text.clone()]),
        Value::Tensor(tensor) => Ok((0..tensor.len())
            .map(|index| {
                format_numeric_label(
                    tensor
                        .numeric_value_at(index)
                        .expect("label index is within numeric storage"),
                )
            })
            .collect()),
        Value::Int(i) => Ok(vec![i.decimal_string()]),
        Value::Num(v) => Ok(vec![v.to_string()]),
        other => Err(plotting_error(
            builtin,
            format!("{builtin}: unsupported {label_context} value {other:?}"),
        )),
    }
}

fn format_numeric_label(value: NumericScalar) -> String {
    match value {
        NumericScalar::F64(value) => value.to_string(),
        NumericScalar::F32(value) => value.to_string(),
        NumericScalar::I8(value) => value.to_string(),
        NumericScalar::I16(value) => value.to_string(),
        NumericScalar::I32(value) => value.to_string(),
        NumericScalar::I64(value) => value.to_string(),
        NumericScalar::U8(value) => value.to_string(),
        NumericScalar::U16(value) => value.to_string(),
        NumericScalar::U32(value) => value.to_string(),
        NumericScalar::U64(value) => value.to_string(),
    }
}

fn heatmap_on_off_value(value: &Value) -> Option<bool> {
    match value {
        Value::Bool(value) => Some(*value),
        Value::CharArray(chars) => {
            let text: String = chars.data.iter().collect();
            heatmap_on_off_text(&text)
        }
        Value::String(text) => heatmap_on_off_text(text),
        Value::Num(value) => numeric_on_off(*value),
        Value::Int(value) => numeric_on_off(value.to_f64()),
        Value::Tensor(tensor) if tensor::is_scalar_tensor(tensor) => tensor
            .numeric_value_at(0)
            .and_then(|value| numeric_on_off(value.materialize_f64())),
        _ => None,
    }
}

fn heatmap_numeric_scalar_f64(value: &Value) -> Option<f64> {
    match value {
        Value::Num(value) => Some(*value),
        Value::Int(value) => Some(value.to_f64()),
        Value::Tensor(tensor) if tensor::is_scalar_tensor(tensor) => tensor
            .numeric_value_at(0)
            .map(NumericScalar::materialize_f64),
        _ => None,
    }
}

fn heatmap_on_off_text(text: &str) -> Option<bool> {
    match text.trim().to_ascii_lowercase().as_str() {
        "on" => Some(true),
        "off" => Some(false),
        _ => None,
    }
}

fn numeric_on_off(value: f64) -> Option<bool> {
    if value == 0.0 {
        Some(false)
    } else if value == 1.0 {
        Some(true)
    } else {
        None
    }
}

fn tensor_from_matrix(data: Vec<Vec<f64>>) -> Value {
    let rows = data.len();
    let cols = data.first().map(|row| row.len()).unwrap_or(0);
    let flat = data.into_iter().flat_map(|row| row.into_iter()).collect();
    Value::Tensor(Tensor::new(flat, vec![rows, cols]).expect("property matrix"))
}

fn surface_x_data_value(surface: &runmat_plot::plots::SurfacePlot) -> Value {
    surface
        .x_grid
        .as_ref()
        .map(|grid| surface_grid_to_tensor(grid))
        .unwrap_or_else(|| tensor_from_vec(surface.x_data.clone()))
}

fn surface_y_data_value(surface: &runmat_plot::plots::SurfacePlot) -> Value {
    surface
        .y_grid
        .as_ref()
        .map(|grid| surface_grid_to_tensor(grid))
        .unwrap_or_else(|| tensor_from_vec(surface.y_data.clone()))
}

fn surface_z_data_value(surface: &runmat_plot::plots::SurfacePlot) -> Value {
    surface
        .z_data
        .as_ref()
        .map(|grid| surface_grid_to_tensor(grid))
        .unwrap_or_else(|| tensor_from_vec(Vec::new()))
}

fn surface_grid_to_tensor(grid: &[Vec<f64>]) -> Value {
    let cols = grid.len();
    let rows = grid.first().map_or(0, Vec::len);
    let mut data = Vec::with_capacity(rows * cols);
    for column in grid {
        data.extend(column.iter().copied());
    }
    Value::Tensor(Tensor::new(data, vec![rows, cols]).expect("surface grid"))
}

fn surface_coordinate_data_from_value(
    value: &Value,
    name: &str,
    builtin: &'static str,
) -> BuiltinResult<SurfaceCoordinateData> {
    let tensor = Tensor::try_from(value)
        .map_err(|_| plotting_error(builtin, format!("{builtin}: {name} must be numeric")))?;
    let data = tensor::tensor_values_f64(&tensor);
    if data.is_empty() {
        return Err(plotting_error(
            builtin,
            format!("{builtin}: {name} must be non-empty"),
        ));
    }
    if tensor.shape.len() > 2 {
        return Err(plotting_error(
            builtin,
            format!("{builtin}: {name} must be a vector or 2-D matrix"),
        ));
    }
    if data.iter().any(|value| !value.is_finite()) {
        return Err(plotting_error(
            builtin,
            format!("{builtin}: {name} must contain finite coordinates"),
        ));
    }
    if tensor.rows == 1 || tensor.cols == 1 {
        return Ok(SurfaceCoordinateData::Vector(data));
    }
    let rows = tensor.rows;
    let cols = tensor.cols;
    let grid = super::common::tensor_to_surface_grid_matlab_xy(tensor, rows, cols, builtin)?;
    Ok(SurfaceCoordinateData::Grid(grid))
}

fn surface_z_grid_from_value(
    value: &Value,
    rows: usize,
    cols: usize,
    builtin: &'static str,
) -> BuiltinResult<Vec<Vec<f64>>> {
    let mut tensor = Tensor::try_from(value)
        .map_err(|_| plotting_error(builtin, format!("{builtin}: ZData must be numeric")))?;
    let expected = rows
        .checked_mul(cols)
        .ok_or_else(|| plotting_error(builtin, format!("{builtin}: grid dimensions overflowed")))?;
    if tensor::tensor_element_len(&tensor) != expected {
        return Err(plotting_error(
            builtin,
            format!("{builtin}: ZData must contain exactly {expected} values"),
        ));
    }
    tensor = tensor::integer_tensor_to_f64(tensor)
        .map_err(|err| plotting_error(builtin, format!("{builtin}: {err}")))?;
    if tensor.rows == rows && tensor.cols == cols {
        return super::common::tensor_to_surface_grid_matlab_xy(tensor, rows, cols, builtin);
    }
    if tensor.rows == 1 || tensor.cols == 1 {
        tensor.rows = rows;
        tensor.cols = cols;
        tensor.shape = vec![rows, cols];
        return super::common::tensor_to_surface_grid_matlab_xy(tensor, rows, cols, builtin);
    }
    Err(plotting_error(
        builtin,
        format!("{builtin}: ZData must have shape {rows}x{cols}"),
    ))
}

fn surface_grid_size(grid: &[Vec<f64>]) -> BuiltinResult<(usize, usize)> {
    let x_len = grid.len();
    let y_len = grid.first().map_or(0, Vec::len);
    if x_len == 0 || y_len == 0 || grid.iter().any(|row| row.len() != y_len) {
        return Err(plotting_error(
            "set",
            "set: surface source data is internally inconsistent",
        ));
    }
    Ok((x_len, y_len))
}

fn validate_surface_grid_shape(
    grid: &[Vec<f64>],
    x_len: usize,
    y_len: usize,
    name: &str,
    builtin: &'static str,
) -> BuiltinResult<()> {
    if grid.len() != x_len || grid.iter().any(|row| row.len() != y_len) {
        return Err(plotting_error(
            builtin,
            format!("{builtin}: {name} matrix must have shape {y_len}x{x_len}"),
        ));
    }
    Ok(())
}

fn axis_to_x_grid(axis: &[f64], y_len: usize) -> Vec<Vec<f64>> {
    axis.iter().map(|&x| vec![x; y_len]).collect()
}

fn axis_to_y_grid(axis: &[f64], x_len: usize) -> Vec<Vec<f64>> {
    vec![axis.to_vec(); x_len]
}

fn surface_face_color_value(surface: &runmat_plot::plots::SurfacePlot) -> Value {
    match &surface.colormap {
        ColorMap::Custom(min, max) if (*min - *max).abs().max_element() < 1e-6 => {
            Value::String(color_to_short_name(*min))
        }
        _ => Value::String("flat".into()),
    }
}

fn surface_edge_color_value(surface: &runmat_plot::plots::SurfacePlot) -> Value {
    Value::String(if surface.wireframe { "flat" } else { "none" }.into())
}

fn surface_shading_name(shading: ShadingMode) -> &'static str {
    match shading {
        ShadingMode::Flat => "flat",
        ShadingMode::Smooth => "interp",
        ShadingMode::Faceted => "faceted",
        ShadingMode::None => "none",
    }
}

fn surface_shading_from_value(value: &Value, builtin: &'static str) -> BuiltinResult<ShadingMode> {
    let text = value_as_string(value)
        .ok_or_else(|| plotting_error(builtin, format!("{builtin}: Shading must be text")))?;
    match text.trim().to_ascii_lowercase().as_str() {
        "flat" => Ok(ShadingMode::Flat),
        "interp" | "gouraud" => Ok(ShadingMode::Smooth),
        "faceted" => Ok(ShadingMode::Faceted),
        "none" => Ok(ShadingMode::None),
        other => Err(plotting_error(
            builtin,
            format!("{builtin}: unsupported Shading `{other}`"),
        )),
    }
}

fn surface_lighting_name(enabled: bool) -> &'static str {
    if enabled {
        "gouraud"
    } else {
        "none"
    }
}

fn surface_lighting_from_value(value: &Value, builtin: &'static str) -> BuiltinResult<bool> {
    if let Some(text) = value_as_string(value) {
        return match text.trim().to_ascii_lowercase().as_str() {
            "none" | "off" => Ok(false),
            "flat" | "gouraud" | "phong" | "auto" | "on" => Ok(true),
            other => Err(plotting_error(
                builtin,
                format!("{builtin}: unsupported Lighting `{other}`"),
            )),
        };
    }
    value_as_bool(value)
        .ok_or_else(|| plotting_error(builtin, format!("{builtin}: Lighting must be logical")))
}

fn surface_face_color_update(
    value: &Value,
    builtin: &'static str,
) -> BuiltinResult<SurfacePropertyUpdate> {
    if let Some(text) = value_as_string(value) {
        return match text.trim().to_ascii_lowercase().as_str() {
            "none" => Ok(SurfacePropertyUpdate::FaceNone),
            "flat" => Ok(SurfacePropertyUpdate::Shading(ShadingMode::Flat)),
            "interp" => Ok(SurfacePropertyUpdate::Shading(ShadingMode::Smooth)),
            "texturemap" => Ok(SurfacePropertyUpdate::FlattenZ(true)),
            _ => parse_color_value(&LineStyleParseOptions::generic(builtin), value)
                .map(SurfacePropertyUpdate::FaceColor),
        };
    }
    parse_color_value(&LineStyleParseOptions::generic(builtin), value)
        .map(SurfacePropertyUpdate::FaceColor)
}

fn vertices_tensor(vertices: &[glam::Vec3]) -> Value {
    let rows = vertices.len();
    let cols = 3;
    let mut data = Vec::with_capacity(rows * cols);
    for col in 0..cols {
        for vertex in vertices {
            data.push(match col {
                0 => vertex.x as f64,
                1 => vertex.y as f64,
                _ => vertex.z as f64,
            });
        }
    }
    Value::Tensor(Tensor::new(data, vec![rows, cols]).expect("patch vertices"))
}

fn faces_tensor(faces: &[Vec<usize>]) -> Value {
    let rows = faces.len();
    let cols = faces.iter().map(|face| face.len()).max().unwrap_or(0);
    let mut data = Vec::with_capacity(rows * cols);
    for col in 0..cols {
        for face in faces {
            data.push(
                face.get(col)
                    .map(|idx| *idx as f64 + 1.0)
                    .unwrap_or(f64::NAN),
            );
        }
    }
    Value::Tensor(Tensor::new(data, vec![rows, cols]).expect("patch faces"))
}

fn patch_color_property(mode: runmat_plot::plots::PatchFaceColorMode, color: glam::Vec4) -> Value {
    match mode {
        runmat_plot::plots::PatchFaceColorMode::None => Value::String("none".into()),
        runmat_plot::plots::PatchFaceColorMode::Flat => Value::String("flat".into()),
        runmat_plot::plots::PatchFaceColorMode::Color => Value::String(color_to_short_name(color)),
    }
}

fn patch_edge_color_property(
    mode: runmat_plot::plots::PatchEdgeColorMode,
    color: glam::Vec4,
) -> Value {
    match mode {
        runmat_plot::plots::PatchEdgeColorMode::None => Value::String("none".into()),
        runmat_plot::plots::PatchEdgeColorMode::Color => Value::String(color_to_short_name(color)),
    }
}

fn insert_line_marker_struct_props(
    st: &mut StructValue,
    marker: Option<&runmat_plot::plots::line::LineMarkerAppearance>,
) {
    if let Some(marker) = marker {
        st.insert(
            "Marker",
            Value::String(marker_style_name(marker.kind).into()),
        );
        st.insert("MarkerSize", Value::Num(marker.size as f64));
        st.insert(
            "MarkerFaceColor",
            Value::String(color_to_short_name(marker.face_color)),
        );
        st.insert(
            "MarkerEdgeColor",
            Value::String(color_to_short_name(marker.edge_color)),
        );
        st.insert("Filled", Value::Bool(marker.filled));
    }
}

fn line_marker_property_value(
    marker: &Option<runmat_plot::plots::line::LineMarkerAppearance>,
    name: &str,
    builtin: &'static str,
) -> BuiltinResult<Value> {
    match name {
        "marker" => Ok(Value::String(
            marker
                .as_ref()
                .map(|m| marker_style_name(m.kind).to_string())
                .unwrap_or_else(|| "none".into()),
        )),
        "markersize" => Ok(Value::Num(
            marker.as_ref().map(|m| m.size as f64).unwrap_or(0.0),
        )),
        "markerfacecolor" => Ok(Value::String(
            marker
                .as_ref()
                .map(|m| color_to_short_name(m.face_color))
                .unwrap_or_else(|| "none".into()),
        )),
        "markeredgecolor" => Ok(Value::String(
            marker
                .as_ref()
                .map(|m| color_to_short_name(m.edge_color))
                .unwrap_or_else(|| "none".into()),
        )),
        "filled" => Ok(Value::Bool(
            marker.as_ref().map(|m| m.filled).unwrap_or(false),
        )),
        other => Err(plotting_error(
            builtin,
            format!("{builtin}: unsupported line property `{other}`"),
        )),
    }
}

fn histogram_labels_from_edges(edges: &[f64]) -> Vec<String> {
    edges
        .windows(2)
        .map(|pair| format!("[{:.3}, {:.3})", pair[0], pair[1]))
        .collect()
}

pub(crate) fn validate_histogram_normalization(
    norm: &str,
    builtin: &'static str,
) -> BuiltinResult<()> {
    match norm {
        "count" | "probability" | "percentage" | "countdensity" | "pdf" | "cumcount" | "cdf" => {
            Ok(())
        }
        other => Err(plotting_error(
            builtin,
            format!("{builtin}: unsupported histogram normalization `{other}`"),
        )),
    }
}

pub(crate) fn apply_histogram_normalization(
    raw_counts: &[f64],
    edges: &[f64],
    norm: &str,
    normalization_denominator: f64,
) -> Vec<f64> {
    let widths: Vec<f64> = edges.windows(2).map(|pair| pair[1] - pair[0]).collect();
    let total = normalization_denominator;
    match norm {
        "count" => raw_counts.to_vec(),
        "probability" => {
            if total > 0.0 {
                raw_counts.iter().map(|&c| c / total).collect()
            } else {
                vec![0.0; raw_counts.len()]
            }
        }
        "percentage" => {
            if total > 0.0 {
                raw_counts.iter().map(|&c| 100.0 * c / total).collect()
            } else {
                vec![0.0; raw_counts.len()]
            }
        }
        "countdensity" => raw_counts
            .iter()
            .zip(widths.iter())
            .map(|(&c, &w)| if w > 0.0 { c / w } else { 0.0 })
            .collect(),
        "pdf" => {
            if total > 0.0 {
                raw_counts
                    .iter()
                    .zip(widths.iter())
                    .map(|(&c, &w)| if w > 0.0 { c / (total * w) } else { 0.0 })
                    .collect()
            } else {
                vec![0.0; raw_counts.len()]
            }
        }
        "cumcount" => {
            let mut acc = 0.0;
            raw_counts
                .iter()
                .map(|&c| {
                    acc += c;
                    acc
                })
                .collect()
        }
        "cdf" => {
            if total > 0.0 {
                let mut acc = 0.0;
                raw_counts
                    .iter()
                    .map(|&c| {
                        acc += c;
                        acc / total
                    })
                    .collect()
            } else {
                vec![0.0; raw_counts.len()]
            }
        }
        _ => raw_counts.to_vec(),
    }
}

fn line_style_name(style: runmat_plot::plots::line::LineStyle) -> &'static str {
    match style {
        runmat_plot::plots::line::LineStyle::None => "none",
        runmat_plot::plots::line::LineStyle::Solid => "-",
        runmat_plot::plots::line::LineStyle::Dashed => "--",
        runmat_plot::plots::line::LineStyle::Dotted => ":",
        runmat_plot::plots::line::LineStyle::DashDot => "-.",
    }
}

fn parse_line_style_name_for_props(name: &str) -> runmat_plot::plots::line::LineStyle {
    match name.trim() {
        "none" => runmat_plot::plots::line::LineStyle::None,
        "--" | "dashed" => runmat_plot::plots::line::LineStyle::Dashed,
        ":" | "dotted" => runmat_plot::plots::line::LineStyle::Dotted,
        "-." | "dashdot" => runmat_plot::plots::line::LineStyle::DashDot,
        _ => runmat_plot::plots::line::LineStyle::Solid,
    }
}

fn marker_style_name(style: runmat_plot::plots::scatter::MarkerStyle) -> &'static str {
    match style {
        runmat_plot::plots::scatter::MarkerStyle::Circle => "o",
        runmat_plot::plots::scatter::MarkerStyle::Square => "s",
        runmat_plot::plots::scatter::MarkerStyle::Triangle => "^",
        runmat_plot::plots::scatter::MarkerStyle::Diamond => "d",
        runmat_plot::plots::scatter::MarkerStyle::Plus => "+",
        runmat_plot::plots::scatter::MarkerStyle::Cross => "x",
        runmat_plot::plots::scatter::MarkerStyle::Star => "*",
        runmat_plot::plots::scatter::MarkerStyle::Hexagon => "h",
    }
}

fn marker_from_name(
    name: &str,
    current: Option<runmat_plot::plots::line::LineMarkerAppearance>,
) -> Option<runmat_plot::plots::line::LineMarkerAppearance> {
    let mut marker = current.unwrap_or(runmat_plot::plots::line::LineMarkerAppearance {
        kind: runmat_plot::plots::scatter::MarkerStyle::Circle,
        size: 6.0,
        edge_color: glam::Vec4::new(0.0, 0.447, 0.741, 1.0),
        face_color: glam::Vec4::new(0.0, 0.447, 0.741, 1.0),
        filled: false,
    });
    marker.kind = match name.trim() {
        "o" => runmat_plot::plots::scatter::MarkerStyle::Circle,
        "s" => runmat_plot::plots::scatter::MarkerStyle::Square,
        "^" => runmat_plot::plots::scatter::MarkerStyle::Triangle,
        "d" => runmat_plot::plots::scatter::MarkerStyle::Diamond,
        "+" => runmat_plot::plots::scatter::MarkerStyle::Plus,
        "x" => runmat_plot::plots::scatter::MarkerStyle::Cross,
        "*" => runmat_plot::plots::scatter::MarkerStyle::Star,
        "h" => runmat_plot::plots::scatter::MarkerStyle::Hexagon,
        "none" => return None,
        _ => marker.kind,
    };
    Some(marker)
}

fn scatter_marker_from_name(
    name: &str,
    current: runmat_plot::plots::scatter::MarkerStyle,
) -> runmat_plot::plots::scatter::MarkerStyle {
    match name.trim() {
        "o" => runmat_plot::plots::scatter::MarkerStyle::Circle,
        "s" => runmat_plot::plots::scatter::MarkerStyle::Square,
        "^" => runmat_plot::plots::scatter::MarkerStyle::Triangle,
        "d" => runmat_plot::plots::scatter::MarkerStyle::Diamond,
        "+" => runmat_plot::plots::scatter::MarkerStyle::Plus,
        "x" => runmat_plot::plots::scatter::MarkerStyle::Cross,
        "*" => runmat_plot::plots::scatter::MarkerStyle::Star,
        "h" => runmat_plot::plots::scatter::MarkerStyle::Hexagon,
        _ => current,
    }
}

trait Unzip3Vec<A, B, C> {
    fn unzip_n_vec(self) -> (Vec<A>, Vec<B>, Vec<C>);
}

impl<I, A, B, C> Unzip3Vec<A, B, C> for I
where
    I: Iterator<Item = (A, B, C)>,
{
    fn unzip_n_vec(self) -> (Vec<A>, Vec<B>, Vec<C>) {
        let mut a = Vec::new();
        let mut b = Vec::new();
        let mut c = Vec::new();
        for (va, vb, vc) in self {
            a.push(va);
            b.push(vb);
            c.push(vc);
        }
        (a, b, c)
    }
}

fn color_to_short_name(color: glam::Vec4) -> String {
    let candidates = [
        (glam::Vec4::new(1.0, 0.0, 0.0, 1.0), "r"),
        (glam::Vec4::new(0.0, 1.0, 0.0, 1.0), "g"),
        (glam::Vec4::new(0.0, 0.0, 1.0, 1.0), "b"),
        (glam::Vec4::new(0.0, 0.0, 0.0, 1.0), "k"),
        (glam::Vec4::new(1.0, 1.0, 1.0, 1.0), "w"),
        (glam::Vec4::new(1.0, 1.0, 0.0, 1.0), "y"),
        (glam::Vec4::new(1.0, 0.0, 1.0, 1.0), "m"),
        (glam::Vec4::new(0.0, 1.0, 1.0, 1.0), "c"),
    ];
    for (candidate, name) in candidates {
        if (candidate - color).abs().max_element() < 1e-6 {
            return name.to_string();
        }
    }
    format!("[{:.3},{:.3},{:.3}]", color.x, color.y, color.z)
}

fn key_name(kind: PlotObjectKind) -> &'static str {
    match kind {
        PlotObjectKind::Title => "Title",
        PlotObjectKind::Subtitle => "Subtitle",
        PlotObjectKind::XLabel => "XLabel",
        PlotObjectKind::YLabel => "YLabel",
        PlotObjectKind::ZLabel => "ZLabel",
        PlotObjectKind::Legend => "Legend",
        PlotObjectKind::SuperTitle => "SGTitle",
        PlotObjectKind::XAxis => "XAxis",
        PlotObjectKind::YAxis => "YAxis",
    }
}

trait AxesMetadataExt {
    fn text_style_for(&self, kind: PlotObjectKind) -> TextStyle;
}

impl AxesMetadataExt for runmat_plot::plots::AxesMetadata {
    fn text_style_for(&self, kind: PlotObjectKind) -> TextStyle {
        match kind {
            PlotObjectKind::Title => self.title_style.clone(),
            PlotObjectKind::Subtitle => self.subtitle_style.clone(),
            PlotObjectKind::XLabel => self.x_label_style.clone(),
            PlotObjectKind::YLabel => self.y_label_style.clone(),
            PlotObjectKind::ZLabel => self.z_label_style.clone(),
            PlotObjectKind::Legend
            | PlotObjectKind::SuperTitle
            | PlotObjectKind::XAxis
            | PlotObjectKind::YAxis => TextStyle::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_builtins::IntegerStorage;

    #[test]
    fn label_strings_preserve_exact_uint64_text() {
        let labels = label_strings_from_value(
            &Value::Int(runmat_builtins::IntValue::U64(u64::MAX)),
            "legend",
            "labels",
        )
        .expect("labels");
        assert_eq!(labels, vec!["18446744073709551615"]);
    }

    #[test]
    fn numeric_property_helpers_read_typed_integer_storage() {
        let aspect_tensor =
            Tensor::new_integer(IntegerStorage::U16(vec![1, 2, 3]), vec![1, 3]).expect("aspect");
        let aspect = Value::Tensor(aspect_tensor);
        let position_tensor =
            Tensor::new_integer(IntegerStorage::U32(vec![10, 20, 300, 200]), vec![1, 4])
                .expect("position");
        let position = Value::Tensor(position_tensor);
        let text_tensor =
            Tensor::new_integer(IntegerStorage::I16(vec![-2, 5, 9]), vec![1, 3]).expect("text");
        let text = Value::Tensor(text_tensor);

        assert_eq!(
            data_aspect_ratio_from_value(&aspect, "daspect").expect("aspect"),
            [1.0, 2.0, 3.0]
        );
        assert_eq!(
            parse_figure_position(&position, "figure").expect("position"),
            [10.0, 20.0, 300.0, 200.0]
        );
        let parsed = parse_text_position(&text, "text").expect("text position");
        assert_eq!(parsed, glam::Vec3::new(-2.0, 5.0, 9.0));

        let zdata =
            Tensor::new_integer(IntegerStorage::I16(vec![1, 2, 3, 4]), vec![2, 2]).expect("zdata");
        let zgrid = surface_z_grid_from_value(&Value::Tensor(zdata), 2, 2, "surf").expect("zdata");
        assert_eq!(zgrid, vec![vec![1.0, 2.0], vec![3.0, 4.0]]);

        let scalar = Tensor::new_integer(IntegerStorage::U16(vec![7]), vec![1, 1]).expect("scalar");
        let scalar = Value::Tensor(scalar);
        assert_eq!(scalar_numeric_value(&scalar), Some(7.0));
        assert_eq!(handle_scalar(&scalar, "plot").expect("handle"), 7.0);

        let empty_labels =
            Tensor::new_integer(IntegerStorage::U16(Vec::new()), vec![1, 0]).expect("empty labels");
        assert_eq!(
            tick_labels_from_value(&Value::Tensor(empty_labels), "xticklabels").expect("labels"),
            Vec::<String>::new()
        );
    }
}
