use runmat_builtins::{StringArray, StructValue, Tensor, Value};
use runmat_plot::plots::{LegendStyle, TextStyle};

use super::state::{
    axes_handle_exists, axes_handles_for_figure, axes_metadata_snapshot, axes_state_snapshot,
    current_axes_handle_for_figure, decode_axes_handle, decode_plot_object_handle,
    figure_handle_exists, legend_entries_snapshot, select_axes_for_figure, set_legend_for_axes,
    set_text_properties_for_axes, FigureHandle, PlotObjectKind,
};
use super::style::{
    parse_color_value, value_as_bool, value_as_f64, value_as_string, LineStyleParseOptions,
};
use super::{plotting_error, plotting_error_with_source};
use crate::builtins::plotting::op_common::limits::limit_value;
use crate::builtins::plotting::op_common::value_as_text_string;
use crate::BuiltinResult;

#[derive(Clone, Debug)]
pub enum PlotHandle {
    Figure(FigureHandle),
    Axes(FigureHandle, usize),
    Text(FigureHandle, usize, PlotObjectKind),
    Legend(FigureHandle, usize),
    PlotChild(super::state::PlotChildHandleState),
}

pub fn resolve_plot_handle(value: &Value, builtin: &'static str) -> BuiltinResult<PlotHandle> {
    let scalar = handle_scalar(value, builtin)?;
    if let Ok(state) = super::state::plot_child_handle_snapshot(scalar) {
        return Ok(PlotHandle::PlotChild(state));
    }
    if let Ok((handle, axes_index, kind)) = decode_plot_object_handle(scalar) {
        if axes_handle_exists(handle, axes_index) {
            return Ok(match kind {
                PlotObjectKind::Legend => PlotHandle::Legend(handle, axes_index),
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
        PlotHandle::Axes(handle, axes_index) => {
            get_axes_property(handle, axes_index, property, builtin)
        }
        PlotHandle::Text(handle, axes_index, kind) => {
            get_text_property(handle, axes_index, kind, property, builtin)
        }
        PlotHandle::Legend(handle, axes_index) => {
            get_legend_property(handle, axes_index, property, builtin)
        }
        PlotHandle::Figure(handle) => get_figure_property(handle, property, builtin),
        PlotHandle::PlotChild(state) => get_plot_child_property(&state, property, builtin),
    }
}

pub fn set_properties(
    handle: PlotHandle,
    args: &[Value],
    builtin: &'static str,
) -> BuiltinResult<()> {
    if args.is_empty() || args.len() % 2 != 0 {
        return Err(plotting_error(
            builtin,
            format!("{builtin}: property/value arguments must come in pairs"),
        ));
    }
    match handle {
        PlotHandle::Figure(handle) => {
            for pair in args.chunks_exact(2) {
                let key = property_name(&pair[0], builtin)?;
                apply_figure_property(handle, &key, &pair[1], builtin)?;
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
        PlotHandle::Text(handle, axes_index, kind) => {
            let mut text: Option<String> = None;
            let mut style = axes_metadata_snapshot(handle, axes_index)
                .map_err(|err| map_figure_error(builtin, err))?
                .text_style_for(kind);
            for pair in args.chunks_exact(2) {
                let key = property_name(&pair[0], builtin)?;
                apply_text_property(&mut text, &mut style, &key, &pair[1], builtin)?;
            }
            set_text_properties_for_axes(handle, axes_index, kind, text, Some(style))
                .map_err(|err| map_figure_error(builtin, err))?;
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
        PlotHandle::PlotChild(state) => {
            for pair in args.chunks_exact(2) {
                let key = property_name(&pair[0], builtin)?;
                apply_plot_child_property(&state, &key, &pair[1], builtin)?;
            }
            Ok(())
        }
    }
}

pub fn parse_text_style_pairs(builtin: &'static str, args: &[Value]) -> BuiltinResult<TextStyle> {
    if args.is_empty() {
        return Ok(TextStyle::default());
    }
    if args.len() % 2 != 0 {
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
    let axes = axes_handles_for_figure(handle).map_err(|err| map_figure_error(builtin, err))?;
    let current_axes =
        current_axes_handle_for_figure(handle).map_err(|err| map_figure_error(builtin, err))?;
    match property.map(canonical_property_name) {
        None => {
            let mut st = StructValue::new();
            st.insert("Handle", Value::Num(handle.as_u32() as f64));
            st.insert("Number", Value::Num(handle.as_u32() as f64));
            st.insert("Type", Value::String("figure".into()));
            st.insert("CurrentAxes", Value::Num(current_axes));
            st.insert("Children", handles_value(axes));
            st.insert("Parent", Value::Num(f64::NAN));
            st.insert("Name", Value::String(format!("Figure {}", handle.as_u32())));
            Ok(Value::Struct(st))
        }
        Some("number") => Ok(Value::Num(handle.as_u32() as f64)),
        Some("type") => Ok(Value::String("figure".into())),
        Some("currentaxes") => Ok(Value::Num(current_axes)),
        Some("children") => Ok(handles_value(axes)),
        Some("parent") => Ok(Value::Num(f64::NAN)),
        Some("name") => Ok(Value::String(format!("Figure {}", handle.as_u32()))),
        Some(other) => Err(plotting_error(
            builtin,
            format!("{builtin}: unsupported figure property `{other}`"),
        )),
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
    match property.map(canonical_property_name) {
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
            st.insert("LegendVisible", Value::Bool(meta.legend_enabled));
            st.insert("Type", Value::String("axes".into()));
            st.insert("Parent", Value::Num(handle.as_u32() as f64));
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
            st.insert("Box", Value::Bool(meta.box_enabled));
            st.insert("AxisEqual", Value::Bool(meta.axis_equal));
            st.insert("Colorbar", Value::Bool(meta.colorbar_enabled));
            st.insert(
                "Colormap",
                Value::String(format!("{:?}", meta.colormap).to_ascii_lowercase()),
            );
            st.insert("XLim", limit_value(meta.x_limits));
            st.insert("YLim", limit_value(meta.y_limits));
            st.insert("ZLim", limit_value(meta.z_limits));
            st.insert("CLim", limit_value(meta.color_limits));
            st.insert(
                "XScale",
                Value::String(if meta.x_log { "log" } else { "linear" }.into()),
            );
            st.insert(
                "YScale",
                Value::String(if meta.y_log { "log" } else { "linear" }.into()),
            );
            Ok(Value::Struct(st))
        }
        Some("title") => Ok(Value::Num(super::state::encode_plot_object_handle(
            handle,
            axes_index,
            PlotObjectKind::Title,
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
        Some("view") => {
            let az = meta.view_azimuth_deg.unwrap_or(-37.5) as f64;
            let el = meta.view_elevation_deg.unwrap_or(30.0) as f64;
            Ok(Value::Tensor(runmat_builtins::Tensor {
                rows: 1,
                cols: 2,
                shape: vec![1, 2],
                data: vec![az, el],
                dtype: runmat_builtins::NumericDType::F64,
            }))
        }
        Some("grid") => Ok(Value::Bool(meta.grid_enabled)),
        Some("box") => Ok(Value::Bool(meta.box_enabled)),
        Some("axisequal") => Ok(Value::Bool(meta.axis_equal)),
        Some("colorbar") => Ok(Value::Bool(meta.colorbar_enabled)),
        Some("colormap") => Ok(Value::String(
            format!("{:?}", meta.colormap).to_ascii_lowercase(),
        )),
        Some("xlim") => Ok(limit_value(meta.x_limits)),
        Some("ylim") => Ok(limit_value(meta.y_limits)),
        Some("zlim") => Ok(limit_value(meta.z_limits)),
        Some("clim") => Ok(limit_value(meta.color_limits)),
        Some("xscale") => Ok(Value::String(
            if meta.x_log { "log" } else { "linear" }.into(),
        )),
        Some("yscale") => Ok(Value::String(
            if meta.y_log { "log" } else { "linear" }.into(),
        )),
        Some("legendvisible") => Ok(Value::Bool(meta.legend_enabled)),
        Some("children") => Ok(handles_value(vec![
            super::state::encode_plot_object_handle(handle, axes_index, PlotObjectKind::Title),
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
    let meta =
        axes_metadata_snapshot(handle, axes_index).map_err(|err| map_figure_error(builtin, err))?;
    let (text, style) = match kind {
        PlotObjectKind::Title => (meta.title, meta.title_style),
        PlotObjectKind::XLabel => (meta.x_label, meta.x_label_style),
        PlotObjectKind::YLabel => (meta.y_label, meta.y_label_style),
        PlotObjectKind::ZLabel => (meta.z_label, meta.z_label_style),
        PlotObjectKind::Legend => unreachable!(),
    };
    match property.map(canonical_property_name) {
        None => {
            let mut st = StructValue::new();
            st.insert("Type", Value::String("text".into()));
            st.insert(
                "Parent",
                Value::Num(super::state::encode_axes_handle(handle, axes_index)),
            );
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
    match property.map(canonical_property_name) {
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
    value_as_string(value)
        .map(|s| canonical_property_name(s.trim()).to_string())
        .ok_or_else(|| {
            plotting_error(
                builtin,
                format!("{builtin}: property names must be strings"),
            )
        })
}

fn canonical_property_name(name: &str) -> &str {
    match name.to_ascii_lowercase().as_str() {
        "textcolor" => "textcolor",
        "color" => "color",
        "fontsize" => "fontsize",
        "fontweight" => "fontweight",
        "fontangle" => "fontangle",
        "interpreter" => "interpreter",
        "visible" => "visible",
        "location" => "location",
        "box" => "box",
        "orientation" => "orientation",
        "string" => "string",
        "title" => "title",
        "xlabel" => "xlabel",
        "ylabel" => "ylabel",
        "zlabel" => "zlabel",
        "view" => "view",
        "grid" => "grid",
        "axisequal" => "axisequal",
        "colorbar" => "colorbar",
        "colormap" => "colormap",
        "xlim" => "xlim",
        "ylim" => "ylim",
        "zlim" => "zlim",
        "clim" => "clim",
        "caxis" => "clim",
        "xscale" => "xscale",
        "yscale" => "yscale",
        "currentaxes" => "currentaxes",
        "children" => "children",
        "parent" => "parent",
        "type" => "type",
        "number" => "number",
        "name" => "name",
        "legend" => "legend",
        "legendvisible" => "legendvisible",
        other => Box::leak(other.to_string().into_boxed_str()),
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
            if tensor.data.len() != 2 || !tensor.data[0].is_finite() || !tensor.data[1].is_finite()
            {
                return Err(plotting_error(
                    builtin,
                    format!("{builtin}: View must be a 2-element finite numeric vector"),
                ));
            }
            crate::builtins::plotting::state::set_view_for_axes(
                handle,
                axes_index,
                tensor.data[0] as f32,
                tensor.data[1] as f32,
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
        "box" => {
            let enabled = value_as_bool(value).ok_or_else(|| {
                plotting_error(builtin, format!("{builtin}: Box must be logical"))
            })?;
            crate::builtins::plotting::state::set_box_enabled_for_axes(handle, axes_index, enabled)
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
        "xscale" => {
            let mode = value_as_string(value).ok_or_else(|| {
                plotting_error(builtin, format!("{builtin}: XScale must be a string"))
            })?;
            let meta = axes_metadata_snapshot(handle, axes_index)
                .map_err(|err| map_figure_error(builtin, err))?;
            crate::builtins::plotting::state::set_log_modes_for_axes(
                handle,
                axes_index,
                mode.trim().eq_ignore_ascii_case("log"),
                meta.y_log,
            )
            .map_err(|err| map_figure_error(builtin, err))?;
            Ok(())
        }
        "yscale" => {
            let mode = value_as_string(value).ok_or_else(|| {
                plotting_error(builtin, format!("{builtin}: YScale must be a string"))
            })?;
            let meta = axes_metadata_snapshot(handle, axes_index)
                .map_err(|err| map_figure_error(builtin, err))?;
            crate::builtins::plotting::state::set_log_modes_for_axes(
                handle,
                axes_index,
                meta.x_log,
                mode.trim().eq_ignore_ascii_case("log"),
            )
            .map_err(|err| map_figure_error(builtin, err))?;
            Ok(())
        }
        other => Err(plotting_error(
            builtin,
            format!("{builtin}: unsupported axes property `{other}`"),
        )),
    }
}

fn apply_figure_property(
    figure_handle: FigureHandle,
    key: &str,
    value: &Value,
    builtin: &'static str,
) -> BuiltinResult<()> {
    match key {
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
            Ok(())
        }
        other => Err(plotting_error(
            builtin,
            format!("{builtin}: unsupported figure property `{other}`"),
        )),
    }
}

fn get_histogram_property(
    hist: &super::state::HistogramHandleState,
    property: Option<&str>,
    builtin: &'static str,
) -> BuiltinResult<Value> {
    let normalized =
        apply_histogram_normalization(&hist.raw_counts, &hist.bin_edges, &hist.normalization);
    match property.map(canonical_property_name) {
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
            st.insert("BinCounts", tensor_from_vec(normalized));
            st.insert("Normalization", Value::String(hist.normalization.clone()));
            st.insert("NumBins", Value::Num(hist.raw_counts.len() as f64));
            Ok(Value::Struct(st))
        }
        Some("type") => Ok(Value::String("histogram".into())),
        Some("parent") => Ok(Value::Num(super::state::encode_axes_handle(
            hist.figure,
            hist.axes_index,
        ))),
        Some("children") => Ok(handles_value(Vec::new())),
        Some("binedges") => Ok(tensor_from_vec(hist.bin_edges.clone())),
        Some("bincounts") => Ok(tensor_from_vec(normalized)),
        Some("normalization") => Ok(Value::String(hist.normalization.clone())),
        Some("numbins") => Ok(Value::Num(hist.raw_counts.len() as f64)),
        Some(other) => Err(plotting_error(
            builtin,
            format!("{builtin}: unsupported histogram property `{other}`"),
        )),
    }
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
        super::state::PlotChildHandleState::Line(plot) => {
            get_line_property(plot, property, builtin)
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
        super::state::PlotChildHandleState::Area(area) => {
            get_area_property(area, property, builtin)
        }
        super::state::PlotChildHandleState::Surface(plot) => {
            get_surface_property(plot, property, builtin)
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
        super::state::PlotChildHandleState::Pie(plot) => get_pie_property(plot, property, builtin),
    }
}

fn apply_plot_child_property(
    state: &super::state::PlotChildHandleState,
    key: &str,
    value: &Value,
    builtin: &'static str,
) -> BuiltinResult<()> {
    match state {
        super::state::PlotChildHandleState::Histogram(hist) => {
            apply_histogram_property(hist, key, value, builtin)
        }
        super::state::PlotChildHandleState::Line(plot) => {
            apply_line_property(plot, key, value, builtin)
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
        super::state::PlotChildHandleState::Area(area) => {
            apply_area_property(area, key, value, builtin)
        }
        super::state::PlotChildHandleState::Surface(plot) => {
            apply_surface_property(plot, key, value, builtin)
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
        super::state::PlotChildHandleState::Pie(plot) => {
            apply_pie_property(plot, key, value, builtin)
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
    match property.map(canonical_property_name) {
        None => {
            let mut st = child_base_struct("line", line_handle.figure, line_handle.axes_index);
            st.insert("XData", tensor_from_vec(line.x_data.clone()));
            st.insert("YData", tensor_from_vec(line.y_data.clone()));
            st.insert("Color", Value::String(color_to_short_name(line.color)));
            st.insert("LineWidth", Value::Num(line.line_width as f64));
            st.insert(
                "LineStyle",
                Value::String(line_style_name(line.line_style).into()),
            );
            if let Some(label) = line.label.clone() {
                st.insert("DisplayName", Value::String(label));
            }
            insert_line_marker_struct_props(&mut st, line.marker.as_ref());
            Ok(Value::Struct(st))
        }
        Some("type") => Ok(Value::String("line".into())),
        Some("parent") => Ok(child_parent_handle(
            line_handle.figure,
            line_handle.axes_index,
        )),
        Some("children") => Ok(handles_value(Vec::new())),
        Some("xdata") => Ok(tensor_from_vec(line.x_data.clone())),
        Some("ydata") => Ok(tensor_from_vec(line.y_data.clone())),
        Some("color") => Ok(Value::String(color_to_short_name(line.color))),
        Some("linewidth") => Ok(Value::Num(line.line_width as f64)),
        Some("linestyle") => Ok(Value::String(line_style_name(line.line_style).into())),
        Some("displayname") => Ok(Value::String(line.label.unwrap_or_default())),
        Some(name) => line_marker_property_value(&line.marker, name, builtin),
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
    match property.map(canonical_property_name) {
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
    match property.map(canonical_property_name) {
        None => {
            let mut st =
                child_base_struct("scatter", scatter_handle.figure, scatter_handle.axes_index);
            st.insert("XData", tensor_from_vec(scatter.x_data.clone()));
            st.insert("YData", tensor_from_vec(scatter.y_data.clone()));
            st.insert(
                "Marker",
                Value::String(marker_style_name(scatter.marker_style).into()),
            );
            st.insert("SizeData", Value::Num(scatter.marker_size as f64));
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
        Some("xdata") => Ok(tensor_from_vec(scatter.x_data.clone())),
        Some("ydata") => Ok(tensor_from_vec(scatter.y_data.clone())),
        Some("marker") => Ok(Value::String(
            marker_style_name(scatter.marker_style).into(),
        )),
        Some("sizedata") => Ok(Value::Num(scatter.marker_size as f64)),
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
    match property.map(canonical_property_name) {
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
    match property.map(canonical_property_name) {
        None => {
            let mut st =
                child_base_struct("surface", surface_handle.figure, surface_handle.axes_index);
            st.insert("XData", tensor_from_vec(surface.x_data.clone()));
            st.insert("YData", tensor_from_vec(surface.y_data.clone()));
            if let Some(z) = surface.z_data.clone() {
                st.insert("ZData", tensor_from_matrix(z));
            }
            st.insert("FaceAlpha", Value::Num(surface.alpha as f64));
            if let Some(label) = surface.label.clone() {
                st.insert("DisplayName", Value::String(label));
            }
            Ok(Value::Struct(st))
        }
        Some("type") => Ok(Value::String("surface".into())),
        Some("parent") => Ok(child_parent_handle(
            surface_handle.figure,
            surface_handle.axes_index,
        )),
        Some("children") => Ok(handles_value(Vec::new())),
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
            format!("{builtin}: unsupported surface property `{other}`"),
        )),
    }
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
    match property.map(canonical_property_name) {
        None => {
            let mut st = child_base_struct("line", line_handle.figure, line_handle.axes_index);
            st.insert("XData", tensor_from_vec(line.x_data.clone()));
            st.insert("YData", tensor_from_vec(line.y_data.clone()));
            st.insert("ZData", tensor_from_vec(line.z_data.clone()));
            st.insert("Color", Value::String(color_to_short_name(line.color)));
            st.insert("LineWidth", Value::Num(line.line_width as f64));
            st.insert(
                "LineStyle",
                Value::String(line_style_name(line.line_style).into()),
            );
            if let Some(label) = line.label.clone() {
                st.insert("DisplayName", Value::String(label));
            }
            Ok(Value::Struct(st))
        }
        Some("type") => Ok(Value::String("line".into())),
        Some("parent") => Ok(child_parent_handle(
            line_handle.figure,
            line_handle.axes_index,
        )),
        Some("children") => Ok(handles_value(Vec::new())),
        Some("xdata") => Ok(tensor_from_vec(line.x_data.clone())),
        Some("ydata") => Ok(tensor_from_vec(line.y_data.clone())),
        Some("zdata") => Ok(tensor_from_vec(line.z_data.clone())),
        Some("color") => Ok(Value::String(color_to_short_name(line.color))),
        Some("linewidth") => Ok(Value::Num(line.line_width as f64)),
        Some("linestyle") => Ok(Value::String(line_style_name(line.line_style).into())),
        Some("displayname") => Ok(Value::String(line.label.unwrap_or_default())),
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
    match property.map(canonical_property_name) {
        None => {
            let mut st =
                child_base_struct("scatter", scatter_handle.figure, scatter_handle.axes_index);
            st.insert("XData", tensor_from_vec(x));
            st.insert("YData", tensor_from_vec(y));
            st.insert("ZData", tensor_from_vec(z));
            st.insert("SizeData", Value::Num(scatter.point_size as f64));
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
        Some("sizedata") => Ok(Value::Num(scatter.point_size as f64)),
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
    match property.map(canonical_property_name) {
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
    match property.map(canonical_property_name) {
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
    match property.map(canonical_property_name) {
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
    match property.map(canonical_property_name) {
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
    match property.map(canonical_property_name) {
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
    match property.map(canonical_property_name) {
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
        Some(other) => Err(plotting_error(
            builtin,
            format!("{builtin}: unsupported quiver property `{other}`"),
        )),
    }
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
    match property.map(canonical_property_name) {
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
            st.insert(
                "CDataMapping",
                Value::String(if surface.color_grid.is_some() {
                    "direct".into()
                } else {
                    "scaled".into()
                }),
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
        Some("cdatamapping") => Ok(Value::String(if surface.color_grid.is_some() {
            "direct".into()
        } else {
            "scaled".into()
        })),
        Some(other) => Err(plotting_error(
            builtin,
            format!("{builtin}: unsupported image property `{other}`"),
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
    match property.map(canonical_property_name) {
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
            let normalized =
                apply_histogram_normalization(&hist.raw_counts, &hist.bin_edges, &norm);
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
        other => Err(plotting_error(
            builtin,
            format!("{builtin}: unsupported histogram property `{other}`"),
        )),
    }
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
                    stem.color = stem.color;
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
    super::state::update_quiver_plot(quiver_handle.figure, quiver_handle.plot_index, |quiver| {
        match key {
            "color" => {
                if let Ok(c) = parse_color_value(&LineStyleParseOptions::generic(builtin), value) {
                    quiver.color = c;
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
                }
            }
            "maxheadsize" => {
                if let Some(v) = value_as_f64(value) {
                    quiver.head_size = v as f32;
                }
            }
            _ => {}
        }
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
    super::state::update_image_plot(image_handle.figure, image_handle.plot_index, |surface| {
        match key {
            "xdata" => {
                if let Ok(tensor) = Tensor::try_from(value) {
                    surface.x_data = tensor.data;
                }
            }
            "ydata" => {
                if let Ok(tensor) = Tensor::try_from(value) {
                    surface.y_data = tensor.data;
                }
            }
            "cdatamapping" => {
                if let Some(text) = value_as_string(value) {
                    if text.trim().eq_ignore_ascii_case("direct") {
                        surface.image_mode = true;
                    }
                }
            }
            _ => {}
        }
    })
    .map_err(|err| map_figure_error(builtin, err))?;
    Ok(())
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
    super::state::update_plot_element(line_handle.figure, line_handle.plot_index, |plot| {
        if let runmat_plot::plots::figure::PlotElement::Line(line) = plot {
            apply_line_plot_properties(line, key, value, builtin);
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
    super::state::update_plot_element(scatter_handle.figure, scatter_handle.plot_index, |plot| {
        if let runmat_plot::plots::figure::PlotElement::Scatter(scatter) = plot {
            match key {
                "marker" => {
                    if let Some(s) = value_as_string(value) {
                        scatter.marker_style = scatter_marker_from_name(&s, scatter.marker_style);
                    }
                }
                "sizedata" => {
                    if let Some(v) = value_as_f64(value) {
                        scatter.marker_size = v as f32;
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
    })
    .map_err(|err| map_figure_error(builtin, err))?;
    Ok(())
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

fn apply_surface_property(
    surface_handle: &super::state::SimplePlotHandleState,
    key: &str,
    value: &Value,
    builtin: &'static str,
) -> BuiltinResult<()> {
    super::state::update_plot_element(surface_handle.figure, surface_handle.plot_index, |plot| {
        if let runmat_plot::plots::figure::PlotElement::Surface(surface) = plot {
            match key {
                "facealpha" => {
                    if let Some(v) = value_as_f64(value) {
                        surface.alpha = v as f32;
                    }
                }
                "displayname" => {
                    surface.label = value_as_string(value).map(|s| s.to_string());
                }
                _ => {}
            }
        }
    })
    .map_err(|err| map_figure_error(builtin, err))?;
    Ok(())
}

fn apply_line3_property(
    line_handle: &super::state::SimplePlotHandleState,
    key: &str,
    value: &Value,
    builtin: &'static str,
) -> BuiltinResult<()> {
    super::state::update_plot_element(line_handle.figure, line_handle.plot_index, |plot| {
        if let runmat_plot::plots::figure::PlotElement::Line3(line) = plot {
            match key {
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
                _ => {}
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
                        scatter.point_size = v as f32;
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
        _ => {}
    }
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

fn parse_colormap_name(
    name: &str,
    builtin: &'static str,
) -> BuiltinResult<runmat_plot::plots::surface::ColorMap> {
    match name.trim().to_ascii_lowercase().as_str() {
        "parula" => Ok(runmat_plot::plots::surface::ColorMap::Parula),
        "viridis" => Ok(runmat_plot::plots::surface::ColorMap::Viridis),
        "plasma" => Ok(runmat_plot::plots::surface::ColorMap::Plasma),
        "inferno" => Ok(runmat_plot::plots::surface::ColorMap::Inferno),
        "magma" => Ok(runmat_plot::plots::surface::ColorMap::Magma),
        "turbo" => Ok(runmat_plot::plots::surface::ColorMap::Turbo),
        "jet" => Ok(runmat_plot::plots::surface::ColorMap::Jet),
        "hot" => Ok(runmat_plot::plots::surface::ColorMap::Hot),
        "cool" => Ok(runmat_plot::plots::surface::ColorMap::Cool),
        "spring" => Ok(runmat_plot::plots::surface::ColorMap::Spring),
        "summer" => Ok(runmat_plot::plots::surface::ColorMap::Summer),
        "autumn" => Ok(runmat_plot::plots::surface::ColorMap::Autumn),
        "winter" => Ok(runmat_plot::plots::surface::ColorMap::Winter),
        "gray" | "grey" => Ok(runmat_plot::plots::surface::ColorMap::Gray),
        "bone" => Ok(runmat_plot::plots::surface::ColorMap::Bone),
        "copper" => Ok(runmat_plot::plots::surface::ColorMap::Copper),
        "pink" => Ok(runmat_plot::plots::surface::ColorMap::Pink),
        "lines" => Ok(runmat_plot::plots::surface::ColorMap::Lines),
        other => Err(plotting_error(
            builtin,
            format!("{builtin}: unknown colormap '{other}'"),
        )),
    }
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
        PlotObjectKind::XLabel => (meta.x_label, meta.x_label_style),
        PlotObjectKind::YLabel => (meta.y_label, meta.y_label_style),
        PlotObjectKind::ZLabel => (meta.z_label, meta.z_label_style),
        PlotObjectKind::Legend => unreachable!(),
    };
    set_text_properties_for_axes(handle, axes_index, kind, text, Some(style))
        .map_err(|err| map_figure_error(builtin, err))?;
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
        Value::Tensor(t) if t.data.len() == 1 => Ok(t.data[0]),
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
    Value::Tensor(runmat_builtins::Tensor {
        rows: 1,
        cols: handles.len(),
        shape: vec![1, handles.len()],
        data: handles,
        dtype: runmat_builtins::NumericDType::F64,
    })
}

fn tensor_from_vec(data: Vec<f64>) -> Value {
    Value::Tensor(runmat_builtins::Tensor {
        rows: 1,
        cols: data.len(),
        shape: vec![1, data.len()],
        data,
        dtype: runmat_builtins::NumericDType::F64,
    })
}

fn tensor_from_matrix(data: Vec<Vec<f64>>) -> Value {
    let rows = data.len();
    let cols = data.first().map(|row| row.len()).unwrap_or(0);
    let flat = data.into_iter().flat_map(|row| row.into_iter()).collect();
    Value::Tensor(runmat_builtins::Tensor {
        rows,
        cols,
        shape: vec![rows, cols],
        data: flat,
        dtype: runmat_builtins::NumericDType::F64,
    })
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

fn validate_histogram_normalization(norm: &str, builtin: &'static str) -> BuiltinResult<()> {
    match norm {
        "count" | "probability" | "countdensity" | "pdf" | "cumcount" | "cdf" => Ok(()),
        other => Err(plotting_error(
            builtin,
            format!("{builtin}: unsupported histogram normalization `{other}`"),
        )),
    }
}

fn apply_histogram_normalization(raw_counts: &[f64], edges: &[f64], norm: &str) -> Vec<f64> {
    let widths: Vec<f64> = edges.windows(2).map(|pair| pair[1] - pair[0]).collect();
    let total: f64 = raw_counts.iter().sum();
    match norm {
        "count" => raw_counts.to_vec(),
        "probability" => {
            if total > 0.0 {
                raw_counts.iter().map(|&c| c / total).collect()
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
        runmat_plot::plots::line::LineStyle::Solid => "-",
        runmat_plot::plots::line::LineStyle::Dashed => "--",
        runmat_plot::plots::line::LineStyle::Dotted => ":",
        runmat_plot::plots::line::LineStyle::DashDot => "-.",
    }
}

fn parse_line_style_name_for_props(name: &str) -> runmat_plot::plots::line::LineStyle {
    match name.trim() {
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
        PlotObjectKind::XLabel => "XLabel",
        PlotObjectKind::YLabel => "YLabel",
        PlotObjectKind::ZLabel => "ZLabel",
        PlotObjectKind::Legend => "Legend",
    }
}

trait AxesMetadataExt {
    fn text_style_for(&self, kind: PlotObjectKind) -> TextStyle;
}

impl AxesMetadataExt for runmat_plot::plots::AxesMetadata {
    fn text_style_for(&self, kind: PlotObjectKind) -> TextStyle {
        match kind {
            PlotObjectKind::Title => self.title_style.clone(),
            PlotObjectKind::XLabel => self.x_label_style.clone(),
            PlotObjectKind::YLabel => self.y_label_style.clone(),
            PlotObjectKind::ZLabel => self.z_label_style.clone(),
            PlotObjectKind::Legend => TextStyle::default(),
        }
    }
}
