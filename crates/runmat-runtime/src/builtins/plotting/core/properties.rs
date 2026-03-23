use runmat_builtins::{StringArray, StructValue, Value};
use runmat_plot::plots::{LegendStyle, TextStyle};

use super::state::{
    axes_handle_exists, axes_metadata_snapshot, axes_state_snapshot, decode_axes_handle,
    decode_plot_object_handle, figure_handle_exists, legend_entries_snapshot, set_legend_for_axes,
    set_text_properties_for_axes, FigureHandle, PlotObjectKind,
};
use super::style::{
    parse_color_value, value_as_bool, value_as_f64, value_as_string, LineStyleParseOptions,
};
use super::{plotting_error, plotting_error_with_source};
use crate::builtins::plotting::op_common::value_as_text_string;
use crate::BuiltinResult;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlotHandle {
    Figure(FigureHandle),
    Axes(FigureHandle, usize),
    Text(FigureHandle, usize, PlotObjectKind),
    Legend(FigureHandle, usize),
}

pub fn resolve_plot_handle(value: &Value, builtin: &'static str) -> BuiltinResult<PlotHandle> {
    let scalar = handle_scalar(value, builtin)?;
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
        PlotHandle::Figure(_) => Err(plotting_error(
            builtin,
            format!("{builtin}: figure property access is not implemented yet"),
        )),
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
        PlotHandle::Figure(_) => Err(plotting_error(
            builtin,
            format!("{builtin}: figure set is not implemented yet"),
        )),
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
                "Legend",
                Value::Num(super::state::encode_plot_object_handle(
                    handle,
                    axes_index,
                    PlotObjectKind::Legend,
                )),
            );
            st.insert("LegendVisible", Value::Bool(meta.legend_enabled));
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
        Some("legend") => Ok(Value::Num(super::state::encode_plot_object_handle(
            handle,
            axes_index,
            PlotObjectKind::Legend,
        ))),
        Some("legendvisible") => Ok(Value::Bool(meta.legend_enabled)),
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
        PlotObjectKind::Legend => unreachable!(),
    };
    match property.map(canonical_property_name) {
        None => {
            let mut st = StructValue::new();
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
        other => Err(plotting_error(
            builtin,
            format!("{builtin}: unsupported axes property `{other}`"),
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
            PlotObjectKind::Legend => TextStyle::default(),
        }
    }
}
