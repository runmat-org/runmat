//! Scatterhist styles domain behavior.
use super::*;

pub(super) fn parse_marker_cycle(value: &Value) -> BuiltinResult<Vec<MarkerStyle>> {
    let entries = style_text_entries(value, "Marker")?;
    let mut markers = Vec::new();
    for entry in entries {
        let trimmed = entry.trim();
        let marker_chars = trimmed
            .chars()
            .filter(|ch| !ch.is_whitespace())
            .collect::<Vec<_>>();
        if marker_chars.len() > 1
            && marker_chars
                .iter()
                .all(|ch| marker_from_text(&ch.to_string()).is_some())
        {
            markers.extend(
                marker_chars
                    .iter()
                    .filter_map(|ch| marker_from_text(&ch.to_string())),
            );
        } else if let Some(marker) = marker_from_text(trimmed) {
            markers.push(marker);
        } else {
            return Err(invalid(format!(
                "scatterhist: unsupported Marker value '{trimmed}'"
            )));
        }
    }
    if markers.is_empty() {
        return Err(invalid("scatterhist: Marker must not be empty"));
    }
    Ok(markers)
}

pub(super) fn marker_from_text(text: &str) -> Option<MarkerStyle> {
    match text {
        "o" | "circle" => Some(MarkerStyle::Circle),
        "+" | "plus" => Some(MarkerStyle::Plus),
        "*" | "star" => Some(MarkerStyle::Star),
        "x" | "cross" => Some(MarkerStyle::Cross),
        "s" | "square" => Some(MarkerStyle::Square),
        "d" | "diamond" => Some(MarkerStyle::Diamond),
        "^" | "triangle" => Some(MarkerStyle::Triangle),
        _ => None,
    }
}

pub(super) fn parse_line_style_cycle(value: &Value) -> BuiltinResult<Vec<LineStyle>> {
    let entries = style_text_entries(value, "LineStyle")?;
    let mut styles = Vec::with_capacity(entries.len());
    for entry in entries {
        styles.push(parse_line_style_text(&entry)?);
    }
    if styles.is_empty() {
        return Err(invalid("scatterhist: LineStyle must not be empty"));
    }
    Ok(styles)
}

pub(super) fn parse_line_style_text(text: &str) -> BuiltinResult<LineStyle> {
    match text.trim().to_ascii_lowercase().as_str() {
        "-" | "solid" => Ok(LineStyle::Solid),
        "--" | "dashed" => Ok(LineStyle::Dashed),
        ":" | "dotted" => Ok(LineStyle::Dotted),
        "-." | "dashdot" => Ok(LineStyle::DashDot),
        "none" => Ok(LineStyle::None),
        other => Err(invalid(format!(
            "scatterhist: unsupported LineStyle value '{other}'"
        ))),
    }
}

pub(super) fn style_text_entries(value: &Value, option_name: &str) -> BuiltinResult<Vec<String>> {
    match value {
        Value::String(runmat_value::StringElement::Text(text)) => {
            Ok(vec![text.as_str().to_owned()])
        }
        Value::StringArray(array) => array
            .data()
            .iter()
            .map(|text| {
                text.as_str().map(str::to_owned).ok_or_else(|| {
                    invalid(format!(
                        "scatterhist: {option_name} must be nonmissing text"
                    ))
                })
            })
            .collect(),
        Value::CharArray(chars) if chars.rows > 1 => (0..chars.rows)
            .map(|row| -> BuiltinResult<String> {
                let start = row * chars.cols;
                let end = start + chars.cols;
                crate::builtins::stats::decode_utf16_with_error(&chars.data()[start..end], || {
                    invalid(format!(
                        "scatterhist: {option_name} row is not valid UTF-16 text"
                    ))
                })
                .map(|text| text.trim().to_string())
            })
            .collect(),
        Value::CharArray(chars) => Ok(vec![crate::builtins::stats::decode_utf16_with_error(
            chars.data(),
            || {
                invalid(format!(
                    "scatterhist: {option_name} is not valid UTF-16 text"
                ))
            },
        )?]),
        Value::Cell(cell) => cell
            .data()
            .iter()
            .map(|entry| {
                scalar_text(entry)?.ok_or_else(|| {
                    invalid(format!(
                        "scatterhist: {option_name} cell entries must be text"
                    ))
                })
            })
            .collect(),
        _ => Err(invalid(format!(
            "scatterhist: {option_name} must be text, string array, or cell text"
        ))),
    }
}

pub(super) fn parse_color_cycle(value: &Value) -> BuiltinResult<Vec<Vec4>> {
    if let Some(text) = scalar_text(value)? {
        let tokens = text
            .chars()
            .filter(|ch| !ch.is_whitespace())
            .collect::<Vec<_>>();
        if tokens.len() > 1 && tokens.iter().all(|ch| color_char(*ch).is_some()) {
            return tokens
                .into_iter()
                .map(|ch| {
                    color_char(ch).ok_or_else(|| {
                        invalid(format!("scatterhist: unsupported Color character '{ch}'"))
                    })
                })
                .collect();
        }
        return Ok(vec![parse_color_text_or_value(value)?]);
    }
    match value {
        Value::StringArray(array) => array
            .data()
            .iter()
            .map(|entry| {
                entry
                    .as_str()
                    .ok_or_else(|| invalid("scatterhist: Color must be nonmissing text"))
                    .and_then(parse_color_text)
            })
            .collect(),
        Value::Cell(cell) => cell.data().iter().map(parse_color_text_or_value).collect(),
        _ => {
            let tensor = tensor::value_to_tensor(value)
                .map_err(|_| invalid("scatterhist: Color must be text or an RGB matrix"))?;
            let values = tensor_values_f64(&tensor);
            let (rows, cols) = controls::control_matrix_dimensions(&tensor);
            if rows > 0 && cols == 3 {
                let mut colors = Vec::with_capacity(rows);
                for row in 0..rows {
                    colors.push(rgb_color(
                        values[row],
                        values[rows + row],
                        values[2 * rows + row],
                    )?);
                }
                return Ok(colors);
            }
            if values.len() == 3 {
                return Ok(vec![rgb_color(values[0], values[1], values[2])?]);
            }
            Err(invalid(
                "scatterhist: Color must be a color token, RGB triplet, or n-by-3 RGB matrix",
            ))
        }
    }
}

pub(super) fn scalar_text(value: &Value) -> BuiltinResult<Option<String>> {
    Ok(match value {
        Value::String(text) => text.as_str().map(str::to_owned),
        Value::StringArray(array) if array.data().len() == 1 => {
            array.data()[0].as_str().map(str::to_owned)
        }
        Value::CharArray(chars) if chars.rows == 1 => Some(
            crate::builtins::stats::decode_utf16_with_error(chars.data(), || {
                invalid("scatterhist: text value is not valid UTF-16")
            })?,
        ),
        _ => None,
    })
}

pub(super) fn parse_color_text_or_value(value: &Value) -> BuiltinResult<Vec4> {
    if let Some(text) = scalar_text(value)? {
        return parse_color_text(&text);
    }
    parse_color_value(&LineStyleParseOptions::generic(NAME), value)
        .map_err(|err| invalid(format!("scatterhist: {}", err.message)))
}

pub(super) fn parse_color_text(text: &str) -> BuiltinResult<Vec4> {
    let trimmed = text.trim();
    if let Some(color) =
        color_char(trimmed.chars().next().unwrap_or('\0')).filter(|_| trimmed.chars().count() == 1)
    {
        return Ok(color);
    }
    parse_color_value(
        &LineStyleParseOptions::generic(NAME),
        &Value::string(trimmed.to_string()),
    )
    .map_err(|err| invalid(format!("scatterhist: {}", err.message)))
}

pub(super) fn color_char(ch: char) -> Option<Vec4> {
    match ch.to_ascii_lowercase() {
        'y' => Some(Vec4::new(1.0, 1.0, 0.0, 1.0)),
        'm' => Some(Vec4::new(1.0, 0.0, 1.0, 1.0)),
        'c' => Some(Vec4::new(0.0, 1.0, 1.0, 1.0)),
        'r' => Some(Vec4::new(1.0, 0.0, 0.0, 1.0)),
        'g' => Some(Vec4::new(0.0, 1.0, 0.0, 1.0)),
        'b' => Some(Vec4::new(0.0, 0.0, 1.0, 1.0)),
        'w' => Some(Vec4::new(1.0, 1.0, 1.0, 1.0)),
        'k' => Some(Vec4::new(0.0, 0.0, 0.0, 1.0)),
        _ => None,
    }
}

pub(super) fn rgb_color(r: f64, g: f64, b: f64) -> BuiltinResult<Vec4> {
    if !(0.0..=1.0).contains(&r) || !(0.0..=1.0).contains(&g) || !(0.0..=1.0).contains(&b) {
        return Err(invalid(
            "scatterhist: RGB color values must be in the range [0,1]",
        ));
    }
    Ok(Vec4::new(r as f32, g as f32, b as f32, 1.0))
}

pub(super) fn parse_parent(value: &Value) -> BuiltinResult<FigureHandle> {
    match resolve_plot_handle(value, NAME) {
        Ok(PlotHandle::Figure(handle)) => Ok(handle),
        Ok(PlotHandle::Axes(handle, _)) => Ok(handle),
        Ok(_) => Err(invalid(
            "scatterhist: Parent must be a figure or axes handle in this build",
        )),
        Err(err) => Err(invalid(format!("scatterhist: {}", err.message))),
    }
}
