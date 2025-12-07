use glam::Vec4;
use runmat_builtins::{CellArray, Tensor, Value};
use runmat_plot::plots::{LineStyle, MarkerStyle as PlotMarkerStyle};

#[derive(Clone, Copy, Debug)]
pub struct LineStyleParseOptions {
    pub builtin_name: &'static str,
    pub forbid_leading_numeric: bool,
    pub forbid_interleaved_numeric: bool,
}

impl LineStyleParseOptions {
    pub const fn plot() -> Self {
        Self {
            builtin_name: "plot",
            forbid_leading_numeric: true,
            forbid_interleaved_numeric: false,
        }
    }

    pub const fn scatter() -> Self {
        Self {
            builtin_name: "scatter",
            forbid_leading_numeric: true,
            forbid_interleaved_numeric: true,
        }
    }

    pub const fn scatter3() -> Self {
        Self {
            builtin_name: "scatter3",
            forbid_leading_numeric: true,
            forbid_interleaved_numeric: true,
        }
    }

    pub const fn stairs() -> Self {
        Self {
            builtin_name: "stairs",
            forbid_leading_numeric: true,
            forbid_interleaved_numeric: true,
        }
    }
}

pub const DEFAULT_LINE_WIDTH: f32 = 2.0;
pub const DEFAULT_LINE_COLOR: Vec4 = Vec4::new(0.0, 0.4, 0.8, 1.0);

fn ctx_err(opts: &LineStyleParseOptions, msg: impl Into<String>) -> String {
    format!("{}: {}", opts.builtin_name, msg.into())
}

#[derive(Clone, Debug)]
pub struct LineAppearance {
    pub color: Vec4,
    pub line_width: f32,
    pub line_style: LineStyle,
    pub marker: Option<MarkerAppearance>,
}

impl Default for LineAppearance {
    fn default() -> Self {
        Self {
            color: DEFAULT_LINE_COLOR,
            line_width: DEFAULT_LINE_WIDTH,
            line_style: LineStyle::Solid,
            marker: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct MarkerAppearance {
    pub kind: MarkerKind,
    pub size: Option<f32>,
    pub edge_color: MarkerColor,
    pub face_color: MarkerColor,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MarkerKind {
    Circle,
    Plus,
    Star,
    Point,
    Cross,
    TriangleUp,
    TriangleDown,
    TriangleLeft,
    TriangleRight,
    Square,
    Diamond,
    Pentagram,
    Hexagram,
}

impl MarkerKind {
    pub fn to_plot_marker(self) -> PlotMarkerStyle {
        match self {
            MarkerKind::Circle => PlotMarkerStyle::Circle,
            MarkerKind::Plus => PlotMarkerStyle::Plus,
            MarkerKind::Star => PlotMarkerStyle::Star,
            MarkerKind::Point => PlotMarkerStyle::Circle,
            MarkerKind::Cross => PlotMarkerStyle::Cross,
            MarkerKind::TriangleUp => PlotMarkerStyle::Triangle,
            MarkerKind::TriangleDown => PlotMarkerStyle::Triangle,
            MarkerKind::TriangleLeft => PlotMarkerStyle::Triangle,
            MarkerKind::TriangleRight => PlotMarkerStyle::Triangle,
            MarkerKind::Square => PlotMarkerStyle::Square,
            MarkerKind::Diamond => PlotMarkerStyle::Diamond,
            MarkerKind::Pentagram => PlotMarkerStyle::Star,
            MarkerKind::Hexagram => PlotMarkerStyle::Star,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum MarkerColor {
    Auto,
    None,
    Flat,
    Color(Vec4),
}

#[derive(Default, Clone)]
struct LineStyleOptions {
    line_style: Option<LineStyle>,
    color: Option<Vec4>,
    line_width: Option<f32>,
    requires_cpu_fallback: bool,
    marker_kind: Option<MarkerKind>,
    marker_disabled: bool,
    marker_size: Option<f32>,
    marker_edge_color: Option<MarkerColor>,
    marker_face_color: Option<MarkerColor>,
    line_style_order: Option<Vec<LineStyle>>,
}

impl LineStyleOptions {
    fn merge(&mut self, other: LineStyleOptions) {
        if other.line_style.is_some() {
            self.line_style = other.line_style;
        }
        if other.color.is_some() {
            self.color = other.color;
        }
        if other.line_width.is_some() {
            self.line_width = other.line_width;
        }
        self.requires_cpu_fallback |= other.requires_cpu_fallback;
        if other.marker_disabled {
            self.marker_disabled = true;
            self.marker_kind = None;
        } else if other.marker_kind.is_some() {
            self.marker_kind = other.marker_kind;
            self.marker_disabled = false;
        }
        if other.marker_size.is_some() {
            self.marker_size = other.marker_size;
        }
        if other.marker_edge_color.is_some() {
            self.marker_edge_color = other.marker_edge_color;
        }
        if other.marker_face_color.is_some() {
            self.marker_face_color = other.marker_face_color;
        }
        if other.line_style_order.is_some() {
            self.line_style_order = other.line_style_order;
        }
    }

    fn resolve(&self) -> LineAppearance {
        let mut appearance = LineAppearance::default();
        if let Some(color) = self.color {
            appearance.color = color;
        }
        if let Some(width) = self.line_width {
            appearance.line_width = width;
        }
        if let Some(style) = self.line_style {
            appearance.line_style = style;
        }
        if !self.marker_disabled {
            if let Some(kind) = self.marker_kind {
                appearance.marker = Some(MarkerAppearance {
                    kind,
                    size: self.marker_size,
                    edge_color: self.marker_edge_color.clone().unwrap_or(MarkerColor::Auto),
                    face_color: self.marker_face_color.clone().unwrap_or(MarkerColor::Auto),
                });
            }
        }
        appearance
    }
}

#[derive(Clone, Debug)]
pub struct ParsedLineStyle {
    pub appearance: LineAppearance,
    pub requires_cpu_fallback: bool,
    pub line_style_explicit: bool,
    pub line_style_order: Option<Vec<LineStyle>>,
}

pub fn parse_line_style_args(
    rest: &[Value],
    opts: &LineStyleParseOptions,
) -> Result<ParsedLineStyle, String> {
    if rest.is_empty() {
        return Ok(ParsedLineStyle {
            appearance: LineAppearance::default(),
            requires_cpu_fallback: false,
            line_style_explicit: false,
            line_style_order: None,
        });
    }

    if opts.forbid_leading_numeric && begins_with_numeric(rest) {
        return Err(ctx_err(
            opts,
            "additional data series must precede style arguments",
        ));
    }

    let mut options = LineStyleOptions::default();
    let mut idx = 0usize;
    if let Some(token) = rest.get(0).and_then(value_as_string) {
        if is_style_token(&token) {
            options.merge(parse_style_string(opts, &token)?);
            idx = 1;
        }
    }

    let remaining = &rest[idx..];
    if !remaining.is_empty() {
        if opts.forbid_interleaved_numeric && begins_with_numeric(remaining) {
            return Err(ctx_err(
                opts,
                "per-series styles interleaved with data are not supported yet",
            ));
        }
        if remaining.len() % 2 != 0 {
            return Err(ctx_err(opts, "name-value arguments must come in pairs"));
        }
        options.merge(parse_name_value_pairs(opts, remaining)?);
    }

    let appearance = options.resolve();
    Ok(ParsedLineStyle {
        requires_cpu_fallback: options.requires_cpu_fallback
            || appearance.line_style != LineStyle::Solid,
        appearance,
        line_style_explicit: options.line_style.is_some(),
        line_style_order: options.line_style_order.clone(),
    })
}

fn begins_with_numeric(rest: &[Value]) -> bool {
    matches!(
        rest.first(),
        Some(
            Value::Tensor(_) | Value::GpuTensor(_) | Value::Num(_) | Value::Int(_) | Value::Bool(_)
        )
    )
}

fn parse_style_string(
    opts: &LineStyleParseOptions,
    token: &str,
) -> Result<LineStyleOptions, String> {
    let mut options = LineStyleOptions::default();
    let mut chars = token.chars().peekable();
    while let Some(ch) = chars.next() {
        match ch {
            '-' => {
                if let Some(next) = chars.peek() {
                    match next {
                        '-' => {
                            chars.next();
                            options.line_style = Some(LineStyle::Dashed);
                        }
                        '.' => {
                            chars.next();
                            options.line_style = Some(LineStyle::DashDot);
                        }
                        _ => options.line_style = Some(LineStyle::Solid),
                    }
                } else {
                    options.line_style = Some(LineStyle::Solid);
                }
            }
            ':' => {
                options.line_style = Some(LineStyle::Dotted);
            }
            'o' | '+' | '*' | '.' | 'x' | '^' | 'v' | '<' | '>' | 's' | 'd' | 'p' | 'h' => {
                if let Some(kind) = marker_kind_from_token(ch) {
                    options.marker_kind = Some(kind);
                }
            }
            c if color_from_token(c).is_some() => {
                options.color = color_from_token(c);
            }
            _ => {
                return Err(ctx_err(opts, format!("unrecognised style token `{ch}`")));
            }
        }
    }
    Ok(options)
}

fn parse_name_value_pairs(
    opts: &LineStyleParseOptions,
    args: &[Value],
) -> Result<LineStyleOptions, String> {
    let mut options = LineStyleOptions::default();
    let mut iter = args.chunks_exact(2);
    for pair in iter.by_ref() {
        let key = value_as_string(&pair[0])
            .ok_or_else(|| ctx_err(opts, "option names must be char arrays or strings"))?;
        let lower = key.trim().to_ascii_lowercase();
        match lower.as_str() {
            "linewidth" => {
                let width = value_as_f64(&pair[1])
                    .ok_or_else(|| ctx_err(opts, "LineWidth must be numeric"))?;
                if width <= 0.0 {
                    return Err(ctx_err(opts, "LineWidth must be positive"));
                }
                options.line_width = Some(width as f32);
            }
            "color" => {
                options.color = Some(parse_color_value(opts, &pair[1])?);
            }
            "linestyle" => {
                let style_text = value_as_string(&pair[1])
                    .ok_or_else(|| ctx_err(opts, "LineStyle must be a string"))?;
                options.line_style = Some(parse_line_style_name(opts, &style_text)?);
            }
            "marker" => {
                let marker_text = value_as_string(&pair[1])
                    .ok_or_else(|| ctx_err(opts, "Marker must be a string"))?;
                let parsed = parse_marker_string(opts, &marker_text)?;
                match parsed {
                    Some(kind) => {
                        options.marker_kind = Some(kind);
                        options.marker_disabled = false;
                    }
                    None => {
                        options.marker_kind = None;
                        options.marker_disabled = true;
                    }
                }
            }
            "markersize" => {
                let size = value_as_f64(&pair[1])
                    .ok_or_else(|| ctx_err(opts, "MarkerSize must be numeric"))?;
                if size <= 0.0 {
                    return Err(ctx_err(opts, "MarkerSize must be positive"));
                }
                options.marker_size = Some(size as f32);
            }
            "markeredgecolor" => {
                options.marker_edge_color = Some(parse_marker_color_value(opts, &pair[1])?);
            }
            "markerfacecolor" => {
                options.marker_face_color = Some(parse_marker_color_value(opts, &pair[1])?);
            }
            "linestyleorder" => {
                let order = parse_line_style_order_value(opts, &pair[1])?;
                options.line_style_order = Some(order);
            }
            other => {
                return Err(ctx_err(opts, format!("unsupported option `{other}`")));
            }
        }
    }
    Ok(options)
}

fn parse_line_style_name(opts: &LineStyleParseOptions, value: &str) -> Result<LineStyle, String> {
    match value.trim() {
        "-" => Ok(LineStyle::Solid),
        "--" => Ok(LineStyle::Dashed),
        ":" => Ok(LineStyle::Dotted),
        "-." => Ok(LineStyle::DashDot),
        other => Err(ctx_err(opts, format!("unsupported LineStyle `{other}`"))),
    }
}

fn parse_marker_string(
    opts: &LineStyleParseOptions,
    value: &str,
) -> Result<Option<MarkerKind>, String> {
    let trimmed = value.trim();
    if trimmed.eq_ignore_ascii_case("none") {
        return Ok(None);
    }
    if trimmed.is_empty() {
        return Err(ctx_err(opts, "Marker cannot be empty"));
    }
    if let Some(kind) = trimmed.chars().next().and_then(marker_kind_from_token) {
        return Ok(Some(kind));
    }
    Err(ctx_err(opts, format!("unsupported Marker `{value}`")))
}

fn marker_kind_from_token(token: char) -> Option<MarkerKind> {
    match token {
        'o' | 'O' => Some(MarkerKind::Circle),
        '+' => Some(MarkerKind::Plus),
        '*' => Some(MarkerKind::Star),
        '.' => Some(MarkerKind::Point),
        'x' | 'X' => Some(MarkerKind::Cross),
        '^' => Some(MarkerKind::TriangleUp),
        'v' => Some(MarkerKind::TriangleDown),
        '<' => Some(MarkerKind::TriangleLeft),
        '>' => Some(MarkerKind::TriangleRight),
        's' | 'S' => Some(MarkerKind::Square),
        'd' | 'D' => Some(MarkerKind::Diamond),
        'p' | 'P' => Some(MarkerKind::Pentagram),
        'h' | 'H' => Some(MarkerKind::Hexagram),
        _ => None,
    }
}

fn parse_marker_color_value(
    opts: &LineStyleParseOptions,
    value: &Value,
) -> Result<MarkerColor, String> {
    if let Some(name) = value_as_string(value) {
        let lower = name.trim().to_ascii_lowercase();
        return match lower.as_str() {
            "auto" => Ok(MarkerColor::Auto),
            "flat" => Ok(MarkerColor::Flat),
            "none" => Ok(MarkerColor::None),
            _ => Ok(MarkerColor::Color(parse_color_value(opts, value)?)),
        };
    }
    Ok(MarkerColor::Color(parse_color_value(opts, value)?))
}

fn parse_line_style_order_value(
    opts: &LineStyleParseOptions,
    value: &Value,
) -> Result<Vec<LineStyle>, String> {
    let strings = match value {
        Value::StringArray(arr) => arr.data.clone(),
        Value::Cell(cell) => {
            collect_cell_strings(cell).map_err(|e| ctx_err(opts, format!("LineStyleOrder: {e}")))?
        }
        _ => {
            if let Some(text) = value_as_string(value) {
                vec![text]
            } else {
                return Err(ctx_err(
                    opts,
                    "LineStyleOrder must be a string, string array, or cell array of strings",
                ));
            }
        }
    };

    if strings.is_empty() {
        return Err(ctx_err(opts, "LineStyleOrder cannot be empty"));
    }

    let mut styles = Vec::with_capacity(strings.len());
    for entry in strings {
        styles.push(parse_line_style_name(opts, &entry)?);
    }
    Ok(styles)
}

fn collect_cell_strings(cell: &CellArray) -> Result<Vec<String>, String> {
    let mut strings = Vec::with_capacity(cell.rows * cell.cols);
    for row in 0..cell.rows {
        for col in 0..cell.cols {
            let value = cell.get(row, col)?;
            if let Some(text) = value_as_string(&value) {
                strings.push(text);
            } else {
                return Err(format!(
                    "cell entry at ({row}, {col}) is not a char array or string"
                ));
            }
        }
    }
    Ok(strings)
}

pub(crate) fn parse_color_value(
    opts: &LineStyleParseOptions,
    value: &Value,
) -> Result<Vec4, String> {
    if let Some(name) = value_as_string(value) {
        if let Some(color) = name.trim().chars().next().and_then(color_from_token) {
            return Ok(color);
        }
        return Err(ctx_err(
            opts,
            format!("unsupported color specification `{name}`"),
        ));
    }
    let tensor = Tensor::try_from(value).map_err(|e| ctx_err(opts, e))?;
    if tensor.data.len() != 3 {
        return Err(ctx_err(opts, "color vectors must contain three elements"));
    }
    Ok(Vec4::new(
        tensor.data[0] as f32,
        tensor.data[1] as f32,
        tensor.data[2] as f32,
        1.0,
    ))
}

pub(crate) fn color_from_token(token: char) -> Option<Vec4> {
    match token {
        'r' | 'R' => Some(Vec4::new(1.0, 0.0, 0.0, 1.0)),
        'g' | 'G' => Some(Vec4::new(0.0, 1.0, 0.0, 1.0)),
        'b' | 'B' => Some(Vec4::new(0.0, 0.0, 1.0, 1.0)),
        'c' | 'C' => Some(Vec4::new(0.0, 1.0, 1.0, 1.0)),
        'm' | 'M' => Some(Vec4::new(1.0, 0.0, 1.0, 1.0)),
        'y' | 'Y' => Some(Vec4::new(1.0, 1.0, 0.0, 1.0)),
        'k' | 'K' => Some(Vec4::new(0.0, 0.0, 0.0, 1.0)),
        'w' | 'W' => Some(Vec4::new(1.0, 1.0, 1.0, 1.0)),
        _ => None,
    }
}

fn is_style_token(token: &str) -> bool {
    !token.trim().is_empty() && !looks_like_option_name(token)
}

pub fn looks_like_option_name(token: &str) -> bool {
    matches!(
        token.trim().to_ascii_lowercase().as_str(),
        "linewidth"
            | "color"
            | "linestyle"
            | "marker"
            | "markersize"
            | "markeredgecolor"
            | "markerfacecolor"
            | "linestyleorder"
    )
}

pub fn value_as_string(value: &Value) -> Option<String> {
    match value {
        Value::CharArray(chars) => Some(chars.data.iter().collect()),
        Value::String(s) => Some(s.clone()),
        _ => None,
    }
}

pub(crate) fn value_as_f64(value: &Value) -> Option<f64> {
    match value {
        Value::Num(v) => Some(*v),
        Value::Int(i) => Some(i.to_f64()),
        Value::Tensor(tensor) => tensor.data.first().copied(),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn style_string_with_marker_no_longer_forces_cpu_fallback() {
        let rest = vec![Value::String("o--".into())];
        let opts = LineStyleParseOptions::plot();
        let parsed = parse_line_style_args(&rest, &opts).expect("parsed");
        assert!(parsed.appearance.marker.is_some());
        assert!(!parsed.requires_cpu_fallback);
    }

    #[test]
    fn marker_none_disables_marker_without_fallback() {
        let rest = vec![Value::String("Marker".into()), Value::String("none".into())];
        let opts = LineStyleParseOptions::plot();
        let parsed = parse_line_style_args(&rest, &opts).expect("parsed");
        assert!(parsed.appearance.marker.is_none());
        assert!(!parsed.requires_cpu_fallback);
    }

    #[test]
    fn marker_color_flat_literal_is_supported() {
        let opts = LineStyleParseOptions::plot();
        let color = parse_marker_color_value(&opts, &Value::String("flat".into())).unwrap();
        assert_eq!(color, MarkerColor::Flat);
    }
}
